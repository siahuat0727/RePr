'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import math
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from models import Vanilla
from average_meter import AverageMeter
from utils import qr_null, test_filter_sparsity, accuracy
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--repr', action='store_true', help="whether to use RePr training scheme")
parser.add_argument('--S1', type=int, default=20, help="S1 epochs for RePr")
parser.add_argument('--S2', type=int, default=10, help="S2 epochs for RePr")
parser.add_argument('--epochs', type=int, default=100, help="total epochs for training")
parser.add_argument('--workers', type=int, default=16, help="number of worker to load data")
parser.add_argument('--print_freq', type=int, default=50, help="print frequency")
parser.add_argument('--gpu', type=int, default=0, help="gpu id")
parser.add_argument('--save_model', type=str, default='best.pt', help="path to save model")
parser.add_argument('--prune_ratio', type=float, default=0.3, help="prune ratio")
parser.add_argument('--comment', type=str, default='', help="tag for tensorboardX event name")

def train(train_loader, criterion, optimizer, epoch, model, writer, mask, args, conv_weights):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None: # TODO None?
            data = data.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        output = model(data)

        loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))

        optimizer.zero_grad()

        loss.backward()

        S1, S2 = args.S1, args.S2
        if args.repr and any(s1 <= epoch < s1+S2 for s1 in range(S1, args.epochs, S1+S2)):
            if i == 0:
                print('freeze for this epoch')
            with torch.no_grad():
                for name, W in conv_weights:
                    W.grad[mask[name]] = 0

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'LR {lr:.3f}\t'
                  .format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1,
                      lr=optimizer.param_groups[0]['lr']))

        end = time.time()
    writer.add_scalar('Train/Acc', top1.avg, epoch)
    writer.add_scalar('Train/Loss', losses.avg, epoch)

def validate(val_loader, criterion, model, writer, args, epoch, best_acc):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            if args.gpu is not None: # TODO None?
                data = data.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      .format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))
            end = time.time()

    print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))
    writer.add_scalar('Test/Acc', top1.avg, epoch)
    writer.add_scalar('Test/Loss', losses.avg, epoch)

    if top1.avg.item() > best_acc:
        print('new best_acc is {top1.avg:.3f}'.format(top1=top1))
        print('saving model {}'.format(args.save_model))
        torch.save(model.state_dict(), args.save_model)
    return top1.avg.item()

def pruning(conv_weights, prune_ratio):
    print('Pruning...')
    # calculate inter-filter orthogonality
    inter_filter_ortho = {}
    for name, W in conv_weights:
        size = W.size()
        W2d = W.view(size[0], -1)
        W2d = F.normalize(W2d, p=2, dim=1)
        W_WT = torch.mm(W2d, W2d.transpose(0, 1))
        I = torch.eye(W_WT.size()[0], dtype=torch.float32).cuda()
        P = torch.abs(W_WT - I)
        P = P.sum(dim=1) / size[0]
        inter_filter_ortho[name] = P.cpu().detach().numpy()
    # the ranking is computed overall the filters in the network
    ranks = np.concatenate([v.flatten() for v in inter_filter_ortho.values()])
    threshold = np.percentile(ranks, 100*(1-prune_ratio))

    prune = {}
    mask = {}
    drop_filters = {}
    for name, W in conv_weights:
        prune[name] = inter_filter_ortho[name] > threshold  # e.g. [True, False, True, True, False]
        # get indice of bad filters
        mask[name] = np.where(prune[name])[0]  # e.g. [0, 2, 3]
        drop_filters[name] = None
        if mask[name].size > 0:
            with torch.no_grad():
                drop_filters[name] = W.data[mask[name]].view(mask[name].size, -1).cpu().numpy()
                W.data[mask[name]] = 0

    test_filter_sparsity(conv_weights)
    return prune, mask, drop_filters

def reinitialize(mask, drop_filters, conv_weights, fc_weights):
    print('Reinitializing...')
    with torch.no_grad():
        prev_layer_name = None
        prev_num_filters = None
        for name, W in conv_weights + fc_weights:
            if W.dim() == 4:  # conv weights
                # find null space
                size = W.size()
                W2d = W.view(size[0], -1).cpu().numpy()
                null_space = qr_null(
                    W2d if drop_filters[name] is None else np.vstack((drop_filters[name], W2d)))
                null_space = torch.from_numpy(null_space).cuda()
                null_space = null_space.transpose(0, 1).view(-1, size[1], size[2], size[3])

                # https://github.com/pytorch/pytorch/blob/08891b0a4e08e2c642deac2042a02238a4d34c67/torch/nn/modules/conv.py#L40-L47
                stdv = 1. / math.sqrt(size[1]*size[2]*size[3])

                null_count = 0
                for mask_idx in mask[name]:
                    if null_count < null_space.size(0):
                        W.data[mask_idx] = null_space.data[null_count].clamp_(-stdv, stdv)
                        null_count += 1
                    else:
                        W.data[mask_idx].uniform_(-stdv, stdv)

            # mask channels of prev-layer-pruned-filters' outputs
            if prev_layer_name is not None:
                if W.dim() == 4:  # conv
                    W.data[:, mask[prev_layer_name]] = 0
                elif W.dim() == 2: # fc
                    W.view(W.size(0), prev_num_filters, -1).data[:, mask[prev_layer_name]] = 0
            prev_layer_name, prev_num_filters = name, W.size(0)
    test_filter_sparsity(conv_weights)

def main():
    if not torch.cuda.is_available():
        raise Exception("Only support GPU training")
    cudnn.benchmark = True

    args = parser.parse_args()

    # Data
    print('==> Preparing data..')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args.workers)

    # Model
    print('==> Building model..')

    model = Vanilla()
    print(model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda()
    else:
        model.cuda()
        model = torch.nn.DataParallel(model)

    conv_weights = []
    fc_weights = []
    for name, W in model.named_parameters():
        if W.dim() == 4:
            conv_weights.append((name, W))
        elif W.dim() == 2:
            fc_weights.append((name, W))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr)
    comment = "-{}-{}-{}".format("repr" if args.repr else "norepr", args.epochs, args.comment)
    writer = SummaryWriter(comment=comment)

    mask = None
    drop_filters = None
    best_acc = 0  # best test accuracy
    prune_map = []
    for epoch in range(args.epochs):
        if args.repr:
            # check if the end of S1 stage
            if any(epoch == s for s in range(args.S1, args.epochs, args.S1+args.S2)):
                prune, mask, drop_filters = pruning(conv_weights, args.prune_ratio)
                prune_map.append(np.concatenate(list(prune.values())))
            # check if the end of S2 stage
            if any(epoch == s for s in range(args.S1+args.S2, args.epochs, args.S1+args.S2)):
                reinitialize(mask, drop_filters, conv_weights, fc_weights)
        train(trainloader, criterion, optimizer, epoch, model, writer, mask, args, conv_weights)
        acc = validate(testloader, criterion, model, writer, args, epoch, best_acc)
        best_acc = max(best_acc, acc)
        test_filter_sparsity(conv_weights)

    writer.close()
    print('overall  best_acc is {}'.format(best_acc))

    # Shows which filters turn off as training progresses
    prune_map = np.array(prune_map).transpose()
    plt.matshow(prune_map.astype(np.int), cmap=ListedColormap(['k', 'w']))
    plt.xticks(np.arange(prune_map.shape[1]))
    plt.yticks(np.arange(prune_map.shape[0]))
    plt.title('Filters on/off map\nwhite: off (pruned)\nblack: on')
    plt.xlabel('Pruning stage')
    plt.ylabel('Filter index from shallower layer to deeper layer')
    plt.savefig('{}-{}.png'.format(
        datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S'),
        comment))


if __name__ == '__main__':
    main()
