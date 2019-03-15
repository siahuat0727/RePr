'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import math
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from models import Vanilla
from scipy.linalg import qr
from tensorboardX import SummaryWriter

def test_filter_sparsity(model):
    for name, W in model.named_parameters():
        if W.dim() != 4:
            continue
        total = W.size(0)
        zero = sum(w.nonzero().size(0) == 0 for w in W)
        print("filter sparsity of layer {} is {}".format(name, zero/total))

def qr_null(A, tol=None):
    Q, R, _ = qr(A.T, mode='full', pivoting=True)
    tol = np.finfo(R.dtype).eps if tol is None else tol
    rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
    return Q[:, rnk:].conj()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--repr', action='store_true', help="whether to use RePr training scheme")
parser.add_argument('--S1', type=int, default=20, help="S1 epochs for RePr")
parser.add_argument('--S2', type=int, default=10, help="S2 epochs for RePr")
parser.add_argument('--epochs', type=int, default=100, help="total epochs for training")
parser.add_argument('--workers', type=int, default=16, help="number of worker to load data")
parser.add_argument('--print_freq', type=int, default=50, help="print frequency")
parser.add_argument('--gpu', type=int, default=6, help="gpu id")
parser.add_argument('--save_model', type=str, default='best.pt', help="path to save model")
parser.add_argument('--prune_ratio', type=float, default=0.3, help="prune ratio")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy


def train(train_loader, criterion, optimizer, epoch, model, writer, mask, args):
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

        if device == 'cuda' and args.gpu is not None:
            data = data.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)


        # compute output
        output = model(data)

        ce_loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))

        losses.update(ce_loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        ce_loss.backward()

        if args.repr and any(s1 <= epoch < s1+args.S2 for s1 in range(args.S1, args.epochs, args.S1+args.S2)):
            if i == 0:
                print('freeze for this epoch')
            with torch.no_grad():
                for name, W in model.named_parameters():
                    if W.dim() != 4:
                        continue
                    mask_one = torch.from_numpy(mask[name].astype(np.float32)).cuda()
                    W.grad *= mask_one.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(W)


        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'LR {lr:.3f}\t'
                  .format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, lr=optimizer.param_groups[0]['lr']))
    writer.add_scalar('Train/Acc', top1.avg, epoch)
    writer.add_scalar('Train/Loss', losses.avg, epoch)



def validate(val_loader, criterion, model, writer, args, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            if device == 'cuda' and args.gpu is not None:
                data = data.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1,5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      .format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

        print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))

        writer.add_scalar('Test/Acc', top1.avg, epoch)
        writer.add_scalar('Test/Loss', losses.avg, epoch)
        global best_acc
        if top1.avg.item() > best_acc:
            best_acc = top1.avg.item()
            print('new best_acc is {top1.avg:.3f}'.format(top1=top1))
            print('saving model {}'.format(args.save_model))
            torch.save(model.state_dict(), args.save_model)

    return top1.avg



def pruning(model, prune_ratio):
    print('Pruning...')
    l = []
    for name, W in model.named_parameters():
        if W.dim() != 4:
            continue
        # calculate inter-filter orthogonality
        size = W.size()
        W2d = W.view(size[0], -1)
        W2d = F.normalize(W2d, p=2, dim=1)
        W_WT = torch.mm(W2d, W2d.transpose(0, 1))
        I = torch.eye(W_WT.size()[0], dtype=torch.float32).cuda()
        P = W_WT - I
        P = P.sum(dim=1)
        P /= size[0]
        print(P.size())
        l.append(P.cpu().detach().numpy())
    # the ranking is computed overall the filters in the network
    l = np.vstack(l)
    mask_list = l < np.percentile(l, 100*(1-prune_ratio))
    mask = {}
    for name, W in model.named_parameters():
        if W.dim() != 4:
            continue
        # convert mask from list to dict
        mask[name] = mask_list[0]
        mask_list = mask_list[1:]  # pop the first row
        with torch.no_grad():
            W.data *= torch.from_numpy(mask[name].astype(np.float32)).cuda().view(-1, 1, 1, 1).expand_as(W)
    return mask

def reinitialize(mask, model):
    print('Reinitializing...')
    for name, W in model.named_parameters():
        if W.dim() != 4:
            continue
        with torch.no_grad():
            size = W.size()
            W2d = W.view(size[0], -1)
            null_space = qr_null(W2d.cpu().detach().numpy())
            null_space = torch.from_numpy(null_space).cuda()
            null_space = null_space.transpose(0, 1).view(-1, size[1], size[2], size[3])

            # https://github.com/pytorch/pytorch/blob/08891b0a4e08e2c642deac2042a02238a4d34c67/torch/nn/modules/conv.py#L40-L47
            stdv = 1. / math.sqrt(size[1]*size[2]*size[3])

            null_count = 0
            for w, keep in zip(W, mask[name]):
                if keep:
                    continue
                if null_count < null_space.size(0):
                    w.data[:] = null_space.data[null_count]
                    null_count += 1
                    w.data.clamp_(-stdv, stdv)
                    # print(w.data.max().item(), w.data.min().item(), w.mean().item(), w.var().item())
                else:
                    w.data.uniform_(-stdv, stdv)
    test_filter_sparsity(model)



def main_worker(args):
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.workers)


    # Model
    print('==> Building model..')

    model = Vanilla()
    print(model)

    if device == 'cuda':
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda()
            # model = torch.nn.DataParallel(model, device_ids = [args.gpu])
        else:
            model.cuda()
            model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr)

    writer = SummaryWriter(comment="-{}-{}".format( \
        "repr-{}-{}".format(args.S1, args.S2) if args.repr else "norepr", \
        args.epochs))

    mask = None
    for epoch in range(args.epochs):
        if args.repr:
            if any(epoch == s for s in range(args.S1, args.epochs, args.S1+args.S2)):
                mask = pruning(model, args.prune_ratio)
            if any(epoch == s for s in range(args.S1+args.S2, args.epochs, args.S1+args.S2)):
                reinitialize(mask, model)
        train(trainloader, criterion, optimizer, epoch, model, writer, mask, args)
        validate(testloader, criterion, model, writer, args, epoch)
        test_filter_sparsity(model)

    writer.close()
    print('overall  best_acc is {}'.format(best_acc))

def main():
    args = parser.parse_args()
    main_worker(args)

if __name__ == '__main__':
    main()
