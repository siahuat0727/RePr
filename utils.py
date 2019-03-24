import numpy as np
from scipy.linalg import qr
import torch

def test_filter_sparsity(conv_weights):
    for name, W in conv_weights:
        zero = sum(w.nonzero().size(0) == 0 for w in W)
        print("filter sparsity of layer {} is {}".format(name, zero/W.size(0)))

def qr_null(A, tol=None):
    Q, R, _ = qr(A.T, mode='full', pivoting=True)
    tol = np.finfo(R.dtype).eps if tol is None else tol
    rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
    return Q[:, rnk:].conj()


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
