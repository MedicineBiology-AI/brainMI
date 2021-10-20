# -*- Coding: utf-8 -*-
from torch.utils.data import Dataset
import numpy as np

from sklearn.preprocessing import minmax_scale, scale

def _scaleSimMat(A):
    """Scale rows of similarity matrix"""
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)
    A = A.astype(np.float)/col[:, None]

    return A


def RWR(A, K=3, alpha=0.98):
    """Random Walk on graph"""
    A = _scaleSimMat(A)
    # Random surfing
    n = A.shape[0]
    P0 = np.eye(n, dtype=float)
    P = P0.copy()
    M = np.zeros((n, n), dtype=float)
    for i in range(0, K):
        P = alpha*np.dot(P, A) + (1. - alpha)*P0
        M = M + P

    return M


def load_networks(path):

    networks, symbols = [], []
    for file in path:
        network = np.load(file)['corr']
        network = RWR(network)
        # network = scale(network, axis=1)
        network = minmax_scale(network)
        
        networks.append(network)
        symbols.append(np.load(file, allow_pickle=True)['symbol'])

    return networks, symbols 


class netsDataset(Dataset):
    def __init__(self, net):
        super(netsDataset, self).__init__()
        self.net = net

    def __len__(self):
        return len(self.net)

    def __getitem__(self, item):
        x = self.net[item]
        y = self.net[item]
        return x, y, item
