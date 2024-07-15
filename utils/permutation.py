import torch


def make_permutation_matrix(seq):
    """
    Generates a permutation matrix that sorts a given sequence
    """
    n = len(seq)
    P = torch.zeros((n, n), dtype=torch.float32)
    target = sorted(list(zip(range(len(seq)), seq)), key=lambda x: x[1])
    for i, x in enumerate(target):
        j = x[0]
        P[i, j] = 1
    return P
