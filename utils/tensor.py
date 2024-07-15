import torch


def zero_pad(A, k):
    m, n = A.shape
    p_rows, p_cols = k - m, k-n
    if p_cols > 0:
        A = torch.hstack([A, torch.zeros((m, p_cols))])
    if p_rows > 0:
        A = torch.vstack([A, torch.zeros((p_rows, k))])
    return A
