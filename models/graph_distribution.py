import torch


def h(G):
    m, m = G.shape
    return torch.trace(torch.linalg.matrix_power((torch.eye(m) + G*G), m)) - m


class SparseDAGDistribution(torch.nn.Module):
    def __init__(self, n_vars, gibbs_temp=10., sparsity_factor=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.gibbs_temp = gibbs_temp
        self.sparsity_factor = sparsity_factor

    def logpmf(self, G):
        dagness = h(G)
        return -self.gibbs_temp*dagness - self.sparsity_factor*G.sum()

    def sample(self, n_samples=1):
        raise NotImplementedError
