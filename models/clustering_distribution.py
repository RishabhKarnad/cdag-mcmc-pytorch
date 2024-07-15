import torch


class ClusteringDistribution(torch.nn.Module):
    def __init__(self, n_vars, n_clusters, alpha=100, lambda_s=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.lambda_s = lambda_s

    def logpmf(self, C):
        return self.lambda_s * torch.prod(torch.tanh(self.alpha * torch.sum(C, axis=0)))

    def sample(self, n_samples=1):
        raise NotImplementedError
