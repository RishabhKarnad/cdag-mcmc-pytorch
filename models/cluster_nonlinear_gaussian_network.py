import torch

from .masked_mlp import MaskedMLP
from utils.cdag import get_covariance_for_clustering
from utils.gauss import mv_gauss_log_prob


class ClusterNonlinearGaussianNetwork(torch.nn.Module):
    def __init__(self, n_vars, n_clus,
                 *, bias=True, sigma=0.1, rho=0.99, optimize_cov=False):
        super().__init__()
        self.n_vars = n_vars
        self.n_clus = n_clus
        self.nns = torch.nn.ModuleList([MaskedMLP(n_vars, n_vars, n_vars, bias=bias)
                                        for _ in range(n_clus)])
        self.sigma = torch.nn.Parameter(
            torch.tensor([sigma]), requires_grad=optimize_cov)
        self.rho = torch.nn.Parameter(
            torch.tensor([rho]), requires_grad=optimize_cov)

    def logpmf(self, X, C, G):
        input_mask = C@G
        output_mask = C
        mean_expected_cluster_wise = [nn(X, input_mask[:, i], output_mask[:, i])
                                      for i, nn in enumerate(self.nns)]

        mean_expected = mean_expected_cluster_wise[0]
        for i in range(1, self.n_clus):
            mean_expected += mean_expected_cluster_wise[i]
        Cov = get_covariance_for_clustering(C, self.sigma, self.rho)

        return (mv_gauss_log_prob(X, mean_expected, covariance_matrix=Cov)
                .mean())

    def sample(self, n_samples=1):
        return NotImplementedError


class ClusterNonlinearGaussianNetworkFullCov(torch.nn.Module):
    def __init__(self, n_vars, n_clus,
                 *, bias=True):
        super().__init__()
        self.n_vars = n_vars
        self.n_clus = n_clus
        self.nns = torch.nn.ModuleList([MaskedMLP(n_vars, n_vars, n_vars, bias=bias)
                                        for _ in range(n_clus)])
        self.L = torch.nn.Parameter(torch.randn((n_vars, n_vars)))
        self.mask = torch.nn.Parameter(torch.tril(
            torch.ones((n_vars, n_vars))), requires_grad=False)

    def logpmf(self, X, C, G):
        input_mask = C@G
        output_mask = C
        mean_expected_cluster_wise = [nn(X, input_mask[:, i], output_mask[:, i])
                                      for i, nn in enumerate(self.nns)]

        mean_expected = mean_expected_cluster_wise[0]
        for i in range(1, self.n_clus):
            mean_expected += mean_expected_cluster_wise[i]

        L = self.L * self.mask
        Cov = L@L.T

        return (mv_gauss_log_prob(X, mean_expected, covariance_matrix=Cov)
                .mean())

    def sample(self, n_samples=1):
        return NotImplementedError
