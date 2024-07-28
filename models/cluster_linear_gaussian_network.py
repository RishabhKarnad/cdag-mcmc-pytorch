import torch

from .masked_linear import MaskedLinear
from utils.cdag import get_covariance_for_clustering
from utils.gauss import mv_gauss_log_prob


class ClusterLinearGaussianNetwork(torch.nn.Module):
    def __init__(self, n_vars, *, bias=True, sigma=0.1, rho=0.99, optimize_cov=False):
        super().__init__()
        self.n_vars = n_vars
        self.fc = MaskedLinear(n_vars, n_vars, bias)
        self.sigma = torch.nn.Parameter(
            torch.tensor([sigma]), requires_grad=optimize_cov)
        self.rho = torch.nn.Parameter(
            torch.tensor([rho]), requires_grad=optimize_cov)

    def logpmf(self, X, C, G):
        G_expand = C@G@C.T
        mean_expected = self.fc(X, G_expand)
        Cov = get_covariance_for_clustering(C, self.sigma, self.rho)
        return (mv_gauss_log_prob(X.unsqueeze(dim=1),
                                  mean_expected,
                                  covariance_matrix=Cov)
                .mean())

    def sample(self, n_samples=1):
        return NotImplementedError


class ClusterLinearGaussianNetworkFullCov(torch.nn.Module):
    def __init__(self, n_vars, *, bias=True, Cov=None, optimize_cov=False):
        super().__init__()
        self.n_vars = n_vars
        self.fc = MaskedLinear(n_vars, n_vars, bias)
        Cov = Cov or torch.eye(n_vars)
        self.Cov = torch.nn.Parameter(Cov, requires_grad=optimize_cov)

    def logpmf(self, X, C, G):
        G_expand = C@G@C.T
        mean_expected = self.fc(X, G_expand)
        return (mv_gauss_log_prob(X.unsqueeze(dim=1),
                                  mean_expected,
                                  covariance_matrix=self.Cov)
                .mean())

    def sample(self, n_samples=1):
        return NotImplementedError
