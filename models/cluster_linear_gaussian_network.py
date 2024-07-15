import torch
from torch.distributions import MultivariateNormal

from .masked_linear import MaskedLinear
from utils.cdag import get_covariance_for_clustering


def zero_pad(A, k):
    m, n = A.shape
    p_rows, p_cols = k - m, k - n
    if p_cols > 0:
        A = torch.hstack([A, torch.zeros((m, p_cols))])
    if p_rows > 0:
        A = torch.vstack([A, torch.zeros((p_rows, k))])
    return A


class ClusterLinearGaussianNetwork(torch.nn.Module):
    def __init__(self, n_vars, *, bias=True, sigma=0.1, rho=0.99):
        super().__init__()
        self.n_vars = n_vars
        self.fc = MaskedLinear(n_vars, n_vars, bias)
        self.sigma = sigma
        self.rho = rho

    def logpmf(self, X, C, G):
        G_expand = C@G@C.T
        mean_expected = self.fc(X, G_expand)
        Cov = get_covariance_for_clustering(C, self.sigma, self.rho)
        return (MultivariateNormal(mean_expected, covariance_matrix=Cov)
                .log_prob(X.unsqueeze(dim=1))
                .mean())
        # m, n = X.shape
        # return (MultivariateNormal(torch.zeros((n, n)), covariance_matrix=torch.eye(n))
        #         .log_prob(X.unsqueeze(dim=1))
        #         .mean())

    def sample(self, n_samples=1):
        return NotImplementedError
