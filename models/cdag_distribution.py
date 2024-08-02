import torch
import scipy.stats as stats

from .clustering_distribution import ClusteringDistribution
from .graph_distribution import SparseDAGDistribution
from .cluster_linear_gaussian_network import ClusterLinearGaussianNetwork, ClusterLinearGaussianNetworkFullCov
from utils.cdag import clustering_to_matrix
from utils.dataset import CustomDataset


class CDAGJointDistribution(torch.nn.Module):
    def __init__(self,
                 n_vars,
                 min_clusters,
                 mean_clusters,
                 max_clusters,
                 optimize_cov=False,
                 full_covariance=False):
        super().__init__()
        self.n_vars = n_vars
        self.min_clusters = min_clusters
        self.mean_clusters = mean_clusters
        self.max_clusters = max_clusters
        self.cluster_prior = ClusteringDistribution(n_vars, max_clusters)
        self.graph_prior = SparseDAGDistribution(n_vars)
        if full_covariance:
            self.likelihood = ClusterLinearGaussianNetworkFullCov(n_vars)
        else:
            self.likelihood = ClusterLinearGaussianNetwork(
                n_vars, optimize_cov=optimize_cov)

    def logpmf(self, C, G, X, batch=False):
        k = len(C)
        C_mat = clustering_to_matrix(C)
        p_k = stats.randint(self.min_clusters, self.max_clusters+1).logpmf(k)
        p_c = self.cluster_prior.logpmf(C_mat)
        p_g = self.graph_prior.logpmf(G)

        p_d = 0
        if batch:
            dataset = CustomDataset(X)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=64, shuffle=True)
            for _, X_i in enumerate(dataloader):
                p_d += (self.likelihood.logpmf(X_i, C_mat, G) * X_i.shape[0])
            p_d /= len(dataset)
        else:
            p_d = self.likelihood.logpmf(X, C_mat, G)

        return p_k + p_c + p_g + p_d

    def sample(self):
        raise NotImplementedError
