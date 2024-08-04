import numpy as np
import torch
from torch.distributions import MultivariateNormal
import networkx as nx

from utils.cdag import clustering_to_matrix, get_covariance_for_clustering


class DataGen:
    def __init__(self, obs_noise):
        self.obs_noise = obs_noise
        self.adjacency_matrix = None
        self.theta = None

    def generate_scm_data(self, n_samples=100):
        G = torch.tensor([[0.,  0.,  0., 1., 1.,  0.,  0.],
                          [0.,  0.,  0., 1.,  1.,  0.,  0.],
                          [0.,  0.,  0.,  1., 1.,  0.,  0.],
                          [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [0.,  0.,  0., 1.,  1.,  0.,  0.],
                          [0.,  0.,  0., 1.,  1.,  0.,  0.]])

        theta = torch.tensor([[0.,  0.,  0., -7., 12.,  0.,  0.],
                              [0.,  0.,  0., 10.,  2.,  0.,  0.],
                              [0.,  0.,  0.,  9., -4.,  0.,  0.],
                              [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                              [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                              [0.,  0.,  0., 26.,  4.,  0.,  0.],
                              [0.,  0.,  0., -15.,  2.,  0.,  0.]])

        mu = torch.zeros(7)
        sigma = self.obs_noise * torch.eye(7)

        Z = MultivariateNormal(mu, sigma).sample((n_samples,))

        W = theta * G

        X = Z @ torch.linalg.inv(torch.eye(7) - W)

        return X, (G, theta, sigma, [{0}, {1}, {2}, {3}, {4}, {5}, {6}], G)

    def generate_group_scm_data(self, n_samples=100, *,
                                vstruct=True, confounded=True, zero_centered=True):
        clus = [{0, 1, 2}, {3, 4}, {5, 6}]
        C = clustering_to_matrix(clus)

        if vstruct:
            G_C = torch.tensor([[0, 1, 0],
                                [0, 0, 0],
                                [0, 1, 0]], dtype=torch.float32)
        else:
            G_C = torch.tensor([[0, 1, 1],
                                [0, 0, 0],
                                [0, 0, 0]], dtype=torch.float32)

        ks = list(map(int, C.sum(axis=0)))
        n_vars = C.shape[0]

        if vstruct:
            theta = torch.tensor([[0.,  0.,  0., -7., 12.,  0.,  0.],
                                  [0.,  0.,  0., 10.,  2.,  0.,  0.],
                                  [0.,  0.,  0.,  9., -4.,  0.,  0.],
                                  [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                  [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                  [0.,  0.,  0., 26.,  4.,  0.,  0.],
                                  [0.,  0.,  0., -15.,  2.,  0.,  0.]])
        else:
            theta = torch.tensor([[0.,  0.,  0.,  7., 12., 26.,  4.],
                                  [0.,  0.,  0., 10.,  2., 15.,  2.],
                                  [0.,  0.,  0.,  9.,  4., 11., -6.],
                                  [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                  [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                  [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                  [0.,  0.,  0.,  0.,  0.,  0.,  0.]])

        if zero_centered:
            mu_1 = torch.zeros(ks[0])
            mu_2 = torch.zeros(ks[1])
            mu_3 = torch.zeros(ks[2])
        else:
            mu_1 = torch.tensor([1.0, 2.0, -1.0])
            mu_2 = torch.tensor([-2.0, 3.0])
            mu_3 = torch.tensor([3.2, 1.5])

        if confounded:
            sigma_1 = self.obs_noise * torch.tensor([[1, 0.99, 0.99],
                                                     [0.99, 1, 0.99],
                                                     [0.99, 0.99, 1]])
            sigma_2 = self.obs_noise * torch.tensor([[1, 0.99],
                                                     [0.99, 1]])
            sigma_3 = self.obs_noise * torch.tensor([[1, 0.99],
                                                     [0.99, 1]])
        else:
            sigma_1 = self.obs_noise * torch.tensor([[1, 0.5, 0.2],
                                                     [0.5, 1, 0.5],
                                                     [0.2, 0.5, 1]])
            sigma_2 = self.obs_noise * torch.tensor([[1, 0.3],
                                                     [0.3, 1]])
            sigma_3 = self.obs_noise * torch.tensor([[1, 0.45],
                                                     [0.45, 1]])

        Z_1 = MultivariateNormal(mu_1, sigma_1).sample((n_samples,))

        Z_2 = MultivariateNormal(mu_2, sigma_2).sample((n_samples,))

        Z_3 = MultivariateNormal(mu_3, sigma_3).sample((n_samples,))

        Z = torch.hstack([Z_1, Z_2, Z_3])

        G_expand = C@G_C@C.T
        W = theta * G_expand

        X = Z @ torch.linalg.inv(torch.eye(n_vars) - W)
        Cov_true = torch.zeros((7, 7))
        Cov_true[0:3, 0:3] = sigma_1
        Cov_true[3:5, 3:5] = sigma_2
        Cov_true[5:7, 5:7] = sigma_3
        return X, (G_expand, W, Cov_true, clus, G_C)

    def generate_group_scm_data_small_dag(self, n_samples=100, *,
                                          zero_centered=True):
        clus = [{0, 1}, {2}]
        C = clustering_to_matrix(clus)

        G_C = torch.tensor([[0., 1.],
                            [0., 0.]])

        ks = list(map(int, C.sum(axis=0)))
        n_vars = C.shape[0]

        theta = torch.tensor([[0., 0., 2.],
                              [0., 0., 3.],
                              [0., 0., 0.]])

        if zero_centered:
            mu_1 = torch.zeros(ks[0])
            mu_2 = torch.zeros(ks[1])
        else:
            mu_1 = torch.tensor([11., 23.])
            mu_2 = torch.tensor([31.])

        sigma_1 = self.obs_noise * torch.tensor([[1, 0.99], [0.99, 1]])
        sigma_2 = self.obs_noise * torch.tensor([[1.]])

        Z_1 = MultivariateNormal(mu_1, sigma_1).sample((n_samples,))

        Z_2 = MultivariateNormal(mu_2, sigma_2).sample((n_samples,))

        Z = torch.hstack([Z_1, Z_2])

        G_expand = C@G_C@C.T
        W = theta * G_expand

        X = Z @ torch.linalg.inv(torch.eye(n_vars) - W)
        Cov_true = torch.zeros((3, 3))
        Cov_true[0:2, 0:2] = sigma_1
        Cov_true[2:3, 2:3] = sigma_2
        return X, (G_expand, W, Cov_true, clus, G_C)

    def generate_group_scm_data_small_dag_4vars(self, n_samples=100, *,
                                                zero_centered=True):
        clus = [{0, 1}, {2, 3}]
        C = clustering_to_matrix(clus)

        G_C = torch.tensor([[0., 1.],
                            [0., 0.]])

        ks = list(map(int, C.sum(axis=0)))
        n_vars = C.shape[0]

        theta = torch.tensor([[0., 0., 2., 5.],
                              [0., 0., 3., 4.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 0.]])

        if zero_centered:
            mu_1 = torch.zeros(ks[0])
            mu_2 = torch.zeros(ks[1])
        else:
            mu_1 = torch.tensor([11., 23.])
            mu_2 = torch.tensor([31., 41.])

        sigma_1 = self.obs_noise * torch.tensor([[1, 0.99], [0.99, 1]])
        sigma_2 = self.obs_noise * torch.tensor([[1, 0.99], [0.99, 1]])

        Z_1 = MultivariateNormal(mu_1, sigma_1).sample((n_samples,))

        Z_2 = MultivariateNormal(mu_2, sigma_2).sample((n_samples,))

        Z = torch.hstack([Z_1, Z_2])

        G_expand = C@G_C@C.T
        W = theta * G_expand

        X = Z @ torch.linalg.inv(torch.eye(n_vars) - W)
        Cov_true = torch.zeros((n_vars, n_vars))
        Cov_true[0:2, 0:2] = sigma_1
        Cov_true[2:4, 2:4] = sigma_2
        return X, (G_expand, W, Cov_true, clus, G_C)

    def generate_er_dag(self, n, p):
        L = torch.tril(torch.distributions.Bernoulli(
            probs=p).sample((n, n)), -1)
        perm = torch.randperm(n)
        return L[np.ix_(perm, perm)]

    def create_clustering(self, group_sizes):
        clustering = []
        total_vars = 0
        for size in group_sizes:
            clustering.append(set(range(total_vars, size + total_vars)))
            total_vars += size
        return clustering

    @staticmethod
    def get_theta(theta, x, y):
        idx = (np.repeat(x, len(y)), np.tile(y, len(x)))
        return theta[idx].reshape((len(x), len(y)))

    def generate_random_group_scm_nonlinear_data(self, n_samples=100, *,
                                                 group_sizes=[3, 2, 2],
                                                 group_dag_density=0.2,
                                                 zero_centered=True,
                                                 sigma=1.0,
                                                 rho=0.9):
        G_C = self.generate_er_dag(len(group_sizes), group_dag_density)

        C_list = self.create_clustering(group_sizes)
        C = clustering_to_matrix(C_list, len(group_sizes))

        Cov_true = get_covariance_for_clustering(C, sigma, rho)

        G_expand = C@G_C@C.T
        n = G_expand.shape[0]

        sign = torch.randint(0, 2, (n, n)) * 2 - 1
        theta_1 = torch.distributions.Uniform(
            low=2, high=5).sample((n, n)) * sign
        theta_2 = torch.distributions.Uniform(
            low=2, high=5).sample((n, n)) * sign

        if zero_centered:
            mu = torch.zeros((2, n))
        else:
            mu = torch.randn((2, n)) * 10

        Z = torch.distributions.MultivariateNormal(
            torch.zeros(n), Cov_true).sample((n_samples,))

        X = torch.zeros((Z.shape))

        Graph_c = nx.from_numpy_array(
            G_C.numpy(), create_using=nx.MultiDiGraph())
        for node in Graph_c.nodes:
            parents = list(Graph_c.predecessors(node))
            if parents:
                idx_p, idx_ch = [
                    val for i in parents for val in C_list[i]], list(C_list[node])

                theta_l1 = self.get_theta(theta_1, idx_p, idx_ch)
                theta_l2 = self.get_theta(theta_2, idx_ch, idx_ch)

                mu_l1 = mu[0, idx_ch]
                mu_l2 = mu[1, idx_ch]

                X[:, idx_ch] = torch.relu(
                    Z[:, idx_p]@theta_l1 + mu_l1)@theta_l2+mu_l2+Z[:, idx_ch]
            else:
                idx_ch = list(C_list[node])
                mu_l2 = mu[1, idx_ch]
                X[:, idx_ch] = Z[:, idx_ch]+mu_l2

        W = torch.concatenate((theta_1 * G_expand, theta_2*(C@C.T)))

        return X, (G_expand, W, Cov_true, C_list, G_C)
