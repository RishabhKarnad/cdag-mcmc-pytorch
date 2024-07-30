import numpy as np
import torch
from torch.distributions import MultivariateNormal
import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig
import scipy
from functools import reduce
import pandas as pd

from utils.cdag import clustering_to_matrix


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

    def viz_graph(adjacency_matrix, graph_lib='igraph'):
        G = nx.from_numpy_array(
            adjacency_matrix, create_using=nx.MultiDiGraph())
        g = ig.Graph.from_networkx(G)
        g.vs['label'] = g.vs['_nx_name']

        if graph_lib == 'igraph':
            fig, ax = plt.subplots()
            ig.plot(g, target=ax)
        elif graph_lib == 'nx':
            nx.draw_circular(G, with_labels=True)

    def compute_fullcov(self):
        n_vars = self.theta.shape[0]
        lambda_ = torch.linalg.inv(torch.eye(n_vars) - self.theta)
        D = self.obs_noise * torch.eye(n_vars)
        Cov = lambda_.T@D@lambda_
        return Cov

    def compute_noisycov(self, C):
        # C: nvars x nclus
        nvars, nclus = C.shape
        Cov_true = self.compute_fullcov()
        G_cov = C@C.T
        G_anticov = torch.ones((nvars, nvars))-G_cov
        Cov_mask_true = G_cov*Cov_true
        Cov_noise = scipy.stats.wishart.rvs(nvars, 100, size=(5, 5))
        Cov_mask_true_noise = Cov_mask_true+G_anticov*Cov_noise
        return Cov_mask_true_noise, Cov_mask_true

    def get_index(self, arr):
        len_ = len(arr)
        idx = torch.zeros((len_*len_, 2))
        idx[:, 0] = np.reshape(np.tile(np.c_[arr], len_), len_*len_)
        idx[:, 1] = np.tile(arr, len_)
        idx = idx.astype(np.int64)
        return idx

    def compute_joint2condcov(self, adj, C_true, Cov_joint):
        C_group = [np.where(C_j > 0)[0].tolist() for C_j in C_true.T]
        G = nx.from_numpy_array(adj, create_using=nx.MultiDiGraph())
        Cond_Cov = []
        for node in G.nodes:
            parents = list(G.predecessors(node))
            if parents:
                node_group = C_group[node]
                parent_group = sorted(sum([C_group[i] for i in parents], []))
                arr = torch.tensor(node_group+parent_group)
                len_ = len(arr)
                split_index = len(node_group)
                idx = self.get_index(arr)
                Cov_full = Cov_joint[(idx[:, 0], idx[:, 1])]
                Cov_full = Cov_full.reshape((len_, len_))
                # CovX|Y=CovX-beta*CovY*beta_T
                # beta=CovXY*CovY-1
                beta = Cov_full[:split_index, split_index:]@torch.linalg.inv(
                    Cov_full[split_index:, split_index:])
                Cov_ = Cov_full[:split_index, :split_index] - \
                    beta@Cov_full[split_index:, split_index:]@beta.T
            else:
                node_group = C_group[node]
                arr = torch.tensor(node_group)
                len_ = len(arr)
                idx = self.get_index(arr)
                Cov_full = Cov_joint[(idx[:, 0], idx[:, 1])]
                Cov_full = Cov_full.reshape((len_, len_))
                Cov_ = Cov_full

            Cond_Cov.append(Cov_)
            Cond_Cov_mat = reduce(
                lambda a, b: scipy.linalg.block_diag(a, b), Cond_Cov)

        return Cond_Cov_mat
