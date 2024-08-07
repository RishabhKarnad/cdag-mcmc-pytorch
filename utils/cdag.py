import numpy as np
import torch
import networkx as nx

from utils.permutation import make_permutation_matrix


def stringify_cdag(G_C):
    C, E_C = G_C
    C_indexed = list(zip(range(len(C)), C))
    C_sorted = sorted(C_indexed, key=lambda s: min(s[1]))
    C_stringified = []
    for elem in C_sorted:
        c = map(str, sorted(elem[1]))
        C_stringified.append(','.join(c))
    C_stringified = '-'.join(C_stringified)
    P = make_permutation_matrix(list(map(lambda x: x[0], C_sorted)))
    E_C_sorted = P.T @ E_C @ P
    E_stringified = ','.join(
        map(lambda x: str(int(x.item())), list(E_C_sorted.flatten())))
    cdag_stringified = C_stringified + ':' + E_stringified
    return cdag_stringified


def unstringify_cdag(cdag_string):
    C_stringified, E_C_stringified = cdag_string.split(':')
    C = C_stringified.split('-')
    C = [set(map(int, c_i.split(','))) for c_i in C]
    n = len(C)
    E_C = list(map(float, map(float, E_C_stringified.split(','))))
    E_C = torch.tensor(E_C).reshape(n, n)
    return C, E_C


def clustering_to_matrix(C, k=None):
    if not k:
        k = len(C)

    n = np.sum([len(C_i) for C_i in C])
    m = torch.zeros((n, k), dtype=torch.float32)
    for i, C_i in enumerate(C):
        for X_j in C_i:
            m[int(X_j), i] = 1
    return m


def matrix_to_clustering(C):
    C_seq = []
    d, k = C.shape
    for i in range(k):
        C_i = set()
        for j in range(d):
            if C[j, i] == 1:
                C_i.add(j)
        C_seq.append(C_i)
    return C_seq


def get_covariance_for_clustering(C, sigma=0.1, rho=0.99):
    d = C.shape[0]
    I = torch.eye(d).to(C.device)
    F = torch.ones((d, d)).to(C.device)
    Cov = sigma * (I + rho*(F - I))
    mask = C@C.T
    Cov = Cov * mask
    return Cov


def count_toposorts(M):
    g = nx.from_numpy_array(M, create_using=nx.DiGraph())
    toposorts = list(nx.all_topological_sorts(g))
    return len(toposorts)


def get_graphs_by_count(samples):
    graph_counts = {}
    for graph in samples:
        graph_string = stringify_cdag(graph)
        if graph_string not in graph_counts:
            graph_counts[graph_string] = 0
        graph_counts[graph_string] += 1
    graphs = sorted(graph_counts, key=graph_counts.get, reverse=True)
    graph_counts = [graph_counts[g] for g in graphs]
    graphs = [unstringify_cdag(g) for g in graphs]
    return graphs, graph_counts
