import numpy as np

import networkx as nx
from sklearn.metrics.cluster import rand_score, mutual_info_score

from models.cluster_linear_gaussian_network import ClusterLinearGaussianNetwork
from utils.cdag import clustering_to_matrix


"""
Clustering metrics
"""


def rand_index(C_true, C_pred):
    labels_true = np.argwhere(C_true > 0)[:, 1]
    labels_pred = np.argwhere(C_pred > 0)[:, 1]
    return rand_score(labels_true, labels_pred)


def expected_rand_index(C_true, C_samples):
    scores = [rand_index(C_true, C) for C in C_samples]
    return np.mean(scores), np.std(scores)


def mutual_information_score(C_true, C_pred):
    labels_true = np.argwhere(C_true > 0)[:, 1]
    labels_pred = np.argwhere(C_pred > 0)[:, 1]
    return mutual_info_score(labels_true, labels_pred)


def expected_mutual_information_score(C_true, C_samples):
    scores = [mutual_information_score(C_true, C) for C in C_samples]
    return np.mean(scores), np.std(scores)


"""
Graph metrics
"""


def compute_metrics(B_est, B_true):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive

        Taken from https://github.com/xunzheng/notears
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        # if not is_dag(B_est):
        #    raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    shd_wc = shd + len(pred_und)
    prc = float(len(true_pos)) / max(
        float(len(true_pos) + len(reverse) + len(false_pos)), 1.0
    )
    rec = tpr
    return {
        'fdr': fdr,
        'tpr': tpr,
        'fpr': fpr,
        'prc': prc,
        'rec': rec,
        'shd': shd,
        'shd_wc': shd_wc,
        'nnz': pred_size,
    }


def shd_expanded_graph(true_cdag, cdag):
    C, G_C = cdag
    C_true, G_C_true = true_cdag
    C = clustering_to_matrix(C)
    G_expand = C@G_C@C.T
    G_expand_true = C_true@G_C_true@C_true.T

    return compute_metrics(G_expand, G_expand_true)['shd']


def shd_expanded_mixed_graph(true_cdag, cdag):
    C, G_C = cdag
    C_true, G_C_true = true_cdag
    C = clustering_to_matrix(C)
    G_expand = C@G_C@C.T
    G_undirected = np.tril(C@C.T, -1) * -1

    G_cpdag = G_expand + G_undirected

    G_expand_true = C_true@G_C_true@C_true.T
    G_undirected_true = np.tril(C_true@C_true.T, -1) * -1

    G_cpdag_true = G_expand_true + G_undirected_true

    return compute_metrics(G_cpdag, G_cpdag_true)['shd']


def expected_shd(cdag_true, cdag_samples):
    shds = [shd_expanded_graph(cdag_true, cdag)
            for cdag in cdag_samples]
    return np.mean(shds), np.std(shds)


def expected_cpdag_shd(cdag_true, cdag_samples):
    shds = [shd_expanded_mixed_graph(cdag_true, cdag)
            for cdag in cdag_samples]
    return np.mean(shds), np.std(shds)


"""
Distribution metrics
"""


def nll(data, C, G, theta):
    n, n = theta.shape
    clgn = ClusterLinearGaussianNetwork(n)
    return -clgn.logpmf(data, theta, clustering_to_matrix(C), G)


def expected_nll(data, samples, theta):
    nlls = [nll(data, C, G, theta) for (C, G) in samples]
    return np.mean(nlls), np.std(nlls)


def mse_theta(G_C, theta, theta_true):
    C, G = G_C
    C = clustering_to_matrix(C)
    G_expand = C@G@C.T
    theta_masked = G_expand*theta
    return np.mean((theta_masked - theta_true)**2)


def expected_mse_theta(samples, theta, theta_true):
    mses = [mse_theta(sample, theta, theta_true) for sample in samples]
    return np.mean(mses), np.std(mses)
