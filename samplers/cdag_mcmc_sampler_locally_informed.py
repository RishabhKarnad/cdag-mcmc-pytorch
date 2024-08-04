import numpy as np
import torch
from scipy.special import comb
from itertools import combinations
from copy import deepcopy
from tqdm import tqdm

from models import UpperTriangular

from utils.sys import debugger_is_active
from utils.cdag import matrix_to_clustering, clustering_to_matrix, stringify_cdag, unstringify_cdag


MAX_PARENTS = 2
MAX_SUBSET_SIZE = 3

N_WARMUP = 100
N_SAMPLES = 500


def safe_set(ns):
    if ns.size == 1:
        return {ns.item()}
    return set(ns)


class ClusteringProposalDistribution:
    def __init__(self, C, min_clusters, max_clusters):
        self.C = C
        self.k = len(C)

        self.neighbours = []
        self.neighbour_counts = []
        self.total_neighbours = 0

        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

        self.populate_neighbours()

    def populate_neighbours(self):
        # No move
        self.neighbours.append(f'no_move')
        self.neighbour_counts.append(1)

        # Exchanges
        for i in range(self.k-1):
            for c1 in range(1, max(MAX_SUBSET_SIZE, len(self.C[i])+1)):
                for c2 in range(1, max(MAX_SUBSET_SIZE, len(self.C[i+1])+1)):
                    combs_c1 = list(combinations(self.C[i], c1))
                    combs_c2 = list(combinations(self.C[i+1], c2))
                    for comb_c1 in combs_c1:
                        for comb_c2 in combs_c2:
                            self.neighbours.append(
                                f'exchange-{i}-{','.join(map(str, comb_c1))}-{','.join(map(str, comb_c2))}')

                    self.neighbour_counts.append(
                        (comb(len(self.C[i]), c1) * comb(len(self.C[i+1]), c2)))

        self.total_neighbours = sum(self.neighbour_counts)
        assert (self.total_neighbours == len(self.neighbours))

    def gen_neighbour(self, spec):
        neighbour = deepcopy(self.C)

        match spec.split('-'):
            case ['no_move']:
                pass
            case ['exchange', i, c1, c2]:
                (i, c1, c2) = (int(i),
                               set(map(int, c1.split(','))),
                               set(map(int, c2.split(','))))
                neighbour[i] -= c1
                neighbour[i+1].update(c1)
                neighbour[i+1] -= c2
                neighbour[i].update(c2)

        return neighbour

    def sample(self):
        p = torch.tensor(self.neighbour_counts) / np.sum(self.neighbour_counts)
        j = torch.distributions.Categorical(probs=p).sample().item()
        return self.gen_neighbour(self.neighbours[j])

    def pdf(self, C_star):
        # Uniform probability over neighbours
        return 1 / self.total_neighbours

    def logpdf(self, C_star):
        # Uniform probability over neighbours
        return -np.log(self.total_neighbours)


class GraphProposalDistribution:
    def __init__(self, G):
        self.G = G
        m, m = G.shape
        self.n_nodes = m

        self.neighbours = []
        self.neighbour_counts = []

        self.total_neighbours = m*(m-1) / 2

        self.populate_neighbours()

    def populate_neighbours(self):
        self.neighbours.append([-1, -1])
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                self.neighbours.append([i, j])
                self.neighbour_counts.append(1)

    def gen_neighbour(self, spec):
        [i, j] = spec

        G = deepcopy(self.G)

        if i == -1 and j == -1:
            return G

        if i > j:
            i, j = j, i

        G[i, j] = 1 - G[i, j]

        return G

    def sample(self):
        """
        Sample uniformly from the neighbourhood of graphs with 1 edge added,
        1 edge removed and the current graph
        """

        n_edges = (self.n_nodes * (self.n_nodes-1)) / 2
        logits = torch.tensor([0., np.log(n_edges)])
        move = torch.distributions.Categorical(logits=logits)

        if not move:
            i, j = -1, -1
        else:
            ns = torch.randperm(self.n_nodes)
            i = ns[0].item()
            j = ns[1].item()

        return self.gen_neighbour([i, j])

    def pdf(self, G_star):
        # Uniform probability over neighbours
        return 1 / self.total_neighbours

    def logpdf(self, G_star):
        # Uniform probability over neighbours
        return -np.log(self.total_neighbours)


class LocallyInformedCDAGProposalDistribution:
    def __init__(self, G_C, min_clusters, max_clusters, *, score):
        C, E_C = G_C
        self.C = C
        self.E_C = E_C
        self.q_c = ClusteringProposalDistribution(
            C, min_clusters, max_clusters)
        self.q_e_c = GraphProposalDistribution(E_C)

        self.logits = []
        self.neighbours = []
        for C_star_spec in self.q_c.neighbours:
            for E_C_star_spec in self.q_e_c.neighbours:
                C_star = self.q_c.gen_neighbour(C_star_spec)
                E_C_star = self.q_e_c.gen_neighbour(E_C_star_spec)
                stringified_cdag = stringify_cdag((C_star, E_C_star))
                self.neighbours.append(stringified_cdag)
                logit = 0.5 * (score(C_star, E_C_star)
                               - score(self.C, self.E_C))
                self.logits.append(logit)
        self.softmax = torch.distributions.Categorical(
            logits=torch.tensor(self.logits))

    def sample(self):
        idx = self.softmax.sample().item()
        return unstringify_cdag(self.neighbours[idx])

    def pdf(self, G_C):
        return np.exp(self.logpdf(G_C))

    def logpdf(self, G_C):
        idx = self.neighbours.index(stringify_cdag(G_C))
        return self.softmax.log_prob(torch.tensor([idx]))


class LocallyInformedCDAGSampler:
    def __init__(self, *, data, score, min_clusters=None, max_clusters=None, initial_sample=None):
        m, n = data.shape

        self.n_nodes = n

        self.min_clusters = min_clusters or 2
        self.max_clusters = max_clusters or n

        if initial_sample is None:
            C_init = matrix_to_clustering(
                self.make_random_clustering(self.max_clusters))
            G_init = UpperTriangular(len(C_init)).sample()
            initial_sample = (C_init, G_init)

        self.samples = [initial_sample]
        self.G_C_proposed = []
        self.U = torch.distributions.Uniform(0, 1)

        self.data = data
        self.score = score
        self.scores = []

        self.n_samples = N_SAMPLES
        self.n_warmup = N_WARMUP

        self.debug = debugger_is_active()

    def _reset(self, C_init, G_init):
        self.samples = [(C_init, G_init)]
        self.scores = []
        self.G_C_proposed = []

    def make_random_clustering(self, n_clusters):
        done = False
        while not done:
            C = np.zeros((self.n_nodes, n_clusters))
            for i in range(self.n_nodes):
                j = torch.randperm(n_clusters)[0].item()
                C[i, j] = 1
            if (np.sum(C, axis=0) > 0).all():
                done = True
        return C

    def sample(self, n_samples=N_SAMPLES, n_warmup=N_WARMUP):
        self.n_samples = n_samples
        self.n_warmup = n_warmup

        it = tqdm(range(n_warmup), 'MCMC warmup')
        for i in it:
            C_t, G_t = self.step(
                cb=lambda K: it.set_postfix_str(f'{len(K)} clusters'))
            self.samples.append((C_t, G_t))
            self.scores.append(self.cdag_score((C_t, G_t)).item())

        it = tqdm(range(n_samples), 'Sampling with MCMC')
        for i in it:
            C_t, G_t = self.step(
                cb=lambda K: it.set_postfix_str(f'{len(K)} clusters'))
            self.samples.append((C_t, G_t))
            self.scores.append(self.cdag_score((C_t, G_t)).item())

    def get_samples(self):
        return self.samples[-(self.n_samples+1):-1]

    def get_scores(self):
        return self.scores[-(self.n_samples+1):-1]

    def step(self, cb=None):
        alpha = self.U.sample().item()

        if alpha < 0.01:
            # Small probability of staying in same state to make Markov Chain ergodic
            return self.samples[-1]
        else:
            C_prev, E_C_prev = self.samples[-1]

            C_new, E_C_new = C_prev, E_C_prev

            C_star, E_C_star = LocallyInformedCDAGProposalDistribution(
                (C_prev, E_C_prev),
                self.min_clusters,
                self.max_clusters,
                score=self.score,
            ).sample()

            self.G_C_proposed.append((
                clustering_to_matrix(C_star),
                E_C_star
            ))

            if cb is not None:
                cb((C_star, E_C_star))

            u = self.U.sample().item()
            a = self.log_prob_accept((C_star, E_C_star))
            if np.log(u) < a:
                C_new = C_star
                E_C_new = E_C_star
            else:
                C_new = C_prev
                E_C_new = E_C_prev

            return C_new, E_C_new

    def log_prob_accept(self, G_C_star):
        G_C_prev = self.samples[-1]

        log_q_G_C_prev = LocallyInformedCDAGProposalDistribution(
            G_C_star,
            self.min_clusters,
            self.max_clusters,
            score=self.score,
        ).logpdf(G_C_prev)

        log_q_G_C_star = LocallyInformedCDAGProposalDistribution(
            G_C_prev,
            self.min_clusters,
            self.max_clusters,
            score=self.score,
        ).logpdf(G_C_star)

        log_prob_G_C_star = self.cdag_score(G_C_star)
        log_prob_G_C_prev = self.cdag_score(G_C_prev)

        rho = ((log_q_G_C_prev + log_prob_G_C_star)
               - (log_q_G_C_star + log_prob_G_C_prev))

        return min(0, rho)

    def cdag_score(self, G_C):
        return self.score(G_C[0], G_C[1])
