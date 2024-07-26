import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
from itertools import chain


from utils.cdag import stringify_cdag, get_graphs_by_count


def plot_graph(g, target):
    c = g[0]
    graphstring = stringify_cdag(g)
    g = ig.Graph.Adjacency(g[1].tolist())
    ig.plot(g,
            target,
            vertex_size=75,
            vertex_color='grey',
            vertex_label=c,
            layout='circle',
            bbox=(0, 0, 300, 300),
            margin=50)


def visualize_graphs(graphs, filename):
    graphs, graph_counts = get_graphs_by_count(graphs)

    selected_graphs = graphs[:5]

    ncols = 2
    nrows = int(np.ceil(len(selected_graphs) / 2))

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*350*px, nrows*350*px))
    ax = ax.reshape(nrows, ncols)
    i, j = 0, 0
    for ni in range(nrows):
        for nj in range(ncols):
            ax[ni, nj].axis('off')
    for graph in selected_graphs:
        plot_graph(graph, ax[i, j])
        if j+1 == ncols:
            j = 0
            i += 1
        else:
            j += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'{filename}.png')
    plt.clf()


def plot_graph_scores(scores, opt_score, filepath):
    lengths = [len(s) for s in scores]
    scores = list(chain(*scores))
    plt.plot(scores)
    for i, l in enumerate(lengths):
        plt.axvline(l*(i+1), color='green')
    if opt_score is not None:
        plt.axhline(opt_score, color='red', linestyle=':')
    plt.savefig(f'{filepath}/scores.png')
    plt.clf()
