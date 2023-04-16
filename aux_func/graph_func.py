import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def gaussian_kernel_distance(vector, band_width):
    euc_dis = pairwise_distances(vector)
    gaus_dis = np.exp(- euc_dis * euc_dis / (band_width * band_width))
    return gaus_dis


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = np.coo_matrix(adj) np.coo_max
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)  # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)  # D^-0.5AD^0.5


def construct_affinity_matrix(data, objects, band_width):
    am_set = []
    obj_nums = np.max(objects)
    for i in range(0, obj_nums + 1):
        sub_object = data[objects == i]
        adj_mat = gaussian_kernel_distance(sub_object, band_width=band_width)
        norm_adj_mat = normalize_adj(adj_mat)
        am_set.append([adj_mat, norm_adj_mat])
    return am_set
