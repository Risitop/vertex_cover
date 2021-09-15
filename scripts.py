#!/usr/bin/env python3

import numpy as np

from pynndescent import NNDescent
from numba import njit
from scipy.sparse import coo_matrix, csr_matrix


def vertex_cover(
        X,
        n_neighbors=15,
        metric="euclidean",
        hops=1
):
    n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
    n_iters = max(5, int(round(np.log2(X.shape[0]))))

    knn_search_index = NNDescent(
        X,
        n_neighbors=15,
        metric="euclidean",
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60
    )
    A = to_symmetric_sparse(*knn_search_index.neighbor_graph)
    return _vertex_cover(
        A.indptr,
        A.indices,
        hops=hops
    )


def to_symmetric_sparse(indices, dists):
    n = indices.shape[0]
    rows, cols, data = fill_coo_matrix(indices, dists)
    A = coo_matrix(
        (data, (rows, cols)), shape=(n, n)
    ).tocsr()
    return A + A.T - A * A.T


@njit
def fill_coo_matrix(indices, dists):
    rows, cols, data = [], [], []
    for i, ni in enumerate(indices):
        for j in ni:
            rows.append(i)
            cols.append(j)
            data.append(1)
    return rows, cols, data


@njit
def _vertex_cover(Lptr, Linds, hops=2):
    # Lptr, Linds: CSR matrix representation
    # of neighbors
    n = Lptr.shape[0] - 1
    anchors = np.zeros(n, dtype='int') - 1
    for v in range(n):
        # If v not visited, add it as an anchor
        if anchors[v] != -1:
            continue
        anchors[v] = v
        # Mark its neighbors as visited
        neighbors = [(v, 0)]
        while len(neighbors):
            nb, d = neighbors.pop()
            anchors[nb] = v
            if d < hops:
                M = (Lptr[nb+1] if nb + 1 < n else n)
                for nb2 in Linds[Lptr[nb]:M]:
                    if anchors[nb2] != -1:
                        continue
                    anchors[nb2] = v
                    neighbors.append( (nb2, d+1) )
    anchors_set = np.zeros(n)
    for i in set(anchors):
        anchors_set[i] = 1
    return anchors_set, anchors # set, map


@njit
def fill_dmatrix(D_anchors, anchors, anchors_map, distances_map):
    # D_i,j ~ D_ri,rj + D_i,ri + D_j,rj
    n = len(anchors_map)
    D = np.zeros((n,n), dtype=np.float32)
    small_idx = [
        np.sum(anchors[:anchors_map[i]])
        for i in range(n)
    ]
    D = (
        D_anchors[small_idx, small_idx]
        + distances_map
        + distances_map[:,None]
    )
    return D - np.diag(np.diag(D))
