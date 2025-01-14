import cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float32_t, ndim=2] calc_inl(
    int n_ax, 
    int nc, 
    int[:] n_node_per_ax,
    int[:, :] nc_per_node, 
    float[:, :] node_l, 
    float[:, :] inl
):

    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros((n_ax, nc - 1), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] node_length = np.zeros((n_ax, nc), dtype=np.float32)
    cdef int i, j, k, nn, n_inl, nc_n, idx

    for i in range(n_ax):
        nn = n_node_per_ax[i]
        idx = 0
        for j in range(nn):
            nc_n = nc_per_node[i, j]
            for k in range(nc_n):
                node_length[i, idx + k] = node_l[i, j] / nc_n
            idx += nc_n

    for i in range(n_ax):
        for j in range(nc - 1):
            result[i, j] = (node_length[i, j] + node_length[i, j + 1]) / 2

    for i in range(n_ax):
        idx = 0
        n_inl = n_node_per_ax[i] - 1
        for j in range(n_inl):
            idx += nc_per_node[i, j]
            result[i, idx - 1] += inl[i, j]

    return result
