# distutils: language=c++

import cython

from libc.math cimport floor, sqrtf  # For fast floor operation in C
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort

import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object calc_inl(
    int n_ax, 
    int nc, 
    int[:] n_node_per_ax,
    int[:] n_internode_per_ax,
    int[:, :] internode_inds,
    int[:, :] nc_per_node, 
    float[:, :] node_l, 
    float[:, :] inls
):
    """
    Calculate internode lengths for each compartment.

    Args:
        n_ax: number of axons
        nc: number of compartments
        n_node_per_ax: number of nodes per axon
        n_internode_per_ax: number of internodes per axon
        internode_inds: indices of internodes
        nc_per_node: number of compartments per node
        node_l: node lengths
        inls: internode lengths
    
    Returns:
        A tuple of two 2D arrays:
        - internode lengths for each compartment
        - node lengths for each compartment
    """

    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros((n_ax, nc - 1), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] node_length = np.zeros((n_ax, nc), dtype=np.float32)
    cdef int i, j, k, nn, n_inl, nc_n, idx, in_ind

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
        in_ind = -1
        n_inl = n_internode_per_ax[i]
        for j in range(n_inl):
            while in_ind < internode_inds[i, j]:
                idx += nc_per_node[i, in_ind]
                in_ind += 1
            result[i, idx - 1] += inls[i, j]

    return (result, node_length)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object calc_ind(
    int n_ax,
    int nc,
    int[:] n_node_per_ax,
    int[:] n_internode_per_ax,
    int[:, :] internode_inds,
    int[:, :] nc_per_node,
    float[:, :] node_d,
    float[:, :] inds
):

    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros((n_ax, nc - 1), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] node_diameter = np.zeros((n_ax, nc), dtype=np.float32)
    cdef int i, j, k, nn, n_ind, nc_n, idx, in_ind

    for i in range(n_ax):
        nn = n_node_per_ax[i]
        idx = 0
        for j in range(nn):
            nc_n = nc_per_node[i, j]
            for k in range(nc_n):
                node_diameter[i, idx + k] = node_d[i, j]
            idx += nc_n

    for i in range(n_ax):
        for j in range(nc - 1):
            result[i, j] = sqrtf(node_diameter[i, j] * node_diameter[i, j + 1])

    for i in range(n_ax):
        idx = 0
        in_ind = -1
        n_ind = n_internode_per_ax[i]
        for j in range(n_ind):
            while in_ind < internode_inds[i, j]:
                idx += nc_per_node[i, in_ind]
                in_ind += 1
            result[i, idx - 1] = inds[i, j]

    return (result, node_diameter)


@cython.boundscheck(True)
@cython.wraparound(False)
cdef distribute_compartments_single(double[:] lengths, int[:, ::1] buff, int row, int P, int N):
    """
    Given:
      - lengths: a 1D array (Cython typed) of unmyelinated section lengths
      - P: the total number of final compartments (>= N)

    Returns:
      A Python list of integers [x_1, x_2, ..., x_N],
      where sum(x_i) = P and x_i >= 1.

    This function attempts to make the sub-piece lengths
    L_i / x_i as close to one another as possible.
    """

    cdef:
        double total_length = 0.0
        double T
        double val, frac
        int i, idx
        int x_floor
        int diff
        int S = 0

    # Compute total length
    for i in range(N):
        total_length += lengths[i]

    T = total_length / P  # ideal sub-piece length (no merging)

    cdef vector[double] x_star
    x_star.reserve(N)  # optional, avoid repeated allocations

    cdef vector[int] x
    x.reserve(N)

    cdef list fractional_parts = []

    # Compute x_star[i] = L_i / T and floor it
    # and store in x[i], ensuring at least 1
    for i in range(N):
        val = lengths[i] / T
        x_star.push_back(val)
        # floor() from libc.math returns a double
        # We'll cast it to int
        x_floor = <int>floor(val)
        if x_floor < 1:
            x_floor = 1
        x.push_back(x_floor)

    # Sum up the initial x[i]
    for i in range(N):
        S += x[i]

    # We need fractional parts for sorting
    for i in range(N):
        frac = x_star[i] - floor(x_star[i])
        fractional_parts.append((frac, i))

    if S == P:
        # Perfect match
        for i in range(N):
            buff[row, i] = x[i]

    elif S < P:
        # Need more pieces: increment some x_i
        diff = P - S

        # Sort by fractional part descending
        fractional_parts.sort(key=lambda x: x[0], reverse=True)
        # round-robin increment
        idx = 0
        while diff > 0:
            frac, i = fractional_parts[idx]
            x[i] += 1
            diff -= 1
            idx = (idx + 1)
            if idx >= N:
                idx = 0

        for i in range(N):
            buff[row, i] = x[i]


    else:
        # S > P, we have too many pieces
        diff = S - P

        # Sort by fractional part ascending
        fractional_parts.sort(key=lambda x: x[0])
        idx = 0
        while diff > 0:
            frac, i = fractional_parts[idx]
            if x[i] > 1:
                x[i] -= 1
                diff -= 1
            idx = idx + 1
            if idx >= N:
                idx = 0
        for i in range(N):
            buff[row, i] = x[i]


@cython.boundscheck(True)
@cython.wraparound(False)
cpdef np.ndarray[np.int32_t, ndim=2] distribute_compartments(
    int n_ax, 
    int max_n_unmyel,
    double[:, :] lengths,
    int nc,
    int[:] n_um
):
    """
    Distribute compartments for multiple axons.

    Args:
        n_ax: number of axons
        max_n_unmyel: maximum number of unmyelinated sections
        lengths: 2D array of unmyelinated section lengths
        nc: number of compartments
        n_um: number of unmyelinated sections per axon
    
    Returns:
        A 2D array of integers, where each row corresponds to an axon
        and each column corresponds to the number of compartments in an unmyelinated section.
    """
    cdef np.ndarray[np.int32_t, ndim=2] result = np.zeros((n_ax, max_n_unmyel), dtype=np.int32)
    cdef int i
    for i in range(n_ax):
        distribute_compartments_single(lengths[i], result, i, nc, n_um[i])
    return result
