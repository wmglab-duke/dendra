# distutils: language=c++

import cython

from libc.math cimport floor  # For fast floor operation in C
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort

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


cdef int compare_descending(const tuple[double, int] &a, const tuple[double, int] &b) nogil:
    """
    Comparator for descending order of fractional parts.
    """
    if a[0] > b[0]:
        return -1
    elif a[0] < b[0]:
        return 1
    return 0


cdef int compare_ascending(const tuple[double, int] &a, const tuple[double, int] &b) nogil:
    """
    Comparator for ascending order of fractional parts.
    """
    if a[0] < b[0]:
        return -1
    elif a[0] > b[0]:
        return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef distribute_compartments_single(double[:] lengths, int[:] buff, int P, int N):
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

    cdef vector[tuple[double, int]] fractional_parts
    fractional_parts.reserve(N)

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
    # fractional_part[i] = x_star[i] - floor(x_star[i])
    for i in range(N):
        frac = x_star[i] - floor(x_star[i])
        # We'll store a tuple: (fractional_part, index)
        fractional_parts.push_back((frac, i))

    if S == P:
        # Perfect match
        for i in range(N):
            buff[i] = x[i]

    elif S < P:
        # Need more pieces: increment some x_i
        diff = P - S

        # Sort by fractional part descending
        sort(fractional_parts.begin(), fractional_parts.end(),
             compare_descending)
        # We'll do a simple round-robin increment
        idx = 0
        while diff > 0:
            frac, i = fractional_parts[idx]
            x[i] += 1
            diff -= 1
            idx = (idx + 1)
            if idx >= N:
                idx = 0

        for i in range(N):
            buff[i] = x[i]


    else:
        # S > P, we have too many pieces
        diff = S - P

        # Sort by fractional part ascending
        sort(fractional_parts.begin(), fractional_parts.end(),
             compare_ascending)
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
            buff[i] = x[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int32_t, ndim=2] distribute_compartments(
    int n_ax, 
    int max_n_unmyel,
    double[:, :] lengths,
    int nc,
    int[:] n_um
):
    cdef np.ndarray[np.int32_t, ndim=2] result = np.zeros((n_ax, max_n_unmyel), dtype=np.int32)
    cdef int i
    for i in range(n_ax):
        distribute_compartments_single(lengths[i], result[i], nc, n_um[i])
    return result
