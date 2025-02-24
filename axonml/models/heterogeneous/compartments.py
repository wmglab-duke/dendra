from typing import Callable
import re

import numba
import numpy as np


def expand_string_list(strings):
    """
    Expands any element of the form "text * number" into
    'number' copies of "text". Other elements remain unchanged.
    
    Parameters:
        strings (list of str): The input list of strings.
    
    Returns:
        list of str: The expanded list.
    """
    pattern = re.compile(r'^(.*?)\s*\*\s*(\d+)$')
    expanded = []
    
    for s in strings:
        match = pattern.match(s)
        if match:
            text, count_str = match.groups()
            count = int(count_str)
            expanded.extend([text] * count)
        else:
            expanded.append(s)
    
    return expanded


@numba.jit
def build_node_diams(
    sequence: list[str],
    n_repeats: int,
    diams: list[float],
    node_d_dict: dict[str, Callable[[float], float]],
):
    sequence = expand_string_list(sequence)
    d2_size = len(sequence) * n_repeats + 1
    d = np.array(diams)[:, np.newaxis]
    out = np.tile(d, (1, d2_size))
    for i in range(len(diams)):
        j = 0
        for r in range(n_repeats):
            for s in sequence:
                out[i, j] = node_d_dict[s](out[i, j])
                j += 1
        out[i, j] = node_d_dict[s[0]](out[i, j])
    return out
