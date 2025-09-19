"""
DTW с лентой Sakoe–Chiba.
"""

from __future__ import annotations
import numpy as np
from math import inf
from typing import Callable, List, Tuple


def dtw_band(A, B, cost_fn, band=15):
    T, S = len(A), len(B)
    D = np.full((T+1, S+1), inf, dtype=np.float32)
    D[0, 0] = 0.0
    for i in range(1, T+1):
        j_min = max(1, i - band)
        j_max = min(S, i + band)
        for j in range(j_min, j_max + 1):
            c = cost_fn(i-1, j-1)
            D[i, j] = c + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    # backtrack
    i, j = T, S
    path = []
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        choices = (D[i-1, j], D[i, j-1], D[i-1, j-1])
        step = int(np.argmin(choices))
        if step == 0: i -= 1
        elif step == 1: j -= 1
        else: i -= 1; j -= 1
    path.reverse()
    return path, D
