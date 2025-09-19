"""
Метрики сходства позы.
Используем нормализованные 2D-координаты (J,2) и видимости (J,) в [0..1].
"""

from __future__ import annotations
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray, vis_mask: np.ndarray | None = None) -> float:
    """
    Косинусная схожесть между двумя позами.
    a, b: (J, 2) нормализованные координаты.
    vis_mask: (J,) bool — True = использовать точку.
    """
    if vis_mask is not None:
        a = a[vis_mask]
        b = b[vis_mask]

    if a.size == 0:
        return 0.0

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    dot = float(np.dot(a_flat, b_flat))
    norm = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    return dot / norm if norm > 1e-6 else 0.0


def weighted_l1(a: np.ndarray, b: np.ndarray, vis_a: np.ndarray, vis_b: np.ndarray) -> float:
    """
    Взвешенная L1-дистанция между позами.
    a, b: (J, 2) нормализованные координаты.
    vis_a, vis_b: (J,) видимости [0..1].
    """
    weights = np.minimum(vis_a, vis_b).astype(np.float32)
    diffs = np.linalg.norm(a - b, ord=1, axis=1)  # L1 по каждой точке → (J,)
    score = float(np.sum(weights * diffs) / (np.sum(weights) + 1e-6))
    return score


def compute_cosine_wl1(ref_xy: np.ndarray, usr_xy: np.ndarray,
                       ref_vis: np.ndarray, usr_vis: np.ndarray) -> tuple[float, float]:
    """
    Вычисляет пару метрик (cos, wL1) между двумя позами.
    ref_xy, usr_xy: (J,2) нормализованные координаты.
    ref_vis, usr_vis: (J,) видимость.
    """
    vis_mask = (ref_vis > 0.3) & (usr_vis > 0.3)
    cos = cosine_similarity(ref_xy, usr_xy, vis_mask)
    wl1 = weighted_l1(ref_xy, usr_xy, ref_vis, usr_vis)
    return cos, wl1
