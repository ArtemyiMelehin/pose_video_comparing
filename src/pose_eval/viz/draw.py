"""
Базовые функции отрисовки.
"""

from __future__ import annotations
import numpy as np
import cv2
from ..core.normalize import BONES  # список рёбер для скелета


def draw_pose(img: np.ndarray, xy_norm01: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    """
    Рисует скелет по нормализованным координатам (x,y в [0..1]).
    """
    H, W = img.shape[:2]
    pts = (xy_norm01 * np.array([W, H])).astype(int)

    # рёбра
    for j1, j2 in BONES:
        p1 = tuple(pts[j1])
        p2 = tuple(pts[j2])
        cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)

    # точки
    for (x, y) in pts:
        cv2.circle(img, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)

    return img


def colormap_val(v: float) -> tuple[int, int, int]:
    """
    Простейшая линейная цветовая карта 0..1 → BGR (синий → красный).
    """
    v = float(np.clip(v, 0.0, 1.0))
    return (int(255 * (1 - v)), 0, int(255 * v))
