"""
Сборка видео-оверлея: heat-кружки по суставам, подсветка сегментов и
PIL-текст с кириллицей (топ-3 подсказки в каждый кадр).
"""

from __future__ import annotations
import os
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .draw import draw_pose, colormap_val
from ..core.tips import per_frame_angle_diffs, ANGLE_BONES, TH_MAJOR, L_SH, R_SH, L_EL, R_EL, L_KNE, R_KNE, L_HIP, R_HIP


def _load_font(size: int = 22) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Пытаемся найти системный TTF с кириллицей."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/seguiemj.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_tips_pil(bgr_img: np.ndarray, tips: List[str], side: str = "left",
                  margin: int = 12, line_h: int = 26) -> np.ndarray:
    """
    Рисует список строк tips кириллицей поверх bgr-кадра.
    side: "left" | "right".
    """
    if not tips:
        return bgr_img

    h, w = bgr_img.shape[:2]
    panel_w = min(int(w * 0.48), 560)
    panel_h = margin * 2 + line_h * len(tips)
    x0 = margin if side == "left" else (w - panel_w - margin)
    y0 = margin

    # затемнённая плашка
    overlay = bgr_img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (32, 32, 32), -1)
    bgr_img = cv2.addWeighted(overlay, 0.35, bgr_img, 0.65, 0)

    # рисуем текст через PIL
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    font = _load_font(size=22)

    tx, ty = x0 + 12, y0 + 8
    for t in tips:
        draw.text((tx, ty), t, font=font, fill=(255, 255, 255))
        ty += line_h

    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def highlight_segments(img: np.ndarray, xy01: np.ndarray, bad_angles: dict) -> None:
    """
    Подсвечивает «проблемные» сегменты (суставы/кости) на кадре.
    xy01: (33,2) в [0..1] image-space.
    bad_angles: dict{name -> abs_delta_deg}, имя из compute_key_angles.
    """
    H, W = img.shape[:2]
    pts = (xy01 * np.array([W, H])).astype(int)

    for name, ad in bad_angles.items():
        bones = ANGLE_BONES.get(name, [])
        # цвет по силе отклонения
        if ad >= TH_MAJOR:
            col, thick = (0, 0, 255), 4    # красный
        else:
            col, thick = (0, 255, 255), 3  # жёлтый
        # линии по костям
        for (j1, j2) in bones:
            cv2.line(img, tuple(pts[j1]), tuple(pts[j2]), col, thick, cv2.LINE_AA)
        # точка-сустав (центр угла)
        if "shoulder" in name:
            j = L_SH if "left" in name else R_SH
        elif "elbow" in name:
            j = L_EL if "left" in name else R_EL
        elif "knee" in name:
            j = L_KNE if "left" in name else R_KNE
        elif "hip" in name:
            j = L_HIP if "left" in name else R_HIP
        else:
            j = None
        if j is not None:
            cv2.circle(img, tuple(pts[j]), 7, col, -1, cv2.LINE_AA)


def save_overlay_with_joint_errors(
    path: str,
    frames_ref: List[np.ndarray],
    frames_usr: List[np.ndarray],
    path_pairs: List[Tuple[int, int]],
    ref_xy_raw: List[np.ndarray],
    usr_xy_raw: List[np.ndarray],
    joint_errs_norm: List[np.ndarray],
    fps: float,
    *,
    ref_xy_norm: np.ndarray | None = None,
    usr_xy_norm: np.ndarray | None = None,
    tips_side: str = "left",
) -> None:
    """
    Собирает видео: слева референс, справа пользователь.
    Поверх — скелет + heat-кружки по суставам; справа — подсветка сегментов и топ-3 советов.

    joint_errs_norm: список длины len(path_pairs), каждый элемент — (33,) в [0..1].
    ref_xy_raw, usr_xy_raw: «сырые» (x,y) в [0..1] для отрисовки.
    ref_xy_norm, usr_xy_norm: нормализованные (для расчёта углов/советов).
    """
    if not frames_ref or not frames_usr:
        return

    # общий стабильный размер кадра
    ref_h0, ref_w0 = frames_ref[0].shape[:2]
    usr_h0, usr_w0 = frames_usr[0].shape[:2]
    tgt_h = min(ref_h0, usr_h0)
    tgt_w = min(ref_w0, usr_w0)

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (tgt_w * 2, tgt_h))

    for step, (i, j) in enumerate(path_pairs):
        li = min(i, len(frames_ref) - 1)
        rj = min(j, len(frames_usr) - 1)

        left = cv2.resize(frames_ref[li], (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(frames_usr[rj], (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)

        # Скелеты
        draw_pose(left,  ref_xy_raw[min(i, len(ref_xy_raw) - 1)])
        draw_pose(right, usr_xy_raw[min(j, len(usr_xy_raw) - 1)], color=(0, 200, 255))

        # Heat-кружки по суставам
        errs = joint_errs_norm[step] if step < len(joint_errs_norm) else np.zeros(33, dtype=np.float32)
        ref_pts = (ref_xy_raw[min(i, len(ref_xy_raw) - 1)] * np.array([tgt_w, tgt_h])).astype(int)
        usr_pts = (usr_xy_raw[min(j, len(usr_xy_raw) - 1)] * np.array([tgt_w, tgt_h])).astype(int)
        for idx in range(33):
            col = colormap_val(float(errs[idx]))
            cv2.circle(left,  tuple(ref_pts[idx]), 5, col, -1, cv2.LINE_AA)
            cv2.circle(right, tuple(usr_pts[idx]), 5, col, -1, cv2.LINE_AA)

        # По-кадровые подсказки и подсветка сегментов
        tips: List[str] = []
        if (ref_xy_norm is not None and usr_xy_norm is not None and
                i < len(ref_xy_norm) and j < len(usr_xy_norm)):
            _, bad_map, tips = per_frame_angle_diffs(ref_xy_norm[i], usr_xy_norm[j])
            highlight_segments(right, usr_xy_raw[min(j, len(usr_xy_raw) - 1)], bad_map)

        # Текст с кириллицей (топ-3)
        right = draw_tips_pil(right, tips, side=tips_side)

        # Склейка в один кадр
        canvas = np.zeros((tgt_h, tgt_w * 2, 3), dtype=np.uint8)
        canvas[:, :tgt_w] = left
        canvas[:, tgt_w:] = right
        writer.write(canvas)

    writer.release()
