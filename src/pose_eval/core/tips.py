"""
Подсказки по углам суставов, пороги и карта подсветки сегментов.
Работает по нормализованным координатам (как в normalize_xy).
"""

from __future__ import annotations
import numpy as np
from math import acos, degrees
from typing import Dict, Tuple, Optional

# Пороги (можно вынести в конфиг)
TH_MINOR = 10   # жёлтая подсветка: |Δ| >= 10°
TH_MAJOR = 20   # красная подсветка: |Δ| >= 20°

# Индексы ключевых точек BlazePose
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28

# Какие «кости» окрашивать для каждого угла (для оверлея)
ANGLE_BONES = {
    "left_shoulder":  [(L_SH, L_EL), (L_SH, R_SH)],
    "right_shoulder": [(R_SH, R_EL), (L_SH, R_SH)],
    "left_elbow":     [(L_SH, L_EL), (L_EL, L_WR)],
    "right_elbow":    [(R_SH, R_EL), (R_EL, R_WR)],
    "left_hip":       [(L_HIP, L_KNE), (L_SH, L_HIP), (L_HIP, R_HIP)],
    "right_hip":      [(R_HIP, R_KNE), (R_SH, R_HIP), (L_HIP, R_HIP)],
    "left_knee":      [(L_HIP, L_KNE), (L_KNE, L_ANK)],
    "right_knee":     [(R_HIP, R_KNE), (R_KNE, R_ANK)],
}

# Для текста
RU_SIDE = {"left": "Лев", "right": "Прав"}


def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    """
    Угол ABC в градусах (между BA и BC). Возвращает None при вырожденных векторах.
    """
    v1 = a - b
    v2 = c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return degrees(acos(cosang))


def compute_key_angles(xy: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Возвращает набор ключевых углов (в градусах) по нормализованным координатам (33,2).
    """
    angles = {
        "left_elbow":  _angle_3pts(xy[L_SH], xy[L_EL], xy[L_WR]),
        "right_elbow": _angle_3pts(xy[R_SH], xy[R_EL], xy[R_WR]),
        "left_knee":   _angle_3pts(xy[L_HIP], xy[L_KNE], xy[L_ANK]),
        "right_knee":  _angle_3pts(xy[R_HIP], xy[R_KNE], xy[R_ANK]),
        "left_shoulder":  _angle_3pts(xy[L_HIP], xy[L_SH], xy[L_EL]),
        "right_shoulder": _angle_3pts(xy[R_HIP], xy[R_SH], xy[R_EL]),
        "left_hip":    _angle_3pts(xy[L_SH], xy[L_HIP], xy[L_KNE]),
        "right_hip":   _angle_3pts(xy[R_SH], xy[R_HIP], xy[R_KNE]),
    }
    return angles


def suggestion_from_angle(name: str, delta_deg: float, thresh: float = TH_MINOR) -> Optional[str]:
    """
    Человеко-читаемая подсказка по углу.
    delta_deg = (угол пользователя) - (угол референса), градусы.
    Положительное значение — пользователь «раскрыл» сустав сильнее (угол больше).
    """
    if delta_deg is None or abs(delta_deg) < thresh:
        return None

    side = "Левый" if "left" in name else "Правый"

    if "elbow" in name:
        return f"{side} локоть {'слишком выпрямлен' if delta_deg > 0 else 'недовыпрямлен'} на ~{abs(int(delta_deg))}°"

    if "knee" in name:
        return f"{side} колено {'слишком выпрямлено' if delta_deg > 0 else 'недовыпрямлено'} на ~{abs(int(delta_deg))}°"

    if "shoulder" in name:
        return f"{side} плечо {'поднято выше' if delta_deg > 0 else 'поднято ниже'} эталона на ~{abs(int(delta_deg))}°"

    if "hip" in name:
        return f"{side} тазобедренный сустав {'сильнее согнут' if delta_deg > 0 else 'меньше согнут'} на ~{abs(int(delta_deg))}°"

    return None


def per_frame_angle_diffs(ref_xy_norm: np.ndarray, usr_xy_norm: np.ndarray) -> tuple[dict, dict, list[str]]:
    """
    Сравнивает углы референса и пользователя на одном кадре.
    Возвращает:
      diffs: {name -> delta_deg},
      bad_map: {name -> abs(delta)}, только те, что выше TH_MINOR,
      tips: список из топ-3 текстовых подсказок по убыванию |delta|.
    """
    a_ref = compute_key_angles(ref_xy_norm)
    a_usr = compute_key_angles(usr_xy_norm)

    diffs, bad = {}, {}
    for k in a_ref.keys():
        ar, au = a_ref[k], a_usr[k]
        if ar is None or au is None:
            continue
        d = au - ar
        diffs[k] = d
        if abs(d) >= TH_MINOR:
            bad[k] = abs(d)

    ordered = sorted(bad.items(), key=lambda x: -x[1])
    tips = []
    for name, _ in ordered[:3]:
        s = suggestion_from_angle(name, diffs[name], thresh=TH_MINOR)
        if s:
            tips.append(s)

    return diffs, dict(ordered), tips
