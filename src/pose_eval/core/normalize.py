import numpy as np
# BlazePose индексы
L_SH, R_SH, PELVIS = 11, 12, 23
BONES = [(11,12),(12,24),(11,23),(23,24),(11,13),(13,15),(12,14),(14,16),
         (23,25),(25,27),(24,26),(26,28),(27,29),(28,30)]

def normalize_xy(xy):
    xy = xy.copy().astype(np.float32)
    origin = xy[PELVIS]
    xy -= origin
    sh = xy[R_SH] - xy[L_SH]
    scale = np.linalg.norm(sh) + 1e-6
    xy /= scale
    th = np.arctan2(sh[1], sh[0] + 1e-9)
    c, s = np.cos(-th), np.sin(-th)
    R = np.array([[c,-s],[s,c]], dtype=np.float32)
    return xy @ R.T
