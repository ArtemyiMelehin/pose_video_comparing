from __future__ import annotations
import argparse, os, numpy as np

from pose_eval.backends.mediapipe_backend import extract_sequence, extract_raw_xy
from pose_eval.core.metrics import compute_cosine_wl1
from pose_eval.core.dtw import dtw_band
from pose_eval.io.exports import make_run_dir, save_frame_metrics, save_joint_stats, save_summary
from pose_eval.viz.overlay import save_overlay_with_joint_errors


# Короткие имена суставов BlazePose (33)
JOINT_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
    "left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle",
    "left_heel","right_heel","left_foot_index","right_foot_index"
]

def per_joint_l1(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    # L1 по каждой точке (2D)
    return np.abs(a_xy - b_xy).sum(axis=1)  # (33,)

def main():
    ap = argparse.ArgumentParser("Pose compare demo (MediaPipe + DTW + tips)")
    ap.add_argument("--ref", required=True, help="путь к видео-референсу")
    ap.add_argument("--usr", required=True, help="путь к видео-пользователя")
    ap.add_argument("--alpha", type=float, default=0.7, help="стоимость: a*(1-cos) + (1-a)*wL1")
    ap.add_argument("--band", type=int, default=15, help="ширина ленты Sakoe–Chiba (в кадрах)")
    ap.add_argument("--outdir", default=None, help="папка результата; по умолчанию outputs/runs/<timestamp>")
    ap.add_argument("--tips_side", default="left", choices=["left","right"], help="сторона плашки с подсказками")
    args = ap.parse_args()

    run_dir = args.outdir or make_run_dir()  # e.g. outputs/runs/2025-09-18_12-03-47
    os.makedirs(run_dir, exist_ok=True)

    # 1) Извлекаем позы и кадры
    ref_xy, ref_vis, ref_frames, (W,H,FPS) = extract_sequence(args.ref)
    usr_xy, usr_vis, usr_frames, _        = extract_sequence(args.usr)
    ref_xy_raw = extract_raw_xy(args.ref)
    usr_xy_raw = extract_raw_xy(args.usr)

    # 2) Локальная стоимость
    def local_cost(i, j):
        cos, wl1 = compute_cosine_wl1(ref_xy[i], usr_xy[j], ref_vis[i], usr_vis[j])
        return args.alpha * (1.0 - cos) + (1.0 - args.alpha) * wl1

    # 3) DTW
    path_pairs, _ = dtw_band(ref_xy, usr_xy, local_cost, band=args.band)

    # 4) Покадровые метрики и пер-суставные ошибки
    rows = []
    per_joint_sum = np.zeros(33, dtype=np.float64)
    per_joint_cnt = np.zeros(33, dtype=np.float64)
    per_step_joint_norm = []

    for (i,j) in path_pairs:
        cos, wl1 = compute_cosine_wl1(ref_xy[i], usr_xy[j], ref_vis[i], usr_vis[j])
        mix = args.alpha * (1.0 - cos) + (1.0 - args.alpha) * wl1
        rows.append([i, j, cos, wl1, mix])

        errs = per_joint_l1(ref_xy[i], usr_xy[j])
        per_joint_sum += errs
        per_joint_cnt += 1.0
        per_step_joint_norm.append((errs / (errs.max() + 1e-6)).astype(np.float32))

    rows = np.array(rows, dtype=float)
    mean_mix = float(rows[:,4].mean()) if len(rows) else float("nan")

    # 5) Сохранения
    save_frame_metrics(
        rows,
        os.path.join(run_dir, "metrics.csv"),
        os.path.join(run_dir, "metrics.png"),
        title=f"DTW mean mix_cost = {mean_mix:.3f}"
    )

    joint_mean = (per_joint_sum / np.maximum(per_joint_cnt, 1e-6)).astype(np.float32)
    save_joint_stats(
        joint_mean, JOINT_NAMES,
        os.path.join(run_dir, "joints_heatmap.png"),
        os.path.join(run_dir, "joints_stats.csv")
    )

    # 6) Видео-оверлей с подсветками и топ-3 советами
    save_overlay_with_joint_errors(
        os.path.join(run_dir, "overlay_heat.mp4"),
        ref_frames, usr_frames, path_pairs,
        ref_xy_raw, usr_xy_raw, per_step_joint_norm, FPS,
        ref_xy_norm=ref_xy, usr_xy_norm=usr_xy,
        tips_side=args.tips_side
    )

    # 7) Сводка
    save_summary({
        "mean_mix_cost": mean_mix,
        "ref_len": int(len(ref_xy)),
        "usr_len": int(len(usr_xy)),
        "alpha": args.alpha,
        "band": args.band
    }, os.path.join(run_dir, "summary.json"))

    print(f"[OK] run saved to: {run_dir}")

if __name__ == "__main__":
    main()
