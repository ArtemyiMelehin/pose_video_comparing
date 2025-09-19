"""
Экстракция поз из референс-видео в файл .npz
Сохраняем:
  - xy:  (T, 33, 2)  нормализованные координаты (pose_eval.core.normalize.normalize_xy)
  - vis: (T, 33)     visibility [0..1]
  - fps: float        FPS видео
  - frames: int       количество использованных кадров (с найденной позой)
Пример:
  python scripts/extract_reference.py --video data/ref/model.mp4
  # -> outputs/ref/model.npz
"""

from __future__ import annotations
import os
import argparse
import numpy as np

from pose_eval.backends.mediapipe_backend import extract_sequence


def main():
    ap = argparse.ArgumentParser("Extract reference pose to .npz")
    ap.add_argument("--video", required=True, help="путь к референс-видео")
    ap.add_argument("--out", default=None, help="путь к .npz (по умолчанию outputs/ref/<name>.npz)")
    args = ap.parse_args()

    # куда сохраняем
    if args.out is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        out_dir = os.path.join("outputs", "ref")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}.npz")
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out_path = args.out

    # извлекаем
    xy, vis, frames, (W, H, FPS) = extract_sequence(args.video)

    np.savez_compressed(out_path,
                        xy=xy.astype(np.float32),
                        vis=vis.astype(np.float32),
                        fps=float(FPS),
                        frames=int(len(xy)),
                        width=int(W),
                        height=int(H))

    print(f"[OK] saved reference poses: {out_path}")
    print(f"  frames(with pose): {len(xy)} | fps≈{FPS:.2f} | shape(xy)={xy.shape}")


if __name__ == "__main__":
    main()
