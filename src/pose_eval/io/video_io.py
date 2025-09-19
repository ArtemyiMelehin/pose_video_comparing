"""
Утилиты для работы с видео:
- VideoReader: итератор по кадрам с метаданными (fps, size)
- make_writer: VideoWriter с проверкой параметров
- ensure_same_size: безопасный ресайз к заданному размеру
"""

from __future__ import annotations
import cv2
from typing import Iterator, Tuple


class VideoReader:
    """
    Простой итератор по кадрам видео.
    Пример:
        vr = VideoReader("file.mp4")
        for frame in vr:
            ...
        print(vr.fps, vr.width, vr.height)
    """
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        ok, frame = self.cap.read()
        if not ok:
            self.cap.release()
            raise StopIteration
        return frame

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()


def make_writer(path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """
    Создаёт cv2.VideoWriter для MP4 (mp4v).
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for: {path}")
    return writer


def ensure_same_size(frame, target_size: Tuple[int, int], interpolation=cv2.INTER_AREA):
    """
    Приводит кадр к (w,h). Возвращает кадр нужного размера.
    """
    w, h = target_size
    if frame.shape[1] == w and frame.shape[0] == h:
        return frame
    return cv2.resize(frame, (w, h), interpolation=interpolation)
