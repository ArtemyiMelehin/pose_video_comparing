import cv2, mediapipe as mp
import numpy as np
from ..core.normalize import normalize_xy

def extract_sequence(video_path, model_complexity=1):
    pose = mp.solutions.pose.Pose(model_complexity=model_complexity, enable_segmentation=False)
    cap = cv2.VideoCapture(video_path)
    seq_xy, seq_vis, frames = [], [], []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            xy = np.array([[lm[i].x, lm[i].y] for i in range(33)], dtype=np.float32)
            vis = np.array([lm[i].visibility for i in range(33)], dtype=np.float32)
            seq_xy.append(normalize_xy(xy))
            seq_vis.append(vis); frames.append(frame)
    cap.release()
    return np.array(seq_xy), np.array(seq_vis), frames, (w,h,fps)

def extract_raw_xy(video_path):
    pose = mp.solutions.pose.Pose(model_complexity=1)
    cap = cv2.VideoCapture(video_path)
    seq = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            seq.append(np.array([[lm[i].x, lm[i].y] for i in range(33)], dtype=np.float32))
    cap.release()
    return seq
