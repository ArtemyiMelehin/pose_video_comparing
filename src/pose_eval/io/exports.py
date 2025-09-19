import os, csv, json, time
import numpy as np
import matplotlib.pyplot as plt

def make_run_dir(base="outputs/runs", name=None):
    ts = name or time.strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base, ts)
    os.makedirs(path, exist_ok=True)
    return path

def save_frame_metrics(rows, out_csv, out_plot, title):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["ref_idx","usr_idx","cosine","wL1","mix_cost"])
        w.writerows(rows)
    arr = np.array(rows, dtype=float)
    if len(arr):
        plt.figure(figsize=(10,4))
        plt.plot(arr[:,4], label="mix_cost")
        plt.plot(1.0 - arr[:,2], label="1 - cosine", alpha=0.7)
        plt.plot(arr[:,3], label="wL1", alpha=0.7)
        plt.title(title); plt.xlabel("DTW step"); plt.ylabel("cost")
        plt.legend(); plt.tight_layout(); plt.savefig(out_plot); plt.close()

def save_joint_stats(joint_mean, names, out_png, out_csv):
    order = np.argsort(-joint_mean)
    vals = joint_mean[order]; labels = [names[i] for i in order]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,10))
    y = np.arange(len(vals)); plt.barh(y, vals); plt.yticks(y, labels)
    plt.gca().invert_yaxis(); plt.xlabel("Средняя L1-ошибка"); plt.tight_layout()
    plt.savefig(out_png); plt.close()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["joint_idx","joint_name","mean_L1"])
        for i in range(len(joint_mean)):
            w.writerow([i, names[i], float(joint_mean[i])])

def save_summary(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
