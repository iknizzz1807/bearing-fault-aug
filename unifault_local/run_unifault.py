"""run_unifault.py — drive UniFault fine-tuning for tiny/base on local GPU.

Runs 6 training jobs: {tiny_foundation, base_scratch} x {K=10,20,50}.
Each run writes checkpoints/.../classification_report.txt, from which we
parse the FAULT-class recall (label 1) as the metric of interest.

Usage (from repo root):
  python unifault_local/run_unifault.py
"""
from __future__ import annotations

import os
import re
import sys
import glob
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.path.dirname(os.path.abspath(__file__))
UNIFAULT = os.path.join(LOCAL, "UniFault")
DATA_ROOT = os.path.join(LOCAL, "data")
CKPT_ROOT = os.path.join(LOCAL, "checkpoints")
RESULTS = os.path.join(LOCAL, "results.txt")

K_LIST = (10, 20, 50)

# model_type -> (description, load_from_pretrained, n_epochs)
MODELS = {
    "tiny": ("tiny_foundation", True, 30),
    "base": ("base_scratch", False, 30),
}

# Which label is "fault"? UniFault writes classes 0 (normal) / 1 (fault).
FAULT_LABEL = 1


def patch_utils() -> None:
    """Upstream utils.py uses importlib.util + ast without importing them. Fix it."""
    p = os.path.join(UNIFAULT, "utils.py")
    src = open(p).read()
    if "import importlib" not in src:
        src = src.replace(
            "from sklearn.metrics import classification_report, accuracy_score\n",
            "from sklearn.metrics import classification_report, accuracy_score\n"
            "import importlib\nimport ast\n",
        )
        open(p, "w").write(src)
        print("[patch] utils.py: added importlib + ast")
    else:
        print("[patch] utils.py: already patched")


def run_finetune(model_type: str, K: int) -> int:
    desc, load_pt, epochs = MODELS[model_type]
    data_id = f"ims_K{K}"
    model_id = f"{desc}_K{K}"
    print(f"\n{'=' * 70}\n[{desc}] K={K} data_id={data_id}\n{'=' * 70}", flush=True)
    cmd = [
        sys.executable, os.path.join(UNIFAULT, "fine_tune.py"),
        "--data_path", DATA_ROOT,
        "--data_id", data_id,
        "--data_percentage", "100",
        "--model_type", model_type,
        "--model_id", model_id,
        "--gpu_id", "0",
        "--num_epochs", str(epochs),
        "--patience", "20",
        "--batch_size", "64",
        "--lr", "3e-4",
        "--load_from_pretrained", str(load_pt),
        "--pretraining_epoch_id", "1",
        "--random_seed", "42",
    ]
    print(" ".join(os.path.basename(c) if "UniFault" in c else c for c in cmd), flush=True)
    env = dict(os.environ)
    r = subprocess.run(cmd, cwd=LOCAL, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{desc} K={K}] FAILED rc={r.returncode}")
        print("STDOUT tail:", (r.stdout or "")[-2500:])
        print("STDERR tail:", (r.stderr or "")[-2000:])
        return r.returncode
    print(f"[{desc} K={K}] OK", flush=True)
    return 0


def find_report(desc: str, K: int) -> str | None:
    pats = sorted(glob.glob(os.path.join(
        CKPT_ROOT, f"*{desc}_K{K}*", "classification_report.txt")))
    return pats[-1] if pats else None


def parse_fault_recall(report_path: str) -> float | None:
    """Parse recall of the fault class (line 'weighted avg'/'1')."""
    if not report_path or not os.path.exists(report_path):
        return None
    lines = open(report_path).read().splitlines()
    for ln in lines:
        m = re.match(r"\s*1\s+[\d.]+\s+([\d.]+)\s+", ln)
        if m:
            return float(m.group(1))
    return None


def extract_all() -> None:
    out = []
    for model_type, (desc, _, _) in MODELS.items():
        for K in K_LIST:
            rep = find_report(desc, K)
            rec = parse_fault_recall(rep)
            out.append(f"{desc:>16} K={K:<3} fault_recall={rec if rec is not None else 'MISSING':>5}")
    txt = "\n".join(out)
    print("\n" + "=" * 70 + "\nRECALL SUMMARY\n" + "=" * 70, flush=True)
    print(txt, flush=True)
    with open(RESULTS, "w") as f:
        f.write(txt + "\n")
    print("\nsaved ->", RESULTS)


def main() -> None:
    patch_utils()
    os.makedirs(CKPT_ROOT, exist_ok=True)
    for model_type in MODELS:
        for K in K_LIST:
            rc = run_finetune(model_type, K)
            if rc != 0:
                print(f"[warn] continue after rc={rc}", flush=True)
    extract_all()


if __name__ == "__main__":
    main()
