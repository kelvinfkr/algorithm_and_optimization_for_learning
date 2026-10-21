"""Small, explicit building blocks shared by the teaching experiments."""
from __future__ import annotations
import json
import platform
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def environment():
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "matplotlib": matplotlib.__version__}
    try:
        import torch
        info.update(torch=torch.__version__, cuda_available=torch.cuda.is_available())
    except ImportError:
        info["torch"] = None
    return info

def plot_style():
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.2, "figure.dpi": 120,
        "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    })

def select_device(name="auto"):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu or install the CUDA build of PyTorch.")
    torch.set_num_threads(1)
    return torch.device(name)

def synchronize(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def polar(matrix, method="ns", steps=5):
    """SVD reference or convergent cubic Newton--Schulz approximation.

    This uses the classical cubic iteration, not the tuned quintic polynomial
    in some production Muon variants. Five steps are not an exact polar factor.
    """
    import torch
    if matrix.ndim != 2:
        raise ValueError("polar expects a matrix")
    scale = torch.linalg.vector_norm(matrix)
    if scale.item() == 0:
        return torch.zeros_like(matrix)
    if method == "svd":
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        cutoff = torch.finfo(s.dtype).eps * max(matrix.shape) * s.max()
        return (u * (s > cutoff).to(s.dtype)) @ vh
    if method != "ns":
        raise ValueError("method must be ns or svd")
    x = matrix / scale
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        x = 1.5 * x - 0.5 * (x @ x.T) @ x
    return x.T if transposed else x

def synthetic_regression(seed=2027, n_train=512, n_val=256, n_test=256, d=12):
    """Independent train/validation/test sets; scale target with train statistics."""
    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scale = torch.logspace(-0.4, 0.4, d, dtype=torch.float64)
    x = torch.randn(n_train+n_val+n_test, d, generator=generator, dtype=torch.float64) * scale
    y = (torch.sin(2*x[:, 0]) + 0.5*x[:, 1]*x[:, 2]
         + 0.2*(x[:, 3].square()-scale[3].square())
         + 0.05*torch.randn(x.shape[0], generator=generator, dtype=torch.float64)).unsqueeze(1)
    target_mean = y[:n_train].mean()
    target_std = y[:n_train].std()
    y = (y-target_mean)/target_std
    a, b = n_train, n_train+n_val
    return [(x[:a], y[:a]), (x[a:b], y[a:b]), (x[b:], y[b:])]
