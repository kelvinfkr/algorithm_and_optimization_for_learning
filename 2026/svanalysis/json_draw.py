import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========== 配置 ==========
JSON_PATH = Path("momentum_sv.json")
CSV_PATH = Path("mp_outliers_summary.csv")
FIG_DIR = Path("mp_outlier_figs")
TOL = 0.01  # 超出 MP 上界 1% 以上算异常


# ========== 读 JSON（兼容 list / jsonl） ==========
def load_momentum_sv(path: Path):
    text = path.read_text().strip()
    if not text:
        raise ValueError("momentum_sv.json 是空的")

    # 以 "[" 开头 → 认为是一个大 list
    if text[0] == "[":
        data = json.loads(text)
        if isinstance(data, dict):
            data = [data]
        return data
    else:
        # 否则按 JSON lines
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records


# ========== MP 理论检测 outlier ==========
def mp_outlier_count(sv, shape, tol=TOL):
    """
    sv: 一维奇异值数组
    shape: [m, n]
    返回: n_outliers, lambda_plus, sigma2_hat
    """
    sv = np.asarray(sv, dtype=np.float64)
    m, n = shape
    m = float(m)
    n = float(n)

    if sv.size == 0:
        return 0, np.nan, np.nan

    # aspect ratio
    q = m / n

    # λ_i = s_i^2 / n  当成 (1/n) G^T G 的特征值
    lambdas = (sv ** 2) / n

    # MP 分布的均值 = σ^2
    sigma2_hat = float(lambdas.mean())

    # MP bulk 上界 λ_+
    lambda_plus = sigma2_hat * (1.0 + np.sqrt(q)) ** 2

    # 超过 λ_+ 一定比例算 outlier
    threshold = lambda_plus * (1.0 + tol)
    n_outliers = int((lambdas > threshold).sum())

    return n_outliers, lambda_plus, sigma2_hat


# ========== 主流程：算 k + 写 CSV + 画图 ==========
def main():
    records = load_momentum_sv(JSON_PATH)

    rows = []
    for rec in records:
        step = rec.get("step")
        layer = rec.get("layer")
        shape = rec.get("shape")
        sv = rec.get("sv")

        if shape is None or sv is None:
            # 跳过不完整记录
            continue

        m, n = shape
        sv_arr = np.array(sv, dtype=np.float64)

        n_out, lam_plus, sigma2 = mp_outlier_count(sv_arr, shape)

        q = m / n

        rows.append(
            {
                "step": step,
                "layer": layer,
                "m": m,
                "n": n,
                "q": q,
                "n_sv": sv_arr.size,
                "n_outliers": n_out,  # k
                "lambda_plus": lam_plus,
                "sigma2_hat": sigma2,
            }
        )

    if not rows:
        print("没有有效数据行，检查一下 momentum_sv.json 格式？")
        return

    # ---- 保存成 CSV ----
    df = pd.DataFrame(rows)
    df.sort_values(["layer", "step"], inplace=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"[INFO] 写入 CSV: {CSV_PATH}，共 {len(df)} 行。")

    # ---- 按 layer 画图 ----
    FIG_DIR.mkdir(exist_ok=True)

    for layer_name, g in df.groupby("layer"):
        # 有些 step 可能 None，先丢掉
        g = g.dropna(subset=["step"])
        if g.empty:
            continue

        g = g.sort_values("step")
        steps = g["step"].to_numpy()
        k_vals = g["n_outliers"].to_numpy()

        plt.figure()
        plt.plot(steps, k_vals, marker="o")
        plt.xlabel("step")
        plt.ylabel("n_outliers (k)")
        plt.title(f"MP outliers vs step\nlayer = {layer_name}")
        plt.grid(True, linestyle="--", alpha=0.5)

        # layer 名字转成安全的文件名
        safe_layer = (
            str(layer_name)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )
        fig_path = FIG_DIR / f"mp_outliers_{safe_layer}.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

        print(f"[INFO] 保存图像: {fig_path}")

    print("[INFO] 全部完成。")


if __name__ == "__main__":
    main()
