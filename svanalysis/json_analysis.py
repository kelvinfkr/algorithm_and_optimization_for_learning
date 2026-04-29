import json
import numpy as np
import csv
from pathlib import Path

# ====== 配置 ======
JSON_PATH = Path("momentum_sv.json")
CSV_PATH = Path("mp_outliers_summary.csv")
TOL = 0.01  # 超出 bulk 上界 1% 以上算异常，可以自己调

# ====== 读 JSON（兼容 list / jsonl 两种格式） ======
def load_momentum_sv(path: Path):
    text = path.read_text().strip()
    if not text:
        raise ValueError("momentum_sv.json 是空的")

    # 如果是以 '[' 开头，认为是一个大 list
    if text[0] == "[":
        data = json.loads(text)
        # 确保是 list[dict]
        if isinstance(data, dict):
            data = [data]
        return data
    else:
        # 否则按 JSON lines 一行一个 dict
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records

# ====== 用 MP 理论检测 outlier 奇异值 ======
def mp_outlier_count(sv, shape, tol=TOL):
    """
    sv: 一维 np.array 的奇异值
    shape: [m, n]
    返回：n_outliers, lambda_plus, sigma2_hat
    """
    sv = np.asarray(sv, dtype=np.float64)
    m, n = shape
    m = float(m)
    n = float(n)

    if sv.size == 0:
        return 0, np.nan, np.nan

    # aspect ratio
    q = m / n

    # 把奇异值当成 G 的奇异值，构造协方差 (1/n) G^T G 的特征值
    # λ_i = s_i^2 / n
    lambdas = (sv ** 2) / n

    # 估计噪声方差 σ^2（MP 分布均值 = σ^2）
    sigma2_hat = float(lambdas.mean())

    # MP bulk 上界 λ_+
    lambda_plus = sigma2_hat * (1.0 + np.sqrt(q)) ** 2

    # 判定 outlier：超出 λ_+ 的一定比例
    threshold = lambda_plus * (1.0 + tol)
    n_outliers = int((lambdas > threshold).sum())

    return n_outliers, lambda_plus, sigma2_hat

def main():
    records = load_momentum_sv(JSON_PATH)

    rows = []
    for rec in records:
        step = rec.get("step")
        layer = rec.get("layer")
        shape = rec.get("shape")
        sv = rec.get("sv")

        if shape is None or sv is None:
            # 跳过不完整的记录
            continue

        m, n = shape
        sv_arr = np.array(sv, dtype=np.float64)

        n_out, lam_plus, sigma2 = mp_outlier_count(sv_arr, shape)

        q = m / n

        rows.append({
            "step": step,
            "layer": layer,
            "m": m,
            "n": n,
            "q": q,
            "n_sv": sv_arr.size,
            "n_outliers": n_out,
            "lambda_plus": lam_plus,
            "sigma2_hat": sigma2,
        })

    # ====== 写 CSV ======
    fieldnames = [
        "step",
        "layer",
        "m",
        "n",
        "q",
        "n_sv",
        "n_outliers",
        "lambda_plus",
        "sigma2_hat",
    ]

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Done. 写入 {CSV_PATH}，共 {len(rows)} 行。")

if __name__ == "__main__":
    main()
