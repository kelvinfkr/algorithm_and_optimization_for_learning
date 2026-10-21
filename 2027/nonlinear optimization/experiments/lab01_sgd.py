#!/usr/bin/env python3
"""Lab 1: exact scalar SGD moments, stability and batch/learning-rate scaling."""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from common import environment, output_dir, plot_style, save_json

@dataclass
class Config:
    seed: int = 2027
    paths: int = 20000
    steps: int = 30
    noise_std: float = 0.2

def second_moment_factor(eta, batch, variance=1.0):
    return (1-eta*variance)**2 + 2*eta**2*variance**2/batch

def exact_moments(eta, batch, steps, noise_std=0.0, variance=1.0, e0=1.0):
    mean = np.empty(steps+1)
    second = np.empty(steps+1)
    mean[0], second[0] = e0, e0**2
    q = second_moment_factor(eta, batch, variance)
    for t in range(steps):
        mean[t+1] = (1-eta*variance)*mean[t]
        second[t+1] = q*second[t] + eta**2*variance*noise_std**2/batch
    return mean, second

def run(config=Config(), output="results/lab01"):
    out = output_dir(output)
    plot_style()
    rng = np.random.default_rng(config.seed)
    # Multiplicative-noise example: rare paths can dominate the second moment.
    eta = 0.9
    e = np.ones(config.paths)
    empirical = [[0, e.mean(), np.mean(e**2), np.median(np.abs(e))]]
    for t in range(config.steps):
        x = rng.normal(size=config.paths)
        e *= 1-eta*x*x
        empirical.append([t+1, e.mean(), np.mean(e**2), np.median(np.abs(e))])
    empirical = np.asarray(empirical)
    mean, second = exact_moments(eta, 1, config.steps)
    np.savetxt(out/"moment_paths.csv",
               np.column_stack([empirical, mean, second]),
               delimiter=",", header="step,empirical_mean,empirical_second,median_abs,theory_mean,theory_second",
               comments="")
    grid = np.linspace(0.0, 2.05, 150)
    one_step = []
    for batch in (1, 4, 16):
        x2 = np.mean(rng.normal(size=(config.paths*4, batch))**2, axis=1)
        for rate in (0.3, 0.65, 0.9, 1.5, 1.8):
            empirical_q = float(np.mean((1-rate*x2)**2))
            theory_q = second_moment_factor(rate, batch)
            one_step.append({"batch": batch, "lr": rate, "empirical_q": empirical_q,
                             "theory_q": theory_q,
                             "mean_square_lr_limit": 2*batch/(batch+2)})
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1), constrained_layout=True)
    t = np.arange(config.steps+1)
    axes[0].semilogy(t, np.abs(mean), label="|E[e]|: exact")
    axes[0].semilogy(t, second, label="E[e^2]: exact")
    axes[0].semilogy(t, empirical[:, 2], "--", label="E[e^2]: Monte Carlo")
    axes[0].set(xlabel="Updates", ylabel="Moment", title="Mean vs mean-square (B=1, lr=0.9)")
    axes[0].legend(fontsize=8)
    for batch in (1, 4, 16):
        line, = axes[1].plot(grid, second_moment_factor(grid, batch), label=f"B={batch}")
        rows = [r for r in one_step if r["batch"] == batch]
        axes[1].scatter([r["lr"] for r in rows], [r["empirical_q"] for r in rows],
                        color=line.get_color(), s=15)
    axes[1].axhline(1, color="black", ls=":", lw=1)
    axes[1].set(xlabel="Learning rate", ylabel="One-step second-moment factor",
                title="Mean-square stability requires q < 1", ylim=(0, 6))
    axes[1].legend()
    scaling_rows = []
    sample_budget = 1600
    for batch in (1, 4, 16):
        lr = 0.04*batch
        steps = sample_budget//batch
        _, mse = exact_moments(lr, batch, steps, config.noise_std)
        axes[2].semilogy(batch*np.arange(steps+1), mse, label=f"B={batch}, lr={lr:g}")
        for step, value in enumerate(mse):
            scaling_rows.append((batch, lr, step, batch*step, value))
    axes[2].set(xlabel="Processed samples", ylabel="E[e^2]: exact",
                title="Linear scaling is a small-step approximation")
    axes[2].legend(fontsize=8)
    fig.savefig(out/"lab01_sgd.pdf")
    fig.savefig(out/"lab01_sgd.png", dpi=170)
    plt.close(fig)
    np.savetxt(out/"batch_scaling.csv", scaling_rows, delimiter=",",
               header="batch,lr,step,processed_samples,exact_mse", comments="")
    report = {"config": asdict(config), "environment": environment(), "one_step_checks": one_step,
              "interpretation": "Finite Monte Carlo can miss rare paths that dominate a growing second moment. "
              "The analytic curve is an expectation, not a guarantee for every trajectory."}
    save_json(out/"report.json", report)
    return report

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/lab01")
    p.add_argument("--paths", type=int, default=20000)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=2027)
    a = p.parse_args()
    run(Config(seed=a.seed, paths=a.paths, steps=a.steps), a.output)
    print(Path(a.output).resolve())
