#!/usr/bin/env python3
"""Build legible, single-column handout figures from the recorded lab outputs."""
from pathlib import Path
import argparse
import csv
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from common import output_dir, plot_style
from lab01_sgd import second_moment_factor


def run(results="results",output="figures"):
    root,out = Path(results),output_dir(output)
    plot_style()
    for folder,stems in [("foundations",["quadratic_geometry","kkt_geometry","norm_geometry"]),
                         ("lab02",["polar_approximation"])]:
        for stem in stems:
            for extension in ("pdf","png"):
                shutil.copyfile(root/folder/f"{stem}.{extension}",out/f"{stem}.{extension}")
    moment = np.genfromtxt(root/"lab01/moment_paths.csv",delimiter=",",names=True)
    report = json.loads((root/"lab01/report.json").read_text())
    fig,axes = plt.subplots(2,1,figsize=(6.4,6.8),constrained_layout=True)
    axes[0].semilogy(moment["step"],abs(moment["theory_mean"]),label="|E[e]|, exact")
    axes[0].semilogy(moment["step"],moment["theory_second"],label="E[e^2], exact")
    axes[0].semilogy(moment["step"],moment["empirical_second"],"--",label="E[e^2], Monte Carlo")
    axes[0].set(xlabel="Updates",ylabel="Moment",title="B=1, learning rate=0.9: different notions of stability")
    axes[0].legend(fontsize=9)
    grid = np.linspace(0,2.05,180)
    for batch in (1,4,16):
        line, = axes[1].plot(grid,second_moment_factor(grid,batch),label=f"B={batch}")
        rows = [r for r in report["one_step_checks"] if r["batch"]==batch]
        axes[1].scatter([r["lr"] for r in rows],[r["empirical_q"] for r in rows],s=15,color=line.get_color())
    axes[1].axhline(1,color="black",ls=":",lw=1)
    axes[1].set(xlabel="Learning rate",ylabel="One-step factor q",title="Mean-square contraction requires q < 1",ylim=(0,6))
    axes[1].legend(fontsize=9)
    fig.savefig(out/"sgd_stability.pdf");fig.savefig(out/"sgd_stability.png",dpi=180)
    plt.close(fig)
    rows = list(csv.DictReader((root/"lab03/training_curves.csv").open()))
    width_report = json.loads((root/"lab03/report.json").read_text())
    rates = width_report["config"]["rates"]
    fixed_eta = rates[len(rates)//2]
    widths = width_report["config"]["widths"]
    fig,axes = plt.subplots(2,1,figsize=(6.4,6.2),constrained_layout=True)
    for regime in ("NTP","MF"):
        for width in (min(widths),max(widths)):
            selected = [r for r in rows if r["regime"]==regime and int(r["width"])==width
                        and float(r["base_lr"])==fixed_eta]
            for ax,field in zip(axes,("w_displacement_rms","feature_gram_relative_change")):
                ax.plot([int(r["step"]) for r in selected],
                        [float(r[field]) if r[field] else np.nan for r in selected],
                        label=f"{regime}, m={width}")
    axes[0].set(xlabel="Updates",ylabel="RMS(W-W0)",title=f"Raw parameter motion, base learning rate={fixed_eta:g}")
    axes[1].set(xlabel="Updates",ylabel="Relative feature Gram change",title="Fixed probe inputs; centered initial functions")
    for ax in axes:ax.legend(fontsize=9)
    fig.savefig(out/"width_dynamics.pdf");fig.savefig(out/"width_dynamics.png",dpi=180)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",default="results")
    p.add_argument("--output",default="figures")
    a = p.parse_args()
    print(run(a.results,a.output))
