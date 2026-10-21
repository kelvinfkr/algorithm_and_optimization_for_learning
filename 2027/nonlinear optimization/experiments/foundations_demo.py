#!/usr/bin/env python3
"""Generate exact geometry figures for lectures 1, 2 and 4; no training required."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from common import output_dir, plot_style, save_json

def run(output="results/foundations"):
    out = output_dir(output)
    plot_style()
    fig, ax = plt.subplots(figsize=(6.3, 4.2), constrained_layout=True)
    h = np.diag([1.0, 9.0])
    xx, yy = np.meshgrid(np.linspace(-.4,3.4,180),np.linspace(-1.2,1.3,180))
    ax.contour(xx, yy, 0.5*(xx**2+9*yy**2), levels=[0.1,0.5,1,2,5,9], colors="0.75")
    for eta, label in [(1/9, "GD, lr=1/9"),(1/5, "GD, lr=1/5")]:
        x = np.array([3.0,1.0]); path = [x.copy()]
        for _ in range(18):
            x = x-eta*h@x; path.append(x.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "o-", ms=3, label=label)
    ax.plot([3,0],[1,0],"--",label="Hessian preconditioning, lr=1")
    ax.set(xlabel="x1", ylabel="x2", title="Quadratic geometry: H=diag(1,9)", aspect="equal")
    ax.legend(fontsize=8)
    fig.savefig(out/"quadratic_geometry.pdf");fig.savefig(out/"quadratic_geometry.png",dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.0, 4.6), constrained_layout=True)
    # Euclidean projection of (2,1) onto x>=0, x1+x2<=2 (lecture 2).
    a = np.array([2.,1.]); optimum = np.array([1.5,0.5])
    xx, yy = np.meshgrid(np.linspace(-.4,2.5,180), np.linspace(-.4,2.0,180))
    ax.contour(xx, yy, .5*((xx-2)**2+(yy-1)**2), levels=[.25,.5,1,2,3,4], colors="0.7")
    ax.fill([0,2,0], [0,0,2], color="#cfe8e8", alpha=.8, label="Feasible set: x1+x2 <= 2")
    ax.plot(*a,"o", label="Unconstrained minimizer (2,1)")
    ax.plot(*optimum,"*", ms=13, label="Constrained optimum (1.5,0.5)")
    ax.annotate("",xy=a,xytext=optimum,arrowprops={"arrowstyle":"->","color":"#147d85","lw":2})
    ax.set(xlabel="x1", ylabel="x2", title="KKT: negative gradient is a normal", aspect="equal")
    ax.legend(fontsize=8,loc="upper right")
    fig.savefig(out/"kkt_geometry.pdf"); fig.savefig(out/"kkt_geometry.png",dpi=170)
    plt.close(fig)
    # Dual-norm direction visualization for g=(3,-1).
    fig, ax = plt.subplots(figsize=(5.8,5.2), constrained_layout=True)
    theta = np.linspace(0,2*np.pi,400)
    ax.plot(np.cos(theta), np.sin(theta), label="l2 ball")
    ax.plot([-1,1,1,-1,-1],[-1,-1,1,1,-1],label="linf ball")
    ax.plot([1,0,-1,0,1],[0,1,0,-1,0],label="l1 ball")
    g = np.array([3.,-1.])
    for name, delta in [("l2",-g/np.linalg.norm(g)),("linf",-np.sign(g)),("l1",np.array([-1.,0.]))]:
        ax.annotate("", xy=delta, xytext=(0,0), arrowprops={"arrowstyle":"->","lw":1.8})
        ax.text(delta[0]*1.05, delta[1]*1.05, name)
    ax.set(xlim=(-1.35,1.35),ylim=(-1.35,1.35),aspect="equal",
           xlabel="delta 1",ylabel="delta 2",title="Same gradient, different unit balls")
    ax.legend(loc="lower right")
    fig.savefig(out/"norm_geometry.pdf"); fig.savefig(out/"norm_geometry.png",dpi=170)
    plt.close(fig)
    save_json(out/"exact_examples.json",{
        "projection": {"objective":"0.5*||x-(2,1)||^2","budget":2.,"x_star":[1.5,0.5],
                       "lambda_sum":0.5,"lambda_nonnegative":[0.,0.],
                       "stationarity_residual":float(np.linalg.norm(optimum-a+.5*np.ones(2)))},
        "quadratic": {"eigenvalues":[1.,9.],"gd_lr":[1/9,1/5],"stable_lr_interval":[0.,2/9]}
    })
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",default="results/foundations")
    print(run(p.parse_args().output))
