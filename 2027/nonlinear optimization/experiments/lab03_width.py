#!/usr/bin/env python3
"""Lab 3: matched initial functions, tangent kernels and two-layer width scaling.

NTP: f=m^(-1/2) sum a_j*tanh(w_j.x), raw-parameter SGD rate eta.
MF:  f=m^(-1)   sum a_j*tanh(w_j.x), raw-parameter SGD rate m*eta.
MF is the two-layer mean-field/muP teaching convention, not a general recipe
for deep networks or for Adam. The frozen initial function is subtracted.
"""
from __future__ import annotations
import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from common import environment, output_dir, plot_style, save_json, select_device, synthetic_regression

@dataclass
class Config:
    seed: int = 2027
    steps: int = 120
    batch: int = 32
    widths: tuple = (32, 64, 128)
    rates: tuple = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    device: str = "auto"
    diagnostic_epsilon: float = 1e-12

class TwoLayer(nn.Module):
    def __init__(self, w, a, regime):
        super().__init__()
        self.w = nn.Parameter(w.clone())
        self.a = nn.Parameter(a.clone())
        self.register_buffer("w_initial", w.clone())
        self.register_buffer("a_initial", a.clone())
        self.width, self.regime = w.shape[0], regime
        self.scale = 1/math.sqrt(self.width) if regime == "NTP" else 1/self.width
        self.lr_multiplier = 1 if regime == "NTP" else self.width

    def features(self, x):
        return torch.tanh(x @ self.w.T)

    def raw(self, x):
        return self.scale*(self.features(x) @ self.a)

    def initial_raw(self, x):
        return self.scale*(torch.tanh(x @ self.w_initial.T) @ self.a_initial)

    def forward(self, x):
        return self.raw(x)-self.initial_raw(x).detach()

@torch.no_grad()
def effective_kernel(model, x):
    """Return the normalized probe kernel: (raw_lr/base_lr) * J J.T / n_probe."""
    phi = model.features(x)
    jac_a = model.scale*phi
    jac_w = model.scale*(1-phi.square())[:, :, None]*model.a[None, :, :]*x[:, None, :]
    jac_w = jac_w.reshape(len(x), -1)
    return model.lr_multiplier*(jac_a@jac_a.T+jac_w@jac_w.T)/len(x)

@torch.no_grad()
def mse(model, pair):
    return float(.5*(model(pair[0])-pair[1]).square().mean())

def run(config=Config(), output="results/lab03"):
    out=output_dir(output)
    plot_style()
    device=select_device(config.device)
    data=[(x.to(device),y.to(device)) for x,y in synthetic_regression(
        config.seed,n_train=256,n_val=128,n_test=128,d=4)]
    probe=data[1][0][:32]
    gen=torch.Generator(device="cpu").manual_seed(config.seed+21)
    max_width=max(config.widths)
    w_all=(torch.randn(max_width,4,generator=gen,dtype=torch.float64)/2).to(device)
    a_all=torch.randn(max_width,1,generator=gen,dtype=torch.float64).to(device)
    ids=torch.randint(len(data[0][0]),(config.steps,config.batch),generator=gen).to(device)
    summaries,curves,selected=[],[],{}
    for regime in ("NTP","MF"):
        for width in config.widths:
            best_val,best_model,best_rate=float("inf"),None,None
            for eta in config.rates:
                model=TwoLayer(w_all[:width],a_all[:width],regime)
                kernel0=effective_kernel(model,probe)
                phi0=model.features(probe).detach().clone()
                gram0=phi0@phi0.T/width
                raw0_rms=float(model.initial_raw(probe).square().mean().sqrt())
                initial_centered_rms=float(model(probe).detach().square().mean().sqrt())
                optimizer=torch.optim.SGD(model.parameters(),lr=eta*model.lr_multiplier)
                status="completed"
                for step,batch_ids in enumerate(ids,1):
                    optimizer.zero_grad(set_to_none=True)
                    loss=.5*(model(data[0][0][batch_ids])-data[0][1][batch_ids]).square().mean()
                    if not torch.isfinite(loss):
                        status="nonfinite_loss";break
                    loss.backward();optimizer.step()
                    if step==1 or step%max(1,config.steps//12)==0 or step==config.steps:
                        train_value,val_value=mse(model,data[0]),mse(model,data[1])
                        with torch.no_grad():
                            current_phi=model.features(probe)
                            current_gram=current_phi@current_phi.T/width
                            dw=float((model.w-model.w_initial).square().mean().sqrt())
                            dc=float(torch.linalg.vector_norm(current_gram-gram0)/
                                     (torch.linalg.vector_norm(gram0)+config.diagnostic_epsilon))
                        curves.append({"regime":regime,"width":width,"base_lr":eta,"step":step,
                                       "train_loss":train_value if math.isfinite(train_value) else None,
                                       "val_loss":val_value if math.isfinite(val_value) else None,
                                       "w_displacement_rms":dw if math.isfinite(dw) else None,
                                       "feature_gram_relative_change":dc if math.isfinite(dc) else None})
                        if not math.isfinite(train_value) or not math.isfinite(val_value):
                            status="nonfinite_evaluation";break
                val=mse(model,data[1]) if status=="completed" else float("inf")
                if not math.isfinite(val):
                    if status=="completed": status="nonfinite_validation"
                    val=float("inf")
                summary={
                    "regime":regime,"width":width,"base_lr":eta,"raw_lr":eta*model.lr_multiplier,
                    "validation_loss":val if math.isfinite(val) else None,"status":status,
                    "raw_initial_output_rms":raw0_rms,"centered_initial_output_rms":initial_centered_rms,
                    "initial_normalized_probe_kernel_trace":float(torch.trace(kernel0)),
                    "w_displacement_rms":None,"feature_relative_change":None,
                    "feature_gram_relative_change":None,"effective_kernel_relative_change":None,
                }
                if status=="completed":
                  with torch.no_grad():
                    phi=model.features(probe)
                    gram=phi@phi.T/width
                    kernel=effective_kernel(model,probe)
                    summary.update({
                        "w_displacement_rms":float((model.w-model.w_initial).square().mean().sqrt()),
                        "feature_relative_change":float(torch.linalg.vector_norm(phi-phi0)/(torch.linalg.vector_norm(phi0)+config.diagnostic_epsilon)),
                        "feature_gram_relative_change":float(torch.linalg.vector_norm(gram-gram0)/(torch.linalg.vector_norm(gram0)+config.diagnostic_epsilon)),
                        "effective_kernel_relative_change":float(torch.linalg.vector_norm(kernel-kernel0)/(torch.linalg.vector_norm(kernel0)+config.diagnostic_epsilon)),
                    })
                summaries.append(summary)
                if val<best_val:
                    best_val,best_model,best_rate=val,model,eta
            selected[f"{regime}:{width}"]={
                "base_lr":best_rate,"validation_loss":best_val if best_model is not None else None,
                "test_loss":mse(best_model,data[2]) if best_model is not None else None,
                "selection_at_grid_boundary":best_rate in (min(config.rates),max(config.rates)),
                "status":"selected" if best_model is not None else "all_configurations_failed"}
    for filename,rows in [("width_summary.csv",summaries),("training_curves.csv",curves)]:
        with (out/filename).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    heatmaps=[]
    for j,regime in enumerate(("NTP","MF")):
        values=np.array([[next(r["validation_loss"] if r["validation_loss"] is not None else np.nan for r in summaries
                         if r["regime"]==regime and r["width"]==w and r["base_lr"]==eta)
                         for eta in config.rates] for w in config.widths])
        heatmaps.append(values)
    finite_values=np.concatenate([v[np.isfinite(v)] for v in heatmaps])
    lo=float(np.log10(finite_values).min()) if len(finite_values) else -1.
    hi=float(np.log10(finite_values).max()) if len(finite_values) else 1.
    for ax,regime,values in zip(axes[:2],("NTP","MF"),heatmaps):
        im=ax.imshow(np.log10(values),aspect="auto",origin="lower",vmin=lo,vmax=hi,cmap="viridis")
        ax.set(xticks=range(len(config.rates)),xticklabels=[f"{v:g}" for v in config.rates],
               yticks=range(len(config.widths)),yticklabels=config.widths,
               xlabel="Base learning rate",ylabel="Width",title=f"{regime}: log10 validation loss")
        ax.grid(False)
        for i in range(len(config.widths)):
            for k in range(len(config.rates)):
                label=f"{values[i,k]:.3f}" if np.isfinite(values[i,k]) else "failed"
                ax.text(k,i,label,ha="center",va="center",color="white" if np.isfinite(values[i,k]) else "black",fontsize=8)
    fig.colorbar(im,ax=list(axes[:2]),shrink=.8)
    # Compare feature motion at the SAME base eta, not at each model's best eta.
    fixed_eta=config.rates[len(config.rates)//2]
    for regime in ("NTP","MF"):
        rows=[r for r in summaries if r["regime"]==regime and r["base_lr"]==fixed_eta and r["status"]=="completed"]
        axes[2].loglog([r["width"] for r in rows],[r["feature_relative_change"] for r in rows],"o-",label=regime)
    axes[2].set(xlabel="Width",ylabel="Relative feature change",
                title=f"Fixed steps, batch and base lr={fixed_eta:g}")
    axes[2].legend()
    fig.savefig(out/"lab03_width.pdf");fig.savefig(out/"lab03_width.png",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for regime in ("NTP","MF"):
        for width in config.widths:
            rows=[r for r in curves if r["regime"]==regime and r["width"]==width and r["base_lr"]==fixed_eta]
            label=f"{regime}, m={width}"
            axes[0].plot([r["step"] for r in rows],[r["w_displacement_rms"] for r in rows],label=label)
            axes[1].plot([r["step"] for r in rows],[r["feature_gram_relative_change"] for r in rows],label=label)
    axes[0].set(xlabel="Updates",ylabel="RMS(W-W0)",title=f"Raw parameter motion, base lr={fixed_eta:g}")
    axes[1].set(xlabel="Updates",ylabel="||C-C0||F / (||C0||F + eps)",title="Feature Gram drift")
    for ax in axes:ax.legend(fontsize=8)
    fig.savefig(out/"feature_dynamics.pdf");fig.savefig(out/"feature_dynamics.png",dpi=170)
    plt.close(fig)
    report={"config":asdict(config),"environment":environment(),"actual_device":str(device),
            "summaries":summaries,"selected":selected,
            "conventions":"Both regimes train raw a,w initialized O(1), with tanh and average half-squared loss. "
             "The frozen initial output is subtracted for every input, so all initial predictions are zero "
             "without changing Jacobians. The effective probe kernel includes the raw/base learning-rate "
             "multiplier AND division by n_probe; it is not the unnormalized training kernel. "
             "Failed configurations remain in the declared grid. Boundary selections require an expanded grid "
             "before claiming a learning-rate optimum. "
             "MF is a two-layer SGD convention; no claim about general deep Adam parameterizations. "
             "Finite-width trends are measured, not assumed; three widths do not establish an asymptotic theorem."}
    save_json(out/"report.json",report)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="results/lab03")
    p.add_argument("--steps",type=int,default=120)
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--device",default="auto")
    p.add_argument("--seed",type=int,default=2027)
    a=p.parse_args()
    report=run(Config(seed=a.seed,steps=a.steps,batch=a.batch,device=a.device),a.output)
    print(report["selected"])
