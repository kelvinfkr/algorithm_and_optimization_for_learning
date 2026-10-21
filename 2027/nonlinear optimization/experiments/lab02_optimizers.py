#!/usr/bin/env python3
"""Lab 2: comparable SGD/AdamW/Muon experiments with explicit parameter roles."""
from __future__ import annotations
import argparse
import copy
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from common import (environment, output_dir, plot_style, polar, save_json,
                    select_device, synchronize, synthetic_regression)

@dataclass
class Config:
    seed: int = 2027
    steps: int = 200
    batch: int = 64
    width: int = 32
    device: str = "auto"
    polar_method: str = "ns"
    ns_steps: int = 5
    weight_decay: float = 0.0

class MLP(nn.Module):
    def __init__(self, d=12, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
    def forward(self, x):
        return self.net(x)

class TeachingMuon:
    """EMA + polar on hidden matrices; AdamW for biases and final readout."""
    def __init__(self, model, lr, aux_lr, config):
        self.matrices = [model.net[0].weight, model.net[2].weight]
        matrix_ids = {id(p) for p in self.matrices}
        other = [p for p in model.parameters() if id(p) not in matrix_ids]
        self.aux = torch.optim.AdamW(other, lr=aux_lr, weight_decay=config.weight_decay)
        self.buffers = [torch.zeros_like(p) for p in self.matrices]
        self.lr, self.config = lr, config

    @torch.no_grad()
    def step(self):
        for p, m in zip(self.matrices, self.buffers):
            if p.grad is None:
                continue
            m.mul_(0.95).add_(p.grad, alpha=0.05)
            direction = polar(m, self.config.polar_method, self.config.ns_steps)
            scale = 0.2*math.sqrt(max(p.shape))
            p.mul_(1-self.lr*self.config.weight_decay)
            p.add_(direction, alpha=-self.lr*scale)
        self.aux.step()

def loss_on(model, pair):
    with torch.no_grad():
        return float(0.5*(model(pair[0])-pair[1]).square().mean())

def run(config=Config(), output="results/lab02"):
    out = output_dir(output)
    plot_style()
    device = select_device(config.device)
    torch.manual_seed(config.seed)
    data = [(x.to(device), y.to(device)) for x, y in synthetic_regression(config.seed)]
    initial = MLP(width=config.width).to(device=device, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(config.seed+11)
    batch_ids = torch.randint(len(data[0][0]), (config.steps, config.batch), generator=generator).to(device)
    grids = {"SGD+momentum": [0.01, 0.03, 0.1],
             "AdamW": [0.001, 0.003, 0.01],
             "Muon+AdamW": [0.01, 0.03, 0.1]}
    rows, candidates, selected = [], [], {}
    log_every = max(1, config.steps//20)
    for name, rates in grids.items():
        best_value, best_state, best_candidate = float("inf"), None, None
        for lr in rates:
            model = copy.deepcopy(initial)
            if name == "SGD+momentum":
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                            weight_decay=config.weight_decay)
            elif name == "AdamW":
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                              weight_decay=config.weight_decay)
            else:
                optimizer = TeachingMuon(model, lr=lr, aux_lr=lr/10, config=config)
            candidate_id = f"{name}:lr={lr:g}"
            synchronize(device); start = time.perf_counter()
            status = "completed"
            for step, ids in enumerate(batch_ids, 1):
                model.zero_grad(set_to_none=True)
                loss = 0.5*(model(data[0][0][ids])-data[0][1][ids]).square().mean()
                if not torch.isfinite(loss):
                    status = "nonfinite_loss"; break
                loss.backward()
                record = step == 1 or step % log_every == 0 or step == config.steps
                before = [p.detach().clone() for p in model.parameters()] if record else None
                optimizer.step()
                if record:
                    with torch.no_grad():
                        delta_sq = weight_sq = grad_sq = descent_inner = 0.0
                        count = 0
                        for p, old in zip(model.parameters(), before):
                            delta = p-old
                            delta_sq += float(delta.square().sum())
                            weight_sq += float(old.square().sum())
                            grad_sq += float(p.grad.square().sum())
                            descent_inner -= float((delta*p.grad).sum())
                            count += p.numel()
                    synchronize(device)
                    rows.append({"candidate":candidate_id,"optimizer":name,"lr":lr,"step":step,
                                 "samples":step*config.batch,
                                 "train_loss":loss_on(model,data[0]),"val_loss":loss_on(model,data[1]),
                                 "update_rms":math.sqrt(delta_sq/count),
                                 "relative_update":math.sqrt(delta_sq/(weight_sq+1e-30)),
                                 "descent_cosine":descent_inner/math.sqrt(max(grad_sq*delta_sq,1e-60)),
                                 "seconds":time.perf_counter()-start})
            synchronize(device)
            elapsed = time.perf_counter()-start
            val = loss_on(model,data[1]) if status=="completed" else float("inf")
            if not math.isfinite(val):
                val = float("inf"); status = "nonfinite_validation"
            candidate = {"id":candidate_id,"optimizer":name,"lr":lr,"status":status,
                         "aux_lr":lr/10 if name=="Muon+AdamW" else None,
                         "validation_loss":val if math.isfinite(val) else None,"seconds":elapsed}
            candidates.append(candidate)
            if val < best_value:
                best_value, best_candidate = val, candidate_id
                best_state = copy.deepcopy(model.state_dict())
        if best_state is None:
            raise RuntimeError(f"All candidate configurations failed for {name}")
        chosen = copy.deepcopy(initial); chosen.load_state_dict(best_state)
        # The held-out test split is touched only after validation-based selection.
        selected[name] = {"candidate":best_candidate,"validation_loss":best_value,
                          "test_loss":loss_on(chosen,data[2])}
    with (out/"all_curves.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(14,4.1),constrained_layout=True)
    for name,info in selected.items():
        r=[v for v in rows if v["candidate"]==info["candidate"]]
        axes[0].semilogy([v["step"] for v in r],[v["val_loss"] for v in r],label=name)
        axes[1].semilogy([v["step"] for v in r],[v["relative_update"] for v in r],label=name)
        axes[2].plot([v["step"] for v in r],[v["descent_cosine"] for v in r],label=name)
    for ax in axes: ax.set_xlabel("Updates"); ax.legend(fontsize=8)
    axes[0].set(ylabel="Validation loss",title="Best of three declared LR configurations")
    axes[1].set(ylabel="||delta theta|| / ||theta||",title="Relative update size")
    axes[2].set(ylabel="-<g,delta> / (||g|| ||delta||)",title="Alignment with current descent direction")
    fig.savefig(out/"lab02_optimizers.pdf");fig.savefig(out/"lab02_optimizers.png",dpi=170)
    plt.close(fig)
    # Finite NS steps versus the exact SVD reference on a prescribed spectrum.
    s=torch.tensor([1.,.4,.1,.01],dtype=torch.float64,device=device)
    g=torch.diag(s)
    singular={"gradient":s.cpu().tolist(),"exact_polar":torch.linalg.svdvals(polar(g,"svd")).cpu().tolist()}
    for k in (1,3,5,10):
        singular[f"ns_{k}"]=torch.linalg.svdvals(polar(g,"ns",k)).cpu().tolist()
    fig,ax=plt.subplots(figsize=(6.2,4.1),constrained_layout=True)
    for label,values in singular.items():
        ax.plot(np.arange(1,5),values,"o-",label=label)
    ax.set(xlabel="Ordered singular-value index",ylabel="Singular value",
           title="Finite polynomial steps are not exact polar factors",xticks=[1,2,3,4])
    ax.legend(fontsize=8)
    fig.savefig(out/"polar_approximation.pdf");fig.savefig(out/"polar_approximation.png",dpi=170)
    plt.close(fig)
    report={"config":asdict(config),"environment":environment(),"actual_device":str(device),
            "candidates":candidates,"selected":selected,"polar_example":singular,
            "protocol":"Common initialization, train/validation/test split and minibatch index stream. "
                       "Three LR configurations per method, final validation-loss selection, one final test evaluation. "
                       "Default weight decay is zero; positive SGD weight decay is coupled L2, unlike AdamW/Muon. "
                       "Timing includes diagnostics and evaluation; this is not a hardware performance benchmark."}
    save_json(out/"report.json",report)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="results/lab02")
    p.add_argument("--steps",type=int,default=200)
    p.add_argument("--width",type=int,default=32)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--device",default="auto")
    p.add_argument("--polar",choices=["ns","svd"],default="ns")
    p.add_argument("--ns-steps",type=int,default=5)
    p.add_argument("--seed",type=int,default=2027)
    p.add_argument("--weight-decay",type=float,default=0.)
    a=p.parse_args()
    r=run(Config(seed=a.seed,steps=a.steps,batch=a.batch,width=a.width,device=a.device,
                 polar_method=a.polar,ns_steps=a.ns_steps,weight_decay=a.weight_decay),a.output)
    print(r["selected"])
