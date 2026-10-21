"""
Lecture 7 Optimizer Comparison: Multi-GPU Benchmark
====================================================

按讲义《从范数约束看优化器：SGD → Muon》的三条主线 + 2 个现代扩展，
在 124M GPT-2 上对比 10 个优化器。每卡一个优化器，独立 log。

  ℓ₂ 路线（§2–§3）:
    SGD         —— 朴素 GD（§2）
    HeavyBall   —— + Polyak 动量（§3.1）
    Nesterov    —— + Nesterov lookahead（§3.2）
    SGD_WD      —— + 权重衰减（§3.4）

  ℓ∞ + 谱范数路线（§4–§10）:
    SignGD      —— ℓ∞ 最优（§4）
    Adam        —— SignGD + 动量 + 二阶矩（§5）
    AdamW       —— + 解耦权重衰减（§5.8）
    Muon        —— 谱范数最优 ∝ UV^T（§10）

  现代扩展（讲义之后几年的代表）:
    Lion        —— SignGD + 双动量（Chen+2023，SignGD 的现代改良）
    Sophia      —— 二阶对角 + 截断（Liu+2023，Adam 的二阶改良）

数据：FineWeb-Edu，micro_batch × grad_accum × seq_len = tokens/step

用法：
    python train_8gpu.py --all                         # 所有 10 个（需≥10 卡）
    python train_8gpu.py --group A                     # 4 GPU：ℓ₂ 路线
    python train_8gpu.py --group B                     # 4 GPU：ℓ∞ + 谱范数
    python train_8gpu.py --group C                     # 2 GPU：Lion + Sophia
    CUDA_VISIBLE_DEVICES=0 python train_8gpu.py --optimizer Sophia --gpu 0
"""

import os
import sys
import math
import time
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW, Adam, SGD

# ============================================================================
# Config
# ============================================================================

CONFIG = {
    # Model: 124M GPT-2
    'n_layer': 12,
    'n_head': 12,
    'n_embd': 768,
    'block_size': 1024,
    'vocab_size': 50257,
    'dropout': 0.0,

    # Training
    'micro_batch_size': int(os.environ.get('MICRO_BATCH', 16)),
    'grad_accum_steps': int(os.environ.get('GRAD_ACCUM', 4)),
    # effective batch = 16 * 4 = 64；262K tokens/step
    'total_tokens': int(os.environ.get('TOTAL_TOKENS', 10_000_000_000)),   # 10B
    'warmup_tokens': int(os.environ.get('WARMUP_TOKENS', 100_000_000)),    # 100M
    'log_interval': 50,
    'val_interval': 500,
    'val_tokens': 10_000_000,
    'save_interval': 1000,

    # Mixed precision
    'use_amp': True,
    'dtype': 'bfloat16',

    # Data / paths
    'data_dir': os.environ.get('DATA_DIR', './data/fineweb'),
    'log_dir': os.environ.get('LOG_DIR', './logs'),
    'checkpoint_dir': os.environ.get('CKPT_DIR', './checkpoints'),
}

CONFIG['batch_size'] = CONFIG['micro_batch_size'] * CONFIG['grad_accum_steps']
TOKENS_PER_STEP = CONFIG['batch_size'] * CONFIG['block_size']
TOTAL_STEPS = CONFIG['total_tokens'] // TOKENS_PER_STEP


# ============================================================================
# Model (unchanged from original)
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg['n_head']
        self.n_embd = cfg['n_embd']
        self.head_dim = cfg['n_embd'] // cfg['n_head']
        self.c_attn = nn.Linear(cfg['n_embd'], 3 * cfg['n_embd'], bias=False)
        self.c_proj = nn.Linear(cfg['n_embd'], cfg['n_embd'], bias=False)
        self.register_buffer("bias", torch.tril(torch.ones(cfg['block_size'], cfg['block_size']))
                                     .view(1, 1, cfg['block_size'], cfg['block_size']))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg['n_embd'], 4 * cfg['n_embd'], bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg['n_embd'], cfg['n_embd'], bias=False)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg['n_embd'])
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg['n_embd'])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg['vocab_size'], cfg['n_embd']),
            wpe=nn.Embedding(cfg['block_size'], cfg['n_embd']),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg['n_layer'])]),
            ln_f=nn.LayerNorm(cfg['n_embd']),
        ))
        self.lm_head = nn.Linear(cfg['n_embd'], cfg['vocab_size'], bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        print(f"Model params: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ============================================================================
# Data (unchanged)
# ============================================================================

class LocalDataLoader:
    def __init__(self, data_dir, batch_size, block_size, device):
        import numpy as np
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

        token_file = f"{data_dir}/tokens.bin"
        print(f"Loading tokens from {token_file}...")
        self.tokens = np.fromfile(token_file, dtype=np.uint16)
        self.total_tokens = len(self.tokens)
        print(f"Loaded {self.total_tokens:,} tokens ({self.total_tokens/1e9:.2f}B) into RAM")

    def get_batch(self):
        import numpy as np
        max_start = self.total_tokens - self.block_size - 1
        starts = np.random.randint(0, max_start, size=self.batch_size)
        x = np.stack([self.tokens[s:s + self.block_size] for s in starts])
        y = np.stack([self.tokens[s + 1:s + self.block_size + 1] for s in starts])
        x = torch.from_numpy(x.astype(np.int64)).to(self.device)
        y = torch.from_numpy(y.astype(np.int64)).to(self.device)
        return x, y


class StreamingDataLoader:
    def __init__(self, batch_size, block_size, device, split='train'):
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

        print("Loading FineWeb-Edu dataset (streaming)...")
        self.ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", "sample-10BT", split=split, streaming=True
        )
        self.ds_iter = iter(self.ds)
        print("Loading GPT-2 tokenizer...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.buffer = []
        self.buffer_size = int(os.environ.get('BUFFER_SIZE', 300_000_000))

    def _fill_buffer(self):
        if len(self.buffer) >= self.buffer_size:
            return
        print(f"Filling buffer to {self.buffer_size/1e6:.0f}M tokens...")
        while len(self.buffer) < self.buffer_size:
            try:
                example = next(self.ds_iter)
                tokens = self.tokenizer.encode(example['text'])
                self.buffer.extend(tokens)
                if len(self.buffer) % 10_000_000 == 0:
                    print(f"  Buffer: {len(self.buffer)/1e6:.0f}M tokens")
            except StopIteration:
                self.ds_iter = iter(self.ds)
        print(f"Buffer filled: {len(self.buffer)/1e6:.0f}M tokens")

    def get_batch(self):
        self._fill_buffer()
        x_list, y_list = [], []
        for _ in range(self.batch_size):
            start = torch.randint(0, len(self.buffer) - self.block_size - 1, (1,)).item()
            x_list.append(self.buffer[start:start + self.block_size])
            y_list.append(self.buffer[start + 1:start + self.block_size + 1])
        x = torch.tensor(x_list, dtype=torch.long, device=self.device)
        y = torch.tensor(y_list, dtype=torch.long, device=self.device)
        if len(self.buffer) > self.buffer_size * 2:
            self.buffer = self.buffer[-self.buffer_size:]
        return x, y


def get_dataloader(cfg, device):
    data_dir = cfg.get('data_dir', './data/fineweb')
    token_file = f"{data_dir}/tokens.bin"
    micro_batch = cfg.get('micro_batch_size', cfg.get('batch_size', 64))
    if os.path.exists(token_file):
        print(f"Using local data from {data_dir}")
        return LocalDataLoader(data_dir, micro_batch, cfg['block_size'], device)
    else:
        print(f"Local data not found at {token_file}, using streaming...")
        return StreamingDataLoader(micro_batch, cfg['block_size'], device)


# ============================================================================
# Optimizers: Lecture 7 的 8 个优化器
# ============================================================================
#
# 讲义里所有优化器都在解同一个问题：
#     u = argmin_{||Δw|| < η} g^T Δw
# 换一个范数，就得到一个优化器。
#
#   范数 ℓ₂      -> Δw ∝ -g/||g||      -> GD （+动量/NAG/WD 都是在同一范数内的改进）
#   范数 ℓ∞      -> Δw ∝ -sign(g)      -> SignGD  （+ 动量 + 二阶矩归一化 -> Adam）
#   谱范数       -> Δw ∝ -UV^T         -> Muon
#
# ============================================================================


class SignGD(Optimizer):
    """
    讲义 §4: ℓ∞ 范数约束下的最优更新
        argmin_{||Δw||_∞ ≤ η} g^T Δw = -η · sign(g)
    
    每个参数独立地走 ±lr，完全抹平梯度幅值差异。这是 Adam 在二次函数上
    的退化极限（讲义 §4.4）：Adam 的 v 稳定后，m / sqrt(v) ≈ sign(g)。
    
    可选带动量（signed momentum），本质上就是 Lion 优化器的去 β₂ 版本。
    """
    def __init__(self, params, lr=1e-4, momentum=0.0, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            lr, mu, wd = g['lr'], g['momentum'], g['weight_decay']
            for p in g['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    # 解耦 weight decay（AdamW 风格）
                    p.mul_(1 - lr * wd)
                if mu > 0:
                    state = self.state[p]
                    if 'buf' not in state:
                        state['buf'] = torch.zeros_like(p)
                    buf = state['buf']
                    buf.mul_(mu).add_(grad)
                    update = buf
                else:
                    update = grad
                p.add_(torch.sign(update), alpha=-lr)


@torch.no_grad()
def muon_polar_ns(G, steps=5, eps=1e-7):
    """讲义 §10.1: Newton–Schulz 迭代逼近 UV^T
    
    用调优多项式 p(x) = 3.4445x - 4.7750x^3 + 2.0315x^5（Muon 论文）
    自动选择 M≤N 或 M>N 形式以减少乘法次数。必须用 fp32，否则会爆。
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    M, N = G.shape
    X = G.float()
    norm = X.norm() + eps
    X = X / norm
    if M <= N:
        for _ in range(steps):
            A = X @ X.T
            B = A @ X
            X = a * X + b * B + c * (A @ B)
    else:
        for _ in range(steps):
            A = X.T @ X
            B = X @ A
            X = a * X + b * B + c * (B @ A)
    return X * math.sqrt(min(M, N))  # norm ≈ sqrt(min(M,N))，对应谱范数 1


class Muon(Optimizer):
    """讲义 §10: 谱范数约束下的最优更新
        argmin_{||ΔW||_2 ≤ η} <G, ΔW>_F = -η · UV^T
    
    实现：动量 + NS 正交化。2D 以上参数走 NS，1D 走 SGD。
    use_kimi_rms=True 时做 RMS 对齐，让 Muon 可以复用 Adam 的 lr（§9）。
    """
    def __init__(self, params, lr=0.02, momentum=0.95, n_iter=5, eps=1e-8,
                 use_kimi_rms=False, target_rms=0.2):
        defaults = dict(lr=lr, momentum=momentum, n_iter=n_iter, eps=eps,
                        use_kimi_rms=use_kimi_rms, target_rms=target_rms)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            lr = g["lr"]
            mu = g["momentum"]
            n_iter = g["n_iter"]
            eps = g["eps"]
            use_kimi_rms = g["use_kimi_rms"]
            target_rms = g["target_rms"]

            for p in g["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p)
                buf = state["buf"]
                buf.mul_(mu).add_(p.grad)

                if p.ndim < 2:
                    # 1D 参数（LayerNorm、bias、embedding wte 共享权重等）退回 SGD
                    p.add_(buf, alpha=-lr)
                    continue

                X = buf.view(p.size(0), -1)
                M, N = X.shape
                Update = muon_polar_ns(X, steps=n_iter, eps=eps)

                if use_kimi_rms:
                    curr_norm = Update.norm()
                    target_norm = target_rms * math.sqrt(M * N)  # RMS·sqrt(numel) = norm
                    Update = Update * (target_norm / (curr_norm + eps))

                p.add_(Update.view_as(p).to(p.dtype), alpha=-lr)


class Lion(Optimizer):
    """
    Lion (EvoLved Sign Momentum, Chen et al. 2023, Google)

    讲义视角：Lion 是 SignGD 家族的一个改良——把 SignGD 的动量换成"双动量"：
        c_t = β₁·m_{t-1} + (1-β₁)·g_t          ← 瞬时方向（mixing）
        w_{t+1} = w_t - lr · sign(c_t) - lr·λ·w_t
        m_t = β₂·m_{t-1} + (1-β₂)·g_t          ← 长期动量（状态更新）
    
    sign(c_t) 部分和 SignGD 完全同构（ℓ∞ 最优更新），但 β₁ ≠ β₂ 让"更新方向"
    和"状态积累"用不同的时间尺度——这是 Lion 相对于 SignGD+Momentum 的核心差异。
    
    默认 betas=(0.9, 0.99)：β₁=0.9 用于更新方向（短期），β₂=0.99 存长期记忆。
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            lr, (b1, b2), wd = g['lr'], g['betas'], g['weight_decay']
            for p in g['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                m = state['exp_avg']

                # 解耦 weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # 更新方向：sign(β₁·m + (1-β₁)·g)
                update = m.mul(b1).add(grad, alpha=1 - b1).sign_()
                p.add_(update, alpha=-lr)

                # 长期动量（用另一组 beta）
                m.mul_(b2).add_(grad, alpha=1 - b2)


class Sophia(Optimizer):
    """
    Sophia-G (Liu et al. 2023): Second-order Clipped Stochastic Optimization
    
    讲义视角：把 Adam 的 v（二阶矩）换成更接近真 Hessian 对角的估计：
      - Adam：h_t ≈ E[g²] = diag(H) + 噪声+偏差
      - Sophia：h_t 用 Gauss-Newton-Bartlett 估计——从模型自己的预测分布采一组
        "假" 标签 y ~ p_θ(·|x)，再 backward 得到 g_sampled，
        E[g_sampled²] = diag(Fisher) ≈ diag(H_GN)（精确无偏）。

    更新规则（clipped Newton）：
        m_t = β₁·m_{t-1} + (1-β₁)·g_t
        h_t = β₂·h_{t-1} + (1-β₂)·g_sampled²      ← 每 K 步采样一次
        u_t = sign(m_t) · clip(|m_t| / (ρ·B·h_t), 0, 1)
        w_{t+1} = w_t·(1-lr·λ) - lr·u_t

    clip 的作用：如果某个方向 h 很小（loss 很平），|m/h| 可能爆炸，clip 把它
    压在 1——"最多走 lr 这么远"，这防止了牛顿法在鞍点附近的不稳定。
    
    注意：Sophia 需要每 K 步额外做一次 forward+backward 用来估计 Hessian，
    train() 里通过 `hasattr(optimizer, 'update_hessian')` 识别并调用。
    """
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=0.1, bs=65536, eps=1e-15):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay,
                        bs=bs, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self):
        """在 G-N-B 采样后调用：此时 p.grad = ∇L(θ, y~p_θ)，用它更新 h。"""
        for g in self.param_groups:
            _, beta2 = g['betas']
            for p in g['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'hessian' not in state:
                    state['hessian'] = torch.zeros_like(p)
                # h_t = β₂·h_{t-1} + (1-β₂)·g_sampled²
                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, _ = group['betas']
            rho = group['rho']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']
            bs = group['bs']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                if 'hessian' not in state:
                    state['hessian'] = torch.zeros_like(p)

                m = state['exp_avg']
                h = state['hessian']

                # m_t = β₁·m_{t-1} + (1-β₁)·g_t
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)

                # 解耦 weight decay
                if wd > 0:
                    p.mul_(1 - lr * wd)

                # Clipped Newton step: sign(m) · min(|m|/(ρ·B·h), 1)
                ratio = (m.abs() / (rho * bs * h + eps)).clamp(max=1.0)
                p.addcmul_(m.sign(), ratio, value=-lr)


class CautiousAdamW(Optimizer):
    """
    Cautious AdamW (Liang et al. 2024, "Cautious Optimizers: Improving Training
    with One Line of Code")

    讲义视角：AdamW 的 update = m̂/√v̂ 的方向有时和瞬时梯度 g 相反（因为动量惯性
    /二阶矩滞后）。这种"逆势"分量往往是错误方向，直接 mask 掉能系统性地改善
    收敛。和 Muon 的 UV^T（谱范数对齐）是不同的修正思路：
      - Muon：对齐"矩阵的哪个方向"应该被等权更新（跨参数维度）
      - Cautious：对齐"这一步该不该更新"（逐元素的符号一致性）

    核心就是一行 mask（论文标题 "One Line of Code"）：
        u_adam = m̂ / (√v̂ + ε)                  ← 标准 AdamW update
        mask = 1[u_adam · g > 0]                ← 和瞬时梯度同号的分量
        u = u_adam · mask / mean(mask)          ← rescale 保持期望幅度不变
        w ← w·(1 - lr·λ) - lr·u
    """
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                m, v = state['exp_avg'], state['exp_avg_sq']

                m.mul_(b1).add_(grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)

                bc1 = 1 - b1 ** state['step']
                bc2 = 1 - b2 ** state['step']

                # 解耦 weight decay（和 AdamW 一致）
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # 标准 AdamW update
                denom = (v / bc2).sqrt_().add_(eps)
                update = (m / bc1) / denom

                # ── Cautious 一行：mask 掉和瞬时梯度 g 方向相反的分量 ──
                mask = (update * grad > 0).to(update.dtype)
                # rescale: 保持"未被 mask 的那些分量"的平均幅度不变
                mask_scale = mask.mean().clamp_(min=1e-3)
                update = update * (mask / mask_scale)

                p.add_(update, alpha=-lr)


class Shampoo(Optimizer):
    """
    Shampoo (Gupta et al. 2018, "Shampoo: Preconditioned Stochastic Tensor
    Optimization")

    讲义视角（§6–§8 的"预条件路线"终点）：对矩阵参数 W ∈ R^{M×N} 维护
    Kronecker 预条件：
        L_t = β·L_{t-1} + (1-β)·G_t G_t^T       (M×M)
        R_t = β·R_{t-1} + (1-β)·G_t^T G_t       (N×N)
    更新：
        W_{t+1} = W_t · (1-lr·λ) - lr · L_t^{-1/4} G_t R_t^{-1/4}

    讲义 §7.3 的 Shampoo 恒等式说：当 L = G G^T, R = G^T G 时，
        L^{-1/4} G R^{-1/4} = UV^T
    所以在"恰好一步"的极限下 Shampoo 和 Muon 给出同一个 UV^T——这也是讲义
    §6 结尾那个"两条路线汇合"的具体含义。Shampoo 和 Muon 的区别只是：
      - Shampoo：把统计量 L, R 跨步累积（显式预条件）
      - Muon：每步用即时的 G 做 NS 迭代逼近 UV^T（隐式预条件）

    实现注意：
      - eigh 是 O(d^3)，只在 M, N ≤ max_precond_dim 时算；超过则该侧退化为
        identity（对应方向不预条件）。
      - 预条件根每 precond_interval 步重算一次，摊销 eigh 代价。
      - 前 precond_interval 步走纯 SGD+动量（统计量还没积累够）。
    """
    def __init__(self, params, lr=3e-4, momentum=0.9, weight_decay=0.0,
                 beta2=0.95, eps=1e-12, precond_interval=200, max_precond_dim=2048):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        beta2=beta2, eps=eps,
                        precond_interval=precond_interval,
                        max_precond_dim=max_precond_dim)
        super().__init__(params, defaults)

    @staticmethod
    def _matrix_inv_root(mat, power=-0.25, eps=1e-12):
        """对称 PSD 矩阵 mat 的 mat^power，默认 -1/4（Shampoo 的那个幂次）。"""
        mat = mat.float()
        n = mat.shape[0]
        # jitter 保数值稳定（小特征值 ↓ → inv root ↑ → 爆炸）
        mat = mat + eps * torch.eye(n, device=mat.device, dtype=mat.dtype)
        L, V = torch.linalg.eigh(mat)
        L = L.clamp_(min=eps)
        return (V * L.pow(power)) @ V.T   # V · diag(L^p) · V^T

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            wd = group['weight_decay']
            b2 = group['beta2']
            eps = group['eps']
            interval = group['precond_interval']
            max_dim = group['max_precond_dim']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # 解耦 weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)

                state = self.state[p]

                # 1D 参数（bias, LayerNorm）：退回 SGD+momentum
                if p.ndim < 2:
                    if 'buf' not in state:
                        state['buf'] = torch.zeros_like(p)
                    buf = state['buf']
                    buf.mul_(mu).add_(grad)
                    p.add_(buf, alpha=-lr)
                    continue

                G = grad.view(p.size(0), -1)
                M, N = G.shape
                G32 = G.float()

                # 初始化
                if 'step' not in state:
                    state['step'] = 0
                    state['buf'] = torch.zeros_like(p)
                    if M <= max_dim:
                        state['L'] = torch.zeros(M, M, device=p.device, dtype=torch.float32)
                        state['L_root'] = None
                    if N <= max_dim:
                        state['R'] = torch.zeros(N, N, device=p.device, dtype=torch.float32)
                        state['R_root'] = None
                state['step'] += 1

                # EMA 更新 L, R 统计
                if 'L' in state:
                    state['L'].mul_(b2).addmm_(G32, G32.T, alpha=1 - b2)
                if 'R' in state:
                    state['R'].mul_(b2).addmm_(G32.T, G32, alpha=1 - b2)

                # 每 interval 步重算预条件根（第一次等统计量积累够）
                if state['step'] % interval == 0 and state['step'] > 0:
                    if 'L' in state:
                        state['L_root'] = self._matrix_inv_root(state['L'], -0.25, eps)
                    if 'R' in state:
                        state['R_root'] = self._matrix_inv_root(state['R'], -0.25, eps)

                # 应用预条件：pG = L^{-1/4} G R^{-1/4}
                # 早期 L_root/R_root 还是 None 时退化为 identity
                pG = G32
                if state.get('L_root') is not None:
                    pG = state['L_root'] @ pG
                if state.get('R_root') is not None:
                    pG = pG @ state['R_root']

                # 动量作用在预条件梯度上
                buf = state['buf']
                buf.mul_(mu).add_(pG.view_as(p).to(buf.dtype))
                p.add_(buf, alpha=-lr)


# ============================================================================
# Learning Rate Schedule
# ============================================================================

def get_lr(step, warmup_steps, total_steps, max_lr, min_lr=0):
    """Cosine decay with linear warmup"""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    elif step >= total_steps:
        return min_lr
    else:
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


# ============================================================================
# Optimizer factory: 把 "名字" -> (optimizer, max_lr)
# ============================================================================

def build_optimizer(opt_name, model):
    """
    默认学习率按讲义三条路线的典型值选。
    SGD 家族：transformer 上通常需要比 vision 小一个量级。
    Adam/Muon：标准 3e-4。
    SignGD：每步 ±lr，需要远小于 Adam 的 lr。
    """
    params = model.parameters()

    if opt_name == "SGD":
        # 讲义 §2：朴素 GD（ℓ₂ 归一化梯度的 Lagrangian 松弛）
        max_lr = 1e-3
        opt = SGD(params, lr=max_lr, momentum=0.0, weight_decay=0.0)

    elif opt_name == "HeavyBall":
        # 讲义 §3.1：Polyak 动量（PyTorch SGD 的默认 momentum 实现就是 HB）
        max_lr = 1e-3
        opt = SGD(params, lr=max_lr, momentum=0.9, nesterov=False, weight_decay=0.0)

    elif opt_name == "Nesterov":
        # 讲义 §3.2：NAG lookahead
        max_lr = 1e-3
        opt = SGD(params, lr=max_lr, momentum=0.9, nesterov=True, weight_decay=0.0)

    elif opt_name == "SGD_WD":
        # 讲义 §3.4：SGD 上 weight decay 与 L2 正则等价
        max_lr = 1e-3
        opt = SGD(params, lr=max_lr, momentum=0.9, nesterov=False, weight_decay=0.1)

    elif opt_name == "SignGD":
        # 讲义 §4：ℓ∞ 最优，带动量
        max_lr = 1e-4
        opt = SignGD(params, lr=max_lr, momentum=0.9, weight_decay=0.0)

    elif opt_name == "Adam":
        # 讲义 §5：SignGD + 动量 + 二阶矩（注意 weight_decay=0，WD 对比在 AdamW 里）
        max_lr = 3e-4
        opt = Adam(params, lr=max_lr, betas=(0.9, 0.95), weight_decay=0.0)

    elif opt_name == "AdamW":
        # 讲义 §5.8：解耦 weight decay —— Adam + WD 但不耦合进二阶矩
        max_lr = 3e-4
        opt = AdamW(params, lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)

    elif opt_name == "Muon":
        # 讲义 §10：谱范数最优 + Kimi/Moonlight RMS 对齐，复用 Adam lr
        max_lr = 3e-4
        opt = Muon(params, lr=max_lr, momentum=0.95, n_iter=5,
                   use_kimi_rms=True, target_rms=0.2)

    elif opt_name == "Lion":
        # 讲义 §4–§5 延伸：SignGD + 双动量（Chen et al. 2023）
        # 和 SignGD 同样"每参数 ±lr"，所以 lr 量级和 SignGD 一致
        max_lr = 1e-4
        opt = Lion(params, lr=max_lr, betas=(0.9, 0.99), weight_decay=0.1)

    elif opt_name == "Sophia":
        # 讲义 §5 延伸：二阶对角牛顿 + 截断（Liu et al. 2023）
        # bs 要和我们的 effective-batch-tokens 对齐（ρ·B·h 是 clip 阈值）
        effective_batch_tokens = CONFIG['batch_size'] * CONFIG['block_size']
        max_lr = 1e-4
        opt = Sophia(params, lr=max_lr, betas=(0.965, 0.99), rho=0.04,
                     weight_decay=0.1, bs=effective_batch_tokens)

    elif opt_name == "CautiousAdamW":
        # 讲义 §5.8 延伸：AdamW + "符号一致" mask（Liang et al. 2024）
        # 和 AdamW 控制变量对比——唯一差别是那行 mask
        max_lr = 3e-4
        opt = CautiousAdamW(params, lr=max_lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.1)

    elif opt_name == "Shampoo":
        # 讲义 §6–§7 预条件路线终点：Kronecker 预条件 L^{-1/4} G R^{-1/4}
        # 讲义 §7.3 恒等式告诉我们：这和 Muon 的 UV^T 殊途同归
        max_lr = 3e-4
        opt = Shampoo(params, lr=max_lr, momentum=0.9, weight_decay=0.1,
                      beta2=0.95, precond_interval=200, max_precond_dim=2048)

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    return opt, max_lr


# ============================================================================
# Logger
# ============================================================================

class Logger:
    def __init__(self, log_dir, name):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{log_dir}/{name}_{timestamp}.jsonl"
        self.csv_file = f"{log_dir}/{name}_{timestamp}.csv"
        with open(self.csv_file, 'w') as f:
            f.write("step,tokens,loss,lr,step_time_ms,opt_time_ms,fb_time_ms,tokens_per_sec\n")
        print(f"Logging to: {self.log_file}")

    def log(self, data):
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
        if 'loss' in data:
            with open(self.csv_file, 'a') as f:
                f.write(f"{data['step']},{data['tokens']},{data['loss']:.6f},"
                        f"{data['lr']:.6f},{data['step_time_ms']:.1f},"
                        f"{data.get('opt_time_ms', 0):.1f},{data.get('fb_time_ms', 0):.1f},"
                        f"{data['tokens_per_sec']:.0f}\n")


# ============================================================================
# Training
# ============================================================================

def train(opt_name, gpu_id, cfg):
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"\n{'='*60}")
    print(f"[GPU {gpu_id}] Starting {opt_name}")
    print(f"{'='*60}")

    logger = Logger(cfg['log_dir'], f"{opt_name}_gpu{gpu_id}")

    # 所有优化器共享初始化
    torch.manual_seed(42)
    model = GPT(cfg).to(device)

    dataloader = get_dataloader(cfg, device)

    # ----- 优化器 -----
    optimizer, max_lr = build_optimizer(opt_name, model)

    micro_batch = cfg['micro_batch_size']
    grad_accum = cfg['grad_accum_steps']
    tokens_per_step = micro_batch * grad_accum * cfg['block_size']
    warmup_steps = cfg['warmup_tokens'] // tokens_per_step
    total_steps = cfg['total_tokens'] // tokens_per_step

    print(f"[GPU {gpu_id}] {opt_name}: max_lr={max_lr:.2e}")
    print(f"[GPU {gpu_id}] Micro batch: {micro_batch}, Grad accum: {grad_accum}, "
          f"Effective batch: {micro_batch * grad_accum}")
    print(f"[GPU {gpu_id}] Warmup: {warmup_steps} steps, Total: {total_steps} steps, "
          f"Tokens/step: {tokens_per_step:,}")
    print(f"[GPU {gpu_id}] AMP: {cfg['use_amp']}, dtype: {cfg['dtype']}")

    use_amp = cfg['use_amp']
    amp_dtype = torch.bfloat16 if cfg['dtype'] == 'bfloat16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and cfg['dtype'] == 'float16'))

    model.train()
    tokens_trained = 0
    step_times, opt_times, fb_times = [], [], []
    using_local = isinstance(dataloader, LocalDataLoader)

    for step in range(1, total_steps + 1):
        step_start = time.time()

        if not using_local and step % 100 == 0:
            token_file = f"{cfg['data_dir']}/tokens.bin"
            if os.path.exists(token_file):
                try:
                    new_loader = LocalDataLoader(cfg['data_dir'], micro_batch, cfg['block_size'], device)
                    dataloader = new_loader
                    using_local = True
                    print(f"[GPU {gpu_id}] Switched to local data at step {step}!")
                except Exception as e:
                    print(f"[GPU {gpu_id}] Failed to switch: {e}")

        lr = get_lr(step, warmup_steps, total_steps, max_lr, min_lr=max_lr * 0.1)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        accumulated_loss = 0.0
        forward_backward_time = 0.0

        for _ in range(grad_accum):
            x, y = dataloader.get_batch()
            fb_start = time.time()
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                _, loss = model(x, y)
                loss = loss / grad_accum
            accumulated_loss += loss.item()
            if use_amp and cfg['dtype'] == 'float16':
                scaler.scale(loss).backward()
            else:
                loss.backward()
            torch.cuda.synchronize()
            forward_backward_time += (time.time() - fb_start) * 1000

        opt_start = time.time()
        if use_amp and cfg['dtype'] == 'float16':
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        torch.cuda.synchronize()
        opt_time = (time.time() - opt_start) * 1000
        optimizer.zero_grad(set_to_none=True)

        # ── Sophia 专用：每 hess_interval 步做一次 G-N-B Hessian 估计 ──
        # 方法：从模型自己的预测分布采一组"假"标签 y ~ p_θ(·|x)，再 backward。
        # 这次 backward 出来的 p.grad 就是 diag(Fisher) 的蒙特卡洛估计，
        # 调用 optimizer.update_hessian() 把它 EMA 进 h。
        if hasattr(optimizer, 'update_hessian') and step % 10 == 0:
            x_h, _ = dataloader.get_batch()
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                logits_h, _ = model(x_h)  # 注意：不传 targets，只要 logits
                # 从 logits 里采样"假"标签
                with torch.no_grad():
                    probs_h = F.softmax(logits_h.float(), dim=-1)
                    Bh, Th, Vh = logits_h.shape
                    sampled = torch.multinomial(
                        probs_h.reshape(-1, Vh), 1
                    ).view(Bh, Th)
                # 用采样标签做 cross entropy，backward
                loss_h = F.cross_entropy(
                    logits_h.view(-1, Vh), sampled.view(-1)
                )
            if use_amp and cfg['dtype'] == 'float16':
                scaler.scale(loss_h).backward()
            else:
                loss_h.backward()
            optimizer.update_hessian()
            optimizer.zero_grad(set_to_none=True)

        step_time = (time.time() - step_start) * 1000
        step_times.append(step_time)
        opt_times.append(opt_time)
        fb_times.append(forward_backward_time)
        tokens_trained += tokens_per_step

        if step % cfg['log_interval'] == 0:
            # accumulated_loss 已经是 4 个 micro-batch per-token loss 的平均，
            # 直接就是 effective batch 的真实 loss，不要再乘 grad_accum。
            avg_loss = accumulated_loss
            avg_step_time = sum(step_times[-cfg['log_interval']:]) / cfg['log_interval']
            avg_opt_time = sum(opt_times[-cfg['log_interval']:]) / cfg['log_interval']
            avg_fb_time = sum(fb_times[-cfg['log_interval']:]) / cfg['log_interval']
            tokens_per_sec = tokens_per_step / (avg_step_time / 1000)

            log_data = {
                'step': step, 'tokens': tokens_trained,
                'loss': avg_loss, 'lr': lr,
                'step_time_ms': avg_step_time,
                'opt_time_ms': avg_opt_time,
                'fb_time_ms': avg_fb_time,
                'tokens_per_sec': tokens_per_sec,
                'gpu': gpu_id, 'optimizer': opt_name,
            }
            logger.log(log_data)

            print(f"[GPU {gpu_id}] {opt_name:<12} | step {step:6d}/{total_steps} | "
                  f"loss {avg_loss:.4f} | lr {lr:.2e} | "
                  f"step {avg_step_time:.0f}ms (fb {avg_fb_time:.0f} + opt {avg_opt_time:.0f}) | "
                  f"{tokens_per_sec/1000:.1f}K tok/s")

        if step % cfg['val_interval'] == 0:
            model.eval()
            val_losses = []
            val_steps = cfg['val_tokens'] // (micro_batch * cfg['block_size'])
            with torch.no_grad():
                for _ in range(val_steps):
                    x, y = dataloader.get_batch()
                    with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                        _, val_loss = model(x, y)
                    val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"[GPU {gpu_id}] {opt_name:<12} | step {step:6d} | VAL LOSS: {avg_val_loss:.4f}")
            logger.log({
                'step': step, 'tokens': tokens_trained,
                'val_loss': avg_val_loss,
                'gpu': gpu_id, 'optimizer': opt_name,
            })
            model.train()

        if step % cfg['save_interval'] == 0:
            ckpt_dir = Path(cfg['checkpoint_dir']) / opt_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'step': step,
                'tokens': tokens_trained,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_dir / f"step_{step}.pt")

    print(f"\n[GPU {gpu_id}] {opt_name} finished! Final loss: {loss.item():.4f}")
    return loss.item()


# ============================================================================
# Multi-GPU Launch
# ============================================================================

# 按讲义三条路线 + 现代扩展排列
OPTIMIZERS = [
    # ── ℓ₂ 路线（§2–§3）───────────────────────────────
    "SGD",              # §2     朴素 GD
    "HeavyBall",        # §3.1   + Polyak 动量
    "Nesterov",         # §3.2   + Nesterov lookahead
    "SGD_WD",           # §3.4   + 权重衰减
    # ── ℓ∞ + 谱范数路线（§4–§10）──────────────────────
    "SignGD",           # §4     ℓ∞ 最优更新
    "Adam",             # §5     SignGD + 动量 + 二阶矩
    "AdamW",            # §5.8   + 解耦 weight decay
    "Muon",             # §10    谱范数最优 ∝ UV^T
    # ── 现代变体（讲义之后几年的代表工作）────────────
    "Lion",             # Chen+2023    SignGD + 双动量
    "Sophia",           # Liu+2023     二阶对角牛顿 + 截断
    "CautiousAdamW",    # Liang+2024   AdamW + 符号一致 mask
    "Shampoo",          # Gupta+2018   Kronecker 预条件（讲义§6–§7 对偶）
]

# 分组：12 个优化器 = 3 组 × 4 个，正好适配 2×4卡 或 3×4卡
#   2×4卡 部署：A + B 并行（讲义核心 8 个），某一节点跑完后再跑 C
#   3×4卡 部署：A + B + C 完全并行
GROUPS = {
    'A': OPTIMIZERS[0:4],     # ℓ₂ 路线
    'B': OPTIMIZERS[4:8],     # ℓ∞ + 谱范数（讲义核心后半）
    'C': OPTIMIZERS[8:12],    # 现代 4 个
}


def run_single(opt_name, gpu_id):
    train(opt_name, gpu_id, CONFIG)


def launch_parallel(opt_list, tag=""):
    """在本机上把 opt_list 里的优化器分配到 GPU 0..len(opt_list)-1 并行跑。
    每个子进程通过 CUDA_VISIBLE_DEVICES 只看到一张卡，内部 --gpu 恒为 0。"""
    print(f"Launching {len(opt_list)} optimizers{' (' + tag + ')' if tag else ''}")
    for local_gpu, opt_name in enumerate(opt_list):
        print(f"  local GPU {local_gpu}: {opt_name}")
    print(f"Config: batch_size={CONFIG['batch_size']}, "
          f"total_tokens={CONFIG['total_tokens']/1e9:.1f}B")

    processes = []
    for local_gpu, opt_name in enumerate(opt_list):
        cmd = [sys.executable, __file__, '--optimizer', opt_name, '--gpu', '0']
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(local_gpu)
        print(f"Starting local GPU {local_gpu}: {opt_name}")
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    for p in processes:
        p.wait()

    print("\n" + "=" * 60)
    print(f"All {len(opt_list)} training runs{' (' + tag + ')' if tag else ''} completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default=None,
                        help='Single optimizer to run')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id for single run')
    parser.add_argument('--all', action='store_true',
                        help='Run all 12 optimizers on one ≥12-GPU node')
    parser.add_argument('--group', type=str, choices=['A', 'B', 'C'], default=None,
                        help='Run a 4-optimizer group on one 4-GPU node. '
                             'A = ℓ₂ 路线 (SGD/HB/NAG/SGD_WD), '
                             'B = ℓ∞ + 谱范数 (SignGD/Adam/AdamW/Muon), '
                             'C = 现代变体 (Lion/Sophia/CautiousAdamW/Shampoo)')
    args = parser.parse_args()

    if args.optimizer:
        run_single(args.optimizer, args.gpu)
    elif args.all:
        launch_parallel(OPTIMIZERS, tag="all 12")
    elif args.group:
        launch_parallel(GROUPS[args.group], tag=f"group {args.group}")
    else:
        print("Usage:")
        print("  Single optimizer:         python train_8gpu.py --optimizer Muon --gpu 0")
        print("  All 12 on 12-GPU node:    python train_8gpu.py --all")
        print("  4 on one 4-GPU node (A):  python train_8gpu.py --group A")
        print("  4 on one 4-GPU node (B):  python train_8gpu.py --group B")
        print("  4 on one 4-GPU node (C):  python train_8gpu.py --group C")
        print("\nGroup layout (12 optimizers = 3 groups × 4):")
        for g, opts in GROUPS.items():
            print(f"  group {g}: {opts}")
        print("\n2×4卡部署：A + B 并行跑，跑完后其中一台跑 C (总共 2 轮)")
        print("3×4卡部署：A + B + C 全并行，1 轮完事")


if __name__ == "__main__":
    main()
