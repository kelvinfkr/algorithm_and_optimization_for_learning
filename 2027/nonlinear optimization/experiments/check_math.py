#!/usr/bin/env python3
"""Independent numerical checks for the identities used in the lecture notes.

These checks validate formulas against autodiff, feasibility certificates and
Monte Carlo. They are not substitutes for the proofs in the handouts.
"""
from __future__ import annotations
import argparse
import math
import numpy as np
import torch
from common import environment, output_dir, polar, save_json
from lab01_sgd import exact_moments
from lab03_width import TwoLayer, effective_kernel


def check_kkt():
    z = np.array([2., 1.])
    x = np.array([1.5, .5])
    lam = .5
    constraints = np.r_[-x, x.sum()-2]
    multipliers = np.array([0., 0., lam])
    stationarity = x-z+lam*np.ones(2)
    primal = .5*np.sum((x-z)**2)
    # Minimizing the Lagrangian over all R^2 gives this dual value.
    dual = lam*(z.sum()-2)-lam**2
    residuals = [np.max(np.maximum(constraints, 0)),
                 np.max(np.maximum(-multipliers, 0)),
                 np.linalg.norm(stationarity),
                 np.max(np.abs(constraints*multipliers)), abs(primal-dual)]
    assert max(residuals) < 1e-12
    return {"max_kkt_and_gap_residual":float(max(residuals)),"primal":primal,"dual":dual}


def check_sgd():
    rng = np.random.default_rng(2027)
    paths, batch, eta, variance, sigma, e0 = 200000, 4, .2, 1.7, .5, 1.3
    x = rng.normal(scale=math.sqrt(variance), size=(paths,batch))
    noise = rng.normal(scale=sigma, size=(paths,batch))
    next_error = (1-eta*np.mean(x*x,axis=1))*e0 + eta*np.mean(x*noise,axis=1)
    means, seconds = exact_moments(eta,batch,1,sigma,variance,e0)
    second_samples = next_error**2
    standard_error = float(second_samples.std(ddof=1)/math.sqrt(paths))
    discrepancy = abs(float(second_samples.mean())-seconds[1])
    assert discrepancy <= 6*standard_error+1e-5
    return {"theoretical_second_moment":float(seconds[1]),
            "empirical_second_moment":float(second_samples.mean()),
            "standard_error":standard_error,"absolute_error":discrepancy,
            "interpretation":"A seeded Monte Carlo consistency check, not a proof."}


def check_polar():
    gen = torch.Generator().manual_seed(2027)
    g = torch.randn(7,4,generator=gen,dtype=torch.float64)
    q = polar(g,"svd")
    nuclear = torch.linalg.svdvals(g).sum()
    support_error = float(torch.abs((g*q).sum()-nuclear))
    spectral_norm = float(torch.linalg.matrix_norm(q,ord=2))
    approximation_error = float(torch.linalg.matrix_norm(polar(g,"ns",40)-q))
    assert support_error < 1e-11 and spectral_norm <= 1+1e-12
    assert approximation_error < 1e-10
    assert torch.count_nonzero(polar(torch.zeros_like(g),"svd")) == 0
    return {"support_function_error":support_error,"spectral_norm":spectral_norm,
            "ns40_frobenius_error":approximation_error}


def check_kernel():
    gen = torch.Generator().manual_seed(9)
    w = torch.randn(4,3,generator=gen,dtype=torch.float64)
    a = torch.randn(4,1,generator=gen,dtype=torch.float64)
    x = torch.randn(5,3,generator=gen,dtype=torch.float64)
    kernels, errors = [], []
    for regime in ("NTP","MF"):
        model = TwoLayer(w,a,regime)
        flat = torch.cat([w.reshape(-1),a.reshape(-1)]).requires_grad_(True)
        def raw_forward(v):
            ww, aa = v[:12].reshape(4,3), v[12:].reshape(4,1)
            return (model.scale*torch.tanh(x@ww.T)@aa).reshape(-1)
        jac = torch.autograd.functional.jacobian(raw_forward,flat)
        reference = model.lr_multiplier*(jac@jac.T)/len(x)
        actual = effective_kernel(model,x)
        errors.append(float(torch.max(torch.abs(reference-actual))))
        kernels.append(actual)
        assert float(model(x).detach().abs().max()) == 0.
    initial_match = float(torch.max(torch.abs(kernels[0]-kernels[1])))
    assert max(errors+[initial_match]) < 1e-12
    return {"analytic_vs_autodiff_errors":errors,"matched_initial_kernel_error":initial_match}


def check_coordinates():
    gen = torch.Generator().manual_seed(11)
    width, d, eta = 4, 3, .02
    w0 = torch.randn(width,d,generator=gen,dtype=torch.float64)
    a0 = torch.randn(width,1,generator=gen,dtype=torch.float64)
    x = torch.randn(6,d,generator=gen,dtype=torch.float64)
    y = torch.randn(6,1,generator=gen,dtype=torch.float64)
    wa, aa = w0.clone().requires_grad_(), a0.clone().requires_grad_()
    wb, bb = w0.clone().requires_grad_(), (a0/width).clone().requires_grad_()
    frozen = (torch.tanh(x@w0.T)@a0/width).detach()
    errors = []
    for _ in range(5):
        fa = torch.tanh(x@wa.T)@aa/width-frozen
        fb = torch.tanh(x@wb.T)@bb-frozen
        ga = torch.autograd.grad(.5*(fa-y).square().mean(),(wa,aa))
        gb = torch.autograd.grad(.5*(fb-y).square().mean(),(wb,bb))
        with torch.no_grad():
            wa -= width*eta*ga[0]; aa -= width*eta*ga[1]
            wb -= width*eta*gb[0]; bb -= eta/width*gb[1]
            errors.append(max(float((wa-wb).abs().max()),float((aa/width-bb).abs().max())))
    assert max(errors) < 1e-12
    return {"max_five_step_coordinate_error":max(errors),
            "rates":{"raw_a":width*eta,"raw_w":width*eta,"physical_b":eta/width}}


def check_factorization():
    rng = np.random.default_rng(17)
    a,b,g = rng.normal(size=(5,3)),rng.normal(size=(3,4)),rng.normal(size=(5,4))
    da,db = -g@b.T,-a.T@g
    balance_derivative = da.T@a+a.T@da-db@b.T-b@db.T
    induced = da@b+a@db
    reference = -a@a.T@g-g@b.T@b
    errors = [np.linalg.norm(balance_derivative),np.linalg.norm(induced-reference)]
    assert max(errors) < 1e-11
    return {"balance_derivative_residual":float(errors[0]),"induced_flow_residual":float(errors[1])}


def run(output="results/checks"):
    torch.set_num_threads(1)
    checks = {"kkt":check_kkt(),"sgd":check_sgd(),"polar":check_polar(),
              "kernel":check_kernel(),"coordinates":check_coordinates(),
              "matrix_factorization":check_factorization()}
    report = {"status":"passed","checks":checks,"environment":environment()}
    save_json(output_dir(output)/"math_checks.json",report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/checks")
    report = run(parser.parse_args().output)
    print("Passed:",", ".join(report["checks"]))
