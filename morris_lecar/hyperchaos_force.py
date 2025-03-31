#!/usr/bin/env python

import sys
from pathlib import Path

# Add the parent directory to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "ml-force"))

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from ml_force.models import MorrisLecar
from ml_force.plots import plot_model
from ml_force.supervisors import HyperChaoticAttractor
from ml_force.utils import minmax_transform


def main():

    seed = 1
    np.random.seed(seed)
    T = 60000
    dt = 1e-2
    t = np.arange(0, T, dt)
    nt = t.size
    x = HyperChaoticAttractor(T, dt, tau=0.02).generate()

    x = x.T
    signal = minmax_transform(x)
    print(signal.shape)

    NE = 2500
    NI = 2500
    N = NI + NE

    # input current for I and E neurons
    Ie = 75
    Ii = 75
    current = np.ones((N, 1))
    middle = N // 2
    current[:middle] *= Ie  # NE bias
    current[middle:] *= Ii  # NI bias
    # current = np.random.rand(N, 1) * Ie

    # RLS params
    rls_start = round(T * 0.02)
    rls_start = 500
    rls_stop = round(T * 0.85)
    rls_step = 20
    Q = 300
    lamda = 1e-5
    gbar = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device <{device}> for PyTorch computations...\n")
    torch.cuda.random.manual_seed(seed)

    try:
        model = MorrisLecar(
            supervisor=signal,
            BIAS=current,
            T=T,
            dt=dt,
            N=N,
            Q=Q,
            gbar=gbar,
            l=lamda,
            device=device,
        )
    except Exception as e:
        print(e)
        print(torch.cuda.memory_summary(device=device))

    render_params = {
        "rls_start": rls_start,
        "rls_stop": rls_stop,
        "rls_step": rls_step,
        "live_plot": False,
        "n_neurons": 10,
        "plt_interval": 300,
        "save_all": False,
    }
    start = time.time()
    random_neurons, voltage_trace, decoder_trace = model.render(**render_params)
    end = time.time()
    print(f"Render took {end - start} seconds to finish...\n")

    plot_params = {
        "rls_start": rls_start,
        "rls_stop": rls_stop,
        "rls_step": rls_step,
        "lamda": lamda,
        "Q": Q,
    }

    plot_model(
        model,
        "hyperrossler",
        random_neurons,
        voltage_trace,
        decoder_trace,
        **plot_params,
    )


if __name__ == "__main__":
    main()
