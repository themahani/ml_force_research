#!/usr/bin/env python

import sys
from pathlib import Path

# Add the parent directory to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "ml-force"))

import matplotlib.pyplot as plt
import numpy as np
from ml_force.models import MorrisLecar
from ml_force.utils import z_transform


def import_results(fp: str):
    f = open(fp, "r")
    data = f.readlines()

    for i, line in enumerate(data):
        if line == "":
            data.pop(i)
            print(f"Popped index {i}")
    return data


def main():
    """main body"""

    np.random.seed(1)

    lamda = 1e-5

    ### Global params for the model
    T = 10000
    dt = 1e-2
    t = np.arange(0, T, dt)
    nt = t.size
    x = np.cos(2 * 2 * np.pi * t / 1000)
    x = x.reshape(-1, 1)
    signal = z_transform(x)
    print(signal.shape)

    # neuron population
    NE = 200
    NI = 200
    N = NI + NE

    # input current for I and E neurons
    Ie = 80
    Ii = 80
    current = np.ones((N, 1))
    middle = N // 2
    current[:middle] *= Ie  # NE bias
    current[middle:] *= Ii  # NI bias

    default_args = {
        "T": T,
        "supervisor": signal,
        "BIAS": current,
        "dt": dt,
        "N": N,
        "l": lamda,
    }
    # RLS params
    rls_start = round(T * 0.02)
    rls_start = 500
    rls_stop = round(T * 0.7)
    rls_step = 20

    render_args = {
        "rls_start": rls_start,
        "rls_stop": rls_stop,
        "rls_step": rls_step,
        "live_plot": True,
        "plt_interval": 100,
        "n_neurons": 10,
        "save_all": False,
    }

    best_params = {"Q": 150, "gbar": 10, "w_rand": 1}
    model_args = default_args | best_params

    model = MorrisLecarBlock(**model_args)
    print(f"Weights sum: {np.sum(model.w)}")
    random_neurons, voltage_trace = model.render(**render_args)


if __name__ == "__main__":
    main()
