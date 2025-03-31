#!/usr/bin/env python

"""
This script runs a parameter sweep for the Lorenz system using the ml-force package.
"""

import torch
import numpy as np
import json

import sys, os
sys.path.append(os.path.join(os.getcwd(), ".."))

from ml_force import LorenzAttractor, MorrisLecarCurrent, minmax_transform
from ml_force.optimizers import BruteForceMesh, NumpyArrayEncoder


def main():
    # Define the output directory
    output_dir = os.path.join(os.getcwd(), "bfm_output", "lorenz_current_focused1")
    # Define the directory to save results
    os.makedirs(output_dir, exist_ok=True)

    seed = 1
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.random.manual_seed(seed)

    # Generate the supervisor signal
    T = 25_000
    dt = 5e-2
    # t = np.arange(0, T, dt)
    x = LorenzAttractor(T, dt, tau=.01).generate(transient_time=500.0)

    x = x.T
    sup = minmax_transform(x, zero_mean=True)
    print(sup.shape)
    np.save(os.path.join(output_dir, "supervisor.npy"), sup)

    # Set the model parameters
    N = 10_000
    BIAS = np.ones((N, 1)) * 75.0
    model_params = {
        "N": N,
        "T": T,
        "dt": dt,
        "supervisor": sup,
        "BIAS": BIAS,
        "p_sparsity": 0.1,
        "l":8e-1,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    }

    try:
        params_file = os.path.join(output_dir, "model_params.json")
        with open(params_file, "w") as f:
            json.dump(model_params, f, cls=NumpyArrayEncoder, indent=4) # Added indent for readability
    except Exception as e:
        print(f"Error exporting JSON file params: {e}")

    Q_range = np.linspace(250, 350, 3)
    gbar_range = np.linspace(15, 25, 5)


    opt_params = {
        "Q": Q_range,
        "gbar": gbar_range
    }

    size = 1
    for _, value_range in opt_params.items():
        size *= value_range.size

    print(f"total size of the mesh is: {size}")

    render_params = dict(rls_start=500, 
                         rls_stop=T-10_000, rls_step=20,
                         live_plot=False, plt_interval=300,
                         n_neurons=10, save_all=False)
    
    n_threads = int(input("Enter number of threads:\n>>"))

    bfm = BruteForceMesh(model=MorrisLecarCurrent, default_args=model_params,
                         render_args=render_params, params=opt_params, num_threads=n_threads)
    
    
    bfm.run(output_dir)
    print(f"Results saved in: {output_dir}")



if __name__ == "__main__":
    main()
