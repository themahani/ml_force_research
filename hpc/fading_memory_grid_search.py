#!/usr/bin/env python

"""
This script runs a parameter sweep for the fading memory test.
"""

import json
import os
from itertools import chain
from tqdm import tqdm

import numpy as np
import torch

from rctorch.models import MorrisLecar, MorrisLecarCurrent
from rctorch.optimizers import KWArgsEncoder


def main():
    # Define the output directory
    output_dir = os.path.join(
        os.getcwd(), "fading_memory", "actual_sine_current_q_300_500_gbar_1_10"
    )
    os.makedirs(output_dir, exist_ok=True)

    seed = 1
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.random.manual_seed(seed)

    # Generate the supervisor signal
    T = 10_000
    dt = 1e-1
    t = np.arange(0, T, dt)
    tau = 0.02
    sup = np.sin(2 * np.pi * 0.01 * t * tau).reshape(-1, 1)

    print(sup.shape)
    np.save(os.path.join(output_dir, "supervisor.npy"), sup)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sup_tensor = torch.tensor(sup, device=device, dtype=torch.float32)

    # Set the model parameters
    Ne = 400
    Ni = 100
    N = Ne + Ni
    BIAS = np.ones((N, 1)) * 70.0
    model_params = {
        "Ne": Ne,
        "Ni": Ni,
        "dt": dt,
        "BIAS": BIAS,
        "p_sparsity": 0.01,
        "device": device,
    }

    try:
        params_file = os.path.join(output_dir, "model_params.json")
        with open(params_file, "w") as f:
            json.dump(
                model_params, f, cls=KWArgsEncoder, indent=4
            )  # Added indent for readability
    except Exception as e:
        print(f"Error exporting JSON file params: {e}")

    q_range = np.linspace(300, 500, 5)
    gbar_range = np.linspace(1, 10, 7)

    opt_params = {"w_in_amp": q_range, "gbar": gbar_range}
    size = 1
    for _, value_range in opt_params.items():
        size *= value_range.size

    grid = [[(Q, gbar) for Q in q_range] for gbar in gbar_range]
    grid_file = os.path.join(output_dir, "grid.json")
    with open(grid_file, "w") as f:
        json.dump(grid, f, cls=KWArgsEncoder, indent=4)


    # run simulations for all elements
    grid_list = list(chain.from_iterable(grid))
    
    for q, gbar in grid_list:
        print(f"Running simulation with params:\n Q={q}, gbar={gbar}")
        try:
            w_in = torch.rand(size=(N, sup_tensor.size(1))).to(device) * 2 - 1
            w_in *= q
            ml0 = MorrisLecarCurrent(gbar=gbar, **model_params)
            ml1 = MorrisLecarCurrent(gbar=gbar, **model_params)
            ml1.w = ml0.w.clone()
            ml1.mem = ml0.mem.clone()
            ml1.s = ml0.s.clone()
            
            nt_transient = round(500 // dt)
            for _ in tqdm(range(nt_transient)):
                ml0.forward(0.0)
                ml1.forward(0.0)
            s_rec0 = torch.zeros((sup_tensor.size(0), ml0.N), device='cpu')
            s_rec1 = torch.zeros((sup_tensor.size(0), ml1.N), device='cpu')
            for i in tqdm(range(sup_tensor.size(0))):
                input_current = w_in @ sup_tensor[i].reshape(-1, 1)
                ml0.forward(input_current)
                ml1.forward(input_current)
                s_rec0[i] = ml0.s.clone().cpu().flatten()
                s_rec1[i] = ml1.s.clone().cpu().flatten()

            torch.save(s_rec0, os.path.join(output_dir, f"s_rec0_q_{q}_gbar_{gbar}.pt"))
            torch.save(s_rec1, os.path.join(output_dir, f"s_rec1_q_{q}_gbar_{gbar}.pt"))
            torch.save(w_in, os.path.join(output_dir, f"w_in_q_{q}_gbar_{gbar}.pt"))
        except Exception as e:
            print(f"Warning: simulation gbar={gbar}, Q={q} failed with error:\n", e)


    print(f"total size of the mesh is: {size}")

    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
