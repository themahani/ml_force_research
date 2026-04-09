#!/usr/bin/env python

"""
This script runs a parameter sweep for the Lorenz system using the ml-force package.
"""

import json
import os

import numpy as np
import torch

from rctorch.models import MorrisLecar, MorrisLecarCurrent
from rctorch.optimizers import BruteForceMesh, KWArgsEncoder
from rctorch.supervisors import LorenzAttractor
from rctorch.utils import minmax_transform


def main():
    # Define the output directory
    output_dir = os.path.join(
        os.getcwd(), "bfm_output", "lorenz_current_q_300_500_gbar_1_10"
    )
    os.makedirs(output_dir, exist_ok=True)

    seed = 1
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.random.manual_seed(seed)

    # Generate the supervisor signal
    T = 20_000
    dt = 1e-1
    x = LorenzAttractor(T, dt, tau=0.01).generate(transient_time=1000.0)

    x = x.T
    sup = minmax_transform(x, zero_mean=True)
    print(sup.shape)
    np.save(os.path.join(output_dir, "supervisor.npy"), sup)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sup_tensor = torch.tensor(sup, device=device, dtype=torch.float32)

    # Set the model parameters
    Ne = 4_000
    Ni = 1_000
    N = Ne + Ni
    BIAS = np.ones((N, 1)) * 70.0
    reservoir_params = {
        "model_cls": MorrisLecarCurrent,
        "n_input": sup_tensor.size(1),
        "n_output": sup_tensor.size(1),
        "Ne": Ne,
        "Ni": Ni,
        "dt": dt,
        "BIAS": BIAS,
        "p_sparsity": 0.01,
        "device": device,
    }

    try:
        params_file = os.path.join(output_dir, "reservoir_params.json")
        with open(params_file, "w") as f:
            json.dump(
                reservoir_params, f, cls=KWArgsEncoder, indent=4
            )  # Added indent for readability
    except Exception as e:
        print(f"Error exporting JSON file params: {e}")

    q_range = np.linspace(300, 500, 5)
    gbar_range = np.linspace(1, 10, 7)

    opt_params = {"w_in_amp": q_range, "gbar": gbar_range}
    opt_params_file = os.path.join(output_dir, "opt_params.json")
    with open(opt_params_file, "w") as f:
        json.dump(opt_params, f, cls=KWArgsEncoder, indent=4)

    size = 1
    for _, value_range in opt_params.items():
        size *= value_range.size

    print(f"total size of the mesh is: {size}")

    train_test_split = 0.5
    nt_split = int(sup_tensor.size(0) * train_test_split)
    sup_train = sup_tensor[:nt_split]
    sup_test = sup_tensor[nt_split:]
    train_kwargs = dict(
        x=sup_train,
        nt_transient=int(500 / dt),
        rls_step=20,
        ridge_reg=1.0,
        ff_coeff=1.0,
    )
    test_kwargs = {
        "x": sup_test,
        "nt_transient": 0,
        "closed_loop": True,
    }

    n_threads = 1

    bfm = BruteForceMesh(
        reservoir_kwargs=reservoir_params,
        train_kwargs=train_kwargs,
        test_kwargs=test_kwargs,
        params=opt_params,
        num_threads=n_threads,
    )

    bfm.run(output_dir)
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
