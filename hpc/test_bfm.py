import json
import os
import sys

import numpy as np
import torch

from rctorch.models import MorrisLecarCurrent
from rctorch.optimizers import BruteForceMesh, KWArgsEncoder
from rctorch.supervisors import LorenzAttractor
from rctorch.utils import minmax_transform

# Define the output directory
output_dir = os.path.join(os.getcwd(), "bfm_output", "test_bfm_long")
# Define the directory to save results
os.makedirs(output_dir, exist_ok=True)

seed = 1
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.random.manual_seed(seed)

# Generate the supervisor signal
T = 20_000
dt = 1e-1
x = LorenzAttractor(T, dt, tau=0.01).generate(transient_time=500.0)

x = x.T
sup = minmax_transform(x, zero_mean=True)
print(sup.shape)
np.save(os.path.join(output_dir, "supervisor.npy"), sup)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sup_tensor = torch.tensor(sup, device=device, dtype=torch.float32)

# Set the model parameters
Ne = 400
Ni = 100
N = Ne + Ni
BIAS = np.ones((N, 1)) * 65.0
reservoir_params = {
    "model_cls": MorrisLecarCurrent,
    "n_input": sup_tensor.size(1),
    "n_output": sup_tensor.size(1),
    "Ne": Ne,
    "Ni": Ni,
    "dt": dt,
    "BIAS": BIAS,
    "p_sparsity": 0.1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


params_file = os.path.join(output_dir, "reservoir_params.json")
with open(params_file, "w") as f:
    json.dump(
        reservoir_params, f, cls=KWArgsEncoder, indent=4
    )  # Added indent for readability

q_range = np.array([20, 450])
gbar_range = np.array([5, 40])

opt_params = {"w_in_amp": q_range, "gbar": gbar_range}
opt_params_file = os.path.join(output_dir, "opt_params.json")
with open(opt_params_file, "w") as f:
    json.dump(
        opt_params, f, cls=KWArgsEncoder, indent=4
    )  # Added indent for readability

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

n_threads = 4

bfm = BruteForceMesh(
    reservoir_kwargs=reservoir_params,
    train_kwargs=train_kwargs,
    test_kwargs=test_kwargs,
    params=opt_params,
    num_threads=n_threads,
)

bfm.run(output_dir)
print(f"Results saved in: {output_dir}")
