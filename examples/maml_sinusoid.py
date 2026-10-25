"""Meta-train MAML on sinusoid regression for a few hundred steps and adapt to a new task."""

import torch
from torch import nn
from torch.nn import functional as F

from metareason import MAML
from metareason.tasks import SinusoidTasks, seed_everything

seed_everything(0)
model = nn.Sequential(nn.Linear(1, 40), nn.ReLU(), nn.Linear(40, 40), nn.ReLU(), nn.Linear(40, 1))
maml = MAML(model, inner_lr=0.01, inner_steps=1)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
tasks = SinusoidTasks(seed=0)
for it in range(500):
    loss = maml.meta_step(tasks.batch(8), opt)
    if it % 100 == 0:
        print(f"iter {it:4d} meta-loss {loss:.3f}")

task = SinusoidTasks(seed=123).sample()
adapted = maml.adapt(task.support_x, task.support_y, steps=10)
with torch.no_grad():
    print("query MSE before:", float(F.mse_loss(model(task.query_x), task.query_y)))
    print("query MSE after 10 steps:", float(F.mse_loss(maml.forward(task.query_x, adapted), task.query_y)))
