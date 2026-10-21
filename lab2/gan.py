# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # GAN on MNIST
#
# Source: https://github.com/PavloZakala/CNN/blob/main/CNN4/gan.py
#
# ## Colab setup
# Use a GPU runtime (Runtime → Change runtime type → T4 GPU). torch/torchvision come preinstalled.

# %%
# !pip install -q einops

# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter

from einops.layers.torch import Rearrange

from IPython.display import Image, display

# %% [markdown]
# ## Models

# %%
class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


# %% [markdown]
# ## Training steps

# %%
def G_train(G, D, batch_size, z_dim, criterion, G_optimizer):
    # =======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(batch_size, z_dim).to(device)
    y = torch.ones(batch_size, 1).to(device)

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()

def D_train(x, G, D, z_dim, criterion, D_optimizer):

    D.zero_grad()
    batch_size = x.shape[0]
    # train discriminator on real
    x_real, y_real = x, torch.ones(batch_size, 1, 1, device=device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)

    # train discriminator on facke
    z = torch.randn(batch_size, z_dim).to(device)

    with torch.no_grad():
        x_fake, y_fake = G(z), torch.zeros(batch_size, 1).to(device)

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


# %% [markdown]
# ## Config

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 128
z_dim = 64
lr = 0.0002
n_epoch = 200

log_root = 'runs'  # each run logs to runs/version_<n>

# %% [markdown]
# ## Data

# %%
# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5)),
    Rearrange('c h w -> c (h w)')
])

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)

# %% [markdown]
# ## Model, loss, optimizers

# %%
G = Generator(g_input_dim=z_dim, g_output_dim=mnist_dim).to(device)
D = Discriminator(mnist_dim).to(device)

# loss
criterion = nn.BCELoss()

# optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

# %% [markdown]
# ## Training
# Every 10 epochs a grid of 64 samples is saved to `./results/` and shown below.
# Losses (per step and per epoch) and the sample grids are logged to TensorBoard.

# %%
# %load_ext tensorboard
# %tensorboard --logdir runs

# %%
os.makedirs(log_root, exist_ok=True)
log_dir = os.path.join(log_root, f'version_{len(os.listdir(log_root))}')
writer = SummaryWriter(log_dir)
print(log_dir)

global_step = 0
for epoch in range(1, n_epoch + 1):
    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(device)

        D_losses.append(D_train(x, G, D, z_dim, criterion, D_optimizer))
        G_losses.append(G_train(G, D, batch_size, z_dim, criterion, G_optimizer))

        writer.add_scalar('train/loss_d', D_losses[-1], global_step)
        writer.add_scalar('train/loss_g', G_losses[-1], global_step)
        global_step += 1

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
        (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    writer.add_scalar('epoch/loss_d', torch.mean(torch.FloatTensor(D_losses)), epoch)
    writer.add_scalar('epoch/loss_g', torch.mean(torch.FloatTensor(G_losses)), epoch)

    # Save trained parameters of model
    torch.save(G.state_dict(), 'G.pth')
    torch.save(D.state_dict(), 'D.pth')

    if epoch % 10 == 0:
        # Visualize generated data
        z = torch.randn(64, z_dim).to(device)

        # Generate image from z
        generated_images = G(z)

        # Make the images as grid
        generated_images = make_grid(generated_images.view(-1, 1, 28, 28), nrow=8, normalize=True)
        os.makedirs('./results', exist_ok=True)
        # Save the generated torch tensor models to disk
        save_image(generated_images, f'./results/gen_img{epoch}.png')
        display(Image(f'./results/gen_img{epoch}.png'))
        writer.add_image('samples', generated_images, epoch)

writer.close()

# %% [markdown]
# ## Training curves

# %%
import math
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def plot_latest_run(log_dir: str = log_root, smooth: float = 0.9) -> None:
    runs = sorted(Path(log_dir).glob("version_*"), key=lambda p: int(p.name.split("_")[1]))
    if not runs:
        raise FileNotFoundError(f"No runs found under {log_dir}")
    run = runs[-1]

    ea = EventAccumulator(str(run), size_guidance={"scalars": 0})  # 0 = load all points
    ea.Reload()
    tags = sorted(ea.Tags()["scalars"])

    cols = 2
    rows = math.ceil(len(tags) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
    for ax, tag in zip(axes.flat, tags):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        if len(values) > 20:
            # noisy per-step series: raw values faint, exponential moving average on top
            ema, s = [], values[0]
            for v in values:
                s = smooth * s + (1 - smooth) * v
                ema.append(s)
            ax.plot(steps, values, color="C0", alpha=0.25, lw=1)
            ax.plot(steps, ema, color="C0", lw=2)
        else:
            ax.plot(steps, values, color="C0", lw=2, marker="o", ms=4)
        ax.set_title(tag)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(tags):]:
        ax.axis("off")
    fig.suptitle(run.name)
    fig.tight_layout()
    plt.show()


plot_latest_run()
