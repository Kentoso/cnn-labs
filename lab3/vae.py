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
# # VAE on MNIST
#
# 1. Train a convolutional VAE and look at its reconstructions and latent space.
# 2. Train the same CNN classifier on original images (**Baseline**) and on VAE reconstructions (**Reconstructed**),
#    plus a third classifier on the VAE latents μ (**Latent**).
# 3. Add Gaussian noise to the test set and check how much the VAE removes.
#
# ## Colab setup
# Use a GPU runtime (Runtime → Change runtime type → T4 GPU). torch/torchvision come preinstalled.

# %%
# !pip install -q umap-learn

# %%
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.utils import make_grid

# %% [markdown]
# ## Config

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.manual_seed(0)

batch_size = 128
lr = 1e-3
z_dim = 16
beta = 1.0  # weight of the KL term
vae_epochs = 20
clf_epochs = 5
sigmas = [0, 0.1, 0.2, 0.3]

# %% [markdown]
# ## Data
# MNIST as plain tensors in [0, 1], kept on the GPU. No normalization: the decoder ends with a sigmoid
# and the noisy images get clamped to [0, 1].

# %%
def load(train):
    ds = datasets.MNIST('data', train=train, download=True)
    return ds.data.float().div(255).unsqueeze(1).to(device), ds.targets.to(device)


x_train, y_train = load(True)
x_test, y_test = load(False)
print(x_train.shape, x_test.shape)

# %% [markdown]
# ## Models

# %%
class VAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),   # 28 -> 14
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # 14 -> 7
            nn.Flatten())
        self.mu = nn.Linear(64 * 7 * 7, z_dim)
        self.logvar = nn.Linear(64 * 7 * 7, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 64 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),  # 7 -> 14
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid())  # 14 -> 28

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)  # reparameterization trick
        return self.dec(z), mu, logvar


def make_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 10)).to(device)


def make_latent_mlp():
    return nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, 10)).to(device)


# %% [markdown]
# ## 1. Train the VAE
# Loss = BCE reconstruction (summed over pixels) + β · KL(q(z|x) ‖ N(0, I)), both averaged over the batch.

# %%
vae = VAE(z_dim).to(device)
opt = torch.optim.Adam(vae.parameters(), lr=lr)
loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)

history = {'recon': [], 'kl': []}
for epoch in range(1, vae_epochs + 1):
    recon_sum, kl_sum = 0.0, 0.0
    for (x,) in loader:
        x_hat, mu, logvar = vae(x)
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum') / len(x)
        kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / len(x)
        loss = recon + beta * kl

        opt.zero_grad()
        loss.backward()
        opt.step()
        recon_sum += recon.item()
        kl_sum += kl.item()

    history['recon'].append(recon_sum / len(loader))
    history['kl'].append(kl_sum / len(loader))
    print(f"[{epoch}/{vae_epochs}] recon: {history['recon'][-1]:.2f}, kl: {history['kl'][-1]:.2f}")

vae.eval()

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for ax, (name, values) in zip(axes, history.items()):
    ax.plot(range(1, vae_epochs + 1), values, marker='o')
    ax.set_title(name)
    ax.set_xlabel('epoch')
    ax.grid(alpha=0.3)
plt.show()


# %% [markdown]
# ### Helpers
# Deterministic encode / reconstruct: use μ, no sampling. Done in chunks to keep GPU memory low.

# %%
@torch.no_grad()
def encode(x):
    return torch.cat([vae.encode(chunk)[0] for chunk in x.split(1000)])


@torch.no_grad()
def reconstruct(x):
    return torch.cat([vae.dec(vae.encode(chunk)[0]) for chunk in x.split(1000)])


def show(rows, titles):
    # rows: list of (n, 1, 28, 28) tensors, shown one under another
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.2 * len(rows)))
    for ax, row, title in zip(axes, rows, titles):
        ax.imshow(make_grid(row.cpu(), nrow=len(row), pad_value=1).permute(1, 2, 0))
        ax.set_title(title, loc='left', fontsize=10)
        ax.axis('off')
    plt.show()


# %% [markdown]
# ### Original vs reconstructed

# %%
show([x_test[:16], reconstruct(x_test[:16])], ['original', 'reconstructed'])

# %% [markdown]
# ### Latent space (UMAP of test μ)

# %%
mu_test = encode(x_test)
emb = umap.UMAP(random_state=0).fit_transform(mu_test.cpu().numpy())

plt.figure(figsize=(7, 6))
plt.scatter(emb[:, 0], emb[:, 1], c=y_test.cpu(), cmap='tab10', s=2)
plt.colorbar(ticks=range(10))
plt.title('UMAP of test-set μ')
plt.show()

# %% [markdown]
# ### Analysis
# - Which features survive reconstruction (overall shape, stroke direction, digit identity)?
# - Which are lost (thin strokes, sharp edges, small gaps or loops, unusual writing styles)?
# - Are any digits reconstructed as a *different* digit? Do those cases sit between clusters in the UMAP?
#
# **TODO:** conclusions after the run.

# %% [markdown]
# ## 2. Classifiers
# Three models, all trained with the same loop:
# - **Baseline**: CNN on original images
# - **Reconstructed**: same CNN on Decoder(Encoder(I))
# - **Latent**: small MLP on μ = Encoder(I), with the VAE frozen

# %%
def train_classifier(model, X, y, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f'  [{epoch}/{epochs}] loss: {loss.item():.4f}')
    model.eval()
    return model


@torch.no_grad()
def accuracy(model, X, y):
    preds = torch.cat([model(chunk).argmax(1) for chunk in X.split(1000)])
    return (preds == y).float().mean().item()


# %%
x_train_rec = reconstruct(x_train)
mu_train = encode(x_train)

print('Baseline')
clf_base = train_classifier(make_cnn(), x_train, y_train, clf_epochs)
print('Reconstructed')
clf_rec = train_classifier(make_cnn(), x_train_rec, y_train, clf_epochs)
print('Latent')
clf_lat = train_classifier(make_latent_mlp(), mu_train, y_train, clf_epochs)

# %%
pd.Series({
    'Baseline': accuracy(clf_base, x_test, y_test),
    'Reconstructed': accuracy(clf_rec, x_test, y_test),
    'Latent': accuracy(clf_lat, mu_test, y_test),
}, name='accuracy on original test set')

# %% [markdown]
# ### Analysis
# - How much accuracy does Reconstructed lose compared to Baseline? What does that say about the information
#   the VAE throws away?
# - How does a 16-dim latent compare to the full 784-pixel image as classifier input?
#
# **TODO:** conclusions after the run.

# %% [markdown]
# ## 3. Input noise
# For each σ: `noisy = clamp(I + N(0, σ²), 0, 1)` and `denoised = Decoder(Encoder(noisy))`.
# Baseline and Reconstructed are tested on both sets; Latent is tested on Encoder(noisy).
# σ = 0 is the clean reference row.

# %%
torch.manual_seed(0)
rows, examples = [], []
for sigma in sigmas:
    noisy = (x_test + sigma * torch.randn_like(x_test)).clamp(0, 1)
    denoised = reconstruct(noisy)
    rows.append({
        'sigma': sigma,
        'Baseline / noisy': accuracy(clf_base, noisy, y_test),
        'Baseline / denoised': accuracy(clf_base, denoised, y_test),
        'Reconstructed / noisy': accuracy(clf_rec, noisy, y_test),
        'Reconstructed / denoised': accuracy(clf_rec, denoised, y_test),
        'Latent / Encoder(noisy)': accuracy(clf_lat, encode(noisy), y_test),
    })
    examples.append((sigma, noisy[:12], denoised[:12]))

results = pd.DataFrame(rows).set_index('sigma')
results

# %%
results.plot(marker='o', figsize=(8, 5), grid=True, ylabel='accuracy', title='Accuracy vs input noise σ')
plt.show()

# %%
for sigma, noisy, denoised in examples:
    show([x_test[:12], noisy, denoised], [f'original (σ={sigma})', 'noisy', 'denoised'])

# %% [markdown]
# ### Analysis
# - Does the VAE remove the noise? Compare the noisy and denoised accuracies for Baseline as σ grows.
# - Is Reconstructed more robust on noisy inputs than Baseline? Why might it be?
# - How does the Latent classifier degrade compared to the image classifiers?
# - At which σ does the VAE start reconstructing the wrong digit?
#
# **TODO:** conclusions after the run.
