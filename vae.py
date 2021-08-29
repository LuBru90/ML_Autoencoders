# %% Imports:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from keras.datasets import mnist

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn

# %% Plot:
def showImage(image):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.window.move(-1200, 100)

    ax.imshow(image, cmap='gray')

    fig.show()

# %% Fetch mnist data:
ishape = (28, 28)

X_raw, _ = mnist.load_data()

X = X_raw[0].reshape(-1, 28*28)
X_ = torch.tensor(X/255).float()

y = X_raw[1]

# %% Show some images:
image = X[24].reshape(28, 28)
#showImage(image)

# %% Variational Autoencoder:
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
                    nn.Linear(28*28, 128),
                    nn.ReLU(),
                    nn.Linear(128, FEATURES*2),
                )

        self.decoder = nn.Sequential(
                    nn.Linear(FEATURES, 128),
                    nn.ReLU(),
                    nn.Linear(128, 28*28),
                )

    def encode(self, x):
        x = self.encoder(x)
        return x

    def repTrick(self, x):
        x = x.view(-1, 2, FEATURES)
        mu = x[:, 0, :]
        var = x[:, 1, :]

        sigma = torch.exp(0.5*var)
        z = torch.randn_like(sigma)
        x =  mu + sigma*z
        return x, mu, var

    def decode(self, x):
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x, mu, var = self.repTrick(x)
        x = self.decode(x)
        return x, mu, var

    def getReconImage(self, image):
        image = torch.tensor(image).view(1, -1).float()
        with torch.no_grad():
            out1 = self.encode(image)
            z, mu, var = self.repTrick(out1)
            recon = self.decode(z).detach().numpy().reshape(ishape)

            zOut = z.detach().numpy().ravel()
        return recon, zOut

    def getXYLatent(self, image):
        with torch.no_grad():
            out1 = self.encode(image)
            recon, _, _ = self.repTrick(out1)

FEATURES = 2
model = VAE()

enc = model.encode(X_[:10])
enc.shape

rep, mu, var = model.repTrick(enc)
rep.shape; print(mu.T, var.T)

z = torch.tensor([0,0]).float()
out = model.decode(z)
out.shape

# Custom loss function:
def VAELoss(const, target, mu, log_var):
    kl_loss =  torch.mean((-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = 1)), dim =0)
    reconLoss = torch.mean((const - target)**2)
    return kl_loss * 0.0001 + reconLoss

optimizer = torch.optim.Adam(model.parameters())

# %% Training:
BS = 32
#nforEpoch = X.shape[0]//BS
for i in range(1500):
    batchIds = np.random.randint(0, X.shape[0], size=BS)
    target = X_[batchIds]

    recon, mu, var = model(target)
    loss = VAELoss(recon, target, mu, var)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 20 == 0:
        print(i, loss.item())

# %% buggy
if 0:
    image = X_[231]
    recon,_,_ = model.forward(image)
    recon = recon.detach().numpy().reshape(ishape)
    grid = np.concatenate([image.reshape(ishape), recon], axis=1)
    showImage(grid)

# %% gen Image:
    z = torch.tensor([-1, -3]).float()
    genImage = model.decode(z).detach().numpy().reshape(28,28)
    showImage(genImage)

# %% Segmentation:
colors = list(cm.rainbow(np.linspace(0, 1, 10)))

with torch.no_grad():
    for i, (image, label) in enumerate(zip(X_, y)):
        out1 = model.encode(image.reshape(1, -1))
        recImage, mu, var = model.repTrick(out1)
        xZ,yZ = mu.detach().numpy().ravel()
        plt.plot(xZ, yZ, 'o', color=colors[label], alpha=0.2)
        if i == 5000:
            break
plt.show()

# %% gen image array:
if 0:
    gridSize = 15
    res = np.linspace(-3, 3, gridSize)
    xg, yg = np.meshgrid(np.linspace(-4, 6, gridSize), np.linspace(-6, 4, gridSize))
    xg = xg.ravel()
    yg = yg.ravel()
    z = np.vstack((xg, yg)).T.reshape(gridSize**2, 2)
    z = torch.tensor(z).float()
    z.shape

    genImage = model.decode(z).detach().numpy()

    wholeImage = list()
    counter = 0
    for xI in range(gridSize):
        xImages = list()
        for imageI in range(gridSize):
            image = genImage[counter].reshape(ishape)
            xImages.append(image)
            counter += 1

        rowImage = np.concatenate([*xImages], axis=1)
        wholeImage.append(rowImage)

    grid = np.concatenate([*wholeImage])
    showImage(grid)
