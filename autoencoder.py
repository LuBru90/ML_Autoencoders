# %%
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

# %% Plot:
def showImage(image):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.window.move(-1200, 100)

    ax.imshow(image, cmap='gray')

    fig.show()

# %% Fetch mnist data:
X_raw, _ = mnist.load_data()
X = X_raw[0].reshape(-1, 28*28)

# %% Show some images:
image = X[51].reshape(28, 28)
showImage(image)

# %% convert to tensor:
X_ = torch.tensor(X/255).float()

# %% const:
ishape = (28, 28)

# %% Encoder:
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Linear(28*28, 64)
        self.act1 = nn.ReLU()

        self.layer2 = nn.Linear(64, 32)
        self.act2 = nn.ReLU()

        self.layer3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.layer3(x)
        return x

# %% Decoder:
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.z = None

        self.layer1 = nn.Linear(10, 32)
        self.act1 = nn.ReLU()

        self.layer2 = nn.Linear(32, 64)
        self.act2 = nn.ReLU()

        self.layer3 = nn.Linear(64, 28*28)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

# %% Init + Training:
mEncoder = Encoder()
mDecoder = Decoder()

lossFn = nn.BCELoss()
optimizer = torch.optim.Adam(list(mEncoder.parameters()) + list(mDecoder.parameters()))

# %%
BS = 32
for i in range(2000):
    batchIds = np.random.randint(0, X.shape[0], size=BS)
    X = X_[batchIds]

    outEncoder = mEncoder(X)
    outDecoder = mDecoder(outEncoder)
    loss = lossFn(outDecoder, X)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(i, loss.item())

# %% forward by model:
image = X_[42].reshape(28,28)
#image[10:11,:10] = 1
image = image.reshape(28*28)
outEnc = mEncoder(image)
print(outEnc)
outDec = mDecoder(outEnc).detach().numpy().reshape(ishape)
grid = np.concatenate([image.detach().numpy().reshape(ishape), outDec], axis=1)
showImage(grid)

# %% set latent space:
z = torch.tensor([10, 100, 150, 0]).reshape(1, 4).float()
z = torch.rand(4).reshape(1, 4).float()
outDec = mDecoder(z).detach().numpy().reshape(ishape)
showImage(outDec)

# %% grid:
images = list()
for i in [10, 50, 100, 200]:
    print(i)
    z = torch.tensor([0, i, 0, 0,0,0,0,0,0,0]).reshape(1, 10).float()
    genImage = mDecoder(z).detach().numpy().reshape(ishape)
    images.append(genImage)

out = np.concatenate([*images], axis=1)
showImage(out)

