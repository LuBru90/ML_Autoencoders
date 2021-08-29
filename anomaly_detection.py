# %%
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns

import torch
import torch.nn as nn

# %% Plot:
def shiftWindow():
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.window.move(-1200, 100)
    return fig, ax

def showImage(image):
    fig, ax = shiftWindow()
    ax.imshow(image)
    fig.show()

def showPlot(*args, **kwargs):
    fig, ax = shiftWindow()
    plt.grid()
    ax.plot(*args, **kwargs)
    fig.show()

# %% Fetch mnist data:
X_raw, _ = mnist.load_data()
X = X_raw[0].reshape(-1, 28*28)
y = X_raw[1].ravel()

#X = X[y == 6]

image = X[14].reshape(28, 28)
showImage(image)

# %% convert to tensor:
X_ = torch.tensor(X/255).float()

# %% const:
ishape = (28, 28)

# %% Encoder:
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1, 8, kernel_size=10, stride=1),
                        nn.MaxPool2d(kernel_size=5, stride=1)
                    )

        self.layer2 = nn.Sequential(
                        nn.Conv2d(8, 16, kernel_size=10, stride=1, padding=1),
                        nn.MaxPool2d(kernel_size=5, stride=2),

                        nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
                        nn.MaxPool2d(kernel_size=3, stride=1),
                    )

    def forward(self, x):
        x = self.layer1(x.reshape(-1, 1, 28, 28))
        x = self.layer2(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layer0 = nn.Flatten()
        
        self.layer1 = nn.Sequential(
                                        nn.Linear(32, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 28*28),
                                    )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = torch.sigmoid(x)
        return x

mEncoder = Encoder()
mDecoder = Decoder()
x_test = X_[:10]
out = mEncoder(x_test)
print('ENCODER:', out.shape)

# %
out2 = mDecoder(out)
print('DECODER:', out2.shape)

# %%
lossFn = nn.BCELoss()
optimizer = torch.optim.Adam(list(mEncoder.parameters()) + list(mDecoder.parameters()))

BS = 64
epochs = 1000
for i in range(epochs):
    batchIds = np.random.randint(0, X_.shape[0], size=BS)
    X_sample = X_[batchIds]

    outEncoder = mEncoder(X_sample)
    outDecoder = mDecoder(outEncoder)

    loss = lossFn(outDecoder, X_sample)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(i, loss.item())

# %% forward by model:
image = X_[128].reshape(28, 28)
image = image.reshape(28*28)
outEnc = mEncoder(image)
#print(outEnc)
outDec = mDecoder(outEnc).detach().numpy().reshape(ishape)
grid = np.concatenate([image.detach().numpy().reshape(ishape), outDec], axis=1)
showImage(grid)

# %% Detect outliers:
imgs = 10000
y0 = mEncoder(X_[:imgs])
y1 = mDecoder(y0[:imgs]).detach().numpy()

x = X_[:imgs].detach().numpy()

diff = np.mean((x - y1)**2, axis = 1)
showPlot(diff)

# %%
shitty = X_[:imgs][diff < 0.004].detach().numpy()
shitty = X_[:imgs][diff > 0.008].detach().numpy()

N = 6
grid = np.concatenate([shitty[i: i+N].reshape(-1, 28) for i in range(0, N*N, N)], axis=1)
showImage(grid)
