import torch
import torch.nn as nn
import numpy as np

def patchify(batch: torch.Tensor, patch_size: tuple = (16, 16)):
    """
    Patchify the batch of images

    Shape:
        batch: (b, h, w, c)
        output: (n, nh*nw, ph*pw*c)
    """
    b, h, w, c = batch.shape # (n, 224, 224, 3)
    ph, pw = patch_size # (16, 16)
    nh, nw = h // ph, w // pw # (14, 14)

    patches = torch.zeros(b, nh*nw, ph*pw*c).to(batch.device) # (n, nh*nw, ph*pw*c) = (n, 196, 768)

    for idx, image in enumerate(batch):
        for i in range(nh):
            for j in range(nw):
                patch = image[i*ph: (i+1)*ph, j*pw: (j+1)*pw, :]
                patches[idx, i*nh + j] = patch.flatten()
    return patches # (n, nh*nw, ph*pw*c) = (n, 196, 768)

def get_mlp(in_features, hidden_units, out_features):
    """
    Returns a MLP head
    """
    dims = [in_features] + hidden_units + [out_features]
    layers = []
    for dim1, dim2 in zip(dims[:-2], dims[1:-1]):
        layers.append(nn.Linear(dim1, dim2))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)

def get_positional_embeddings(sequence_length, d):
    """
    Returns position embeddings
    """
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result # (s, d)