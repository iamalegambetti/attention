#!/usr/bin/env python

import torch
import sys
sys.path.insert(0, '../Attention/')
from attention.encoder import Encoder

n_layers = 8
n_encoder_layers = 6
embed_dim = 512
head_dim = 64
batch_size = 16
context_window = 100

x = torch.randn(batch_size, embed_dim)
encoder = Encoder(n_encoder_layers, n_layers, embed_dim, head_dim)
out = encoder(x)
print(out.shape)

# number of parameters in the encoder in millions
params = int(sum(p.numel() for p in encoder.parameters()) // 1e6)
print(f"Number of Parameters in the Model: {params}M.")
