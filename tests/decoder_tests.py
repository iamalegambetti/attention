#!/usr/bin/env python

import sys
sys.path.insert(0, '../Attention/')

from attention.decoder import Decoder
import torch

n_blocks = 6
n_heads = 8
embed_dim = 512
head_dim = 64
batch_size = 16
context_window = 100

x = torch.randn(batch_size, context_window, embed_dim)
z = torch.randn(batch_size, context_window, embed_dim)
decoder = Decoder(n_blocks, n_heads, embed_dim, head_dim)
out = decoder(x, z)
print(out.shape)
