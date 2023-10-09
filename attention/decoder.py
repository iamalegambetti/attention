import torch 
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)

class MaskedAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super(MaskedAttentionHead, self).__init__()
        self.embed_dim = embed_dim # dimension of the embedding 
        self.head_dim = head_dim # dimension of the head

        self.values = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.embed_dim, self.head_dim, bias=False)

    def forward(self, x):
        v = self.values(x)
        k = self.keys(x)
        q = self.queries(x)
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** .5
        weights = torch.tril(weights) # set to 0s the upper diagonal of the matrix -> do not attend prior tokens 
        weights = torch.where(weights == 0, float('-inf'), weights) # set to -inf the 0s -> when taking the softmax the will be 0
        weights = F.softmax(weights, dim = -1) 
        logits = weights @ v
        return logits

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, head_dim):
        super(MaskedMultiHeadAttention, self).__init__()
        self.n_heads = n_heads 
        self.multi_head_attention = nn.ModuleList([MaskedAttentionHead(embed_dim, head_dim) for _ in range(n_heads)])
        self.project = nn.Linear(n_heads * head_dim, embed_dim)
    
    def forward(self, x):
        x = torch.cat([h(x) for h in self.multi_head_attention], dim = -1)
        x = self.project(x)
        return x
    
class CrossAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super(CrossAttentionHead, self).__init__()
        self.embed_dim = embed_dim # dimension of the embedding 
        self.head_dim = head_dim
        self.values = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.embed_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.embed_dim, self.head_dim, bias=False)

    def forward(self, x, z):
        v = self.values(z)
        k = self.keys(z)
        q = self.queries(x)
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** .5
        weights = F.softmax(weights, dim = -1) # B, T, T
        logits = weights @ v
        return logits
    

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, head_dim):
        super(MultiHeadCrossAttention, self).__init__()
        self.multi_head_cross_attention = nn.ModuleList([CrossAttentionHead(embed_dim, head_dim) for _ in range(n_heads)])
        self.project = nn.Linear(n_heads * head_dim, embed_dim)
    
    def forward(self, x, z):
        out = torch.cat([h(x, z) for h in self.multi_head_cross_attention], dim = -1)
        return self.project(out)
    
class BlockDecoder(nn.Module):
    def __init__(self, n_heads, embed_dim, head_dim):
        super(BlockDecoder, self).__init__()
        self.MaskedMHA = MaskedMultiHeadAttention(n_heads, embed_dim, head_dim)
        self.MHCA = MultiHeadCrossAttention(n_heads, embed_dim, head_dim)
        self.FF = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, z):
        x = self.ln1(x + self.MaskedMHA(x))
        out = self.ln2(x + self.MHCA(x, z))
        out = self.ln3(out + self.FF(out))
        return out
    
class Decoder(nn.Module):
    def __init__(self, n_blocks, n_heads, embed_dim, head_dim):
        super(Decoder, self).__init__()
        #self.decoder = nn.Sequential(*[self.wrapped_model(n_heads, embed_dim, head_dim) for _ in range(n_encoder_layers)])    
        #self.decoder = SequentialWrapper(*[BlockDecoder(n_heads, embed_dim, head_dim) for _ in range(n_encoder_layers)])
        self.decoder = nn.ModuleList([BlockDecoder(n_heads, embed_dim, head_dim) for _ in range(n_blocks)])
        self.project = nn.Linear(n_blocks * embed_dim, embed_dim)

    def forward(self, x, z):
        out = torch.cat([d(x, z) for d in self.decoder], dim = -1)
        return self.project(out)