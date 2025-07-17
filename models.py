
import numpy as np
import math
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
from torch.nn import functional as F


def positional_encoding(pos, d_model=512):
    # Computes the sinusoidal positional encoding for a given position `pos`,
    # as introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017).

    pe = np.zeros((1, d_model))
    for i in range(0, d_model, 2):
        pe[0][i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
        pe[0][i+1] = math.cos(pos / (10000 ** ((2 * i)/d_model)))
    return pe


# Diffusers implementatin https://github.com/huggingface/diffusers/blob/a98a839de75f1ad82d8d200c3bc2e4ff89929081/src/diffusers/models/embeddings.py#L162
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)                 # 2i/d
    omega = 1.0 / (temperature ** omega)                            # 1 / (10000 ^ (2i/d))
    # print(omega[None, :].size())
    # print(y.flatten().size(), y.flatten()[:, None].size())
    # print(x.flatten())
    y = y.flatten()[:, None] * omega[None, :]                       # pos * (1 / (10000 ^ (2i/d)))  y.size ()
    # print(y)
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)     # sin(pos * (1 / (10000 ^ (2i/d)))) cos(pos * (1 / (10000 ^ (2i/d))))
    return pe.type(dtype)


class ImageEmbedder(nn.Module):
  def __init__(self, patch_size, embed_size, channels = 3):
    super().__init__()
    self.patch_size = patch_size
    self.embed_size = embed_size
    self.patch_dim = self.patch_size * self.patch_size * channels

    self.to_flat_patches = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2=self.patch_size)

    self.embedder = nn.Linear(self.patch_dim, self.embed_size)    # Differs from transformers with trainable vocabulary

  def forward(self, x):
    # x (B,H,W,C)
    x = self.to_flat_patches(x) # x-> patches  shape = (B,(H*W)/(patch_size**2), patch_size*patch_size*C)
    print(x.size())
    out = self.embedder(x)      # B, (H*W)/(patch_size**2), embed_size
    return out


class ImageEmbedderPE(nn.Module):
  def __init__(self, patch_size, embed_size, img_h, img_w, channels = 3):
    super().__init__()
    self.patch_size = patch_size
    self.embed_size = embed_size
    self.patch_dim = self.patch_size * self.patch_size * channels

    self.to_flat_patches = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2=self.patch_size)

    self.embedder = nn.Linear(self.patch_dim, self.embed_size)    # Differs from transformers with trainable vocabulary
    self.pe_table = posemb_sincos_2d(img_h//self.patch_size, img_w//self.patch_size, dim=self.embed_size)

  def forward(self, x):
    # x (B,H,W,C)
    device = x.device
    x = self.to_flat_patches(x)                         # x-> patches  shape = (B,(H*W)/(patch_size**2), patch_size*patch_size*C)
    x = self.embedder(x)                                # B, (H*W)/(patch_size**2), embed_size
    out = x + self.pe_table.to(device, dtype=x.dtype)   # But it has constant size so it wouldn't work with images of different sizes
                                                        # To fix it PE could be calculated during forward call
    return out
  

class Attention(nn.Module):
  def __init__(self, embed_size, head_size):
    super().__init__()
    self.q_proj = nn.Linear(embed_size, head_size, bias=False)
    self.k_proj = nn.Linear(embed_size, head_size, bias=False)
    self.v_proj = nn.Linear(embed_size, head_size, bias=False)
    self.scale = head_size ** (-0.5)                                    #  1 / sqrt(D)

  def forward(self, q, k, v):

    q = self.q_proj(q)
    k = self.k_proj(k)

    scores = F.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)     # (B,T,C) @ (B,C,T) -> (B,T,T)
    v = self.v_proj(v)
    out = scores @ v                                                     # (B,T,T) @ (B,T,C) -> (B,T,C)
    return out


class NotViT(nn.Module):
  def __init__(self, embed_size, input_h, input_w, patch_size=2, num_classes=10):
    """
    Just naive embedder, one block of self attention and classification head
    embed_size  -   dimensionality of the embeddings produced by the patch encoder
    d_model     -   dimensionality of the self-attention output (equal to embed_size in this implementation)
    """
    super().__init__()

    self.embedder = ImageEmbedderPE(patch_size, embed_size, img_h=input_h, img_w=input_w)
    self.sa = Attention(embed_size, head_size=embed_size)

    self.classification_head = nn.Linear(embed_size, num_classes)

  def forward(self, x):
    emb = self.embedder(x)                  # (B, T, embed_size)
    emb = emb + self.sa(emb, emb, emb)      # Add self-attention output (B, T, embed_size)

    out = self.classification_head(emb)     # (B, T, num_classes)

    # However, we need a prediction of shape (B, num_classes)
    # Since each token represents a patch of the image,
    # to obtain a single prediction for the whole image,
    # we can simply average the token-level predictions across the sequence dimension
    out = out.mean(dim = 1)
    return out

