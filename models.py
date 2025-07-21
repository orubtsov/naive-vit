
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
    y = y.flatten()[:, None] * omega[None, :]                       # pos * (1 / (10000 ^ (2i/d)))  y.size ()
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)     # sin(pos * (1 / (10000 ^ (2i/d)))) cos(pos * (1 / (10000 ^ (2i/d))))
    return pe.type(dtype)


class ImageEmbedder(nn.Module):
    """
    Converts an input image into a sequence of flat patch embeddings using a linear projection.

    This class does not apply any positional encoding. It simply splits the image into
    non-overlapping patches, flattens them, and projects each patch to a fixed embedding size.

    Args:
        patch_size (int): Size of the square image patches (e.g., 16).
        embed_size (int): Dimensionality of each patch embedding.
        channels (int): Number of image input channels (default: 3 for RGB).

    Inputs:
        x (Tensor): Input image tensor of shape (B, C, H, W)

    Returns:
        out (Tensor): Patch embeddings of shape (B, T, embed_size),
                      where T = (H * W) / (patch_size ** 2)
    """
    def __init__(self, patch_size, embed_size, channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.patch_dim = patch_size * patch_size * channels

        # Convert image to flattened patches: (B, C, H, W) → (B, T, patch_dim)
        self.to_flat_patches = Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=patch_size, p2=patch_size
        )

        # Linear projection of each flattened patch to embedding space
        self.embedder = nn.Linear(self.patch_dim, embed_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.to_flat_patches(x)  # (B, T, patch_dim)
        print(x.size())              # For debugging: show patch shape
        out = self.embedder(x)       # (B, T, embed_size)
        return out


class ImageEmbedderPE(nn.Module):
    """
    Converts an input image into a sequence of patch embeddings and adds 2D sinusoidal positional encoding.

    The positional encoding is fixed and precomputed in __init__, meaning this module
    only supports a single fixed image size. For variable-size support, consider computing
    PE dynamically in the forward pass.

    Args:
        patch_size (int): Size of the square image patches.
        embed_size (int): Dimensionality of each patch embedding.
        img_h (int): Height of the input image in pixels.
        img_w (int): Width of the input image in pixels.
        channels (int): Number of input channels (default: 3 for RGB).

    Inputs:
        x (Tensor): Input image tensor of shape (B, C, H, W)

    Returns:
        out (Tensor): Patch embeddings with positional encodings of shape (B, T, embed_size)
    """
    def __init__(self, patch_size, embed_size, img_h, img_w, channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.patch_dim = patch_size * patch_size * channels

        # Patch flattening: (B, C, H, W) → (B, T, patch_dim)
        self.to_flat_patches = Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=patch_size, p2=patch_size
        )

        self.embedder = nn.Linear(self.patch_dim, embed_size)

        # Fixed positional encoding for a fixed image size
        # PE shape: (T, D) → will be broadcasted to (B, T, D)
        self.pe_table = posemb_sincos_2d(
            img_h // patch_size,
            img_w // patch_size,
            dim=embed_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        device = x.device
        x = self.to_flat_patches(x)           # (B, T, patch_dim)
        x = self.embedder(x)                  # (B, T, embed_size)
        out = x + self.pe_table.to(device, dtype=x.dtype)  # (B, T, embed_size)
        return out
  

class Attention(nn.Module):
  """
  Simple Attention module implementation with no optimizations

  Args:
        embed_size (int): Dimensionality of patch embeddings.
        head_size (int): Dimensionality of inner dim.

    Inputs:
        q (Tensor): Input tensor of shape (B, T, C)
        k (Tensor): Input tensor of shape (B, T, C)
        v (Tensor): Input tensor of shape (B, T, C)

    Returns:
        out (Tensor): Tensor of shape (B,T,C)

  """
  def __init__(self, embed_size, head_size):
    super().__init__()
    self.q_proj = nn.Linear(embed_size, head_size, bias=False)
    self.k_proj = nn.Linear(embed_size, head_size, bias=False)
    self.v_proj = nn.Linear(embed_size, head_size, bias=False)
    self.scale = head_size ** (-0.5)                                    #  1 / sqrt(D)

  def forward(self, q, k, v):

    q = self.q_proj(q)      # q, k, v: (B, T, head_size)
    k = self.k_proj(k)
    v = self.v_proj(v)
    scores = F.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)     # (B,T,head_size) @ (B,head_size,T) -> (B, T, T) — attention weights between tokens
    out = scores @ v                                                     # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
    return out


class NotViT(nn.Module):
  """
    Just naive embedder, one block of self attention and classification head

    Args:
        embed_size (int): Dimensionality of patch embeddings.
        input_h (int): Height of input images.
        input_w (int): Width of input images.
        patch_size (int): Size of square patches to split the image into.
        num_classes (int): Number of output classes.

    Inputs:
        x (Tensor): Input image tensor of shape (B, C, H, W)

    Returns:
        out (Tensor): Logits of shape (B, num_classes)
    """
  def __init__(self, embed_size, input_h, input_w, patch_size=2, num_classes=10):
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

class FeedForward(nn.Module):
    """
    A two-layer MLP used in Transformer blocks.

    Args:
        embed_size (int): Size of the input and output features.
        mlp_ratio (int): Expansion factor for the hidden layer (typically 4).
    """
    def __init__(self, embed_size, mlp_ratio=4):
        super().__init__()
        hidden_dim = embed_size * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_size)
        )

    def forward(self, x):
        return self.net(x)

class NaiveViT(nn.Module):
  def __init__(self, embed_size, input_h, input_w, patch_size=2, num_classes=10, mlp_ratio=4):
    """
    embed_size  -   size of embedding produced by patch encoder
    d_model     -   size of self attention output
    """
    super().__init__()

    self.embedder = ImageEmbedderPE(patch_size, embed_size, img_h=input_h, img_w=input_w)
    self.sa = Attention(embed_size, head_size=embed_size)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)
    self.ffn = FeedForward(embed_size, mlp_ratio=mlp_ratio)

    self.classification_head = nn.Linear(embed_size, num_classes)

  def forward(self, x):

    emb = self.embedder(x)                  # (B, T, embed_size)
    res = emb
    emb = self.norm1(emb)
    emb = res + self.sa(emb, emb, emb)      # (B, T, embed_size)

    res = emb
    emb = self.norm2(emb)
    emb = res + self.ffn(emb)
    out = self.classification_head(emb)     # (B, T, num_classes)
    # But we need prediction of shape B x num_classes.
    # If eache token represents part of an image to get class of an whole image for now we can simply average predictions over the first dimention
    out = out.mean(dim = 1)                 # (B, num_classes)
    return out
  

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer using multiple independent attention heads.
    This is a naïve implementation where each head is a separate Attention module.

    Args:
        embed_size (int): The dimensionality of the input embeddings.
        dim (int): The output dimensionality of each attention output (typically equal to embed_size).
        num_heads (int): The number of attention heads.

    Inputs:
        q, k, v (Tensor): Query, Key, and Value tensors of shape (B, T, D), where
                          B = batch size, T = sequence length, D = embed_size.

    Returns:
        out (Tensor): Output tensor of shape (B, T, dim).
    """
    def __init__(self, embed_size, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads."

        self.heads = nn.ModuleList([
            Attention(embed_size, head_size=dim // num_heads)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        # Concatenate the outputs from all heads along the last dimension
        x = torch.cat([h(q, k, v) for h in self.heads], dim=-1)  # (B, T, dim)
        out = self.proj(x)  # Final projection to match input dim
        return out


class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of multi-head self-attention, LayerNorms, and a feedforward network.

    Args:
        embed_size (int): Dimensionality of input embeddings.
        dim (int): Dimensionality of internal representations (typically equal to embed_size).
        num_heads (int): Number of attention heads in the MultiHeadAttention layer.

    Inputs:
        x (Tensor): Input tensor of shape (B, T, D)

    Returns:
        out (Tensor): Output tensor of shape (B, T, D)
    """
    def __init__(self, embed_size, dim, num_heads):
        super().__init__()
        self.sa = MultiHeadAttention(embed_size, dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x):
        # Self-attention block with residual connection
        res = x
        x = self.norm1(x)
        x = res + self.sa(x, x, x)

        # Feedforward block with residual connection
        res = x
        x = self.norm2(x)
        out = res + self.ffn(x)
        return out


class SimpleViT(nn.Module):
    """
    A simplified Vision Transformer for image classification.

    Args:
        embed_size (int): Dimensionality of patch embeddings.
        input_h (int): Height of input images.
        input_w (int): Width of input images.
        patch_size (int): Size of square patches to split the image into.
        num_classes (int): Number of output classes.
        depth (int): Number of Transformer blocks in the encoder.
        num_heads (int): Number of attention heads in each block.

    Inputs:
        x (Tensor): Input image tensor of shape (B, C, H, W)

    Returns:
        out (Tensor): Logits of shape (B, num_classes)
    """
    def __init__(self, embed_size, input_h, input_w, patch_size=2, num_classes=10, depth=4, num_heads=8):
        super().__init__()

        self.embedder = ImageEmbedderPE(
            patch_size,
            embed_size,
            img_h=input_h,
            img_w=input_w
        )

        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_size=embed_size, dim=embed_size, num_heads=num_heads)
              for _ in range(depth)]
        )

        self.classification_head = nn.Linear(embed_size, num_classes, bias=False)

    def forward(self, x):
        # Convert image to patch embeddings + positional encodings
        emb = self.embedder(x)  # (B, T, D)

        # Pass through transformer encoder
        latents = self.transformer(emb)  # (B, T, D)

        # Aggregate over sequence dimension (average pooling)
        latents = latents.mean(dim=1)  # (B, D)

        # Final classification layer
        out = self.classification_head(latents)  # (B, num_classes)
        return out
    

class ImageEmbedderClsPE(nn.Module):
    """
    Converts an image into a sequence of patch embeddings with fixed 2D sinusoidal positional encoding
    and prepends a learnable CLS token.

    Args:
        patch_size (int): Size of square patches to divide the image.
        embed_size (int): Size of output embedding per patch.
        img_h (int): Input image height.
        img_w (int): Input image width.
        channels (int): Number of input channels. Default is 3.

    Inputs:
        x (Tensor): Input tensor of shape (B, C, H, W)

    Returns:
        out (Tensor): Output tensor of shape (B, T+1, embed_size),
                      where the first token is the learnable CLS token.
    """
    def __init__(self, patch_size, embed_size, img_h, img_w, channels=3):
        super().__init__()
        self.cls_emb = nn.Parameter(torch.zeros((1, 1, embed_size)))
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.patch_dim = self.patch_size * self.patch_size * channels

        self.to_flat_patches = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                         p1=patch_size, p2=patch_size)

        self.embedder = nn.Linear(self.patch_dim, embed_size)
        self.pe_table = posemb_sincos_2d(img_h // patch_size, img_w // patch_size, dim=embed_size)

    def forward(self, x):
        device = x.device
        x = self.to_flat_patches(x)              # (B, T, patch_dim)
        x = self.embedder(x)                     # (B, T, embed_size)
        b, _, _ = x.size()

        # Add positional encoding (broadcasted)
        x = x + self.pe_table.to(device, dtype=x.dtype)  # (B, T, D)

        # Prepend CLS token
        x = torch.cat([self.cls_emb.expand(b, -1, -1), x], dim=1)  # (B, T+1, D)
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT) model using CLS token for image classification.

    Args:
        embed_size (int): Dimensionality of patch embeddings.
        input_h (int): Input image height.
        input_w (int): Input image width.
        patch_size (int): Size of image patches (square).
        num_classes (int): Number of classification categories.
        depth (int): Number of transformer blocks.
        num_heads (int): Number of attention heads per block.

    Inputs:
        x (Tensor): Input image tensor of shape (B, C, H, W)

    Returns:
        out (Tensor): Output logits of shape (B, num_classes)
    """
    def __init__(self, embed_size, input_h, input_w, patch_size=2, num_classes=10, depth=4, num_heads=8):
        super().__init__()
        self.embedder = ImageEmbedderClsPE(patch_size, embed_size, img_h=input_h, img_w=input_w)

        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_size=embed_size, dim=embed_size, num_heads=num_heads)
              for _ in range(depth)]
        )

        self.classification_head = nn.Linear(embed_size, num_classes, bias=False)

    def forward(self, x):
        emb = self.embedder(x)           # (B, T+1, D)
        latents = self.transformer(emb)  # (B, T+1, D)
        cls_token = latents[:, 0, :]     # Extract [CLS] token (B, D)
        out = self.classification_head(cls_token)  # (B, num_classes)
        return out