import math

import torch
import torch.nn as nn
from timm.models.layers import Mlp


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: (B, T, P)
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t * freqs.reshape(1, 1, 1, -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning for ego and Cross-Attention.
    """

    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)

        self.mlp2 = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )

    def forward(self, x, cross_c):
        normed_x = self.norm1(x)
        x = x + self.attn(normed_x, normed_x, normed_x)[0]

        normed_x = self.norm2(x)
        x = x + self.mlp1(normed_x)

        x = x + self.cross_attn(self.norm3(x), cross_c, cross_c)[0]

        x = x + self.mlp2(self.norm4(x))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True),
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.proj(x)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        depth,
        output_dim,
        hidden_dim=192,
        heads=6,
        dropout=0.1,
        mlp_ratio=4.0,
    ):
        super().__init__()

        P = 33
        D = 4

        self.agent_embedding = nn.Embedding(2, D)
        self.time_embedding = nn.Embedding(100, hidden_dim)
        self.preproj = Mlp(
            in_features=P * D,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.t_embedder = TimestepEmbedder(D)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_dim, P * D)

    def forward(self, x, t, cross_c, neighbor_current_mask):
        """
        Forward pass of DiT.
        x: (B, P, T, D)   -> Embedded out of DiT
        t: (B, P, T, 1)         -> Timestep embedding
        cross_c: (B, N, D)      -> Cross-Attention context
        """
        assert x.dim() == 4, f"{x.dim()=}"
        assert t.dim() == 4, f"{t.dim()=}"
        assert x.shape[2] == t.shape[2], f"{x.shape[2]=} {t.shape[2]=}"
        B, P, T, D = x.shape

        # Add agent type embedding
        agent_embedding = torch.cat(
            [
                self.agent_embedding.weight[0][None, :],
                self.agent_embedding.weight[1][None, :].expand(P - 1, -1),
            ],
            dim=0,
        )  # (P, D)
        agent_embedding = agent_embedding[None, :, None, :].expand(B, -1, -1, -1)  # (B, P, 1, D)
        x = x + agent_embedding

        # Added denoising timestep embedding
        y = self.t_embedder(t)  # (B, T, P, D)
        x = x + y

        x = x.permute(0, 2, 1, 3)  # (B, T, P, D)

        # Reshape to (B, T, P*D)
        x = x.reshape(B, T, P * D)
        x = self.preproj(x)  # (B, T, hidden_dim)

        # Added time embedding
        time_embedding = self.time_embedding.weight[:T]  # (T, hidden_dim)
        x = x + time_embedding[None, :, :]

        for block in self.blocks:
            x = block(x, cross_c)

        x = self.final_layer(x)  # (B, T, P*D)
        x = x.reshape(B, T, P, D)
        x = x.permute(0, 2, 1, 3)  # (B, P, T, D)
        return x
