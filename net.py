import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :]
        embd = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(embd)


class EnergyEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        if e.dim() == 1:
            e = e.unsqueeze(-1)
        return self.mlp(e)


class ResBlock3D(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, t_dim: int, e_dim: int):
        super().__init__()
        self.conv1  = nn.Conv3d(in_ch,  out_ch, 3, padding=0)
        self.conv2  = nn.Conv3d(out_ch, out_ch, 3, padding=0)
        self.norm1  = nn.GroupNorm(min(8, in_ch),  in_ch)
        self.norm2  = nn.GroupNorm(min(8, out_ch), out_ch)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.e_proj = nn.Linear(e_dim, out_ch)
        self.skip   = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    @staticmethod
    def _pad(x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, 0, 0, 1, 1), mode='constant', value=0)
        x = F.pad(x, (0, 0, 1, 1, 0, 0), mode='circular')
        return x

    def forward(self, x: torch.Tensor, t_embd: torch.Tensor, e_embd: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self._pad(h)
        h = self.conv1(h)
        h = h + self.t_proj(t_embd)[:, :, None, None, None]
        h = h + self.e_proj(e_embd)[:, :, None, None, None]
        h = F.silu(self.norm2(h))
        h = self._pad(h)
        h = self.conv2(h)
        return h + self.skip(x)


class UNet3D(nn.Module):

    def __init__(self, t_dim: int = 128, e_dim: int = 128):
        super().__init__()
        self.t_embd = SinusoidalEmbedding(t_dim)
        self.e_embd = EnergyEmbedding(e_dim)

        self.d1   = ResBlock3D(1,   32,  t_dim, e_dim)
        self.d2   = ResBlock3D(32,  64,  t_dim, e_dim)
        self.d3   = ResBlock3D(64,  128, t_dim, e_dim)
        self.pool = nn.MaxPool3d((1, 2, 2))

        self.mid  = ResBlock3D(128, 128, t_dim, e_dim)

        self.u3   = ResBlock3D(128 + 128, 64,  t_dim, e_dim)
        self.u2   = ResBlock3D(64  + 64,  32,  t_dim, e_dim)
        self.u1   = ResBlock3D(32  + 32,  32,  t_dim, e_dim)

        self.out  = nn.Conv3d(32, 1, 1)

    @staticmethod
    def _up(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.shape[2:], mode='nearest')

    def forward(self, x: torch.Tensor, t: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        t_e = self.t_embd(t)
        e_e = self.e_embd(energy)

        x1 = self.d1(x,              t_e, e_e)
        x2 = self.d2(self.pool(x1),  t_e, e_e)
        x3 = self.d3(self.pool(x2),  t_e, e_e)
        h  = self.mid(self.pool(x3), t_e, e_e)

        h = self.u3(torch.cat([self._up(h, x3), x3], 1), t_e, e_e)
        h = self.u2(torch.cat([self._up(h, x2), x2], 1), t_e, e_e)
        h = self.u1(torch.cat([self._up(h, x1), x1], 1), t_e, e_e)

        return self.out(h)