import torch
import math


class VPSchedule:
    def __init__(self, s=0.008, clamp_min=1e-4):
        self.s = s
        self.clamp_min = clamp_min

    def __call__(self, t):
        t_s = (t + self.s) / (1.0 + self.s)
        angle = t_s * (math.pi / 2)

        signal_rate = torch.cos(angle)
        noise_rate  = torch.sin(angle)

        signal_rate = torch.clamp(signal_rate, min=self.clamp_min)
        noise_rate = torch.sqrt(1.0 - signal_rate**2)

        return signal_rate, noise_rate

    def beta(self, t):
        """VP-SDE drift coefficient: β(t) = -d/dt log(ᾱ(t))

        ᾱ(t) = sr(t)²,  analytical derivation: β(t) = π/(1+s) * nr(t) / sr(t)
        """
        sr, nr = self(t)
        beta_t = (math.pi / (1.0 + self.s)) * nr / (sr + self.clamp_min)
        return beta_t

    def g(self, t):
        """VP-SDE diffusion coefficient: g(t) = √β(t)"""
        return torch.sqrt(self.beta(t))