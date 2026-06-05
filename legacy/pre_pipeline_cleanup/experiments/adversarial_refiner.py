"""Adversarial latent refiner experiment.

This experiment is kept out of the production workflow because previous results
showed it did not improve the assay_only SCANVI model. Use it only if you want
to revisit that idea.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return GradientReversalFn.apply(x, self.lambd)


class LatentAdversarialRefiner(nn.Module):
    def __init__(
        self,
        z_dim,
        n_nk,
        n_dataset,
        n_assay,
        hidden=64,
        lambda_dataset=0.05,
        lambda_assay=0.05,
    ):
        super().__init__()
        self.refiner = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, z_dim))
        self.nk_head = nn.Linear(z_dim, n_nk)
        self.grl_dataset = GradientReversal(lambda_dataset)
        self.dataset_head = nn.Linear(z_dim, n_dataset)
        self.grl_assay = GradientReversal(lambda_assay)
        self.assay_head = nn.Linear(z_dim, n_assay)

    def forward(self, z):
        z_refined = z + self.refiner(z)
        return (
            z_refined,
            self.nk_head(z_refined),
            self.dataset_head(self.grl_dataset(z_refined)),
            self.assay_head(self.grl_assay(z_refined)),
        )
