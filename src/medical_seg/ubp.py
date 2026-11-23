"""Uncertainty-guided, structure-aware per-image prototype refinement (UBP)."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class StructureAwarePrototypeRefiner(nn.Module):
    """Refine late decoder features using reliable class context and structure."""

    def __init__(self, channels: int = 96, classes: int = 4, projected: int = 16,
                 confidence_threshold: float = 0.7, max_entropy: float = 0.6,
                 min_support: float = 1.0, temperature: float = 0.1,
                 selective_gating: bool = True):
        super().__init__()
        self.classes = classes
        self.confidence_threshold = confidence_threshold
        self.max_entropy = max_entropy
        self.min_support = min_support
        self.temperature = temperature
        self.selective_gating = selective_gating
        self.embedding = nn.Conv2d(channels, projected, 1, bias=False)
        self.query = nn.Conv2d(projected, projected, 1, bias=False)
        self.key = nn.Linear(projected, projected, bias=False)
        self.value = nn.Linear(projected, projected, bias=False)
        self.context_projection = nn.Conv2d(projected, channels, 1, bias=False)
        self.structural_depthwise = nn.Conv2d(channels, channels, 3, padding=1,
                                              groups=channels, bias=False)
        self.structural_projection = nn.Conv2d(channels, projected, 1, bias=False)
        self.discrepancy = nn.Sequential(
            nn.Conv2d(projected * 2 + 1, projected, 3, padding=1),
            nn.GELU(), nn.Conv2d(projected, 1, 1),
        )
        self.gate = nn.Conv2d(projected + 2, 1, 1)
        self.correction = nn.Sequential(
            nn.Conv2d(channels * 3, 32, 1, bias=False), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.GELU(),
            nn.Conv2d(32, channels, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _probability_edge(probability: torch.Tensor) -> torch.Tensor:
        dx = (probability[..., 1:] - probability[..., :-1]).abs().mean(1, keepdim=True)
        dy = (probability[..., 1:, :] - probability[..., :-1, :]).abs().mean(1, keepdim=True)
        return F.pad(dx, (0, 1, 0, 0)) + F.pad(dy, (0, 0, 0, 1))

    def forward(self, features: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        if coarse_logits.shape != (batch, self.classes, height, width):
            raise ValueError("Coarse logits must match decoder feature geometry and class count")
        probability = coarse_logits.float().softmax(dim=1)
        entropy = -(probability * probability.clamp_min(1e-7).log()).sum(1, keepdim=True)
        entropy = entropy / math.log(self.classes)
        detached = probability.detach()
        confidence, predicted = detached.max(dim=1)
        reliable = ((confidence >= self.confidence_threshold) &
                    (entropy[:, 0].detach() <= self.max_entropy))
        predicted_one_hot = F.one_hot(predicted, self.classes).permute(0, 3, 1, 2).float()
        weights = predicted_one_hot * reliable[:, None].float()
        weights = weights * (confidence * (1.0 - entropy[:, 0].detach()))[:, None]
        weights = weights.flatten(2)
        support = weights.sum(-1)
        valid = support >= self.min_support

        embedded = self.embedding(features)
        flat_features = embedded.flatten(2).transpose(1, 2).float()
        prototypes = torch.bmm(weights, flat_features) / support.clamp_min(1e-6)[..., None]
        keys = F.normalize(self.key(prototypes), dim=-1, eps=1e-6)
        values = self.value(prototypes)
        queries = F.normalize(self.query(embedded).flatten(2).transpose(1, 2).float(),
                              dim=-1, eps=1e-6)
        similarity = torch.bmm(queries, keys.transpose(1, 2)) / self.temperature
        attention = torch.softmax(similarity.masked_fill(~valid[:, None, :], -1e4), dim=-1)
        attention = attention * valid[:, None, :].to(similarity.dtype)
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-6)
        context = torch.bmm(attention, values).transpose(1, 2).reshape(batch, -1, height, width)
        context = self.context_projection(context.to(features.dtype))

        edge = self._probability_edge(probability).to(features.dtype)
        structural = self.structural_projection(self.structural_depthwise(features))
        discrepancy = torch.sigmoid(self.discrepancy(torch.cat((structural, edge, embedded), dim=1)))
        gate = torch.sigmoid(self.gate(torch.cat((entropy.to(features.dtype), discrepancy, embedded), dim=1)))
        if not self.selective_gating:
            gate = torch.ones_like(gate)
        delta = self.correction(torch.cat((features, context, features - context), dim=1))
        has_prototype = valid.any(1).view(batch, 1, 1, 1).to(features.dtype)
        return features + self.alpha * gate * delta * has_prototype


class UBPModel(nn.Module):
    """Attach UBP at the decoder's final normalization layer."""

    def __init__(self, backbone: nn.Module, classes: int, enabled: bool = True,
                 auxiliary_weight: float = 0.3, **refiner_options):
        super().__init__()
        self.backbone = backbone
        self.classes = classes
        self.auxiliary_weight = auxiliary_weight
        self.refiner = StructureAwarePrototypeRefiner(classes=classes, **refiner_options)
        self._auxiliary_logits = None
        self._hook = None
        transformer = self.backbone.base_model.CS_Unet
        self.coarse_head = nn.Conv2d(transformer.embed_dim, classes, 1)
        self.feature_height, self.feature_width = transformer.patches_resolution
        if enabled:
            self._hook = transformer.norm_up.register_forward_hook(self._refine_decoder_feature)

    def _refine_decoder_feature(self, module, inputs, tokens):
        if tokens.ndim != 3 or tokens.shape[1] != self.feature_height * self.feature_width:
            raise ValueError(f"Unexpected late decoder token shape: {tuple(tokens.shape)}")
        batch, _, channels = tokens.shape
        features = tokens.transpose(1, 2).reshape(
            batch, channels, self.feature_height, self.feature_width)
        coarse = self.coarse_head(features)
        self._auxiliary_logits = coarse
        refined = self.refiner(features, coarse)
        return refined.flatten(2).transpose(1, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self._auxiliary_logits = None
        return self.backbone(image)

    def auxiliary_logits(self):
        return self._auxiliary_logits


def build_ubp(backbone: nn.Module, classes: int, auxiliary_weight: float = 0.3,
              enabled: bool = True, **refiner_options):
    return UBPModel(backbone, classes, enabled=enabled,
                    auxiliary_weight=auxiliary_weight, **refiner_options)
