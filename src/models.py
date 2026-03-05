from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    efficientnet_b2,
    resnet18,
)


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
    return _GradReverse.apply(x, lambda_grl)


def build_backbone(
    name: Literal["resnet18", "efficientnet_b0", "efficientnet_b2"],
    pretrained: bool = True,
):
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = resnet18(weights=weights)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feat_dim

    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = efficientnet_b0(weights=weights)
        feat_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
        return net, feat_dim

    if name == "efficientnet_b2":
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        net = efficientnet_b2(weights=weights)
        feat_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
        return net, feat_dim

    raise ValueError(f"Unsupported backbone: {name}")


class BaselineClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_extractor, feat_dim = build_backbone(backbone, pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        return self.classifier(feats)


class DANN(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        class_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_extractor, feat_dim = build_backbone(backbone, pretrained=pretrained)
        self.class_classifier = nn.Sequential(
            nn.Dropout(class_dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        return self.class_classifier(feats)

    def forward(
        self,
        x: torch.Tensor,
        lambda_grl: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.feature_extractor(x)
        class_logits = self.class_classifier(feats)
        domain_logits = self.domain_classifier(grad_reverse(feats, lambda_grl))
        return class_logits, domain_logits
