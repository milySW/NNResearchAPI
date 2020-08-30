from dataclasses import dataclass
from src.models.lightning import LitModel
import torch.nn as nn
import torch
from configs.models.model_template import DefaultModel


@dataclass
class DefaultResnet(DefaultModel):
    name: str = "xresnet18"
    expansion = 1
    layers = []
    model = LitModel()
    depth = 4
    in_channels = 1
    out_channels = 4
    bias = None
    activation: torch.nn.Module = nn.ReLU(inplace=True)
