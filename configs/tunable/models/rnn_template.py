# IN PROGRESS
# TODO: Replace it with custom config after
# implementing rnn from scratch

import torch
import torch.nn as nn

from configs.tunable.models.model_template import DefaultModel


class DefaultRNN(DefaultModel):
    """
    DOCS
    """

    name: str = "rnn"  # final
    data_dim: str = "1D"  # final

    activation: torch.nn.Module = nn.ELU()

    input_dim: int = 1024
    hidden_dim: int = 512
    layer_dim: int = 5
    output_dim: int = 2
    dropout: float = 0.2

    # Pretrained weights
    pretrained: bool = False
    unfreezing_epoch: int = 0
    freezing_start: int = 0
    freezing_stop: int = 0
