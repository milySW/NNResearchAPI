# IN PROGRESS
# NOTE: It's not working with gpu
# TODO: Replace it with RNN implementation from scratch

import torch

from torch import nn
from torch.autograd import Variable

from src.base.model import LitModel


class RNNModel(LitModel):
    def __init__(self, config):
        super(RNNModel, self).__init__(config)

        # Number of hidden dimensions
        self.hidden_dim = config.model.hidden_dim

        # Number of hidden layers
        self.layer_dim = config.model.layer_dim

        # RNN
        self.rnn = nn.RNN(
            input_size=config.model.input_dim,
            hidden_size=config.model.hidden_dim,
            num_layers=config.model.layer_dim,
            dropout=config.model.dropout,
            batch_first=True,
            nonlinearity="relu",
        )

        # Readout layer
        self.fc = nn.Linear(config.model.hidden_dim, config.model.output_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # One time step
        self.rnn.flatten_parameters()
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
