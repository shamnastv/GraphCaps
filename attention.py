import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=1, num_layers=2, activation=F.leaky_relu):
        super(Attention, self).__init__()

        self.num_layers = num_layers
        self.activation = activation
        self.linears = torch.nn.ModuleList()
        # self.batch_norms = torch.nn.ModuleList()

        if hidden_dim is None:
            hidden_dim = input_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linears.append(nn.Linear(input_dim, output_dim, bias=False))
        else:
            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))

            # for layer in range(num_layers - 1):
            #     self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        h = x
        for layer in range(self.num_layers - 1):
            h = self.linears[layer](h)
            # h = self.batch_norms[layer](h)
            h = self.activation(h)

        return self.linears[self.num_layers - 1](h)
