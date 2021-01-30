import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PrimaryCapsuleLayer(torch.nn.Module):
    """
    Primary Convolutional Capsule Layer class based on:
    https://github.com/timomernick/pytorch-capsule.
    """

    def __init__(self, in_units, in_channels, num_units, capsule_dimensions):
        super(PrimaryCapsuleLayer, self).__init__()
        """
        :param in_units: Number of input units (GCN layers).
        :param in_channels: Number of channels.
        :param num_units: Number of capsules.
        :param capsule_dimensions: Number of neurons in capsule.
        """
        self.num_units = num_units
        self.units = []
        for i in range(self.num_units):
            unit = torch.nn.Conv1d(in_channels=in_channels,
                                   out_channels=capsule_dimensions,
                                   kernel_size=(in_units, 1),
                                   stride=1,
                                   bias=True)

            self.add_module("unit_" + str(i), unit)
            self.units.append(unit)

    @staticmethod
    def squash(s):
        """
        Squash activations.
        :param s: Signal.
        :return s: Activated signal.
        """
        mag_sq = torch.sum(s ** 2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        """
        Forward propagation pass.
        :param x: Input features.
        :return : Primary capsule features.
        """
        u = [self.units[i](x) for i in range(self.num_units)]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_units, -1)
        return PrimaryCapsuleLayer.squash(u)


class SecondaryCapsuleLayer(torch.nn.Module):
    """
    Secondary Convolutional Capsule Layer class based on this repostory:
    https://github.com/timomernick/pytorch-capsule
    """

    def __init__(self, in_channels, in_dim, out_channels, out_dim):
        super(SecondaryCapsuleLayer, self).__init__()
        """
        :param in_units: Number of input units (GCN layers).
        :param in_channels: Number of channels.
        :param num_units: Number of capsules.
        :param capsule_dimensions: Number of neurons in capsule.
        """
        self.in_dim = in_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_dim = out_dim
        # self.W = torch.nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))
        self.W = torch.nn.Parameter(torch.randn(1, 1, in_channels, out_channels, in_dim, out_dim), requires_grad=True)

    @staticmethod
    def squash(s):
        """
        Squash activations.
        :param s: Signal.
        :return s: Activated signal.
        """
        mag_sq = torch.sum(s ** 2, dim=3, keepdim=True)
        mag = torch.sqrt(mag_sq) + 1e-11
        a_j = mag_sq / (1.0 + mag_sq)
        s = a_j * (s / mag)
        return s, a_j

    def forward(self, x):
        """
        Forward propagation pass.
        :param x: Input features.
        :return : Capsule output.
        """
        input_shape = x.size()  # b x n x c x d
        batch_size = input_shape[0]
        n = input_shape[1]

        x = x.unsqueeze(3).repeat(1, 1, 1, self.out_channels, 1).unsqueeze(4)
        W = self.W.repeat(batch_size, n, 1, 1, 1, 1)    # b x n x ci x co x di x do
        # W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(x, W).squeeze(4)  # b x n x ci x co x d
        u_hat = u_hat.reshape(batch_size, n * self.in_channels, self.out_channels, self.out_dim)   # b x n*ci x co x d

        b_ij = torch.zeros(batch_size, n * self.in_channels, self.out_channels, 1)  # b x n*ci x co x 1

        num_iterations = 3
        v_j = None
        a_j = None
        for _ in range(num_iterations):
            c_ij = torch.nn.functional.softmax(b_ij, dim=1)
            # c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)   # b x 1 x co x d
            v_j, a_j = SecondaryCapsuleLayer.squash(s_j)  # b x 1 x co x d

            v_j1 = torch.cat([v_j] * n * self.in_channels, dim=1)  # b x n*ci x co x d
            u_vj1 = torch.sum(u_hat * v_j1, dim=3, keepdim=True)  # b x n*ci x co x 1
            b_ij = b_ij + u_vj1

        return v_j, a_j


def margin_loss(scores, target, loss_lambda):
    """
    The margin loss from the original paper. Based on:
    https://github.com/timomernick/pytorch-capsule
    :param scores: Capsule scores.
    :param target: Target groundtruth.
    :param loss_lambda: Regularization parameter.
    :return L_c: Classification loss.
    """
    scores = scores.squeeze()
    v_mag = torch.sqrt((scores ** 2).sum(dim=1, keepdim=True))
    zero = torch.zeros(1)
    m_plus = torch.tensor(0.9)
    m_minus = torch.tensor(0.1)
    max_l = torch.max(m_plus - v_mag, zero).view(1, -1) ** 2
    max_r = torch.max(v_mag - m_minus, zero).view(1, -1) ** 2
    T_c = target
    L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
    L_c = L_c.sum(dim=1)
    L_c = L_c.mean()
    return L_c


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()

        self.linear1 = nn.Linear(input_dim, output_dim)

    def forward(self, adj, x):
        h = self.linear1(x)
        h = torch.bmm(adj, h)
        return h
