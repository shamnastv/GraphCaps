import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from attention import Attention
from layer import SecondaryCapsuleLayer, GCN
from util import normalize_adj

epsilon = 1e-11


def to_torch(adj, node_inputs, label, reconstructs, device):
    adj = torch.from_numpy(adj).float().to(device)
    tmp = []
    for a in node_inputs:
        tmp.append(torch.tensor(a).long().to(device))

    node_inputs = tmp
    label = torch.tensor(label).long().to(device)
    reconstructs = torch.tensor(reconstructs).float().to(device)
    return adj, node_inputs, label, reconstructs


class Model(nn.Module):
    def __init__(self, args, num_features, num_classes, recon_dim, device):
        super(Model, self).__init__()

        self.args = args
        self.device = device
        self.num_features = num_features
        self.num_classes = num_classes

        self.recon_dim = recon_dim
        self.embeddings = nn.ModuleList()
        for i, d in enumerate(num_features):
            self.embeddings.append(nn.Embedding(d, args.node_embedding_size))

        self.gcn_input_dim = args.node_embedding_size * len(num_features)
        self.num_gcn_channels = 2
        args.num_gcn_layers = 5
        self.graph_embd_str = [16]

        self.attention = Attention(args.node_embedding_size * self.num_gcn_channels * args.num_gcn_layers)
        self._init_gcn(args)
        self._init_capsules()
        self._init_reconstruction_layers()

    def _init_gcn(self, args):
        self.gcn_layers = nn.ModuleList()
        hidden_dim = args.node_embedding_size * self.num_gcn_channels
        # self.gcn_layers.append(GCNConv(self.gcn_input_dim, hidden_dim))
        # for _ in range(args.num_gcn_layers - 1):
        #     self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.gcn_layers.append(GCN(self.gcn_input_dim, hidden_dim))
        for _ in range(args.num_gcn_layers - 1):
            self.gcn_layers.append(GCN(hidden_dim, hidden_dim))

    def _init_capsules(self):
        # self.first_capsule = PrimaryCapsuleLayer(in_units=self.args.gcn_filters, in_channels=self.args.gcn_layers,
        #                                          num_units=self.args.gcn_layers,
        #                                          capsule_dimensions=self.args.capsule_dimensions)

        self.graph_capsule = SecondaryCapsuleLayer(self.num_gcn_channels * self.args.num_gcn_layers,
                                                   self.args.node_embedding_size, self.args.num_graph_capsules,
                                                   self.args.graph_embedding_size, self.device)

        self.class_capsule = SecondaryCapsuleLayer(self.args.num_graph_capsules, self.args.graph_embedding_size,
                                                   self.num_classes, 16, self.device)

    def _init_reconstruction_layers(self):
        self.reconstruction_layer_1 = nn.Linear(16,
                                                int((self.gcn_input_dim * 2) / 3))

        self.reconstruction_layer_2 = nn.Linear(int((self.gcn_input_dim * 2) / 3),
                                                int((self.gcn_input_dim * 3) / 2))

        self.reconstruction_layer_3 = nn.Linear(int((self.gcn_input_dim * 3) / 2),
                                                self.recon_dim)

    def forward(self, adj, node_inputs, label, reconstructs):
        args = self.args

        adj, node_inputs, label, reconstructs = to_torch(adj, node_inputs, label, reconstructs, self.device)
        features = []
        for i, att in enumerate(self.num_features):
            features.append(self.embeddings[i](node_inputs[i]))

        masks = torch.max(adj, dim=-1, keepdim=True)[0]
        number_of_nodes = torch.sum(masks, dim=1, keepdim=True).float().unsqueeze(3)

        b, n, _ = adj.shape
        c = self.num_gcn_channels
        # edge_index = (adj > 0).nonzero().t()
        adj_norm = normalize_adj(adj)
        features = torch.cat(features, dim=-1)  # b x n x c*d
        hidden_representations = []
        for layer in self.gcn_layers:
            features = layer(adj_norm, features)
            features = torch.tanh(features)
            features = torch.dropout(features)
            hidden_representations.append(features.reshape(b, n, c, -1))

        hidden_representations = torch.cat(hidden_representations, dim=2)  # b x n x c x d
        # hidden_representations = hidden_representations.view(1, self.args.gcn_layers, self.args.gcn_filters, -1)

        attn = self.attention(hidden_representations.reshape(b, n, -1))

        attn = F.softmax(attn.masked_fill(masks.eq(0), -np.inf), dim=1).unsqueeze(-1)
        num_nodes = torch.sum(masks, dim=1, keepdim=True).unsqueeze(-1)
        hidden_representations = hidden_representations * attn * num_nodes  # b x n x c x d

        # first_capsule_output = self.first_capsule(hidden_representations)
        # first_capsule_output = first_capsule_output.view(-1, self.args.gcn_layers * self.args.capsule_dimensions)
        #
        # rescaled_capsule_output = self.attention(first_capsule_output)
        # rescaled_first_capsule_output = rescaled_capsule_output.view(-1, self.args.gcn_layers,
        #                                                              self.args.capsule_dimensions)

        graph_capsule_output, a_j = self.graph_capsule(hidden_representations, number_of_nodes)

        class_capsule_output, a_j = self.class_capsule(graph_capsule_output, 1.0)
        class_capsule_output = class_capsule_output.squeeze()

        loss, margin_loss, reconstruction_loss, pred = self.calculate_loss(args, class_capsule_output, label,
                                                                           reconstructs)
        return class_capsule_output, loss, margin_loss, reconstruction_loss, label, pred

    def calculate_loss(self, args, capsule_input, target, reconstructs):
        """
        Calculating the reconstruction loss of the model.
        :param reconstructs:
        :param target:
        :param args:
        :param capsule_input: Output of class capsule.
        :param features: Feature matrix.
        :return reconstrcution_loss: Loss of reconstruction.
        """

        input_shape = capsule_input.shape
        batch_size = input_shape[0]
        num_class = input_shape[1]

        capsule_input = capsule_input.squeeze()
        v_mag = torch.sqrt((capsule_input ** 2).sum(dim=2))
        pred = v_mag.max(dim=1)[1]

        zero = torch.zeros(1, device=self.device)
        m_plus = torch.tensor(0.9, device=self.device)
        m_minus = torch.tensor(0.1, device=self.device)
        max_l = torch.max(m_plus - v_mag, zero) ** 2
        max_r = torch.max(v_mag - m_minus, zero) ** 2

        T_c = torch.zeros(batch_size, num_class, device=self.device)
        T_c[torch.arange(batch_size, device=self.device), target] = 1
        L_c = T_c * max_l + args.lambda_val * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)
        margin_loss = L_c.mean()

        # _, v_max_index = v_mag.max(dim=0)
        # v_max_index = v_max_index.data
        # capsule_masked = torch.zeros(capsule_input.size())
        # capsule_masked[v_max_index, :] = 1

        T_c = T_c.unsqueeze(2)
        capsule_masked = capsule_input * T_c
        capsule_masked = capsule_masked.sum(dim=1)

        reconstruction_output = F.relu(self.reconstruction_layer_1(capsule_masked))
        reconstruction_output = F.relu(self.reconstruction_layer_2(reconstruction_output))
        reconstruction_output = torch.sigmoid(self.reconstruction_layer_3(reconstruction_output))

        neg_indicator = torch.where(reconstructs < 1e-5, torch.ones(reconstructs.shape, device=self.device),
                                    torch.zeros(reconstructs.shape, device=self.device))
        pos_indicator = 1 - neg_indicator
        reconstructs_max = torch.max(reconstructs, dim=1, keepdim=True)[0]
        reconstruct_value = reconstructs / (reconstructs_max + epsilon)
        diff = (reconstruction_output - reconstruct_value) ** 2

        neg_loss = torch.max(diff * neg_indicator, dim=-1)[0]
        pos_loss = torch.max(diff * pos_indicator, dim=-1)[0]
        reconstruction_loss = torch.mean(pos_loss + neg_loss)

        loss = margin_loss + reconstruction_loss * args.reg_scale
        # reconstruction_loss = torch.sum((reconstruction_output - reconstruction_output) ** 2)
        return loss, margin_loss, reconstruction_loss, pred
