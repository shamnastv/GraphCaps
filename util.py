import torch


def normalize_adj(adj):
    row_sum = torch.sum(adj, dim=-1)  # (batch , N, 1)
    r_inv = torch.pow(row_sum, -0.5)
    r_inv[torch.isinf(r_inv)] = 0.0
    r_mat_inv = torch.diag_embed(r_inv)
    norm_adj = torch.matmul(torch.matmul(r_mat_inv, adj), r_mat_inv)

    return norm_adj

