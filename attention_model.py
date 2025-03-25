import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedAttentionLayer(nn.Module):
    """
    Enhanced Attention Layer:
      - Uses a two-layer MLP and integrates the global information α in advance.
      - Amplifies raw score differences using sigmoid, then applies group-wise softmax to obtain a distinctive final distribution.
      - Adjusts the temperature to clearly distinguish attention scores between edges.
    """

    def __init__(self, input_dim, hidden_dim=128, num_heads=4, temperature=1.0):
        super(EnhancedAttentionLayer, self).__init__()
        self.num_heads = num_heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.head_dim = input_dim // num_heads

        # Deeper MLP architecture increases dimensionality first, then reduces it
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.ln = nn.LayerNorm(input_dim)

        self.fc_att = nn.Linear(input_dim, num_heads)
        self.temperature = temperature

    def forward(self, x, row=None, alpha=None):
        # Integrate global α information upfront to enhance its influence on attention calculation
        if alpha is not None:
            alpha_expanded = alpha.expand(x.size(0), 1)  # [num_edges, 1]
            x_alpha = torch.cat([x, alpha_expanded], dim=1)  # [num_edges, input_dim+1]
        else:
            x_alpha = torch.cat([x, torch.zeros(x.size(0), 1, device=x.device)], dim=1)

        h = F.relu(self.fc1(x_alpha))
        h = F.relu(self.fc2(h))
        h = self.ln(h)

        raw_att = self.fc_att(h)  # [num_edges, num_heads]

        # Amplify score differences using sigmoid
        att_sigmoid = torch.sigmoid(raw_att)

        if row is not None:
            unique, inv = torch.unique(row, return_inverse=True)
            norm_att = torch.zeros_like(att_sigmoid)  # [num_edges, num_heads]
            for i in range(unique.size(0)):
                mask = (inv == i)
                group_scores = att_sigmoid[mask]  # [group_size, num_heads]
                # Group-wise softmax to sharply distinguish attention scores
                norm_scores = F.softmax(group_scores / self.temperature, dim=0)
                norm_att[mask] = norm_scores
            att_final = norm_att.mean(dim=1, keepdim=True)  # [num_edges, 1]
        else:
            att_final = att_sigmoid.mean(dim=1, keepdim=True)

        return att_final
