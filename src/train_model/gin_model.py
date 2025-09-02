### Min
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch import nn

class GIN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=64, out_dim=5):
        super(GIN, self).__init__()

        ### First GIN layer
        nn1 = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(nn1)

        ### Second GIN layer
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINConv(nn2)

        ### Output layers for graph features
        self.lin_graph = nn.Linear(hidden_dim, 64)
        self.lin_out = nn.Linear(64, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.relu(self.lin_graph(x))
        return self.lin_out(x)
### #%#