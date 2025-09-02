### Min
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=64, out_dim=5):
        super().__init__()
        ### GraphSAGE convolution layers
        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        ### Linear layers for graph features and output
        self.lin_graph = torch.nn.Linear(hidden_dim, 64)
        self.lin_out = torch.nn.Linear(64, out_dim)

    def forward(self, x, edge_index, batch):
        ### GNN layers and pooling
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin_graph(x))
        out = self.lin_out(x)
        return out
### #%#