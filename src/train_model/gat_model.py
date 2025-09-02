### Min
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class GAT(torch.nn.Module):
    ### Multi-task GAT model for predicting 5 properties with attention visualization
    def __init__(self, num_node_features, hidden_channels=64, heads=4, out_dim=5):
        super(GAT, self).__init__()

        ### Enhanced GAT layers with residual connections
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.6)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6)

        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.norm2 = nn.LayerNorm(hidden_channels * heads)
        self.norm3 = nn.LayerNorm(hidden_channels)

        self.lin_graph = torch.nn.Linear(hidden_channels, 128)
        self.lin_graph2 = torch.nn.Linear(128, 64)

        self.lin_out = torch.nn.Linear(64, out_dim)

        ### Attention weights storage for visualization
        self.attention_weights = {
            'layer1': None,
            'layer2': None,
            'layer3': None
        }

        self.attention_masks = {
            'layer1': None,
            'layer2': None,
            'layer3': None
        }

        self.dropout = nn.Dropout(0.3)

        ### Enable attention weight collection
        self.collect_attention = True

    def forward(self, x, edge_index, batch):
        identity = x

        ### First GAT layer with attention collection
        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        if self.collect_attention:
            self.attention_weights['layer1'] = attn1[1].detach().cpu()
            self.attention_masks['layer1'] = edge_index.detach().cpu()
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)

        ### Second GAT layer with attention collection
        x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
        if self.collect_attention:
            self.attention_weights['layer2'] = attn2[1].detach().cpu()
            self.attention_masks['layer2'] = edge_index.detach().cpu()
        x = self.norm2(x)
        x = F.elu(x)
        x = self.dropout(x)

        ### Third GAT layer with attention collection
        x, attn3 = self.conv3(x, edge_index, return_attention_weights=True)
        if self.collect_attention:
            self.attention_weights['layer3'] = attn3[1].detach().cpu()
            self.attention_masks['layer3'] = edge_index.detach().cpu()
        x = self.norm3(x)
        x = F.elu(x)

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin_graph(x))
        x = self.dropout(x)
        x = F.relu(self.lin_graph2(x))
        x = self.dropout(x)

        out = self.lin_out(x)
        return out

    def get_attention_weights(self, layer=None):
        ### Get attention weights for specified layer or all layers
        if layer is None:
            return self.attention_weights
        elif layer in self.attention_weights:
            return self.attention_weights[layer]
        else:
            return None

    def get_attention_masks(self, layer=None):
        ### Get edge indices for attention weights
        if layer is None:
            return self.attention_masks
        elif layer in self.attention_masks:
            return self.attention_masks[layer]
        else:
            return None

    def clear_attention_weights(self):
        ### Clear stored attention weights to free memory
        self.attention_weights = {
            'layer1': None,
            'layer2': None,
            'layer3': None
        }
        self.attention_masks = {
            'layer1': None,
            'layer2': None,
            'layer3': None
        }

    def set_attention_collection(self, enabled=True):
        ### Enable or disable attention weight collection
        self.collect_attention = enabled

    def get_node_attention_scores(self, layer='layer3'):
        ### Calculate node-level attention scores by aggregating edge attention
        if layer not in self.attention_weights or self.attention_weights[layer] is None:
            return None

        attn_weights = self.attention_weights[layer]
        edge_index = self.attention_masks[layer]

        if attn_weights is None or edge_index is None:
            return None

        ### Aggregate attention scores for each node
        num_nodes = edge_index.max().item() + 1
        node_attention = torch.zeros(num_nodes, device=attn_weights.device)
        node_count = torch.zeros(num_nodes, device=attn_weights.device)

        ### Sum attention weights for incoming edges to each node
        for i, (src, dst) in enumerate(edge_index.t()):
            node_attention[dst] += attn_weights[i]
            node_count[dst] += 1

        ### Average attention scores
        node_count = torch.clamp(node_count, min=1)
        node_attention = node_attention / node_count

        return node_attention

    def visualize_attention_heatmap(self, layer='layer3', save_path=None, figsize=(12, 8)):
        ### Generate attention heatmap visualization
        attn_weights = self.get_attention_weights(layer)
        edge_index = self.get_attention_masks(layer)

        if attn_weights is None or edge_index is None:
            print(f"No attention weights available for layer {layer}")
            return None

        ### Create attention matrix
        num_nodes = edge_index.max().item() + 1
        attention_matrix = torch.zeros(num_nodes, num_nodes)

        for i, (src, dst) in enumerate(edge_index.t()):
            attention_matrix[src, dst] = attn_weights[i]

        ### Convert to numpy for plotting
        attention_matrix = attention_matrix.numpy()

        ### Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(attention_matrix, cmap='viridis', annot=False, cbar=True)
        plt.title(f'GAT Attention Heatmap - {layer}')
        plt.xlabel('Target Node')
        plt.ylabel('Source Node')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention heatmap saved to: {save_path}")

        plt.show()
        return attention_matrix

    def visualize_node_importance(self, layer='layer3', save_path=None, figsize=(10, 6)):
        ### Visualize node importance based on attention scores
        node_attention = self.get_node_attention_scores(layer)

        if node_attention is None:
            print(f"No node attention scores available for layer {layer}")
            return None

        ### Convert to numpy
        node_attention = node_attention.numpy()

        ### Create bar plot
        plt.figure(figsize=figsize)
        plt.bar(range(len(node_attention)), node_attention, alpha=0.7, color='skyblue')
        plt.title(f'Node Importance Scores - {layer}')
        plt.xlabel('Node Index')
        plt.ylabel('Attention Score')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Node importance plot saved to: {save_path}")

        plt.show()
        return node_attention

    def get_attention_statistics(self, layer='layer3'):
        ### Get statistical information about attention weights
        attn_weights = self.get_attention_weights(layer)

        if attn_weights is None:
            return None

        attn_np = attn_weights.numpy()

        stats = {
            'mean': float(np.mean(attn_np)),
            'std': float(np.std(attn_np)),
            'min': float(np.min(attn_np)),
            'max': float(np.max(attn_np)),
            'median': float(np.median(attn_np)),
            'num_edges': int(len(attn_np)),
            'sparsity': float(np.sum(attn_np == 0) / len(attn_np))
        }

        return stats

    def save_attention_data(self, save_dir='attention_data'):
        ### Save attention weights and masks to files
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for layer in self.attention_weights:
            if self.attention_weights[layer] is not None:
                ### Save attention weights
                attn_path = os.path.join(save_dir, f'{layer}_attention.pt')
                torch.save(self.attention_weights[layer], attn_path)

                ### Save edge indices
                mask_path = os.path.join(save_dir, f'{layer}_edges.pt')
                torch.save(self.attention_masks[layer], mask_path)

        print(f"Attention data saved to directory: {save_dir}")

    def load_attention_data(self, load_dir='attention_data'):
        ### Load attention weights and masks from files
        for layer in self.attention_weights:
            attn_path = os.path.join(load_dir, f'{layer}_attention.pt')
            mask_path = os.path.join(load_dir, f'{layer}_edges.pt')

            if os.path.exists(attn_path) and os.path.exists(mask_path):
                self.attention_weights[layer] = torch.load(attn_path)
                self.attention_masks[layer] = torch.load(mask_path)

        print(f"Attention data loaded from directory: {load_dir}")

### #%#