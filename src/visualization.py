### Min
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import warnings
import sys
import os

warnings.filterwarnings('ignore')

### Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from data_part.dataset import PolymerDataset
    from model.gat_model import GAT
    from model.gin_model import GIN
    from model.graphsage_model import GraphSAGE
except ImportError:
    try:
        from dataset import PolymerDataset
        from gat_model import GAT
        from gin_model import GIN
        from graphsage_model import GraphSAGE
    except ImportError:
        print("Error: Could not import required modules. Please check file paths.")
        sys.exit(1)

from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


### Attention visualization for GAT
class AttentionVisualizer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.attention_weights = []

    def capture_attention(self, x, edge_index, batch):
        ### Hook to capture attention weights
        def hook_fn(module, input, output):
            if hasattr(module, 'att_src'):
                self.attention_weights.append(module.att_src.detach().cpu())

        hooks = []
        for name, module in self.model.named_modules():
            if 'conv' in name and hasattr(module, 'att_src'):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)

        with torch.no_grad():
            output = self.model(x, edge_index, batch)

        for hook in hooks:
            hook.remove()

        return output, self.attention_weights

    def visualize_attention(self, data, save_path='attention_vis.png'):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

        output, attention_weights = self.capture_attention(x, edge_index, batch)

        G = to_networkx(data, to_undirected=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('GAT Attention Visualization', fontsize=16, fontweight='bold')

        ### Graph structure
        ax1 = axes[0, 0]
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue',
                node_size=500, font_size=8, font_weight='bold')
        ax1.set_title('Graph Structure', fontweight='bold')

        ### Attention heatmap
        ax2 = axes[0, 1]
        if attention_weights:
            attention_matrix = attention_weights[0].numpy()
            sns.heatmap(attention_matrix, ax=ax2, cmap='viridis', cbar=True)
            ax2.set_title('Attention Weights Heatmap', fontweight='bold')
            ax2.set_xlabel('Attention Heads')
            ax2.set_ylabel('Nodes')

        ### Node importance
        ax3 = axes[1, 0]
        if attention_weights:
            node_importance = attention_weights[0].mean(dim=1).numpy()
            nodes = list(G.nodes())
            colors = plt.cm.viridis(node_importance / node_importance.max())
            nx.draw(G, pos, ax=ax3, with_labels=True, node_color=colors,
                    node_size=500, font_size=8, font_weight='bold')
            ax3.set_title('Node Importance (Attention)', fontweight='bold')

        ### Attention distribution
        ax4 = axes[1, 1]
        if attention_weights:
            attention_flat = attention_weights[0].flatten().numpy()
            ax4.hist(attention_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.set_title('Attention Distribution', fontweight='bold')
            ax4.set_xlabel('Attention Weight')
            ax4.set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

        return attention_weights


### Training visualization
class TrainingVisualizer:
    def __init__(self, save_dir='visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def plot_training_curves(self, train_losses, val_losses, model_name, fold=1):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title(f'{model_name} Training Curves (Fold {fold})', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ### Overfitting analysis
        loss_diff = np.array(val_losses) - np.array(train_losses)
        ax2.plot(epochs, loss_diff, 'g-', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title(f'Overfitting Analysis (Fold {fold})', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation - Training Loss')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_model_comparison(self, results_dict):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

        models = list(results_dict.keys())
        avg_losses = []
        std_losses = []
        training_times = []

        for model in models:
            results = results_dict[model]
            avg_loss = np.mean([r['best_val_loss'] for r in results])
            std_loss = np.std([r['best_val_loss'] for r in results])
            total_time = sum([r['training_time'] for r in results]) / 3600

            avg_losses.append(avg_loss)
            std_losses.append(std_loss)
            training_times.append(total_time)

        ### Average validation loss
        ax1 = axes[0, 0]
        bars1 = ax1.bar(models, avg_losses, yerr=std_losses, capsize=5,
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        ax1.set_title('Average Validation Loss', fontweight='bold')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)

        for bar, loss in zip(bars1, avg_losses):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')

        ### Training time
        ax2 = axes[0, 1]
        bars2 = ax2.bar(models, training_times,
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        ax2.set_title('Total Training Time', fontweight='bold')
        ax2.set_ylabel('Hours')
        ax2.grid(True, alpha=0.3)

        for bar, time in zip(bars2, training_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{time:.2f}h', ha='center', va='bottom', fontweight='bold')

        ### Loss distribution
        ax3 = axes[1, 0]
        all_losses = []
        labels = []
        for model in models:
            results = results_dict[model]
            losses = [r['best_val_loss'] for r in results]
            all_losses.extend(losses)
            labels.extend([model] * len(losses))

        df = pd.DataFrame({'Model': labels, 'Loss': all_losses})
        sns.boxplot(data=df, x='Model', y='Loss', ax=ax3,
                    palette=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax3.set_title('Loss Distribution Across Folds', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        ### Performance vs Time
        ax4 = axes[1, 1]
        scatter = ax4.scatter(training_times, avg_losses, s=200,
                              c=range(len(models)), cmap='viridis', alpha=0.8)
        for i, model in enumerate(models):
            ax4.annotate(model, (training_times[i], avg_losses[i]),
                         xytext=(5, 5), textcoords='offset points',
                         fontweight='bold', fontsize=10)
        ax4.set_xlabel('Training Time (Hours)')
        ax4.set_ylabel('Average Validation Loss')
        ax4.set_title('Performance vs Training Time', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_ensemble_analysis(self, ensemble_results):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ensemble Analysis', fontsize=16, fontweight='bold')

        methods = list(ensemble_results.keys())
        scores = [ensemble_results[method]['score'] for method in methods]
        times = [ensemble_results[method]['time'] for method in methods]

        ### Ensemble scores
        ax1 = axes[0, 0]
        bars1 = ax1.bar(methods, scores,
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
        ax1.set_title('Ensemble Method Scores', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)

        for bar, score in zip(bars1, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                     f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

        ### Ensemble time
        ax2 = axes[0, 1]
        bars2 = ax2.bar(methods, times,
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
        ax2.set_title('Ensemble Computation Time', fontweight='bold')
        ax2.set_ylabel('Time (seconds)')
        ax2.grid(True, alpha=0.3)

        for bar, time in zip(bars2, times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{time:.2f}s', ha='center', va='bottom', fontweight='bold')

        ### Score vs Time
        ax3 = axes[1, 0]
        scatter = ax3.scatter(times, scores, s=200,
                              c=range(len(methods)), cmap='viridis', alpha=0.8)
        for i, method in enumerate(methods):
            ax3.annotate(method, (times[i], scores[i]),
                         xytext=(5, 5), textcoords='offset points',
                         fontweight='bold', fontsize=10)
        ax3.set_xlabel('Computation Time (seconds)')
        ax3.set_ylabel('Score')
        ax3.set_title('Score vs Computation Time', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        ### Improvement over best single model
        ax4 = axes[1, 1]
        best_single = min(scores)
        improvements = [(best_single - score) / best_single * 100 for score in scores]
        bars4 = ax4.bar(methods, improvements,
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
        ax4.set_title('Improvement over Best Single Model', fontweight='bold')
        ax4.set_ylabel('Improvement (%)')
        ax4.grid(True, alpha=0.3)

        for bar, imp in zip(bars4, improvements):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{imp:.2f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()


### Feature importance visualization
class FeatureVisualizer:
    def __init__(self, save_dir='visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def plot_feature_importance(self, feature_names, importance_scores, model_name):
        fig, ax = plt.subplots(figsize=(12, 8))

        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_scores = [importance_scores[i] for i in sorted_indices]

        bars = ax.barh(range(len(sorted_features)), sorted_scores,
                       color='skyblue', alpha=0.8, edgecolor='black')

        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{score:.4f}', ha='left', va='center', fontweight='bold')

        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{model_name} Feature Importance', fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, feature_matrix, feature_names, model_name):
        corr_matrix = np.corrcoef(feature_matrix.T)

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                    center=0, square=True, linewidths=0.5,
                    xticklabels=feature_names, yticklabels=feature_names, ax=ax)

        ax.set_title(f'{model_name} Feature Correlation Matrix', fontweight='bold')

        plt.tight_layout()
        plt.show()


### Main visualization function
def create_comprehensive_visualization(model_name='GAT', model_path=None, data_path='data/train.csv',
                                       save_dir='visualizations'):
    print(f"Creating visualizations for {model_name}...")

    train_viz = TrainingVisualizer(save_dir)
    feature_viz = FeatureVisualizer(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ### Create sample training data for demonstration
    sample_train_losses = [0.8, 0.6, 0.5, 0.4, 0.35, 0.32, 0.30, 0.29, 0.28, 0.27]
    sample_val_losses = [0.85, 0.65, 0.55, 0.45, 0.40, 0.38, 0.37, 0.36, 0.36, 0.37]

    ### Create sample model comparison data
    sample_results = {
        'GAT': [{'best_val_loss': 0.892, 'training_time': 3600}],
        'GIN': [{'best_val_loss': 0.638, 'training_time': 7200}],
        'GraphSAGE': [{'best_val_loss': 0.583, 'training_time': 5400}]
    }

    ### Create sample ensemble data
    sample_ensemble = {
        'Simple Average': {'score': 0.583, 'time': 1.2},
        'Weighted Average': {'score': 0.571, 'time': 2.1},
        'Stacking': {'score': 0.568, 'time': 5.3},
        'Blending': {'score': 0.565, 'time': 3.8}
    }

    ### Generate training curves
    print("Generating training curves...")
    train_viz.plot_training_curves(sample_train_losses, sample_val_losses, model_name)

    ### Generate model comparison
    print("Generating model comparison...")
    train_viz.plot_model_comparison(sample_results)

    ### Generate ensemble analysis
    print("Generating ensemble analysis...")
    train_viz.plot_ensemble_analysis(sample_ensemble)

    ### Create feature importance visualization
    print("Generating feature importance...")
    feature_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    importance_scores = [0.25, 0.20, 0.30, 0.15, 0.10]
    feature_viz.plot_feature_importance(feature_names, importance_scores, model_name)

    ### Create correlation matrix
    print("Generating correlation matrix...")
    sample_feature_matrix = np.random.randn(100, 5)
    feature_viz.plot_correlation_matrix(sample_feature_matrix, feature_names, model_name)

    print(f"All visualizations for {model_name} created successfully!")


if __name__ == "__main__":
    print("Starting comprehensive visualization generation...")
    create_comprehensive_visualization()
    print("Visualization generation completed!")
### #%#