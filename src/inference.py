### Min inference.py
### #%#

import torch
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import argparse
import sys
from pathlib import Path

### Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

### Import models and dataset with fallback
try:
    from data_part.dataset import PolymerDataset
    from model.gat_model import GAT
    from model.gin_model import GIN
    from model.graphsage_model import GraphSAGE
    from model.mpnn_model import MPNN
except ImportError:
    try:
        from dataset import PolymerDataset
        from gat_model import GAT
        from gin_model import GIN
        from graphsage_model import GraphSAGE
        from mpnn_model import MPNN
    except ImportError:
        print("Error: Could not import required modules. Please check file paths.")
        sys.exit(1)

from torch_geometric.data import DataLoader
import warnings

warnings.filterwarnings('ignore')

### Performance optimization settings
OPTIMIZATION_CONFIG = {
    'pin_memory': True,
    'num_workers': 4,
    'persistent_workers': True,
    'prefetch_factor': 2,
    'batch_size': 64,
    'cache_embeddings': True,
    'use_mixed_precision': True
}

### Embedding cache for avoiding recomputation
class EmbeddingCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            ### Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value

    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

### Optimized DataLoader with performance settings
def create_optimized_loader(dataset, batch_size=None, shuffle=False):
    if batch_size is None:
        batch_size = OPTIMIZATION_CONFIG['batch_size']

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=OPTIMIZATION_CONFIG['pin_memory'],
        num_workers=OPTIMIZATION_CONFIG['num_workers'],
        persistent_workers=OPTIMIZATION_CONFIG['persistent_workers'],
        prefetch_factor=OPTIMIZATION_CONFIG['prefetch_factor']
    )
    return loader

### Fast inference with caching and optimization
def fast_inference(model, data_loader, device, model_name, cache=None):
    model.eval()
    predictions = []
    total_time = 0
    batch_times = []

    ### Enable mixed precision if available
    if OPTIMIZATION_CONFIG['use_mixed_precision'] and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        use_amp = True
    else:
        use_amp = False

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            batch_start = time.time()

            data = data.to(device, non_blocking=True)

            ### Check cache for embeddings
            if cache is not None and OPTIMIZATION_CONFIG['cache_embeddings']:
                cache_key = f"{model_name}_{hash(str(data.x.shape) + str(data.edge_index.shape))}"
                cached_pred = cache.get(cache_key)

                if cached_pred is not None:
                    predictions.append(cached_pred)
                    continue

            ### Forward pass with optimization
            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(data.x, data.edge_index, data.batch)
            else:
                preds = model(data.x, data.edge_index, data.batch)

            preds_cpu = preds.cpu().numpy()
            predictions.append(preds_cpu)

            ### Cache embeddings if enabled
            if cache is not None and OPTIMIZATION_CONFIG['cache_embeddings']:
                cache.put(cache_key, preds_cpu)

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            total_time += batch_time

            ### Progress reporting
            if (batch_idx + 1) % 10 == 0:
                avg_batch_time = np.mean(batch_times[-10:])
                samples_per_sec = data.num_graphs / avg_batch_time
                print(f"{model_name}: Batch {batch_idx + 1}, Avg time: {avg_batch_time:.4f}s, "
                      f"Throughput: {samples_per_sec:.2f} samples/sec")

    predictions = np.concatenate(predictions, axis=0)

    performance_stats = {
        'total_time': total_time,
        'avg_batch_time': np.mean(batch_times),
        'throughput': len(predictions) / total_time,
        'cache_stats': cache.get_stats() if cache else None
    }

    return predictions, performance_stats

### Load trained models with error handling and architecture matching
def load_trained_models(device, num_node_features=12):
    models = {}

    ### Try to load GAT model with correct architecture
    try:
        ### Load GAT with exact training architecture (heads=4, hidden_channels=64)
        gat_model = GAT(num_node_features=num_node_features, hidden_channels=64, heads=4, out_dim=5)

        ### Try different checkpoint paths
        gat_paths = [
            'best_gat_model_fold_1.pt',
            'best_gat_model_fold_2.pt',
            'best_gat_model_fold_3.pt',
            'best_gat_model.pt'
        ]

        gat_loaded = False
        for path in gat_paths:
            if os.path.exists(path):
                try:
                    gat_model.load_state_dict(torch.load(path, map_location=device))
                    gat_model.to(device)
                    gat_model.eval()
                    models['GAT'] = gat_model
                    print(f"GAT loaded successfully from {path}")
                    gat_loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load GAT from {path}: {e}")
                    continue

        if not gat_loaded:
            print("Could not load GAT model with any checkpoint")

    except Exception as e:
        print(f"Error creating GAT model: {e}")

    ### Try to load GIN model with correct architecture
    try:
        ### Load GIN with exact training architecture
        gin_model = GIN(num_node_features=num_node_features, hidden_dim=64, out_dim=5)

        gin_paths = [
            'best_gin_model_fold_1.pt',
            'best_gin_model_fold_2.pt',
            'best_gin_model_fold_3.pt',
            'best_gin_model.pt'
        ]

        gin_loaded = False
        for path in gin_paths:
            if os.path.exists(path):
                try:
                    gin_model.load_state_dict(torch.load(path, map_location=device))
                    gin_model.to(device)
                    gin_model.eval()
                    models['GIN'] = gin_model
                    print(f"GIN loaded successfully from {path}")
                    gin_loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load GIN from {path}: {e}")
                    continue

        if not gin_loaded:
            print("Could not load GIN model with any checkpoint")

    except Exception as e:
        print(f"Error creating GIN model: {e}")

    ### Try to load GraphSAGE model with correct architecture
    try:
        ### Load GraphSAGE with exact training architecture
        sage_model = GraphSAGE(num_node_features=num_node_features, hidden_dim=64, out_dim=5)

        sage_paths = [
            'best_graphsage_model_fold_1.pt',
            'best_graphsage_model_fold_2.pt',
            'best_graphsage_model_fold_3.pt',
            'best_graphsage_model.pt'
        ]

        sage_loaded = False
        for path in sage_paths:
            if os.path.exists(path):
                try:
                    sage_model.load_state_dict(torch.load(path, map_location=device))
                    sage_model.to(device)
                    sage_model.eval()
                    models['GraphSAGE'] = sage_model
                    print(f"GraphSAGE loaded successfully from {path}")
                    sage_loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load GraphSAGE from {path}: {e}")
                    continue

        if not sage_loaded:
            print("Could not load GraphSAGE model with any checkpoint")

    except Exception as e:
        print(f"Error creating GraphSAGE model: {e}")

    return models

### Ensemble predictions with weighted averaging
def ensemble_predictions(predictions_dict, weights=None):
    if weights is None:
        weights = {name: 1.0 for name in predictions_dict.keys()}

    ### Normalize weights
    total_weight = sum(weights.values())
    weights = {name: w / total_weight for name, w in weights.items()}

    ### Weighted ensemble
    ensemble_pred = np.zeros_like(list(predictions_dict.values())[0])
    for name, pred in predictions_dict.items():
        weight = weights.get(name, 1.0)
        ensemble_pred += weight * pred

    return ensemble_pred

### Main inference function with optimization
def main_inference(test_csv='data/test.csv', output_dir='predictions', ensemble_method='average'):
    print("Starting optimized inference...")

    ### Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ### Load dataset
    print("Loading dataset...")
    try:
        dataset = PolymerDataset(csv_file=test_csv)
        print(f"Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    ### Get feature dimensions
    sample = dataset[0]
    num_node_features = sample.x.shape[1]
    edge_dim = getattr(sample, 'edge_attr', torch.zeros(1, )).shape[0] if hasattr(sample, 'edge_attr') else 4

    print(f"Node features: {num_node_features}, Edge features: {edge_dim}")

    ### Create optimized loader
    loader = create_optimized_loader(dataset, batch_size=OPTIMIZATION_CONFIG['batch_size'])
    print(f"Created optimized DataLoader with {len(loader)} batches")

    ### Initialize embedding cache
    cache = EmbeddingCache(max_size=1000) if OPTIMIZATION_CONFIG['cache_embeddings'] else None

    ### Load models
    models = load_trained_models(device, num_node_features)
    if not models:
        print("No models loaded, exiting...")
        return

    print(f"Loaded {len(models)} models: {list(models.keys())}")

    ### Run inference for each model
    all_predictions = {}
    all_performance = {}

    for model_name, model in models.items():
        print(f"\nRunning inference for {model_name}...")

        predictions, performance = fast_inference(model, loader, device, model_name, cache)
        all_predictions[model_name] = predictions
        all_performance[model_name] = performance

        print(f"{model_name} inference completed:")
        print(f"  Total time: {performance['total_time']:.2f}s")
        print(f"  Throughput: {performance['throughput']:.2f} samples/sec")
        if performance['cache_stats']:
            print(f"  Cache hit rate: {performance['cache_stats']['hit_rate']:.2%}")

    ### Create ensemble predictions
    print("\nCreating ensemble predictions...")
    ensemble_pred = ensemble_predictions(all_predictions)

    ### Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ### Save individual model predictions
    for model_name, pred in all_predictions.items():
        output_file = os.path.join(output_dir, f'{model_name.lower()}_preds_{timestamp}.npy')
        np.save(output_file, pred)
        print(f"Saved {model_name} predictions to {output_file}")

    ### Save ensemble predictions as CSV
    ensemble_csv = os.path.join(output_dir, f'final_predictions_{timestamp}.csv')
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    ensemble_df = pd.DataFrame(ensemble_pred, columns=target_cols)
    ensemble_df.to_csv(ensemble_csv, index=False)
    print(f"Saved final predictions to {ensemble_csv}")

    ### Save performance report
    performance_file = os.path.join(output_dir, f'inference_performance_{timestamp}.csv')
    performance_df = pd.DataFrame(all_performance).T
    performance_df.to_csv(performance_file)
    print(f"Saved performance report to {performance_file}")

    ### Print summary
    print(f"\n{'=' * 50}")
    print("INFERENCE COMPLETE")
    print(f"{'=' * 50}")
    print(f"Total samples processed: {len(ensemble_pred)}")
    print(f"Ensemble predictions shape: {ensemble_pred.shape}")
    print(f"First 5 ensemble predictions:")
    print(ensemble_pred[:5])

    ### Cache statistics
    if cache:
        cache_stats = cache.get_stats()
        print(f"\nCache Statistics:")
        print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
        print(f"  Cache size: {cache_stats['cache_size']}")

    return ensemble_pred, all_predictions, all_performance

### Command line interface
def main():
    parser = argparse.ArgumentParser(description='Optimized Inference with Caching')
    parser.add_argument('--test_csv', type=str, default='data/test.csv', help='Test data CSV file')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--ensemble_method', type=str, default='average', help='Ensemble method')
    parser.add_argument('--cache_embeddings', action='store_true', help='Enable embedding caching')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    args = parser.parse_args()

    ### Update optimization config
    OPTIMIZATION_CONFIG['batch_size'] = args.batch_size
    OPTIMIZATION_CONFIG['cache_embeddings'] = args.cache_embeddings
    OPTIMIZATION_CONFIG['num_workers'] = args.num_workers

    ### Run inference
    main_inference(args.test_csv, args.output_dir, args.ensemble_method)

if __name__ == "__main__":
    main()
### #%#