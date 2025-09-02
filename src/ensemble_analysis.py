### Min
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from torch_geometric.data import DataLoader
import os
import time
from datetime import datetime
import argparse
import sys
import warnings

warnings.filterwarnings('ignore')

### Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

### Import models and dataset with fallback
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

### Import advanced ensemble methods
try:
    from advanced_ensemble import AdvancedEnsembleMethods
    ADVANCED_ENSEMBLE_AVAILABLE = True
except ImportError:
    ADVANCED_ENSEMBLE_AVAILABLE = False
    print("Warning: Advanced ensemble methods not available")


### Advanced ensemble methods implementation with meta-learning
class AdvancedEnsembleMethods:
    def __init__(self):
        self.meta_learners = {
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.01),
            'linear': LinearRegression(),
            'rf': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
            'gbm': GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=6),
            'et': ExtraTreesRegressor(n_estimators=200, random_state=42, max_depth=10),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }

        self.ensemble_weights = {}
        self.validation_performance = {}
        self.uncertainty_estimates = {}

    def weighted_ensemble(self, predictions, weights=None, validation_performance=None):
        ### Weighted average ensemble with validation-based weights
        if weights is None:
            if validation_performance:
                ### Use validation performance to determine weights
                weights = []
                for name in predictions.keys():
                    if name in validation_performance:
                        loss = validation_performance[name]
                        weight = 1.0 / (loss + 1e-8)
                        weights.append(weight)
                    else:
                        weights.append(1.0)

                weights = np.array(weights)
                weights = weights / weights.sum()
            else:
                weights = [1.0 / len(predictions)] * len(predictions)

        if len(weights) != len(predictions):
            weights = [1.0 / len(predictions)] * len(predictions)

        weighted_pred = np.zeros_like(list(predictions.values())[0])
        for i, (name, pred) in enumerate(predictions.items()):
            weight = weights[i] if i < len(weights) else weights[-1]
            weighted_pred += weight * pred

        return weighted_pred

    def stacking_ensemble(self, predictions, true_values=None, masks=None, meta_model='ridge',
                          validation_split=0.2, cross_validation=True):
        ### Advanced stacking ensemble with cross-validation
        if len(predictions) < 2:
            print("Need at least 2 models for stacking")
            return np.mean(list(predictions.values()), axis=0)

        meta_features = np.column_stack(list(predictions.values()))

        if meta_model not in self.meta_learners:
            meta_model = 'ridge'

        meta_learner = self.meta_learners[meta_model]

        if true_values is not None and masks is not None:
            ### Prepare validation data
            valid_mask = masks.sum(axis=1) > 0
            if valid_mask.sum() > 0:
                X_val = meta_features[valid_mask]
                y_val = true_values[valid_mask]

                if cross_validation and len(X_val) > 10:
                    ### Use cross-validation for meta-learner training
                    cv_scores = cross_val_score(meta_learner, X_val, y_val, cv=min(5, len(X_val) // 2))
                    print(f"Meta-learner CV scores: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

                ### Train meta-learner
                meta_learner.fit(X_val, y_val)

                ### Store meta-learner for later use
                self.meta_learner = meta_learner

        ensemble_pred = meta_learner.predict(meta_features)
        return ensemble_pred

    def blending_ensemble(self, predictions, true_values=None, masks=None, validation_split=0.2):
        ### Advanced blending ensemble with validation split
        if len(predictions) < 2:
            return np.mean(list(predictions.values()), axis=0)

        if true_values is not None and masks is not None:
            ### Use validation data to optimize blending weights
            valid_mask = masks.sum(axis=1) > 0
            if valid_mask.sum() > 0:
                meta_features = np.column_stack(list(predictions.values()))
                X_val = meta_features[valid_mask]
                y_val = true_values[valid_mask]

                ### Optimize blending weights using validation data
                blend_learner = Ridge(alpha=0.1)
                blend_learner.fit(X_val, y_val)

                ### Apply optimized blending
                blend_pred = blend_learner.predict(meta_features)
                return blend_pred

        ### Fallback to simple blending
        blend_pred = np.mean(list(predictions.values()), axis=0)
        return blend_pred

    def dynamic_ensemble(self, predictions, validation_performance=None,
                         uncertainty_estimates=None, adaptive_weights=True):
        ### Dynamic ensemble with adaptive weight adjustment
        if validation_performance is None:
            return np.mean(list(predictions.values()), axis=0)

        ### Calculate dynamic weights
        weights = []
        for name in predictions.keys():
            if name in validation_performance:
                loss = validation_performance[name]
                weight = 1.0 / (loss + 1e-8)
                weights.append(weight)
            else:
                weights.append(1.0)

        weights = np.array(weights)

        if adaptive_weights and uncertainty_estimates:
            ### Adjust weights based on uncertainty
            for i, name in enumerate(predictions.keys()):
                if name in uncertainty_estimates:
                    uncertainty = uncertainty_estimates[name]
                    uncertainty_factor = 1.0 / (uncertainty + 1e-8)
                    weights[i] *= uncertainty_factor

        weights = weights / weights.sum()

        ### Apply dynamic weights
        dynamic_pred = np.zeros_like(list(predictions.values())[0])
        for i, (name, pred) in enumerate(predictions.items()):
            weight = weights[i] if i < len(weights) else weights[-1]
            dynamic_pred += weight * pred

        return dynamic_pred

    def uncertainty_ensemble(self, predictions, uncertainty_estimates=None,
                             confidence_threshold=0.8):
        ### Uncertainty-weighted ensemble with confidence thresholding
        if uncertainty_estimates is None:
            return np.mean(list(predictions.values()), axis=0)

        ### Calculate uncertainty-based weights
        weights = []
        valid_predictions = {}

        for name in predictions.keys():
            if name in uncertainty_estimates:
                uncertainty = uncertainty_estimates[name]
                confidence = 1.0 - uncertainty

                if confidence >= confidence_threshold:
                    weight = confidence
                    weights.append(weight)
                    valid_predictions[name] = predictions[name]
                else:
                    print(f"Excluding {name} due to low confidence: {confidence:.3f}")
            else:
                weights.append(1.0)
                valid_predictions[name] = predictions[name]

        if not valid_predictions:
            return np.mean(list(predictions.values()), axis=0)

        weights = np.array(weights)
        weights = weights / weights.sum()

        ### Apply uncertainty weights
        uncertainty_pred = np.zeros_like(list(valid_predictions.values())[0])
        for i, (name, pred) in enumerate(valid_predictions.items()):
            weight = weights[i] if i < len(weights) else weights[-1]
            uncertainty_pred += weight * pred

        return uncertainty_pred

    def adaptive_ensemble(self, predictions, method='auto', validation_performance=None,
                          uncertainty_estimates=None):
        ### Adaptive ensemble method selection with performance monitoring
        if method == 'auto':
            num_models = len(predictions)

            if num_models <= 2:
                method = 'weighted'
            elif num_models <= 5:
                method = 'stacking'
            elif validation_performance:
                method = 'dynamic'
            else:
                method = 'weighted'

        ### Select and apply ensemble method
        if method == 'weighted':
            return self.weighted_ensemble(predictions, validation_performance=validation_performance)
        elif method == 'stacking':
            return self.stacking_ensemble(predictions)
        elif method == 'blending':
            return self.blending_ensemble(predictions)
        elif method == 'dynamic':
            return self.dynamic_ensemble(predictions, validation_performance, uncertainty_estimates)
        elif method == 'uncertainty':
            return self.uncertainty_ensemble(predictions, uncertainty_estimates)
        else:
            return np.mean(list(predictions.values()), axis=0)

    def multi_level_ensemble(self, predictions, true_values=None, masks=None):
        ### Multi-level ensemble combining multiple strategies
        if len(predictions) < 3:
            return self.weighted_ensemble(predictions)

        ### Level 1: Individual model predictions
        level1_preds = predictions

        ### Level 2: Apply different ensemble methods
        level2_methods = ['weighted', 'stacking', 'blending']
        level2_preds = {}

        for method in level2_methods:
            if method == 'weighted':
                level2_preds[f'{method}_ensemble'] = self.weighted_ensemble(level1_preds)
            elif method == 'stacking':
                level2_preds[f'{method}_ensemble'] = self.stacking_ensemble(level1_preds, true_values, masks)
            elif method == 'blending':
                level2_preds[f'{method}_ensemble'] = self.blending_ensemble(level1_preds, true_values, masks)

        ### Level 3: Final ensemble of level 2 predictions
        final_pred = self.weighted_ensemble(level2_preds)

        return final_pred

    def get_ensemble_weights(self):
        ### Get current ensemble weights
        return self.ensemble_weights.copy()

    def set_ensemble_weights(self, weights):
        ### Set custom ensemble weights
        self.ensemble_weights = weights.copy()

    def update_validation_performance(self, performance_dict):
        ### Update validation performance for dynamic weighting
        self.validation_performance.update(performance_dict)

    def update_uncertainty_estimates(self, uncertainty_dict):
        ### Update uncertainty estimates for uncertainty weighting
        self.uncertainty_estimates.update(uncertainty_dict)


### Load trained models with error handling
def load_trained_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}

    ### Load dataset with stable path handling
    data_path = os.path.join('data', 'train_cleaned.csv')
    if not os.path.exists(data_path):
        possible_paths = [
            os.path.join('data', 'train_cleaned.csv'),
            os.path.join('..', 'data', 'train_cleaned.csv'),
            os.path.join(os.path.dirname(__file__), '..', 'data', 'train_cleaned.csv')
        ]

        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                print(f"Found data at: {data_path}")
                break
        else:
            raise FileNotFoundError(f"Could not find training data. Tried: {possible_paths}")

    dataset = PolymerDataset(csv_file=data_path)
    sample = dataset[0]
    num_features = sample.x.shape[1]

    try:
        gat_model = GAT(num_node_features=num_features, out_dim=5)
        best_gat_path = None
        best_gat_loss = float('inf')

        for fold in range(1, 4):
            fold_path = f'best_gat_model_fold_{fold}.pt'
            if os.path.exists(fold_path):
                val_loss_file = f'gat_val_loss_fold_{fold}.txt'
                if os.path.exists(val_loss_file):
                    val_losses = np.loadtxt(val_loss_file)
                    min_loss = np.min(val_losses)
                    if min_loss < best_gat_loss:
                        best_gat_loss = min_loss
                        best_gat_path = fold_path

        if not best_gat_path:
            for ablation in range(1, 3):
                ablation_path = f'best_gat_model_ablation_{ablation}.pt'
                if os.path.exists(ablation_path):
                    best_gat_path = ablation_path
                    break

        if not best_gat_path:
            base_path = 'best_gat_model.pt'
            if os.path.exists(base_path):
                best_gat_path = base_path

        if best_gat_path:
            gat_model.load_state_dict(torch.load(best_gat_path, map_location=device))
            gat_model.to(device)
            gat_model.eval()
            models['GAT'] = gat_model
            print(f"GAT model loaded successfully from {best_gat_path}")
        else:
            print("No GAT model found, skipping...")
    except Exception as e:
        print(f"GAT model loading failed: {e}")

    try:
        gin_model = GIN(num_node_features=num_features, out_dim=5)
        best_gin_path = None
        best_gin_loss = float('inf')

        for fold in range(1, 4):
            fold_path = f'best_gin_model_fold_{fold}.pt'
            if os.path.exists(fold_path):
                val_loss_file = f'gin_val_loss_fold_{fold}.txt'
                if os.path.exists(val_loss_file):
                    val_losses = np.loadtxt(val_loss_file)
                    min_loss = np.min(val_losses)
                    if min_loss < best_gin_loss:
                        best_gin_loss = min_loss
                        best_gin_path = fold_path

        if best_gin_path:
            gin_model.load_state_dict(torch.load(best_gin_path, map_location=device))
            gin_model.to(device)
            gin_model.eval()
            models['GIN'] = gin_model
            print(f"GIN model loaded successfully from {best_gin_path}")
        else:
            print("No GIN model found, skipping...")
    except Exception as e:
        print(f"GIN model loading failed: {e}")

    try:
        sage_model = GraphSAGE(num_node_features=num_features, out_dim=5)
        best_sage_path = None
        best_sage_loss = float('inf')

        for fold in range(1, 4):
            fold_path = f'best_graphsage_model_fold_{fold}.pt'
            if os.path.exists(fold_path):
                val_loss_file = f'graphsage_val_loss_fold_{fold}.txt'
                if os.path.exists(val_loss_file):
                    val_losses = np.loadtxt(val_loss_file)
                    min_loss = np.min(val_losses)
                    if min_loss < best_sage_loss:
                        best_sage_loss = min_loss
                        best_sage_path = fold_path

        if best_sage_path:
            sage_model.load_state_dict(torch.load(best_sage_path, map_location=device))
            sage_model.to(device)
            sage_model.eval()
            models['GraphSAGE'] = sage_model
            print(f"GraphSAGE model loaded successfully from {best_sage_path}")
        else:
            print("No GraphSAGE model found, skipping...")
    except Exception as e:
        print(f"GraphSAGE model loading failed: {e}")

    return models, device


### Fast batch prediction with performance monitoring
def fast_batch_predict(models, data_loader, device, ensemble_method='weighted',
                       validation_performance=None, uncertainty_estimates=None):
    all_predictions = {name: [] for name in models.keys()}

    start_time = time.time()
    total_samples = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            batch_size = data.num_graphs
            total_samples += batch_size

            for model_name, model in models.items():
                pred = model(data.x, data.edge_index, data.batch)
                all_predictions[model_name].append(pred.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = total_samples / elapsed
                print(f"Processed {batch_idx + 1} batches, {total_samples} samples, {samples_per_sec:.2f} samples/sec")

    for name in all_predictions:
        all_predictions[name] = np.concatenate(all_predictions[name], axis=0)

    ### Apply advanced ensemble methods
    ensemble_methods = AdvancedEnsembleMethods()

    ### Update performance and uncertainty if provided
    if validation_performance:
        ensemble_methods.update_validation_performance(validation_performance)
    if uncertainty_estimates:
        ensemble_methods.update_uncertainty_estimates(uncertainty_estimates)

    if ensemble_method == 'weighted':
        final_pred = ensemble_methods.weighted_ensemble(all_predictions, validation_performance=validation_performance)
    elif ensemble_method == 'stacking':
        final_pred = ensemble_methods.stacking_ensemble(all_predictions)
    elif ensemble_method == 'blending':
        final_pred = ensemble_methods.blending_ensemble(all_predictions)
    elif ensemble_method == 'dynamic':
        final_pred = ensemble_methods.dynamic_ensemble(all_predictions, validation_performance, uncertainty_estimates)
    elif ensemble_method == 'uncertainty':
        final_pred = ensemble_methods.uncertainty_ensemble(all_predictions, uncertainty_estimates)
    elif ensemble_method == 'adaptive':
        final_pred = ensemble_methods.adaptive_ensemble(all_predictions, validation_performance=validation_performance,
                                                        uncertainty_estimates=uncertainty_estimates)
    elif ensemble_method == 'multi_level':
        final_pred = ensemble_methods.multi_level_ensemble(all_predictions)
    else:
        final_pred = np.mean(list(all_predictions.values()), axis=0)

    total_time = time.time() - start_time
    print(f"Total inference time: {total_time:.2f}s")
    print(f"Average time per sample: {total_time / len(final_pred):.4f}s")
    print(f"Throughput: {len(final_pred) / total_time:.2f} samples/sec")

    return final_pred, all_predictions, total_time


### Fast single SMILES prediction
def fast_single_predict(models, smiles, device, ensemble_method='weighted'):
    try:
        from dataset import smiles_to_graph
    except ImportError:
        try:
            from data_part.dataset import smiles_to_graph
        except ImportError:
            print("Error: Could not import smiles_to_graph function")
            return None, {}, 0

    graph = smiles_to_graph(smiles)
    graph = graph.to(device)

    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)

    predictions = {}

    start_time = time.time()

    with torch.no_grad():
        for model_name, model in models.items():
            pred = model(graph.x, graph.edge_index, graph.batch)
            predictions[model_name] = pred.cpu().numpy()

    ### Apply advanced ensemble methods
    ensemble_methods = AdvancedEnsembleMethods()

    if ensemble_method == 'weighted':
        final_pred = ensemble_methods.weighted_ensemble(predictions)
    elif ensemble_method == 'stacking':
        final_pred = ensemble_methods.stacking_ensemble(predictions)
    elif ensemble_method == 'blending':
        final_pred = ensemble_methods.blending_ensemble(predictions)
    elif ensemble_method == 'dynamic':
        final_pred = ensemble_methods.dynamic_ensemble(predictions)
    elif ensemble_method == 'uncertainty':
        final_pred = ensemble_methods.uncertainty_ensemble(predictions)
    elif ensemble_method == 'adaptive':
        final_pred = ensemble_methods.adaptive_ensemble(predictions)
    elif ensemble_method == 'multi_level':
        final_pred = ensemble_methods.multi_level_ensemble(predictions)
    else:
        final_pred = np.mean(list(predictions.values()), axis=0)

    inference_time = time.time() - start_time
    print(f"Single SMILES inference time: {inference_time:.4f}s")

    return final_pred[0], predictions, inference_time


### Get model predictions
def get_model_predictions(models, data_loader, device):
    predictions = {name: [] for name in models.keys()}

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            for name, model in models.items():
                pred = model(data.x, data.edge_index, data.batch)
                predictions[name].append(pred.cpu().numpy())

    for name in predictions:
        predictions[name] = np.concatenate(predictions[name], axis=0)

    return predictions


### Advanced ensemble prediction
def advanced_ensemble_predict(predictions, ensemble_method='weighted', **kwargs):
    ensemble_methods = AdvancedEnsembleMethods()

    if ensemble_method == 'weighted':
        return ensemble_methods.weighted_ensemble(predictions, **kwargs)
    elif ensemble_method == 'stacking':
        return ensemble_methods.stacking_ensemble(predictions, **kwargs)
    elif ensemble_method == 'blending':
        return ensemble_methods.blending_ensemble(predictions, **kwargs)
    elif ensemble_method == 'dynamic':
        return ensemble_methods.dynamic_ensemble(predictions, **kwargs)
    elif ensemble_method == 'uncertainty':
        return ensemble_methods.uncertainty_ensemble(predictions, **kwargs)
    elif ensemble_method == 'adaptive':
        return ensemble_methods.adaptive_ensemble(predictions, **kwargs)
    elif ensemble_method == 'multi_level':
        return ensemble_methods.multi_level_ensemble(predictions, **kwargs)
    else:
        return np.mean(list(predictions.values()), axis=0)


### Evaluate ensemble strategies
def evaluate_ensemble_strategies(predictions, true_values, masks):
    num_models = len(predictions)
    if num_models == 2:
        weights = [0.5, 0.5]
    elif num_models == 3:
        weights = [0.4, 0.3, 0.3]
    else:
        weights = [1.0 / num_models] * num_models

    strategies = {
        'Simple_Average': lambda preds: np.mean(list(preds.values()), axis=0),
        'Weighted_Average': lambda preds: np.average(list(preds.values()), axis=0, weights=weights),
        'Median': lambda preds: np.median(list(preds.values()), axis=0),
        'Min': lambda preds: np.min(list(preds.values()), axis=0),
        'Max': lambda preds: np.max(list(preds.values()), axis=0)
    }

    results = {}

    for strategy_name, strategy_func in strategies.items():
        if len(predictions) > 0:
            ensemble_pred = strategy_func(predictions)

            loss = masked_mse_loss_numpy(ensemble_pred, true_values, masks)
            results[strategy_name] = loss

            print(f"{strategy_name}: {loss:.6f}")

    return results


### Advanced ensemble evaluation with all methods
def evaluate_advanced_ensemble_strategies(predictions, true_values, masks):
    ensemble_methods = AdvancedEnsembleMethods()
    results = {}

    ### Test all advanced ensemble methods
    methods = ['weighted', 'stacking', 'blending', 'dynamic', 'adaptive', 'multi_level']

    for method in methods:
        try:
            if method == 'stacking':
                ensemble_pred = ensemble_methods.stacking_ensemble(predictions, true_values, masks)
            elif method == 'blending':
                ensemble_pred = ensemble_methods.blending_ensemble(predictions, true_values, masks)
            elif method == 'dynamic':
                ### Create dummy validation performance for demonstration
                validation_performance = {name: 0.1 + 0.1 * i for i, name in enumerate(predictions.keys())}
                ensemble_pred = ensemble_methods.dynamic_ensemble(predictions, validation_performance)
            elif method == 'multi_level':
                ensemble_pred = ensemble_methods.multi_level_ensemble(predictions, true_values, masks)
            else:
                ensemble_pred = ensemble_methods.__getattribute__(f'{method}_ensemble')(predictions)

            loss = masked_mse_loss_numpy(ensemble_pred, true_values, masks)
            results[f'Advanced_{method.capitalize()}'] = loss
            print(f"Advanced {method.capitalize()}: {loss:.6f}")

        except Exception as e:
            print(f"Advanced {method} failed: {e}")
            results[f'Advanced_{method.capitalize()}'] = float('inf')

    return results


### Stacking ensemble
def stacking_ensemble(predictions, true_values, masks, meta_model='ridge'):
    if len(predictions) < 2:
        print("Need at least 2 models for stacking")
        return None

    meta_features = np.column_stack(list(predictions.values()))

    if meta_model == 'ridge':
        meta_learner = Ridge(alpha=1.0)
    elif meta_model == 'linear':
        meta_learner = LinearRegression()
    elif meta_model == 'rf':
        meta_learner = RandomForestRegressor(n_estimators=100, random_state=42)

    valid_mask = masks.sum(axis=1) > 0
    if valid_mask.sum() > 0:
        meta_learner.fit(meta_features[valid_mask], true_values[valid_mask])

        ensemble_pred = meta_learner.predict(meta_features)

        loss = masked_mse_loss_numpy(ensemble_pred, true_values, masks)

        print(f"Stacking ({meta_model}): {loss:.6f}")
        return loss, meta_learner

    return None, None


### Masked MSE loss for numpy arrays
def masked_mse_loss_numpy(pred, target, mask):
    loss = (pred - target) ** 2
    loss = loss * mask
    return loss.sum() / mask.sum()


### Save predictions to CSV
def save_predictions(predictions, output_file):
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

    results = []
    for i in range(len(predictions)):
        row = {}
        for j, col in enumerate(target_cols):
            row[col] = predictions[i, j]
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


### Create ensemble visualization
def create_ensemble_plots(results, predictions, true_values, masks):
    plt.figure(figsize=(12, 6))

    strategies = list(results.keys())
    losses = list(results.values())

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    bars = plt.bar(strategies, losses, color=colors[:len(strategies)])
    plt.xlabel('Ensemble Strategy')
    plt.ylabel('Validation Loss')
    plt.title('Ensemble Strategy Performance Comparison')
    plt.xticks(rotation=45)

    for bar, loss in zip(bars, losses):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{loss:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    if len(predictions) > 1:
        plt.figure(figsize=(10, 8))

        pred_arrays = list(predictions.values())
        model_names = list(predictions.keys())

        first_prop_preds = np.column_stack([pred[:, 0] for pred in pred_arrays])
        correlation_matrix = np.corrcoef(first_prop_preds.T)

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    xticklabels=model_names, yticklabels=model_names, fmt='.3f')
        plt.title('Model Prediction Correlation Matrix (First Property)')
        plt.tight_layout()
        plt.show()


### Generate ensemble report
def generate_ensemble_report(results, predictions, true_values, masks):
    report = []
    report.append("# Ensemble Analysis Report")
    report.append("=" * 50)
    report.append("")

    report.append("## Model Information")
    report.append(f"Number of models: {len(predictions)}")
    report.append(f"Models: {', '.join(predictions.keys())}")
    report.append("")

    report.append("## Ensemble Strategy Performance")
    report.append("")
    for strategy, loss in results.items():
        report.append(f"- **{strategy}**: {loss:.6f}")
    report.append("")

    best_strategy = min(results, key=results.get)
    best_loss = results[best_strategy]
    report.append(f"## Best Strategy")
    report.append(f"**{best_strategy}** achieved the best performance with validation loss: **{best_loss:.6f}**")
    report.append("")

    if len(predictions) > 1:
        report.append("## Individual Model Performance")
        report.append("")
        for model_name, pred in predictions.items():
            loss = masked_mse_loss_numpy(pred, true_values, masks)
            report.append(f"- **{model_name}**: {loss:.6f}")
        report.append("")

    report.append("## Recommendations")
    report.append("")
    report.append("1. **Use the best ensemble strategy**: " + best_strategy)
    report.append("2. **Consider model diversity**: Models with lower correlation may improve ensemble performance")
    report.append("3. **Weight optimization**: Fine-tune ensemble weights based on validation performance")
    report.append("4. **Meta-learning**: Consider stacking with different meta-learners")
    report.append("5. **Multi-level ensemble**: Try combining multiple ensemble strategies")
    report.append("")

    with open('ensemble_analysis_report.md', 'w') as f:
        f.write('\n'.join(report))

    return report


### Main function
def main():
    parser = argparse.ArgumentParser(description='Advanced Ensemble Analysis and Fast Inference')
    parser.add_argument('--mode', type=str, default='analysis',
                        choices=['analysis', 'prediction', 'fast_inference', 'single_test', 'both'],
                        help='Mode: analysis, prediction, fast_inference, single_test, or both')
    parser.add_argument('--test_csv', type=str, help='Test data CSV file for prediction')
    parser.add_argument('--ensemble_method', type=str, default='adaptive',
                        choices=['average', 'median', 'max', 'min', 'weighted', 'stacking', 'blending', 'dynamic',
                                 'uncertainty', 'adaptive', 'multi_level'],
                        help='Ensemble method for prediction')
    parser.add_argument('--output_file', type=str, default='ensemble_predictions.csv',
                        help='Output file for predictions')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for fast inference')
    parser.add_argument('--single_smiles', type=str, help='Single SMILES string for quick test')

    args = parser.parse_args()

    print("Starting advanced ensemble analysis and fast inference...")

    models, device = load_trained_models()

    if len(models) == 0:
        print("No models found. Please ensure trained models are available.")
        return

    print(f"Loaded {len(models)} models: {list(models.keys())}")

    if args.mode == 'single_test':
        if not args.single_smiles:
            print("Error: Single SMILES required for single_test mode")
            return

        print(f"\n=== Testing Single SMILES ===")
        print(f"SMILES: {args.single_smiles}")

        final_pred, individual_preds, inference_time = fast_single_predict(
            models, args.single_smiles, device, args.ensemble_method
        )

        print(f"\nPrediction Results:")
        print(f"Final prediction: {final_pred}")
        print(f"Individual model predictions:")
        for model_name, pred in individual_preds.items():
            print(f"  {model_name}: {pred[0]}")
        return

    if args.mode == 'fast_inference':
        if not args.test_csv:
            print("Error: Test CSV file required for fast_inference mode")
            return

        print(f"\n=== Running Fast Inference ===")

        print("Loading test dataset...")
        test_dataset = PolymerDataset(csv_file=args.test_csv)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        print(f"Dataset size: {len(test_dataset)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Number of batches: {len(test_loader)}")

        final_predictions, individual_predictions, total_time = fast_batch_predict(
            models, test_loader, device, args.ensemble_method
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"fast_inference_{timestamp}.csv"
        save_predictions(final_predictions, output_file)

        print(f"\nFast Inference Results:")
        print(f"Total samples: {len(final_predictions)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {len(final_predictions) / total_time:.2f} samples/sec")
        print(f"Output file: {output_file}")

    if args.mode in ['analysis', 'both']:
        print("\n=== Running Advanced Ensemble Analysis ===")

        print("Loading validation dataset...")
        dataset = PolymerDataset(csv_file=os.path.join('data', 'train_cleaned.csv'))

        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        print("Getting model predictions...")
        predictions = get_model_predictions(models, val_loader, device)

        true_values = []
        masks = []
        for data in val_loader:
            true_values.append(data.y.numpy())
            masks.append(data.mask.numpy())

        true_values = np.concatenate(true_values, axis=0)
        masks = np.concatenate(masks, axis=0)

        true_values = true_values.reshape(-1, 5)
        masks = masks.reshape(-1, 5)

        print(f"Prediction shapes: {[(name, pred.shape) for name, pred in predictions.items()]}")

        print("\nEvaluating basic ensemble strategies...")
        results = evaluate_ensemble_strategies(predictions, true_values, masks)

        print("\nEvaluating advanced ensemble strategies...")
        advanced_results = evaluate_advanced_ensemble_strategies(predictions, true_values, masks)

        print("\nTesting stacking ensemble...")
        stacking_results = {}
        for meta_model in ['ridge', 'linear', 'rf', 'gbm', 'et']:
            try:
                loss, meta_learner = stacking_ensemble(predictions, true_values, masks, meta_model)
                if loss is not None:
                    stacking_results[f'Stacking_{meta_model}'] = loss
            except Exception as e:
                print(f"Stacking with {meta_model} failed: {e}")

        all_results = {**results, **advanced_results, **stacking_results}

        print("\nCreating visualizations...")
        create_ensemble_plots(all_results, predictions, true_values, masks)

        print("\nGenerating ensemble report...")
        report = generate_ensemble_report(all_results, predictions, true_values, masks)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df = pd.DataFrame(list(all_results.items()), columns=['Strategy', 'Validation_Loss'])
        results_df.to_csv(f'ensemble_strategy_results_{timestamp}.csv', index=False)

        print(f"\nAnalysis Results:")
        print(f"Best strategy: {min(all_results, key=all_results.get)}")
        print(f"Best loss: {min(all_results.values()):.6f}")

    if args.mode in ['prediction', 'both']:
        print("\n=== Running Ensemble Prediction ===")

        if not args.test_csv:
            print("Error: Test CSV file required for prediction mode")
            return

        print("Loading test dataset...")
        test_dataset = PolymerDataset(csv_file=args.test_csv)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        print("Getting model predictions...")
        predictions = get_model_predictions(models, test_loader, device)

        print(f"Applying {args.ensemble_method} ensemble...")
        final_predictions = advanced_ensemble_predict(predictions, args.ensemble_method)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{args.output_file.replace('.csv', '')}_{timestamp}.csv"
        save_predictions(final_predictions, output_file)

        print(f"\nPrediction Results:")
        print(f"Number of samples: {len(final_predictions)}")
        print(f"Output file: {output_file}")

    print(f"\n{'=' * 50}")
    print("ADVANCED ENSEMBLE ANALYSIS AND FAST INFERENCE COMPLETE")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
### #%#