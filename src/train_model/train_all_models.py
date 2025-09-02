### Min
import torch
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import argparse
from pathlib import Path
import warnings
import sys

warnings.filterwarnings('ignore')

### Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

### Import models and dataset
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
from sklearn.model_selection import KFold
from torch.utils.data import Subset

### Import advanced modules
try:
    from advanced_losses import get_loss_function
    ADVANCED_LOSSES_AVAILABLE = True
except ImportError:
    ADVANCED_LOSSES_AVAILABLE = False
    print("Warning: Advanced losses not available")

try:
    from advanced_schedulers import get_scheduler
    ADVANCED_SCHEDULERS_AVAILABLE = True
except ImportError:
    ADVANCED_SCHEDULERS_AVAILABLE = False
    print("Warning: Advanced schedulers not available")

try:
    from advanced_regularization import RegularizationManager
    ADVANCED_REG_AVAILABLE = True
except ImportError:
    ADVANCED_REG_AVAILABLE = False
    print("Warning: Advanced regularization not available")

### Import hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available, using scikit-learn alternatives")

try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available")

### Set random seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

### Enhanced masked loss with wMAE support
def masked_loss(pred, target, mask, loss_type='wmae', reg_manager=None):
    if ADVANCED_LOSSES_AVAILABLE:
        criterion = get_loss_function(loss_type)
        loss = criterion(pred, target, mask)
    else:
        if loss_type == 'wmae':
            ### Implement weighted MAE loss
            loss = torch.abs(pred - target)
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            ### Default MSE loss
            loss = (pred - target) ** 2
            loss = loss * mask
            loss = loss.sum() / mask.sum()

    if ADVANCED_REG_AVAILABLE and reg_manager is not None:
        reg_loss = reg_manager.compute_regularization_loss()
        loss += reg_loss

    return loss

### Training function - Fixed parameter order
def train_epoch(model, train_loader, optimizer, device, loss_type='wmae', reg_manager=None):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)
        y = batch.y.view(out.shape[0], out.shape[1])
        mask = batch.mask.view(out.shape[0], out.shape[1])

        loss = masked_loss(out, y, mask, loss_type, reg_manager)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

### Validation function - Fixed parameter order
def validate_epoch(model, val_loader, device, loss_type='wmae', reg_manager=None):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index, batch.batch)
            y = batch.y.view(out.shape[0], out.shape[1])
            mask = batch.mask.view(out.shape[0], out.shape[1])

            loss = masked_loss(out, y, mask, loss_type, reg_manager)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches

### Early stopping class
class EarlyStopping:
    def __init__(self, patience=10, verbose=True, min_delta=1e-4):
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter} epochs")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")

### Hyperparameter search with Optuna
class OptunaHyperparameterSearch:
    def __init__(self, model_type='GAT', n_trials=50, timeout=1800):
        self.model_type = model_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None
        self.best_score = float('inf')

    def get_search_space(self, trial):
        ### Define hyperparameter search space
        if self.model_type == 'GAT':
            return {
                'hidden_channels': trial.suggest_int('hidden_channels', 32, 256),
                'heads': trial.suggest_int('heads', 2, 8),
                'dropout': trial.suggest_float('dropout', 0.1, 0.7),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
                'scheduler': trial.suggest_categorical('scheduler', ['plateau', 'cosine', 'onecycle']),
                'regularization': trial.suggest_categorical('regularization', ['dropout', 'label_smoothing', 'mixup'])
            }
        elif self.model_type == 'GIN':
            return {
                'hidden_channels': trial.suggest_int('hidden_channels', 32, 256),
                'num_layers': trial.suggest_int('num_layers', 3, 6),
                'dropout': trial.suggest_float('dropout', 0.1, 0.7),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256])
            }
        else:  ### GraphSAGE
            return {
                'hidden_channels': trial.suggest_int('hidden_channels', 32, 256),
                'num_layers': trial.suggest_int('num_layers', 2, 5),
                'dropout': trial.suggest_float('dropout', 0.1, 0.7),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256])
            }

    def create_model(self, params, num_features):
        ### Create model with given parameters
        if self.model_type == 'GAT':
            return GAT(
                num_node_features=num_features,
                hidden_channels=params['hidden_channels'],
                heads=params['heads'],
                out_dim=5
            )
        elif self.model_type == 'GIN':
            return GIN(
                num_node_features=num_features,
                hidden_channels=params['hidden_channels'],
                num_layers=params['num_layers'],
                out_dim=5
            )
        else:  ### GraphSAGE
            return GraphSAGE(
                num_node_features=num_features,
                hidden_channels=params['hidden_channels'],
                num_layers=params['num_layers'],
                out_dim=5
            )

    def objective(self, trial, dataset, device, k_folds=3, max_epochs=30):
        ### Optuna objective function
        params = self.get_search_space(trial)

        try:
            ### Create model
            sample = dataset[0]
            num_features = sample.x.shape[1]
            model = self.create_model(params, num_features).to(device)

            ### K-fold cross validation
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
                train_dataset = Subset(dataset, train_idx)
                val_dataset = Subset(dataset, val_idx)

                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

                ### Initialize optimizer
                if params['optimizer'] == 'adamw':
                    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'],
                                                  weight_decay=params['weight_decay'])
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                                 weight_decay=params['weight_decay'])

                ### Initialize scheduler
                if ADVANCED_SCHEDULERS_AVAILABLE:
                    scheduler = get_scheduler(params['scheduler'], optimizer, max_epochs, len(train_loader))
                else:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=5
                    )

                ### Initialize regularization
                if ADVANCED_REG_AVAILABLE:
                    reg_manager = RegularizationManager(model, reg_type=params['regularization'])
                else:
                    reg_manager = None

                ### Training loop
                best_val_loss = float('inf')
                early_stopping = EarlyStopping(patience=5, verbose=False)

                for epoch in range(max_epochs):
                    train_loss = train_epoch(model, train_loader, optimizer, device, 'wmae', reg_manager)
                    val_loss = validate_epoch(model, val_loader, device, 'wmae', reg_manager)

                    if scheduler is not None:
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(val_loss)
                        else:
                            scheduler.step()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                    early_stopping(val_loss)
                    if early_stopping.early_stop:
                        break

                fold_scores.append(best_val_loss)

            ### Return average validation loss
            avg_score = np.mean(fold_scores)

            ### Update best score
            if avg_score < self.best_score:
                self.best_score = avg_score
                self.best_params = params.copy()

            return avg_score

        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')

    def search(self, dataset, device, k_folds=3, max_epochs=30):
        ### Run hyperparameter search
        if not OPTUNA_AVAILABLE:
            print("Optuna not available, skipping hyperparameter search")
            return None, None

        print(f"Starting Optuna hyperparameter search for {self.model_type}...")
        print(f"Number of trials: {self.n_trials}")
        print(f"Timeout: {self.timeout} seconds")

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        objective_with_args = lambda trial: self.objective(trial, dataset, device, k_folds, max_epochs)

        study.optimize(
            objective_with_args,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_trial.value:.6f}")
        print(f"Best params: {study.best_trial.params}")

        return study.best_trial.params, study.best_trial.value

### Grid search alternative (when Optuna is not available)
class GridSearchHyperparameterSearch:
    def __init__(self, model_type='GAT'):
        self.model_type = model_type
        self.best_params = None
        self.best_score = float('inf')

    def get_parameter_grid(self):
        ### Define parameter grid for grid search
        if self.model_type == 'GAT':
            return {
                'hidden_channels': [64, 128, 256],
                'heads': [4, 8],
                'dropout': [0.3, 0.5, 0.7],
                'learning_rate': [1e-4, 5e-4, 1e-3],
                'batch_size': [64, 128],
                'optimizer': ['adam', 'adamw']
            }
        elif self.model_type == 'GIN':
            return {
                'hidden_channels': [64, 128, 256],
                'num_layers': [3, 4, 5],
                'dropout': [0.3, 0.5, 0.7],
                'learning_rate': [1e-4, 5e-4, 1e-3],
                'batch_size': [64, 128]
            }
        else:  ### GraphSAGE
            return {
                'hidden_channels': [64, 128, 256],
                'num_layers': [2, 3, 4],
                'dropout': [0.3, 0.5, 0.7],
                'learning_rate': [1e-4, 5e-4, 1e-3],
                'batch_size': [64, 128]
            }

    def search(self, dataset, device, k_folds=3, max_epochs=20):
        ### Run grid search
        if not SKLEARN_AVAILABLE:
            print("scikit-learn not available, skipping grid search")
            return None, None

        print(f"Starting Grid Search for {self.model_type}...")

        param_grid = self.get_parameter_grid()
        param_combinations = list(ParameterGrid(param_grid))

        print(f"Total parameter combinations: {len(param_combinations)}")

        best_score = float('inf')
        best_params = None

        for i, params in enumerate(param_combinations):
            print(f"Testing combination {i + 1}/{len(param_combinations)}: {params}")

            try:
                ### Create model
                sample = dataset[0]
                num_features = sample.x.shape[1]

                if self.model_type == 'GAT':
                    model = GAT(num_node_features=num_features, out_dim=5).to(device)
                elif self.model_type == 'GIN':
                    model = GIN(num_node_features=num_features, out_dim=5).to(device)
                else:
                    model = GraphSAGE(num_node_features=num_features, out_dim=5).to(device)

                ### Quick evaluation (reduced epochs for grid search)
                score = self.evaluate_params(model, params, dataset, device, k_folds, max_epochs)

                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"New best score: {best_score:.6f}")

            except Exception as e:
                print(f"Combination failed: {e}")
                continue

        print(f"Grid search completed. Best score: {best_score:.6f}")
        print(f"Best params: {best_params}")

        return best_params, best_score

    def evaluate_params(self, model, params, dataset, device, k_folds=3, max_epochs=20):
        ### Evaluate a single parameter combination
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

            ### Initialize optimizer
            if params['optimizer'] == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'],
                                              weight_decay=0.01)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                             weight_decay=0.01)

            ### Quick training
            best_val_loss = float('inf')
            for epoch in range(max_epochs):
                train_loss = train_epoch(model, train_loader, optimizer, device, 'wmae')
                val_loss = validate_epoch(model, val_loader, device, 'wmae')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                if epoch > 5 and val_loss > best_val_loss * 1.1:  ### Early stopping
                    break

            fold_scores.append(best_val_loss)

        return np.mean(fold_scores)

### Train single model with hyperparameter search
def train_single_model_with_search(model_name, model, dataset, device, config,
                                   enable_hp_search=True, search_type='optuna'):
    print(f"\n=== Training {model_name} ===")

    ### Hyperparameter search
    best_params = None
    if enable_hp_search:
        print("Running hyperparameter search...")

        if search_type == 'optuna' and OPTUNA_AVAILABLE:
            hp_searcher = OptunaHyperparameterSearch(
                model_type=model_name,
                n_trials=config.get('hp_search_trials', 30),
                timeout=config.get('hp_search_timeout', 900)
            )
            best_params, best_score = hp_searcher.search(
                dataset, device,
                k_folds=config.get('hp_search_k_folds', 2),
                max_epochs=config.get('hp_search_epochs', 20)
            )
        else:
            hp_searcher = GridSearchHyperparameterSearch(model_type=model_name)
            best_params, best_score = hp_searcher.search(
                dataset, device,
                k_folds=config.get('hp_search_k_folds', 2),
                max_epochs=config.get('hp_search_epochs', 20)
            )

        if best_params:
            print(f"Best hyperparameters found: {best_params}")
            ### Update config with best parameters
            config.update(best_params)
        else:
            print("Hyperparameter search failed, using default parameters")

    ### Continue with regular training using best parameters
    kfold = KFold(n_splits=config['k_folds'], shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n--- Fold {fold + 1}/{config['k_folds']} ---")

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        ### Initialize optimizer
        if config['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        ### Initialize scheduler
        if ADVANCED_SCHEDULERS_AVAILABLE:
            scheduler = get_scheduler(config['scheduler'], optimizer, config['epochs'], len(train_loader))
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )

        ### Initialize regularization
        if ADVANCED_REG_AVAILABLE:
            reg_manager = RegularizationManager(model, reg_type=config['regularization'])
        else:
            reg_manager = None

        ### Initialize early stopping
        early_stopping = EarlyStopping(patience=config['patience'], verbose=True)

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        start_time = time.time()

        for epoch in range(config['epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, device, config['loss_type'], reg_manager)
            val_loss = validate_epoch(model, val_loader, device, config['loss_type'], reg_manager)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            ### Update scheduler
            if hasattr(scheduler, 'step'):
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            ### Update regularization
            if ADVANCED_REG_AVAILABLE and reg_manager is not None:
                reg_manager.update_epoch(epoch)

            ### Early stopping
            early_stopping(val_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_{model_name.lower()}_model_fold_{fold + 1}.pt')

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        end_time = time.time()
        training_time = end_time - start_time

        fold_results.append({
            'fold': fold + 1,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'epochs': len(train_losses),
            'best_params': best_params
        })

        ### Save loss curves
        np.savetxt(f'{model_name.lower()}_train_loss_fold_{fold + 1}.txt', train_losses)
        np.savetxt(f'{model_name.lower()}_val_loss_fold_{fold + 1}.txt', val_losses)

        print(f"Fold {fold + 1} completed in {training_time / 3600:.2f} hours")

    return fold_results

### Train single model (original function)
def train_single_model(model_name, model, dataset, device, config):
    return train_single_model_with_search(model_name, model, dataset, device, config,
                                          enable_hp_search=False)

### Main training function
def train_all_models(config):
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Starting unified training for all models...")
    print(f"Using device: {device}")
    print(f"Configuration: {config}")

    ### Load dataset with stable path handling
    print("Loading dataset...")
    data_path = config['data_path']

    ### Check if data path exists, try multiple locations
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
    print(f"Dataset size: {len(dataset)}")

    ### Get feature dimensions
    sample = dataset[0]
    num_features = sample.x.shape[1]

    ### Define models
    models = {
        'GAT': GAT(num_node_features=num_features, out_dim=5),
        'GIN': GIN(num_node_features=num_features, out_dim=5),
        'GraphSAGE': GraphSAGE(num_node_features=num_features, out_dim=5)
    }

    ### Train each model
    all_results = {}

    for model_name, model in models.items():
        if model_name in config['models_to_train']:
            model = model.to(device)
            results = train_single_model_with_search(
                model_name, model, dataset, device, config,
                enable_hp_search=config.get('enable_hp_search', True),
                search_type=config.get('hp_search_type', 'optuna')
            )
            all_results[model_name] = results

    ### Generate summary
    print(f"\n{'=' * 50}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 50}")

    for model_name, results in all_results.items():
        avg_val_loss = np.mean([r['best_val_loss'] for r in results])
        std_val_loss = np.std([r['best_val_loss'] for r in results])
        total_time = sum([r['training_time'] for r in results])

        print(f"\n{model_name}:")
        print(f"  Average Validation Loss: {avg_val_loss:.6f} ± {std_val_loss:.6f}")
        print(f"  Total Training Time: {total_time / 3600:.2f} hours")
        print(f"  Average Time per Fold: {total_time / 3600 / len(results):.2f} hours")

    ### Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ### Save summary
    summary_data = []
    for model_name, results in all_results.items():
        avg_val_loss = np.mean([r['best_val_loss'] for r in results])
        std_val_loss = np.std([r['best_val_loss'] for r in results])
        total_time = sum([r['training_time'] for r in results])

        summary_data.append({
            'model': model_name,
            'avg_val_loss': avg_val_loss,
            'std_val_loss': std_val_loss,
            'total_training_time_hours': total_time / 3600,
            'avg_training_time_hours': total_time / 3600 / len(results),
            'k_folds': len(results),
            'loss_type': config['loss_type'],
            'scheduler': config['scheduler'],
            'regularization': config['regularization'],
            'hyperparameter_search': config.get('enable_hp_search', False),
            'search_type': config.get('hp_search_type', 'none')
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'training_summary_{timestamp}.csv', index=False)

    print(f"\nResults saved to: training_summary_{timestamp}.csv")
    print("Training completed!")

### Main function
def main():
    parser = argparse.ArgumentParser(description='Unified Training for All Models with Hyperparameter Search')
    parser.add_argument('--data_path', type=str, default='data/train_cleaned.csv', help='Path to training data')
    parser.add_argument('--models', type=str, nargs='+', default=['GAT', 'GIN', 'GraphSAGE'],
                        help='Models to train')
    parser.add_argument('--k_folds', type=int, default=3, help='Number of K-fold splits')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--loss_type', type=str, default='wmae',
                        choices=['mse', 'wmae', 'weighted_mse', 'huber', 'focal', 'combined', 'uncertainty',
                                 'adaptive'],
                        help='Loss function type')
    parser.add_argument('--scheduler', type=str, default='cosine_warmup',
                        choices=['plateau', 'cosine_warmup', 'onecycle', 'adaptive', 'cyclical', 'polynomial'],
                        help='Learning rate scheduler')
    parser.add_argument('--regularization', type=str, default='dropout',
                        choices=['dropout', 'label_smoothing', 'mixup', 'cutmix', 'weight_decay', 'gradient_penalty'],
                        help='Regularization type')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    ### Hyperparameter search arguments
    parser.add_argument('--enable_hp_search', action='store_true', help='Enable hyperparameter search')
    parser.add_argument('--hp_search_type', type=str, default='optuna',
                        choices=['optuna', 'grid', 'random'],
                        help='Hyperparameter search type')
    parser.add_argument('--hp_search_trials', type=int, default=30, help='Number of search trials')
    parser.add_argument('--hp_search_timeout', type=int, default=900, help='Search timeout in seconds')
    parser.add_argument('--hp_search_k_folds', type=int, default=2, help='K-folds for search evaluation')
    parser.add_argument('--hp_search_epochs', type=int, default=20, help='Epochs for search evaluation')

    args = parser.parse_args()

    ### Configuration dictionary
    config = {
        'data_path': args.data_path,
        'models_to_train': args.models,
        'k_folds': args.k_folds,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'loss_type': args.loss_type,
        'scheduler': args.scheduler,
        'regularization': args.regularization,
        'optimizer': args.optimizer,
        'seed': args.seed,
        'enable_hp_search': args.enable_hp_search,
        'hp_search_type': args.hp_search_type,
        'hp_search_trials': args.hp_search_trials,
        'hp_search_timeout': args.hp_search_timeout,
        'hp_search_k_folds': args.hp_search_k_folds,
        'hp_search_epochs': args.hp_search_epochs
    }

    ### Start training
    train_all_models(config)

if __name__ == "__main__":
    main()
### #%#