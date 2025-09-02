### Min
import os
import torch
from torch_geometric.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Subset
import random
import numpy as np
import time
from datetime import datetime
import pandas as pd
import sys

### Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

### Import models and dataset with fallback
try:
    from data_part.dataset import PolymerDataset
    from model.gat_model import GAT
except ImportError:
    try:
        from dataset import PolymerDataset
        from gat_model import GAT
    except ImportError:
        print("Error: Could not import required modules. Please check file paths.")
        sys.exit(1)

### Import advanced regularization
try:
    from advanced_regularization import RegularizationManager

    ADVANCED_REG_AVAILABLE = True
except ImportError:
    ADVANCED_REG_AVAILABLE = False
    print("Warning: Advanced regularization not available")

### Import advanced losses
try:
    from advanced_losses import get_loss_function

    ADVANCED_LOSSES_AVAILABLE = True
except ImportError:
    ADVANCED_LOSSES_AVAILABLE = False
    print("Warning: Advanced losses not available, using default MSE loss")


### Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


### Enhanced masked loss with regularization and wMAE support
def masked_loss(pred, target, mask, loss_type='mse', reg_manager=None):
    if ADVANCED_LOSSES_AVAILABLE:
        criterion = get_loss_function(loss_type)
        loss = criterion(pred, target, mask)
    else:
        if loss_type == 'wmae':
            # Implement weighted MAE loss
            loss = torch.abs(pred - target)
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            # Default MSE loss
            loss = (pred - target) ** 2
            loss = loss * mask
            loss = loss.sum() / mask.sum()

    ### Add regularization if available
    if ADVANCED_REG_AVAILABLE and reg_manager is not None:
        reg_loss = reg_manager.compute_regularization_loss()
        loss += reg_loss

    return loss


### Training loop with enhanced regularization
def train(model, loader, optimizer, device, loss_type='mse', reg_manager=None):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.batch)
        y = data.y.view(out.shape[0], out.shape[1])
        mask = data.mask.view(out.shape[0], out.shape[1])

        loss = masked_loss(out, y, mask, loss_type, reg_manager)
        loss.backward()

        ### Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


### Evaluation with regularization
def evaluate(model, loader, device, loss_type='mse', reg_manager=None):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            y = data.y.view(out.shape[0], out.shape[1])
            mask = data.mask.view(out.shape[0], out.shape[1])

            loss = masked_loss(out, y, mask, loss_type, reg_manager)
            total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


### Enhanced early stopping utility
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


### K-fold cross validation with enhanced regularization
def k_fold_training(dataset, k=3, max_epochs=50, batch_size=64, augmentation=False,
                    augmentation_ratio=0.5, loss_type='wmae', reg_type='dropout'):
    ### Optimized K-fold training with regularization
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n=== Fold {fold + 1}/{k} ===")

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        sample = dataset[0]
        num_features = sample.x.shape[1]
        model = GAT(num_node_features=num_features, out_dim=5).to(device)

        ### Enhanced optimizer with weight decay
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

        ### Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.3, patience=3
        )

        ### Early stopping
        early_stopping = EarlyStopping(patience=10, verbose=True)

        ### Initialize regularization manager
        if ADVANCED_REG_AVAILABLE:
            reg_manager = RegularizationManager(model, reg_type=reg_type)
        else:
            reg_manager = None

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        start_time = time.time()

        for epoch in range(1, max_epochs + 1):
            ### Training with regularization
            train_loss = train(model, train_loader, optimizer, device, loss_type, reg_manager)
            val_loss = evaluate(model, val_loader, device, loss_type, reg_manager)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)
            early_stopping(val_loss)

            ### Update regularization if available
            if ADVANCED_REG_AVAILABLE and reg_manager is not None:
                reg_manager.update_epoch(epoch)

            if epoch % 5 == 0:
                print(f"Epoch {epoch:03d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_gat_model_fold_{fold + 1}.pt')

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
            'loss_type': loss_type,
            'reg_type': reg_type
        })

        np.savetxt(f'gat_train_loss_fold_{fold + 1}.txt', train_losses)
        np.savetxt(f'gat_val_loss_fold_{fold + 1}.txt', val_losses)

        print(f"Fold {fold + 1} completed in {training_time / 3600:.2f} hours")

    return fold_results


### Main training routine with enhanced regularization
def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Starting optimized GAT training with regularization...")
    print(f"Using device: {device}")

    ### Load dataset with full feature configuration and augmentation
    feature_config = {
        "atomic_num": True, "mass": True, "aromatic": True,
        "degree": True, "formal_charge": True, "num_hs": True,
        "hybridization": True, "bond_type": True,
        "bond_aromatic": True, "bond_ring": True
    }

    ### Enable data augmentation for training
    augmentation_enabled = True
    augmentation_ratio = 0.5

    ### Enhanced loss and regularization settings
    loss_type = 'wmae'
    reg_type = 'dropout'

    print("Loading dataset with augmentation...")

    ### Fixed data path with intelligent detection
    data_path = os.path.join('data', 'train_cleaned.csv')

    # Check if data path exists, try multiple locations
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

    dataset = PolymerDataset(
        csv_file=data_path,
        feature_config=feature_config,
        augmentation=augmentation_enabled,
        augmentation_ratio=augmentation_ratio
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Data augmentation: {'Enabled' if augmentation_enabled else 'Disabled'}")
    print(f"Augmentation ratio: {augmentation_ratio}")
    print(f"Loss function: {loss_type}")
    print(f"Regularization: {reg_type}")

    ### Run enhanced K-fold training with regularization
    print("\nStarting K-fold cross validation with regularization...")
    fold_results = k_fold_training(
        dataset,
        k=3,
        max_epochs=50,
        batch_size=64,
        augmentation=augmentation_enabled,
        augmentation_ratio=augmentation_ratio,
        loss_type=loss_type,
        reg_type=reg_type
    )

    ### Calculate and save results
    avg_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
    std_val_loss = np.std([r['best_val_loss'] for r in fold_results])
    total_time = sum([r['training_time'] for r in fold_results])

    results_summary = {
        'model': 'GAT',
        'augmentation': augmentation_enabled,
        'augmentation_ratio': augmentation_ratio,
        'loss_type': loss_type,
        'reg_type': reg_type,
        'avg_val_loss': avg_val_loss,
        'std_val_loss': std_val_loss,
        'total_training_time_hours': total_time / 3600,
        'avg_training_time_hours': total_time / 3600 / len(fold_results),
        'k_folds': len(fold_results),
        'max_epochs': 50,
        'batch_size': 64
    }

    ### Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ### Save fold details
    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(f'gat_fold_results_{timestamp}.csv', index=False)

    ### Save summary
    summary_df = pd.DataFrame([results_summary])
    summary_df.to_csv(f'gat_training_summary_{timestamp}.csv', index=False)

    print(f"\n{'=' * 50}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'=' * 50}")
    print(f"Data Augmentation: {'Enabled' if augmentation_enabled else 'Disabled'}")
    print(f"Augmentation Ratio: {augmentation_ratio}")
    print(f"Loss Function: {loss_type}")
    print(f"Regularization: {reg_type}")
    print(f"Average Validation Loss: {avg_val_loss:.6f} ± {std_val_loss:.6f}")
    print(f"Total Training Time: {total_time / 3600:.2f} hours")
    print(f"Average Time per Fold: {total_time / 3600 / len(fold_results):.2f} hours")
    print(f"Results saved with timestamp: {timestamp}")


if __name__ == "__main__":
    main()
### #%#