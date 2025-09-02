### Min
import os
import torch
from torch_geometric.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Subset
from dataset import PolymerDataset
from graphsage_model import GraphSAGE
import random
import numpy as np
import time
from datetime import datetime
import pandas as pd


### Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


### Masked MSE loss for multi-task with missing labels
def masked_mse_loss(pred, target, mask):
    loss = (pred - target) ** 2
    loss = loss * mask
    return loss.sum() / mask.sum()


### Training loop for one epoch with optimization
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        y = data.y.view(out.shape[0], out.shape[1])
        mask = data.mask.view(out.shape[0], out.shape[1])
        loss = masked_mse_loss(out, y, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


### Evaluation on validation set
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            y = data.y.view(out.shape[0], out.shape[1])
            mask = data.mask.view(out.shape[0], out.shape[1])
            loss = masked_mse_loss(out, y, mask)
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


### K-fold cross validation function with augmentation support
def k_fold_training(dataset, k=3, max_epochs=50, batch_size=64, augmentation=False, augmentation_ratio=0.5):
    ### Optimized K-fold training for time efficiency with data augmentation
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
        print(f"Number of node features: {num_features}")

        model = GraphSAGE(num_node_features=num_features, out_dim=5).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

        optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.3, patience=3
        )
        early_stopping = EarlyStopping(patience=10, verbose=True)

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        start_time = time.time()

        for epoch in range(1, max_epochs + 1):
            train_loss = train(model, train_loader, optimizer, device)
            val_loss = evaluate(model, val_loader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)
            early_stopping(val_loss)

            if epoch % 5 == 0:
                print(f"Epoch {epoch:03d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_graphsage_model_fold_{fold + 1}.pt')

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        end_time = time.time()
        training_time = end_time - start_time

        fold_results.append({
            'fold': fold + 1,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'epochs': len(train_losses)
        })

        np.savetxt(f'graphsage_train_loss_fold_{fold + 1}.txt', train_losses)
        np.savetxt(f'graphsage_val_loss_fold_{fold + 1}.txt', val_losses)

        print(f"Fold {fold + 1} completed in {training_time / 3600:.2f} hours")

    return fold_results


### Main training routine with optimization and augmentation
def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Starting optimized GraphSAGE training with data augmentation...")
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

    print("Loading dataset with augmentation...")
    dataset = PolymerDataset(
        csv_file=os.path.join('..', 'data', 'train_cleaned.csv'),
        feature_config=feature_config,
        augmentation=augmentation_enabled,
        augmentation_ratio=augmentation_ratio
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Data augmentation: {'Enabled' if augmentation_enabled else 'Disabled'}")
    print(f"Augmentation ratio: {augmentation_ratio}")

    ### Run optimized K-fold training with augmentation
    print("\nStarting K-fold cross validation with augmentation...")
    fold_results = k_fold_training(
        dataset,
        k=3,
        max_epochs=50,
        batch_size=64,
        augmentation=augmentation_enabled,
        augmentation_ratio=augmentation_ratio
    )

    ### Calculate and save results
    avg_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
    std_val_loss = np.std([r['best_val_loss'] for r in fold_results])
    total_time = sum([r['training_time'] for r in fold_results])

    results_summary = {
        'model': 'GraphSAGE',
        'augmentation': augmentation_enabled,
        'augmentation_ratio': augmentation_ratio,
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
    fold_df.to_csv(f'graphsage_fold_results_{timestamp}.csv', index=False)

    ### Save summary
    summary_df = pd.DataFrame([results_summary])
    summary_df.to_csv(f'graphsage_training_summary_{timestamp}.csv', index=False)

    print(f"\n{'=' * 50}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'=' * 50}")
    print(f"Data Augmentation: {'Enabled' if augmentation_enabled else 'Disabled'}")
    print(f"Augmentation Ratio: {augmentation_ratio}")
    print(f"Average Validation Loss: {avg_val_loss:.6f} ± {std_val_loss:.6f}")
    print(f"Total Training Time: {total_time / 3600:.2f} hours")
    print(f"Average Time per Fold: {total_time / 3600 / len(fold_results):.2f} hours")
    print(f"Results saved with timestamp: {timestamp}")


if __name__ == "__main__":
    main()
### #%#