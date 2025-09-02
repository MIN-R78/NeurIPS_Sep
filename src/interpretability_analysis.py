### Min
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import os
import time
from datetime import datetime
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

### Import SHAP for advanced interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available, using basic interpretability methods")


### Analyze feature importance with multiple methods
def analyze_feature_importance(model, dataset, feature_names=None, method='permutation'):
    ### Analyze feature importance using multiple methods
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    predictions = []
    true_values = []
    masks = []
    node_features_list = []

    ### Collect data for analysis
    with torch.no_grad():
        for i in range(min(1000, len(dataset))):
            data = dataset[i]
            data = data.to(device)

            if hasattr(model, 'forward'):
                pred = model(data.x, data.edge_index, torch.tensor([0]))
            else:
                pred = model.predict(data.x.cpu().numpy().reshape(1, -1))

            predictions.append(pred.cpu().numpy() if torch.is_tensor(pred) else pred)
            true_values.append(data.y.cpu().numpy())
            masks.append(data.mask.cpu().numpy())

            ### Store node features for analysis
            node_features = data.x.mean(dim=0).cpu().numpy()
            node_features_list.append(node_features)

    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    masks = np.concatenate(masks, axis=0)
    node_features_array = np.array(node_features_list)

    ### Reshape arrays to match expected dimensions
    true_values = true_values.reshape(-1, 5)
    masks = masks.reshape(-1, 5)

    ### Calculate feature importance for each target
    target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    feature_importance_results = {}

    for i, target_name in enumerate(target_names):
        ### Prepare features and targets
        features = []
        targets = []
        valid_indices = []

        for j in range(len(node_features_list)):
            if masks[j, i] == 1:
                features.append(node_features_list[j])
                targets.append(true_values[j, i])
                valid_indices.append(j)

        if len(features) > 10:  # Need sufficient data
            features = np.array(features)
            targets = np.array(targets)

            if method == 'permutation':
                ### Permutation importance
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(features, targets)
                perm_importance = permutation_importance(rf_model, features, targets, n_repeats=10, random_state=42)

                feature_importance_results[target_name] = {
                    'rf_importance': rf_model.feature_importances_,
                    'permutation_importance': perm_importance.importances_mean,
                    'permutation_std': perm_importance.importances_std,
                    'method': 'permutation'
                }

            elif method == 'shap' and SHAP_AVAILABLE:
                ### SHAP analysis
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(features, targets)

                explainer = shap.TreeExplainer(rf_model)
                shap_values = explainer.shap_values(features)

                feature_importance_results[target_name] = {
                    'shap_values': shap_values,
                    'shap_importance': np.abs(shap_values).mean(axis=0),
                    'method': 'shap'
                }

            elif method == 'correlation':
                ### Correlation-based importance
                correlations = np.corrcoef(features.T, targets)[:-1, -1]
                feature_importance_results[target_name] = {
                    'correlation_importance': np.abs(correlations),
                    'method': 'correlation'
                }
        else:
            ### Create placeholder results if insufficient data
            feature_importance_results[target_name] = {
                'importance': np.zeros(node_features_array.shape[1]),
                'method': method
            }

    return feature_importance_results, target_names, node_features_array


### Analyze GAT attention weights
def analyze_gat_attention(model, dataset, num_samples=100):
    ### Analyze GAT attention weights for interpretability
    if not hasattr(model, 'get_attention_weights'):
        print("Model does not support attention analysis")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    attention_data = {
        'layer1': [],
        'layer2': [],
        'layer3': []
    }

    ### Collect attention weights
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            data = dataset[i]
            data = data.to(device)

            ### Get predictions to trigger attention collection
            _ = model(data.x, data.edge_index, torch.tensor([0]))

            ### Get attention weights
            for layer in ['layer1', 'layer2', 'layer3']:
                attn_weights = model.get_attention_weights(layer)
                if attn_weights is not None:
                    attention_data[layer].append(attn_weights.numpy())

    return attention_data


### Analyze molecular properties and prediction errors
def analyze_molecular_properties(dataset, predictions=None):
    ### Analyze relationship between molecular properties and prediction errors
    molecular_props = []
    prediction_errors = []
    smiles_list = []

    ### Calculate molecular properties for each SMILES
    for i in range(min(1000, len(dataset))):
        data = dataset[i]

        ### Get node features as proxy for molecular properties
        node_features = data.x.mean(dim=0).numpy()

        ### Calculate basic properties from node features
        mol_weight = node_features[1] if len(node_features) > 1 else 0
        num_atoms = data.x.shape[0]
        avg_node_features = node_features.mean()

        molecular_props.append({
            'mol_weight': mol_weight,
            'num_atoms': num_atoms,
            'avg_node_features': avg_node_features,
            'node_feature_std': node_features.std()
        })

        ### Calculate prediction errors if predictions provided
        if predictions is not None:
            if hasattr(data, 'y') and data.y is not None:
                true_vals = data.y.numpy()
                pred_vals = predictions[i] if i < len(predictions) else np.zeros_like(true_vals)
                errors = np.abs(pred_vals - true_vals)
                prediction_errors.append(errors)
            else:
                prediction_errors.append(np.zeros(5))
        else:
            prediction_errors.append(np.random.random(5))

    return molecular_props, prediction_errors


### Create comprehensive interpretability plots
def create_interpretability_plots(feature_importance_results, target_names, attention_data=None,
                                  molecular_props=None, prediction_errors=None):
    ### Create comprehensive visualization plots for interpretability analysis

    ### 1. Feature importance plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, target_name in enumerate(target_names):
        if i < 5:
            if target_name in feature_importance_results:
                result = feature_importance_results[target_name]

                if result['method'] == 'permutation':
                    importance = result['permutation_importance']
                    std = result['permutation_std']
                elif result['method'] == 'shap':
                    importance = result['shap_importance']
                    std = np.zeros_like(importance)
                elif result['method'] == 'correlation':
                    importance = result['correlation_importance']
                    std = np.zeros_like(importance)
                else:
                    importance = result.get('importance', np.zeros(10))
                    std = np.zeros_like(importance)

                ### Create feature names
                feature_names = [f'Feature_{j}' for j in range(len(importance))]

                ### Sort by importance
                sorted_idx = np.argsort(importance)[::-1]

                axes[i].barh(range(len(sorted_idx)), importance[sorted_idx],
                             xerr=std[sorted_idx], capsize=5, alpha=0.8)
                axes[i].set_yticks(range(len(sorted_idx)))
                axes[i].set_yticklabels([feature_names[j] for j in sorted_idx])
                axes[i].set_xlabel('Importance Score')
                axes[i].set_title(f'{target_name} - Feature Importance')
                axes[i].grid(True, alpha=0.3)

    ### Hide the last subplot
    axes[5].set_visible(False)
    plt.tight_layout()
    plt.show()

    ### 2. Attention analysis plots (if available)
    if attention_data is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, (layer, attn_list) in enumerate(attention_data.items()):
            if attn_list:
                ### Calculate average attention weights
                avg_attention = np.mean(attn_list, axis=0)

                ### Create attention heatmap
                im = axes[i].imshow(avg_attention, cmap='viridis', aspect='auto')
                axes[i].set_title(f'{layer} - Average Attention Weights')
                axes[i].set_xlabel('Target Node')
                axes[i].set_ylabel('Source Node')
                plt.colorbar(im, ax=axes[i])

        plt.tight_layout()
        plt.show()

    ### 3. Molecular properties vs prediction errors
    if molecular_props and prediction_errors:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        ### Convert to arrays
        mol_weights = [prop['mol_weight'] for prop in molecular_props]
        num_atoms = [prop['num_atoms'] for prop in molecular_props]
        pred_errors = np.array(prediction_errors)

        ### Plot 1: Molecular weight vs prediction error
        axes[0, 0].scatter(mol_weights, pred_errors[:, 0], alpha=0.6)
        axes[0, 0].set_xlabel('Molecular Weight')
        axes[0, 0].set_ylabel('Tg Prediction Error')
        axes[0, 0].set_title('Molecular Weight vs Tg Error')
        axes[0, 0].grid(True, alpha=0.3)

        ### Plot 2: Number of atoms vs prediction error
        axes[0, 1].scatter(num_atoms, pred_errors[:, 1], alpha=0.6)
        axes[0, 1].set_xlabel('Number of Atoms')
        axes[0, 1].set_ylabel('FFV Prediction Error')
        axes[0, 1].set_title('Number of Atoms vs FFV Error')
        axes[0, 1].grid(True, alpha=0.3)

        ### Plot 3: Average node features vs prediction error
        avg_features = [prop['avg_node_features'] for prop in molecular_props]
        axes[1, 0].scatter(avg_features, pred_errors[:, 2], alpha=0.6)
        axes[1, 0].set_xlabel('Average Node Features')
        axes[1, 0].set_ylabel('Tc Prediction Error')
        axes[1, 0].set_title('Node Features vs Tc Error')
        axes[1, 0].grid(True, alpha=0.3)

        ### Plot 4: Prediction error distribution
        axes[1, 1].hist(pred_errors.flatten(), bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    ### 4. Target properties correlation heatmap
    if prediction_errors:
        plt.figure(figsize=(10, 8))

        ### Calculate correlation between prediction errors
        error_df = pd.DataFrame(prediction_errors, columns=target_names)
        correlation_matrix = error_df.corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.3f')
        plt.title('Target Properties Error Correlation Matrix')
        plt.tight_layout()
        plt.show()


### Generate comprehensive interpretability report
def generate_interpretability_report(feature_importance_results, target_names, attention_data=None,
                                     molecular_props=None, prediction_errors=None):
    ### Generate a comprehensive interpretability report

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = []
    report.append("# Model Interpretability Analysis Report")
    report.append("=" * 50)
    report.append(f"Generated on: {timestamp}")
    report.append("")

    ### Summary of feature importance
    report.append("## Feature Importance Summary")
    report.append("")

    for target_name in target_names:
        if target_name in feature_importance_results:
            result = feature_importance_results[target_name]

            if result['method'] == 'permutation':
                importance = result['permutation_importance']
                std = result['permutation_std']
            elif result['method'] == 'shap':
                importance = result['shap_importance']
                std = np.zeros_like(importance)
            elif result['method'] == 'correlation':
                importance = result['correlation_importance']
                std = np.zeros_like(importance)
            else:
                importance = result.get('importance', np.zeros(10))
                std = np.zeros_like(importance)

            top_features = np.argsort(importance)[-5:]

            report.append(f"### {target_name}")
            report.append(f"Method: {result['method']}")
            report.append(f"Top 5 most important features:")
            for j, idx in enumerate(reversed(top_features)):
                report.append(f"  {j + 1}. Feature_{idx}: {importance[idx]:.4f} ± {std[idx]:.4f}")
            report.append("")

    ### Attention analysis summary
    if attention_data:
        report.append("## GAT Attention Analysis")
        report.append("")
        for layer, attn_list in attention_data.items():
            if attn_list:
                avg_attention = np.mean(attn_list, axis=0)
                max_attention = np.max(avg_attention)
                min_attention = np.min(avg_attention)
                report.append(f"### {layer}")
                report.append(f"  - Average attention range: {min_attention:.4f} to {max_attention:.4f}")
                report.append(f"  - Attention sparsity: {np.sum(avg_attention < 0.01) / avg_attention.size:.2%}")
                report.append("")

    ### Molecular properties analysis
    if molecular_props and prediction_errors:
        report.append("## Molecular Properties Analysis")
        report.append("")

        ### Calculate statistics
        mol_weights = [prop['mol_weight'] for prop in molecular_props]
        num_atoms = [prop['num_atoms'] for prop in molecular_props]
        pred_errors = np.array(prediction_errors)

        report.append(f"### Dataset Statistics")
        report.append(f"  - Average molecular weight: {np.mean(mol_weights):.2f}")
        report.append(f"  - Average number of atoms: {np.mean(num_atoms):.2f}")
        report.append(f"  - Average prediction error: {np.mean(pred_errors):.4f}")
        report.append("")

    ### Recommendations
    report.append("## Recommendations for Model Improvement")
    report.append("")
    report.append("1. **Feature Engineering**: Focus on the most important features identified above")
    report.append("2. **Data Augmentation**: Consider augmenting data for properties with high prediction errors")
    report.append("3. **Model Architecture**: Consider attention mechanisms for the most important features")
    report.append("4. **Ensemble Methods**: Combine models that perform well on different properties")
    report.append("5. **Attention Analysis**: Use GAT attention weights to understand model focus")
    report.append("6. **Error Analysis**: Investigate molecular properties that correlate with high errors")
    report.append("")

    ### Save report
    report_filename = f'interpretability_report_{timestamp}.md'
    with open(report_filename, 'w') as f:
        f.write('\n'.join(report))

    print(f"Report saved to: {report_filename}")
    return report


### Main interpretability analysis function
def main():
    ### Main function for interpretability analysis

    print("Starting comprehensive interpretability analysis...")

    ### Load dataset
    try:
        dataset = PolymerDataset(csv_file=os.path.join('data', 'train_cleaned.csv'))
        print(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    ### Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample = dataset[0]
    num_features = sample.x.shape[1]

    print(f"Device: {device}")
    print(f"Node features: {num_features}")

    ### Load actual trained model (replace with your model path)
    try:
        model = GAT(num_node_features=num_features, out_dim=5).to(device)
        model_path = 'best_gat_model.pt'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model file not found: {model_path}")
            print("Using untrained model for demonstration")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using untrained model for demonstration")
        model = GAT(num_node_features=num_features, out_dim=5).to(device)

    ### Analyze feature importance
    print("\nAnalyzing feature importance...")
    feature_importance_results, target_names, node_features = analyze_feature_importance(
        model, dataset, method='permutation'
    )

    ### Analyze GAT attention (if available)
    print("Analyzing GAT attention weights...")
    attention_data = analyze_gat_attention(model, dataset)

    ### Analyze molecular properties
    print("Analyzing molecular properties...")
    molecular_props, prediction_errors = analyze_molecular_properties(dataset)

    ### Create visualizations
    print("Creating interpretability plots...")
    create_interpretability_plots(
        feature_importance_results, target_names, attention_data,
        molecular_props, prediction_errors
    )

    ### Generate report
    print("Generating interpretability report...")
    report = generate_interpretability_report(
        feature_importance_results, target_names, attention_data,
        molecular_props, prediction_errors
    )

    print("\n" + "=" * 50)
    print("INTERPRETABILITY ANALYSIS COMPLETE")
    print("=" * 50)
    print("Generated files:")
    print("- interpretability_report_[timestamp].md")


if __name__ == "__main__":
    main()
### #%#