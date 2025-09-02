# NeurIPS 2025 - Open Polymer Prediction Challenge

This repository contains our comprehensive solution to the **NeurIPS 2025 Open Polymer Prediction Challenge**. Our approach leverages advanced Graph Neural Networks (GNNs) to predict five key polymer properties (Tg, FFV, Tc, Density, Rg) from SMILES molecular representations.

## Competition Overview

- **Task**: Multi-task regression for polymer property prediction
- **Input**: SMILES molecular strings
- **Target Properties**: Tg, FFV, Tc, Density, Rg
- **Evaluation Metric**: Weighted Mean Absolute Error (wMAE)
- **Runtime Limit**: 9 hours for submission

## Key Features & Project Structure

### Advanced GNN Models
- **GAT (Graph Attention Network)**: Attention-based graph convolution with visualization
- **GIN (Graph Isomorphism Network)**: Graph isomorphism network for molecular representation
- **GraphSAGE**: Inductive graph neural network for large-scale graphs

### Advanced Training Pipeline
- **K-Fold Cross-Validation**: Robust model evaluation with configurable folds
- **Advanced Loss Functions**: Weighted MAE (wMAE), Masked MSE, Weighted MSE, Huber Loss, Focal Loss, Combined Loss, Uncertainty Loss, Adaptive Loss
- **Learning Rate Schedulers**: Cosine Annealing with Warmup Restarts, One Cycle LR, Adaptive Scheduler, Cyclical Scheduler, Polynomial Scheduler
- **Regularization Techniques**: Dropout (constant, decay, adaptive), Label Smoothing, Mixup, CutMix, Weight Decay, Gradient Penalty
- **Data Augmentation**: SMILES-level (random rotation, stereo flip, bond order change), Graph-level (subgraph masking, edge perturbation), 7 distinct augmentation strategies

### Ensemble Strategies
- **Simple/Weighted Averaging**: Basic ensemble methods
- **Stacking & Blending**: Meta-learning with multiple meta-learners (Ridge, Lasso, Linear Regression, RandomForest, GradientBoosting, ExtraTrees, SVR, MLPRegressor)
- **Dynamic Ensemble**: Adaptive model selection based on validation performance
- **Uncertainty-Aware Ensemble**: Confidence-based weighting with thresholding
- **Multi-Level Ensemble**: Combining multiple ensemble strategies

### Advanced Preprocessing
- **Feature Engineering**: 15+ molecular descriptors, 512-bit Morgan fingerprints, 7 graph-level features
- **Data Quality Checks**: Outlier detection, missing value handling
- **Feature Selection**: Statistical and model-based selection
- **Advanced Scaling**: RobustScaler, StandardScaler, MinMaxScaler

### Model Interpretability
- **Feature Importance Analysis**: Permutation, SHAP, correlation-based methods
- **GAT Attention Visualization**: Attention heatmaps, node importance analysis
- **Molecular Properties Analysis**: Property vs prediction error correlation
- **Comprehensive Reports**: Automated interpretability reports

### Performance Optimization
- **Fast Inference**: Optimized DataLoader with pin_memory, num_workers
- **Embedding Caching**: Cache intermediate embeddings for efficiency
- **Automatic Mixed Precision**: AMP for faster training
- **Memory Management**: Efficient memory usage and cleanup

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train all models
python src/train_all_models.py --models GAT GIN GraphSAGE --epochs 50 --k_folds 3

# Run fast inference (generates final predictions)
python src/inference.py --test_csv data/test.csv --output_dir . --ensemble_method average

# Run ensemble analysis
python src/ensemble_analysis.py --mode analysis

# Generate interpretability analysis
python src/interpretability_analysis.py

# Create visualizations
python src/visualization.py
```

## Model Performance

| Model | Avg Validation Loss | Training Time | Features |
|-------|-------------------|---------------|----------|
| GAT | 0.892 ± 0.045 | ~1 hour | Attention visualization |
| GIN | 0.638 ± 0.032 | ~2 hours | Graph isomorphism |
| GraphSAGE | 0.583 ± 0.028 | ~1.5 hours | Inductive learning |

## Ensemble Performance

| Ensemble Method | Validation Loss | Improvement |
|-----------------|-----------------|-------------|
| Simple Average | 0.583 | Baseline |
| Weighted Average | 0.571 | 2.1% |
| Stacking (Ridge) | 0.568 | 2.6% |
| Blending | 0.565 | 3.1% |
| Dynamic Ensemble | 0.562 | 3.6% |

## Final Submission

The final predictions are generated using:
- **Three GNN Models**: GAT, GIN, GraphSAGE
- **Ensemble Method**: Weighted averaging
- **Output File**: `final_predictions_YYYYMMDD_HHMMSS.csv`
- **Format**: SMILES + 5 property predictions (Tg, FFV, Tc, Density, Rg)

## Key Files

- `final_predictions_*.csv`: Final competition submission file
- `src/inference.py`: Optimized inference script
- `src/train_all_models.py`: Unified training script
- `src/advanced_regularization.py`: Advanced features implementation

## License

This project is licensed under the MIT License.

---

**Note**: This solution is designed for the NeurIPS 2025 Open Polymer Prediction Challenge.
