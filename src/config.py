### Min
import os
from dataclasses import dataclass
from typing import List, Dict, Any
import json


@dataclass
class ModelConfig:
    name: str
    hidden_channels: int = 64
    heads: int = 4
    dropout: float = 0.3
    num_layers: int = 3
    out_dim: int = 5


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 50
    patience: int = 10
    k_folds: int = 3
    seed: int = 42
    optimizer: str = 'adamw'
    loss_type: str = 'wmae'
    scheduler: str = 'cosine_warmup'
    regularization: str = 'dropout'


@dataclass
class DataConfig:
    train_path: str = 'data/train_cleaned.csv'
    test_path: str = 'data/test_cleaned.csv'
    augmentation_ratio: float = 0.5
    augmentation_enabled: bool = True
    feature_scaling: str = 'robust'
    outlier_threshold: float = 3.0


@dataclass
class AdvancedConfig:
    gradient_clipping: float = 1.0
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    max_lr: float = 1e-3
    cycle_length: int = 10
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_prob: float = 0.5


@dataclass
class EnsembleConfig:
    methods: List[str] = None
    weights: Dict[str, float] = None
    meta_learner: str = 'ridge'
    cv_folds: int = 5
    uncertainty_threshold: float = 0.1

    def __post_init__(self):
        if self.methods is None:
            self.methods = ['weighted', 'stacking']
        if self.weights is None:
            self.weights = {'GAT': 0.3, 'GIN': 0.35, 'GraphSAGE': 0.35}


@dataclass
class CompleteConfig:
    models: Dict[str, ModelConfig]
    training: TrainingConfig
    data: DataConfig
    advanced: AdvancedConfig
    ensemble: EnsembleConfig

    def __post_init__(self):
        if self.models is None:
            self.models = {
                'GAT': ModelConfig(name='GAT', hidden_channels=64, heads=4),
                'GIN': ModelConfig(name='GIN', hidden_channels=64, num_layers=3),
                'GraphSAGE': ModelConfig(name='GraphSAGE', hidden_channels=64, num_layers=3)
            }


class ConfigManager:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or 'config.json'
        self.config = self._load_default_config()

    def _load_default_config(self) -> CompleteConfig:
        models = {
            'GAT': ModelConfig(
                name='GAT',
                hidden_channels=64,
                heads=4,
                dropout=0.3,
                num_layers=3,
                out_dim=5
            ),
            'GIN': ModelConfig(
                name='GIN',
                hidden_channels=64,
                heads=1,
                dropout=0.3,
                num_layers=3,
                out_dim=5
            ),
            'GraphSAGE': ModelConfig(
                name='GraphSAGE',
                hidden_channels=64,
                heads=1,
                dropout=0.3,
                num_layers=3,
                out_dim=5
            )
        }

        training = TrainingConfig(
            batch_size=64,
            learning_rate=1e-4,
            weight_decay=0.01,
            epochs=50,
            patience=10,
            k_folds=3,
            seed=42,
            optimizer='adamw',
            loss_type='wmae',
            scheduler='cosine_warmup',
            regularization='dropout'
        )

        data = DataConfig(
            train_path='data/train_cleaned.csv',
            test_path='data/test_cleaned.csv',
            augmentation_ratio=0.5,
            augmentation_enabled=True,
            feature_scaling='robust',
            outlier_threshold=3.0
        )

        advanced = AdvancedConfig(
            gradient_clipping=1.0,
            warmup_epochs=5,
            min_lr=1e-6,
            max_lr=1e-3,
            cycle_length=10,
            label_smoothing=0.1,
            mixup_alpha=0.2,
            cutmix_prob=0.5
        )

        ensemble = EnsembleConfig(
            methods=['weighted', 'stacking'],
            weights={'GAT': 0.3, 'GIN': 0.35, 'GraphSAGE': 0.35},
            meta_learner='ridge',
            cv_folds=5,
            uncertainty_threshold=0.1
        )

        return CompleteConfig(
            models=models,
            training=training,
            data=data,
            advanced=advanced,
            ensemble=ensemble
        )

    def load_config(self) -> CompleteConfig:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
            return self._dict_to_config(config_dict)
        return self.config

    def save_config(self, config: CompleteConfig = None):
        if config is None:
            config = self.config

        config_dict = self._config_to_dict(config)
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def _config_to_dict(self, config: CompleteConfig) -> Dict[str, Any]:
        return {
            'models': {
                name: {
                    'name': model.name,
                    'hidden_channels': model.hidden_channels,
                    'heads': model.heads,
                    'dropout': model.dropout,
                    'num_layers': model.num_layers,
                    'out_dim': model.out_dim
                }
                for name, model in config.models.items()
            },
            'training': {
                'batch_size': config.training.batch_size,
                'learning_rate': config.training.learning_rate,
                'weight_decay': config.training.weight_decay,
                'epochs': config.training.epochs,
                'patience': config.training.patience,
                'k_folds': config.training.k_folds,
                'seed': config.training.seed,
                'optimizer': config.training.optimizer,
                'loss_type': config.training.loss_type,
                'scheduler': config.training.scheduler,
                'regularization': config.training.regularization
            },
            'data': {
                'train_path': config.data.train_path,
                'test_path': config.data.test_path,
                'augmentation_ratio': config.data.augmentation_ratio,
                'augmentation_enabled': config.data.augmentation_enabled,
                'feature_scaling': config.data.feature_scaling,
                'outlier_threshold': config.data.outlier_threshold
            },
            'advanced': {
                'gradient_clipping': config.advanced.gradient_clipping,
                'warmup_epochs': config.advanced.warmup_epochs,
                'min_lr': config.advanced.min_lr,
                'max_lr': config.advanced.max_lr,
                'cycle_length': config.advanced.cycle_length,
                'label_smoothing': config.advanced.label_smoothing,
                'mixup_alpha': config.advanced.mixup_alpha,
                'cutmix_prob': config.advanced.cutmix_prob
            },
            'ensemble': {
                'methods': config.ensemble.methods,
                'weights': config.ensemble.weights,
                'meta_learner': config.ensemble.meta_learner,
                'cv_folds': config.ensemble.cv_folds,
                'uncertainty_threshold': config.ensemble.uncertainty_threshold
            }
        }

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> CompleteConfig:
        models = {}
        for name, model_dict in config_dict.get('models', {}).items():
            models[name] = ModelConfig(**model_dict)

        training = TrainingConfig(**config_dict.get('training', {}))
        data = DataConfig(**config_dict.get('data', {}))
        advanced = AdvancedConfig(**config_dict.get('advanced', {}))
        ensemble = EnsembleConfig(**config_dict.get('ensemble', {}))

        return CompleteConfig(
            models=models,
            training=training,
            data=data,
            advanced=advanced,
            ensemble=ensemble
        )

    def update_model_config(self, model_name: str, **kwargs):
        if model_name in self.config.models:
            for key, value in kwargs.items():
                if hasattr(self.config.models[model_name], key):
                    setattr(self.config.models[model_name], key, value)

    def update_training_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config.training, key):
                setattr(self.config.training, key, value)

    def get_model_config(self, model_name: str) -> ModelConfig:
        return self.config.models.get(model_name)

    def get_training_config(self) -> TrainingConfig:
        return self.config.training

    def get_data_config(self) -> DataConfig:
        return self.config.data

    def get_advanced_config(self) -> AdvancedConfig:
        return self.config.advanced

    def get_ensemble_config(self) -> EnsembleConfig:
        return self.config.ensemble


def get_optimized_config() -> CompleteConfig:
    models = {
        'GAT': ModelConfig(
            name='GAT',
            hidden_channels=128,
            heads=8,
            dropout=0.4,
            num_layers=4,
            out_dim=5
        ),
        'GIN': ModelConfig(
            name='GIN',
            hidden_channels=128,
            heads=1,
            dropout=0.4,
            num_layers=4,
            out_dim=5
        ),
        'GraphSAGE': ModelConfig(
            name='GraphSAGE',
            hidden_channels=128,
            heads=1,
            dropout=0.4,
            num_layers=4,
            out_dim=5
        )
    }

    training = TrainingConfig(
        batch_size=32,
        learning_rate=5e-5,
        weight_decay=0.001,
        epochs=100,
        patience=15,
        k_folds=5,
        seed=42,
        optimizer='adamw',
        loss_type='wmae',
        scheduler='onecycle',
        regularization='mixup'
    )

    data = DataConfig(
        train_path='data/train_cleaned.csv',
        test_path='data/test_cleaned.csv',
        augmentation_ratio=0.8,
        augmentation_enabled=True,
        feature_scaling='robust',
        outlier_threshold=2.5
    )

    advanced = AdvancedConfig(
        gradient_clipping=0.5,
        warmup_epochs=10,
        min_lr=1e-7,
        max_lr=5e-4,
        cycle_length=20,
        label_smoothing=0.05,
        mixup_alpha=0.3,
        cutmix_prob=0.7
    )

    ensemble = EnsembleConfig(
        methods=['weighted', 'stacking'],
        weights={'GAT': 0.25, 'GIN': 0.35, 'GraphSAGE': 0.4},
        meta_learner='ridge',
        cv_folds=5,
        uncertainty_threshold=0.05
    )

    return CompleteConfig(
        models=models,
        training=training,
        data=data,
        advanced=advanced,
        ensemble=ensemble
    )


def get_fast_config() -> CompleteConfig:
    models = {
        'GAT': ModelConfig(
            name='GAT',
            hidden_channels=32,
            heads=2,
            dropout=0.2,
            num_layers=2,
            out_dim=5
        ),
        'GIN': ModelConfig(
            name='GIN',
            hidden_channels=32,
            heads=1,
            dropout=0.2,
            num_layers=2,
            out_dim=5
        ),
        'GraphSAGE': ModelConfig(
            name='GraphSAGE',
            hidden_channels=32,
            heads=1,
            dropout=0.2,
            num_layers=2,
            out_dim=5
        )
    }

    training = TrainingConfig(
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=0.01,
        epochs=20,
        patience=5,
        k_folds=2,
        seed=42,
        optimizer='adam',
        loss_type='wmae',
        scheduler='plateau',
        regularization='dropout'
    )

    data = DataConfig(
        train_path='data/train_cleaned.csv',
        test_path='data/test_cleaned.csv',
        augmentation_ratio=0.2,
        augmentation_enabled=True,
        feature_scaling='standard',
        outlier_threshold=3.0
    )

    advanced = AdvancedConfig(
        gradient_clipping=1.0,
        warmup_epochs=2,
        min_lr=1e-6,
        max_lr=1e-3,
        cycle_length=5,
        label_smoothing=0.1,
        mixup_alpha=0.1,
        cutmix_prob=0.3
    )

    ensemble = EnsembleConfig(
        methods=['weighted'],
        weights={'GAT': 0.33, 'GIN': 0.33, 'GraphSAGE': 0.34},
        meta_learner='ridge',
        cv_folds=3,
        uncertainty_threshold=0.1
    )

    return CompleteConfig(
        models=models,
        training=training,
        data=data,
        advanced=advanced,
        ensemble=ensemble
    )


if __name__ == "__main__":
    config_manager = ConfigManager('optimized_config.json')

    optimized_config = get_optimized_config()
    config_manager.save_config(optimized_config)

    fast_config = get_fast_config()
    config_manager = ConfigManager('fast_config.json')
    config_manager.save_config(fast_config)

    print("Configuration files created successfully!")
### #%#