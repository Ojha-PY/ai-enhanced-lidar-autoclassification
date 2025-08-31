"""
Enhanced Automated Machine Learning (AutoML) System - Conda Optimized
Compatible implementation with conda environment support
"""

import optuna
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import pickle
import warnings
import os
warnings.filterwarnings('ignore')


@dataclass
class CondaOptimizedHPOResult:
    """Enhanced hyperparameter optimization result for conda environments"""
    best_params: Dict[str, Any]
    best_value: float
    best_trial: optuna.trial.Trial
    study: optuna.study.Study
    optimization_time: float
    n_trials: int
    convergence_history: List[float]
    param_importance: Dict[str, float]
    conda_environment_info: Dict[str, Any]


class CondaAwareOptimizerFactory:
    """Conda-aware optimizer factory with environment detection"""

    def __init__(self):
        self.conda_env = self._detect_conda_environment()
        self.cuda_available = torch.cuda.is_available()

        # Conda-optimized optimizer configs
        self.optimizer_configs = {
            'adamw': {
                'lr_range': (1e-5, 1e-2),
                'weight_decay_range': (1e-6, 1e-2),
                'betas_range': ((0.8, 0.95), (0.95, 0.999)),
                'eps_range': (1e-9, 1e-6)
            },
            'sgd': {
                'lr_range': (1e-4, 1e-1),
                'momentum_range': (0.8, 0.99),
                'weight_decay_range': (1e-6, 1e-2),
                'nesterov': [True, False]
            },
            'adam': {
                'lr_range': (1e-5, 1e-2),
                'weight_decay_range': (1e-6, 1e-2),
                'betas_range': ((0.8, 0.95), (0.95, 0.999))
            }
        }

        # Apply conda-specific optimizations
        if self.conda_env:
            self._apply_conda_optimizations()

    def _detect_conda_environment(self) -> bool:
        """Detect conda environment"""
        return any([
            os.environ.get('CONDA_DEFAULT_ENV'),
            os.environ.get('CONDA_PREFIX'),
            os.path.exists(os.path.join(os.sys.prefix, 'conda-meta'))
        ])

    def _apply_conda_optimizations(self):
        """Apply conda-specific optimizations"""
        if self.conda_env:
            # Set optimal threading for conda
            os.environ['OMP_NUM_THREADS'] = str(min(8, os.cpu_count()))
            os.environ['MKL_NUM_THREADS'] = str(min(8, os.cpu_count()))
            print("🐍 Applied conda optimizations for AutoML")

    def create_optimizer(self, model: nn.Module, trial: optuna.trial.Trial) -> torch.optim.Optimizer:
        """Create conda-optimized optimizer"""
        optimizer_name = trial.suggest_categorical("optimizer", list(self.optimizer_configs.keys()))
        config = self.optimizer_configs[optimizer_name]

        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=trial.suggest_float("learning_rate", *config['lr_range'], log=True),
                weight_decay=trial.suggest_float("weight_decay", *config['weight_decay_range'], log=True),
                betas=(
                    trial.suggest_float("beta1", *config['betas_range'][0]),
                    trial.suggest_float("beta2", *config['betas_range'][1])
                ),
                eps=trial.suggest_float("eps", *config['eps_range'], log=True)
            )
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=trial.suggest_float("learning_rate", *config['lr_range'], log=True),
                momentum=trial.suggest_float("momentum", *config['momentum_range']),
                weight_decay=trial.suggest_float("weight_decay", *config['weight_decay_range'], log=True),
                nesterov=trial.suggest_categorical("nesterov", config['nesterov'])
            )
        else:  # adam
            return torch.optim.Adam(
                model.parameters(),
                lr=trial.suggest_float("learning_rate", *config['lr_range'], log=True),
                weight_decay=trial.suggest_float("weight_decay", *config['weight_decay_range'], log=True),
                betas=(
                    trial.suggest_float("beta1", *config['betas_range'][0]),
                    trial.suggest_float("beta2", *config['betas_range'][1])
                )
            )


class CondaEnhancedAutoMLOptimizer:
    """Conda-enhanced AutoML system with environment awareness"""

    def __init__(self, 
                 model_factory: Callable,
                 train_loader,
                 val_loader,
                 device: str,
                 num_classes: int = 256):

        self.model_factory = model_factory
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes

        # Conda environment detection
        self.conda_env_info = self._get_conda_environment_info()

        # Enhanced components
        self.optimizer_factory = CondaAwareOptimizerFactory()

        # Results storage
        self.optimization_history = []
        self.best_configs = []
        self.convergence_history = []

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.trial_times = []

        if self.conda_env_info['is_conda']:
            print(f"🐍 Conda AutoML: Environment '{self.conda_env_info['name']}' detected")
            print(f"🔥 PyTorch: {self.conda_env_info['pytorch_version']}")
            print(f"🚀 CUDA: {'Available' if self.conda_env_info['cuda_available'] else 'Not available'}")

    def _get_conda_environment_info(self) -> Dict[str, Any]:
        """Get detailed conda environment information"""
        info = {
            'is_conda': any([
                os.environ.get('CONDA_DEFAULT_ENV'),
                os.environ.get('CONDA_PREFIX'),
                os.path.exists(os.path.join(os.sys.prefix, 'conda-meta'))
            ]),
            'name': os.environ.get('CONDA_DEFAULT_ENV', 'unknown'),
            'prefix': os.environ.get('CONDA_PREFIX', 'unknown'),
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'optuna_version': optuna.__version__
        }

        return info

    def objective(self, trial: optuna.trial.Trial) -> float:
        """Conda-optimized Optuna objective function"""
        trial_start_time = time.time()

        try:
            # Model hyperparameters
            model_params = self._suggest_model_params(trial)

            # Training hyperparameters with conda optimizations
            train_params = self._suggest_conda_training_params(trial)

            # Create model
            model = self.model_factory(**model_params)
            model.to(self.device)

            # Create conda-optimized optimizer
            optimizer = self.optimizer_factory.create_optimizer(model, trial)

            # Conda-aware mixed precision
            use_amp = train_params['use_mixed_precision'] and self.device == 'cuda'
            scaler = torch.cuda.amp.GradScaler() if use_amp else None

            # Loss function
            criterion = nn.CrossEntropyLoss()

            # Training configuration
            num_epochs = train_params['num_epochs']
            early_stopping_patience = train_params['early_stopping_patience']

            # Training loop with conda optimizations
            best_val_acc = 0.0
            patience_counter = 0

            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_correct = 0
                train_total = 0

                # Limit training batches for faster HPO
                max_train_batches = min(50, len(self.train_loader))

                for batch_idx, batch in enumerate(self.train_loader):
                    if batch_idx >= max_train_batches:
                        break

                    try:
                        # Extract batch data
                        points = batch.get('points', batch[0] if isinstance(batch, (list, tuple)) else None)
                        features = batch.get('features', batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else points)
                        labels = batch.get('labels', batch[2] if isinstance(batch, (list, tuple)) and len(batch) > 2 else torch.randint(0, self.num_classes, (points.shape[0],)))
                        num_points = batch.get('num_points', torch.full((points.shape[0],), points.shape[1]))

                        # Move to device
                        points = points.to(self.device)
                        features = features.to(self.device)
                        labels = labels.to(self.device)
                        num_points = num_points.to(self.device)

                        optimizer.zero_grad()

                        # Forward pass with conda-optimized mixed precision
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = model(points, features, num_points)
                                loss = criterion(outputs, labels)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(points, features, num_points)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                        # Statistics
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += labels.size(0)
                        train_correct += (predicted == labels).sum().item()

                    except Exception as e:
                        continue

                # Validation phase
                model.eval()
                val_correct = 0
                val_total = 0

                max_val_batches = min(25, len(self.val_loader))

                with torch.no_grad():
                    for batch_idx, batch in enumerate(self.val_loader):
                        if batch_idx >= max_val_batches:
                            break

                        try:
                            points = batch.get('points', batch[0] if isinstance(batch, (list, tuple)) else None)
                            features = batch.get('features', batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else points)
                            labels = batch.get('labels', batch[2] if isinstance(batch, (list, tuple)) and len(batch) > 2 else torch.randint(0, self.num_classes, (points.shape[0],)))
                            num_points = batch.get('num_points', torch.full((points.shape[0],), points.shape[1]))

                            points = points.to(self.device)
                            features = features.to(self.device)
                            labels = labels.to(self.device)
                            num_points = num_points.to(self.device)

                            outputs = model(points, features, num_points)

                            _, predicted = torch.max(outputs.data, 1)
                            val_total += labels.size(0)
                            val_correct += (predicted == labels).sum().item()

                        except Exception as e:
                            continue

                # Calculate accuracy
                val_acc = val_correct / val_total if val_total > 0 else 0.0

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    break

                # Report for pruning
                trial.report(val_acc, epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # Track trial performance
            trial_time = time.time() - trial_start_time
            self.trial_times.append(trial_time)

            # Memory cleanup
            del model, optimizer
            torch.cuda.empty_cache()

            return best_val_acc

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(f"Trial failed: {e}")
            return 0.0

    def _suggest_conda_training_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Suggest conda-optimized training parameters"""
        params = {
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 24, 32]),
            'num_epochs': trial.suggest_int('num_epochs', 10, 30),  # Reduced for conda HPO
            'early_stopping_patience': trial.suggest_int('early_stopping_patience', 3, 10),
            'use_mixed_precision': trial.suggest_categorical('use_mixed_precision', [True, False]) if self.device == 'cuda' else False
        }

        return params

    def _suggest_model_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Suggest model parameters"""
        return {
            'dropout': trial.suggest_float('dropout_rate', 0.0, 0.5),
            'num_classes': self.num_classes
        }

    def optimize(self, 
                n_trials: int = 50, 
                timeout: int = 1800,
                study_name: Optional[str] = None) -> CondaOptimizedHPOResult:
        """Run conda-optimized hyperparameter optimization"""

        self.logger.info(f"🐍 Starting Conda-Enhanced AutoML with {n_trials} trials")
        start_time = time.time()

        if study_name is None:
            study_name = f"conda_lidar_automl_{int(time.time())}"

        # Conda-optimized pruning and sampling
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=3,
            reduction_factor=3
        )

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=5,
            n_ei_candidates=12,
            multivariate=True
        )

        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            sampler=sampler,
            study_name=study_name
        )

        # Optimize
        try:
            study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            self.logger.info("⚠️ Optimization interrupted by user")

        optimization_time = time.time() - start_time

        # Parameter importance
        param_importance = {}
        try:
            param_importance = optuna.importance.get_param_importances(study)
        except Exception:
            pass

        # Create conda-aware result
        result = CondaOptimizedHPOResult(
            best_params=study.best_params,
            best_value=study.best_value,
            best_trial=study.best_trial,
            study=study,
            optimization_time=optimization_time,
            n_trials=len(study.trials),
            convergence_history=self.convergence_history,
            param_importance=param_importance,
            conda_environment_info=self.conda_env_info
        )

        # Save results with conda info
        self._save_conda_results(result)

        self.logger.info(f"✅ Conda AutoML completed!")
        self.logger.info(f"🏆 Best accuracy: {result.best_value:.4f}")
        self.logger.info(f"🐍 Environment: {self.conda_env_info['name']}")

        return result

    def _save_conda_results(self, result: CondaOptimizedHPOResult):
        """Save conda-aware results"""
        save_dir = Path("experiments/automl")
        save_dir.mkdir(parents=True, exist_ok=True)

        results_dict = {
            'best_params': result.best_params,
            'best_value': result.best_value,
            'optimization_time': result.optimization_time,
            'n_trials': result.n_trials,
            'param_importance': result.param_importance,
            'conda_environment_info': result.conda_environment_info,
            'trial_statistics': {
                'avg_trial_time': np.mean(self.trial_times) if self.trial_times else 0,
                'total_trials': len(self.trial_times)
            }
        }

        results_file = save_dir / f"conda_automl_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        self.logger.info(f"💾 Conda AutoML results saved to: {save_dir}")


def create_conda_enhanced_automl_optimizer(model_factory: Callable,
                                          train_loader,
                                          val_loader,
                                          device: str,
                                          num_classes: int = 256) -> CondaEnhancedAutoMLOptimizer:
    """Create conda-enhanced AutoML optimizer"""

    return CondaEnhancedAutoMLOptimizer(
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=num_classes
    )


def run_conda_enhanced_automl_search(train_loader, val_loader, device: str, 
                                    n_trials: int = 50) -> CondaOptimizedHPOResult:
    """Run complete conda-enhanced AutoML search"""

    def model_factory(**kwargs):
        """Enhanced model factory for conda environments"""
        import torch.nn as nn

        class CondaOptimizedModel(nn.Module):
            def __init__(self, num_classes=256, dropout=0.1):
                super().__init__()

                # Conda-optimized feature extraction
                self.features = nn.Sequential(
                    nn.Linear(32, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, 256),
                    nn.ReLU(), 
                    nn.Dropout(dropout),
                    nn.Linear(256, num_classes)
                )

            def forward(self, points, features, num_points=None):
                batch_size = features.shape[0]
                pooled = features.mean(dim=1)  # Global average pooling
                return self.features(pooled)

        return CondaOptimizedModel(**kwargs)

    # Create conda-enhanced optimizer
    automl = create_conda_enhanced_automl_optimizer(
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # Run optimization
    result = automl.optimize(n_trials=n_trials)

    print("\n🐍 CONDA-ENHANCED AUTOML RESULTS")
    print("=" * 60)
    print(f"🏆 Best Accuracy: {result.best_value:.4f}")
    print(f"📊 Total Trials: {result.n_trials}")
    print(f"⏱️ Optimization Time: {result.optimization_time:.1f}s")
    print(f"🐍 Conda Environment: {result.conda_environment_info['name']}")
    print(f"🔥 PyTorch Version: {result.conda_environment_info['pytorch_version']}")

    return result


# Export functions for compatibility
create_enhanced_automl_optimizer = create_conda_enhanced_automl_optimizer
run_enhanced_automl_search = run_conda_enhanced_automl_search
