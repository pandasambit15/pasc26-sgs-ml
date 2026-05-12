#!/usr/bin/env python3
"""
Task Weight Optimization for Richardson-Conditioned Models (COMPLETE)
====================================================================

Supports ALL four methods: Manual, Uncertainty, GradNorm, DWA

Usage:
    python task_weight_optimization_ri_conditioned.py \
        --config config.yaml \
        --architecture MLP \
        --methods Manual Uncertainty GradNorm DWA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
import yaml
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from experiment_utils import (
    load_experiment_data,
    create_ri_conditioned_model,  # Use Ri-conditioned models
    predictions_dict_from_model_output,
    compute_regression_metrics
)

# Import ALL weighting classes from original script
from task_weight_optimization_complete import (
    UncertaintyWeighting,
    GradNormWeighter,
    DWAWeighter
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskWeightExperimentRiConditioned:
    """
    Task weight optimization for Ri-conditioned models.
    Supports: Manual, Uncertainty, GradNorm, DWA
    """
    
    def __init__(self, architecture: str, config: Dict, output_dir: Path, resume: bool = False):
        self.architecture = architecture
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task_names = ['viscosity', 'diffusivity', 'richardson', 'regime', 'positivity']

        self.setup_logging()
        
        logger.info("🔬 Using Richardson-conditioned models")

        self.results = {}
        self.checkpoint_path = output_dir / "experiment_checkpoint.pt"

        if resume and self.checkpoint_path.exists():
            self.load_checkpoint()
        else:
            self.completed_methods = []

    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "experiment.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _compute_task_losses(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """Compute individual task losses."""
        losses = {}

        losses['viscosity'] = F.huber_loss(predictions['viscosity'], targets['viscosity'])
        losses['diffusivity'] = F.huber_loss(predictions['diffusivity'], targets['diffusivity'])
        losses['richardson'] = F.huber_loss(predictions['richardson'], targets['richardson'])
        losses['regime'] = F.cross_entropy(predictions['regime'], targets['regime'])
        losses['positivity'] = (
            F.softplus(-predictions['viscosity']).mean() +
            F.softplus(-predictions['diffusivity']).mean()
        )

        return losses

    def train_manual(self, train_loader, val_loader, test_loader,
                    manual_weights: Dict, seed: int = 42) -> Dict:
        """Train with manual (equal) task weights."""
        self.logger.info(f"Training Ri-{self.architecture} with Manual weights (seed={seed})")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = create_ri_conditioned_model(self.architecture)
        model = model.to(self.device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-5)
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        best_val_loss = float('inf')
        best_metrics = None
        epochs_no_improve = 0
        weight_history = {k: [v] for k, v in manual_weights.items()}

        run_checkpoint = self.output_dir / f"checkpoint_Manual_Ri{self.architecture}_seed{seed}.pt"
        start_epoch = 0

        if run_checkpoint.exists():
            checkpoint = torch.load(run_checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            best_metrics = checkpoint['best_metrics']
            epochs_no_improve = checkpoint['epochs_no_improve']
            weight_history = checkpoint['weight_history']
            self.logger.info(f"Resumed from epoch {start_epoch}")

        for epoch in range(start_epoch, self.config['epochs']):
            model.train()
            train_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                features, targets = batch
                features = features.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                model_output = model(features)
                predictions = predictions_dict_from_model_output(model_output)
                losses = self._compute_task_losses(predictions, targets)

                total_loss = sum(manual_weights[k] * v for k, v in losses.items())

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += total_loss.item()

            val_loss, val_metrics = self.evaluate(model, val_loader, manual_weights)
            scheduler.step(val_loss)

            self.logger.info(
                f"Epoch {epoch+1}: Train={train_loss/len(train_loader):.4f}, "
                f"Val={val_loss:.4f}, Visc R²={val_metrics['viscosity_r2']:.4f}"
            )

            for key in weight_history:
                weight_history[key].append(manual_weights[key])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                epochs_no_improve = 0

                best_model_path = self.output_dir / f"best_model_Manual_Ri{self.architecture}_seed{seed}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'metrics': best_metrics,
                    'config': self.config,
                    'architecture': f'Ri-{self.architecture}',
                    'method': 'Manual',
                    'seed': seed
                }, best_model_path)
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_metrics': best_metrics,
                    'epochs_no_improve': epochs_no_improve,
                    'weight_history': weight_history
                }, run_checkpoint)

            if epochs_no_improve >= self.config.get('early_stopping_patience', 20):
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        test_loss, test_metrics = self.evaluate(model, test_loader, manual_weights)

        if run_checkpoint.exists():
            run_checkpoint.unlink()

        return {
            'model': model,
            'best_val_metrics': best_metrics,
            'test_metrics': test_metrics,
            'weight_history': weight_history,
            'final_weights': manual_weights
        }

    def train_uncertainty(self, train_loader, val_loader, test_loader, seed: int = 42) -> Dict:
        """Train with uncertainty-based weighting."""
        self.logger.info(f"Training Ri-{self.architecture} with Uncertainty weighting (seed={seed})")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = create_ri_conditioned_model(self.architecture)
        model = model.to(self.device)

        weighter = UncertaintyWeighting(n_tasks=5)
        weighter = weighter.to(self.device)

        model_optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-5)
        )

        weight_optimizer = optim.Adam(
            [weighter.log_vars],
            lr=self.config.get('weight_learning_rate', 0.025)
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            model_optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        best_val_loss = float('inf')
        best_metrics = None
        epochs_no_improve = 0
        weight_history = {'viscosity': [], 'diffusivity': [], 'richardson': [],
                         'regime': [], 'positivity': []}

        run_checkpoint = self.output_dir / f"checkpoint_Uncertainty_Ri{self.architecture}_seed{seed}.pt"
        start_epoch = 0

        if run_checkpoint.exists():
            checkpoint = torch.load(run_checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            weighter.log_vars.data = checkpoint['log_vars']
            model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
            weight_optimizer.load_state_dict(checkpoint['weight_optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            best_metrics = checkpoint['best_metrics']
            epochs_no_improve = checkpoint['epochs_no_improve']
            weight_history = checkpoint['weight_history']
            self.logger.info(f"Resumed from epoch {start_epoch}")

        for epoch in range(start_epoch, self.config['epochs']):
            model.train()
            weighter.train()

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                features, targets = batch
                features = features.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                model_output = model(features)
                predictions = predictions_dict_from_model_output(model_output)
                losses = self._compute_task_losses(predictions, targets)

                total_loss, weights_dict = weighter.compute_weighted_loss(losses)

                model_optimizer.zero_grad()
                weight_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                model_optimizer.step()
                weight_optimizer.step()

            current_weights = weighter.get_weights()
            task_names = ['viscosity', 'diffusivity', 'richardson', 'regime', 'positivity']
            for i, task in enumerate(task_names):
                weight_history[task].append(current_weights[i].item())

            val_loss, val_metrics = self.evaluate_uncertainty(model, val_loader, weighter)
            scheduler.step(val_loss)

            self.logger.info(
                f"Epoch {epoch+1}: Val={val_loss:.4f}, Visc R²={val_metrics['viscosity_r2']:.4f}, "
                f"Weights: Visc={current_weights[0]:.2f}, Diff={current_weights[1]:.2f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                epochs_no_improve = 0

                best_model_path = self.output_dir / f"best_model_Uncertainty_Ri{self.architecture}_seed{seed}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'log_vars': weighter.log_vars.data,
                    'metrics': best_metrics,
                    'config': self.config,
                    'architecture': f'Ri-{self.architecture}',
                    'method': 'Uncertainty',
                    'seed': seed
                }, best_model_path)
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'log_vars': weighter.log_vars.data,
                    'model_optimizer_state_dict': model_optimizer.state_dict(),
                    'weight_optimizer_state_dict': weight_optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_metrics': best_metrics,
                    'epochs_no_improve': epochs_no_improve,
                    'weight_history': weight_history
                }, run_checkpoint)

            if epochs_no_improve >= self.config.get('early_stopping_patience', 20):
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        test_loss, test_metrics = self.evaluate_uncertainty(model, test_loader, weighter)

        if run_checkpoint.exists():
            run_checkpoint.unlink()

        final_weights_tensor = weighter.get_weights()
        final_weights = {
            task: final_weights_tensor[i].item()
            for i, task in enumerate(['viscosity', 'diffusivity', 'richardson', 'regime', 'positivity'])
        }

        return {
            'model': model,
            'weighter': weighter,
            'best_val_metrics': best_metrics,
            'test_metrics': test_metrics,
            'weight_history': weight_history,
            'final_weights': final_weights
        }

    def train_gradnorm(self, train_loader, val_loader, test_loader, seed: int = 42) -> Dict:
        """Train with GradNorm weighting."""
        self.logger.info(f"Training Ri-{self.architecture} with GradNorm (seed={seed})")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = create_ri_conditioned_model(self.architecture)
        model = model.to(self.device)

        weighter = GradNormWeighter(
            n_tasks=5,
            alpha=self.config.get('gradnorm_alpha', 1.5)
        )

        model_params = list(model.parameters())
        shared_param = model_params[len(model_params)//4]

        model_optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-5)
        )

        weight_optimizer = optim.Adam(
            [weighter.weights],
            lr=self.config.get('weight_learning_rate', 0.025)
        )

        weighter.weights.data = weighter.weights.data.to(self.device)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            model_optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        best_val_loss = float('inf')
        best_metrics = None
        epochs_no_improve = 0
        weight_history = {'viscosity': [], 'diffusivity': [], 'richardson': [],
                         'regime': [], 'positivity': []}

        run_checkpoint = self.output_dir / f"checkpoint_GradNorm_Ri{self.architecture}_seed{seed}.pt"
        start_epoch = 0

        if run_checkpoint.exists():
            checkpoint = torch.load(run_checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            weighter.weights.data = checkpoint['gradnorm_weights'].to(self.device)
            weighter.initial_losses = checkpoint.get('initial_losses')
            weighter.loss_ratios = checkpoint.get('loss_ratios')

            model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
            weight_optimizer.load_state_dict(checkpoint['weight_optimizer_state_dict'])

            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            best_metrics = checkpoint['best_metrics']
            epochs_no_improve = checkpoint['epochs_no_improve']
            weight_history = checkpoint['weight_history']
            self.logger.info(f"Resumed from epoch {start_epoch}")

        for epoch in range(start_epoch, self.config['epochs']):
            model.train()

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                features, targets = batch
                features = features.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                model_output = model(features)
                predictions = predictions_dict_from_model_output(model_output)
                losses = self._compute_task_losses(predictions, targets)

                weighter.update_weights(losses, shared_param, weight_optimizer)

                current_weights = weighter.get_weights()

                total_loss = sum(current_weights[i] * losses[task]
                               for i, task in enumerate(['viscosity', 'diffusivity', 'richardson', 'regime', 'positivity']))

                model_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                model_optimizer.step()

            current_weights = weighter.get_weights()
            task_names = ['viscosity', 'diffusivity', 'richardson', 'regime', 'positivity']
            for i, task in enumerate(task_names):
                weight_history[task].append(current_weights[i].item())

            val_loss, val_metrics = self.evaluate(model, val_loader,
                                                  {task: current_weights[i].item() for i, task in enumerate(task_names)})
            scheduler.step(val_loss)

            self.logger.info(
                f"Epoch {epoch+1}: Val={val_loss:.4f}, Visc R²={val_metrics['viscosity_r2']:.4f}, "
                f"Weights: Visc={current_weights[0]:.2f}, Diff={current_weights[1]:.2f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                epochs_no_improve = 0

                best_model_path = self.output_dir / f"best_model_GradNorm_Ri{self.architecture}_seed{seed}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'gradnorm_weights': weighter.weights.data,
                    'initial_losses': weighter.initial_losses,
                    'metrics': best_metrics,
                    'config': self.config,
                    'architecture': f'Ri-{self.architecture}',
                    'method': 'GradNorm',
                    'seed': seed
                }, best_model_path)
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'gradnorm_weights': weighter.weights.data,
                    'initial_losses': weighter.initial_losses,
                    'loss_ratios': weighter.loss_ratios,
                    'model_optimizer_state_dict': model_optimizer.state_dict(),
                    'weight_optimizer_state_dict': weight_optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_metrics': best_metrics,
                    'epochs_no_improve': epochs_no_improve,
                    'weight_history': weight_history
                }, run_checkpoint)

            if epochs_no_improve >= self.config.get('early_stopping_patience', 20):
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        final_weights_tensor = weighter.get_weights()
        final_weights = {
            task: final_weights_tensor[i].item()
            for i, task in enumerate(['viscosity', 'diffusivity', 'richardson', 'regime', 'positivity'])
        }
        test_loss, test_metrics = self.evaluate(model, test_loader, final_weights)

        if run_checkpoint.exists():
            run_checkpoint.unlink()

        return {
            'model': model,
            'weighter': weighter,
            'best_val_metrics': best_metrics,
            'test_metrics': test_metrics,
            'weight_history': weight_history,
            'final_weights': final_weights
        }

    def train_dwa(self, train_loader, val_loader, test_loader, seed: int = 42) -> Dict:
        """Train with Dynamic Weight Averaging."""
        self.logger.info(f"Training Ri-{self.architecture} with DWA (seed={seed})")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = create_ri_conditioned_model(self.architecture)
        model = model.to(self.device)

        weighter = DWAWeighter(
            n_tasks=5,
            temperature=self.config.get('dwa_temperature', 2.0),
            device=self.device
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-5)
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        best_val_loss = float('inf')
        best_metrics = None
        epochs_no_improve = 0
        weight_history = {'viscosity': [], 'diffusivity': [], 'richardson': [],
                         'regime': [], 'positivity': []}

        run_checkpoint = self.output_dir / f"checkpoint_DWA_Ri{self.architecture}_seed{seed}.pt"
        start_epoch = 0

        if run_checkpoint.exists():
            checkpoint = torch.load(run_checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            dwa_state = {
                'weights': checkpoint['dwa_weights'],
                'prev_losses': checkpoint.get('prev_losses'),
                'temperature': checkpoint.get('dwa_temperature', 2.0),
                'n_tasks': 5
            }
            weighter.load_state_dict(dwa_state)
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            best_metrics = checkpoint['best_metrics']
            epochs_no_improve = checkpoint['epochs_no_improve']
            weight_history = checkpoint['weight_history']
            self.logger.info(f"Resumed from epoch {start_epoch}")

        for epoch in range(start_epoch, self.config['epochs']):
            model.train()
            epoch_task_losses = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                features, targets = batch
                features = features.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                model_output = model(features)
                predictions = predictions_dict_from_model_output(model_output)
                losses = self._compute_task_losses(predictions, targets)

                loss_tensor = torch.stack([losses[task] for task in ['viscosity', 'diffusivity', 'richardson', 'regime', 'positivity']])
                epoch_task_losses.append(loss_tensor.detach())

                current_weights = weighter.get_weights().to(self.device)

                total_loss = (current_weights * loss_tensor).sum()

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            avg_epoch_losses = torch.stack(epoch_task_losses).mean(dim=0)
            losses_dict = {
                task: avg_epoch_losses[i]
                for i, task in enumerate(['viscosity', 'diffusivity', 'richardson', 'regime', 'positivity'])
            }
            weighter.update_weights(losses_dict, epoch)

            current_weights = weighter.get_weights()
            task_names = ['viscosity', 'diffusivity', 'richardson', 'regime', 'positivity']
            for i, task in enumerate(task_names):
                weight_history[task].append(current_weights[i].item())

            final_weights = {task: current_weights[i].item() for i, task in enumerate(task_names)}
            val_loss, val_metrics = self.evaluate(model, val_loader, final_weights)
            scheduler.step(val_loss)

            self.logger.info(
                f"Epoch {epoch+1}: Val={val_loss:.4f}, Visc R²={val_metrics['viscosity_r2']:.4f}, "
                f"Weights: Visc={current_weights[0]:.2f}, Diff={current_weights[1]:.2f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                epochs_no_improve = 0

                best_model_path = self.output_dir / f"best_model_DWA_Ri{self.architecture}_seed{seed}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'dwa_weights': weighter.weights,
                    'dwa_temperature': weighter.temperature,
                    'prev_losses': weighter.prev_losses,
                    'metrics': best_metrics,
                    'config': self.config,
                    'architecture': f'Ri-{self.architecture}',
                    'method': 'DWA',
                    'seed': seed
                }, best_model_path)
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'dwa_weights': weighter.weights,
                    'dwa_temperature': weighter.temperature,
                    'prev_losses': weighter.prev_losses,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_metrics': best_metrics,
                    'epochs_no_improve': epochs_no_improve,
                    'weight_history': weight_history
                }, run_checkpoint)

            if epochs_no_improve >= self.config.get('early_stopping_patience', 20):
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        final_weights = {
            task: current_weights[i].item()
            for i, task in enumerate(['viscosity', 'diffusivity', 'richardson', 'regime', 'positivity'])
        }
        test_loss, test_metrics = self.evaluate(model, test_loader, final_weights)

        if run_checkpoint.exists():
            run_checkpoint.unlink()

        return {
            'model': model,
            'weighter': weighter,
            'best_val_metrics': best_metrics,
            'test_metrics': test_metrics,
            'weight_history': weight_history,
            'final_weights': final_weights
        }

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader, weights: Dict) -> Tuple[float, Dict]:
        """Evaluate model with fixed weights."""
        model.eval()

        all_preds = {'viscosity': [], 'diffusivity': [], 'richardson': [], 'regime': []}
        all_targets = {'viscosity': [], 'diffusivity': [], 'richardson': [], 'regime': []}
        total_loss = 0.0

        for batch in loader:
            features, targets = batch
            features = features.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            model_output = model(features)
            predictions = predictions_dict_from_model_output(model_output)
            losses = self._compute_task_losses(predictions, targets)

            batch_loss = sum(weights[k] * v for k, v in losses.items())
            total_loss += batch_loss.item()

            for key in all_preds.keys():
                if key == 'regime':
                    all_preds[key].append(predictions[key].argmax(dim=1).cpu())
                    all_targets[key].append(targets[key].cpu())
                else:
                    all_preds[key].append(predictions[key].cpu())
                    all_targets[key].append(targets[key].cpu())

        for key in all_preds.keys():
            all_preds[key] = torch.cat(all_preds[key]).numpy()
            all_targets[key] = torch.cat(all_targets[key]).numpy()

        metrics = {}

        for key in ['viscosity', 'diffusivity', 'richardson']:
            reg_metrics = compute_regression_metrics(all_preds[key], all_targets[key])
            for metric_name, value in reg_metrics.items():
                metrics[f'{key}_{metric_name}'] = value

        metrics['regime_accuracy'] = (all_preds['regime'] == all_targets['regime']).mean()

        return total_loss / len(loader), metrics

    @torch.no_grad()
    def evaluate_uncertainty(self, model: nn.Module, loader: DataLoader,
                           weighter: UncertaintyWeighting) -> Tuple[float, Dict]:
        """Evaluate with uncertainty module."""
        model.eval()
        weighter.eval()

        all_preds = {'viscosity': [], 'diffusivity': [], 'richardson': [], 'regime': []}
        all_targets = {'viscosity': [], 'diffusivity': [], 'richardson': [], 'regime': []}
        total_loss = 0.0

        for batch in loader:
            features, targets = batch
            features = features.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            model_output = model(features)
            predictions = predictions_dict_from_model_output(model_output)
            losses = self._compute_task_losses(predictions, targets)

            batch_loss, _ = weighter.compute_weighted_loss(losses)
            total_loss += batch_loss.item()

            for key in all_preds.keys():
                if key == 'regime':
                    all_preds[key].append(predictions[key].argmax(dim=1).cpu())
                    all_targets[key].append(targets[key].cpu())
                else:
                    all_preds[key].append(predictions[key].cpu())
                    all_targets[key].append(targets[key].cpu())

        for key in all_preds.keys():
            all_preds[key] = torch.cat(all_preds[key]).numpy()
            all_targets[key] = torch.cat(all_targets[key]).numpy()

        metrics = {}

        for key in ['viscosity', 'diffusivity', 'richardson']:
            reg_metrics = compute_regression_metrics(all_preds[key], all_targets[key])
            for metric_name, value in reg_metrics.items():
                metrics[f'{key}_{metric_name}'] = value

        metrics['regime_accuracy'] = (all_preds['regime'] == all_targets['regime']).mean()

        return total_loss / len(loader), metrics

    def run_experiment(self, method: str, train_loader, val_loader, test_loader, seed: int = 42) -> Dict:
        """Run a single experiment configuration."""
        manual_weights = {
            'viscosity': 1.0,
            'diffusivity': 1.0,
            'richardson': 1.0,
            'regime': 1.0,
            'positivity': 1.0
        }

        if method == 'Manual':
            return self.train_manual(train_loader, val_loader, test_loader, manual_weights, seed)
        elif method == 'Uncertainty':
            return self.train_uncertainty(train_loader, val_loader, test_loader, seed)
        elif method == 'GradNorm':
            return self.train_gradnorm(train_loader, val_loader, test_loader, seed)
        elif method == 'DWA':
            return self.train_dwa(train_loader, val_loader, test_loader, seed)
        else:
            raise ValueError(f"Unknown method: {method}")

    def save_checkpoint(self):
        """Save experiment-level checkpoint."""
        checkpoint = {
            'completed_methods': self.completed_methods,
            'results': self.results,
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_path)
        self.logger.info("Saved experiment checkpoint")

    def load_checkpoint(self):
        """Load experiment-level checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        self.completed_methods = checkpoint['completed_methods']
        self.results = checkpoint['results']
        self.logger.info(f"Loaded checkpoint: {len(self.completed_methods)} methods completed")

    def run_all_experiments(self, data_dir: Path, scaler_dir: Path, methods: List[str],
                           seeds: List[int] = [42, 123, 456]):
        """Run all requested experiments with checkpointing."""

        self.logger.info("Loading data...")
        train_data, val_data, test_data = load_experiment_data(data_dir, scaler_dir)

        batch_size = self.config['batch_size']
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        for method in methods:
            if method in self.completed_methods:
                self.logger.info(f"Skipping {method} (already completed)")
                continue

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Starting Ri-conditioned experiments for {method}")
            self.logger.info(f"{'='*60}\n")

            method_results = {}
            any_success = False

            for seed in seeds:
                self.logger.info(f"\nRunning: {method} - Ri-{self.architecture} - Seed {seed}")

                try:
                    result = self.run_experiment(method, train_loader, val_loader, test_loader, seed)
                    method_results[f"seed_{seed}"] = result
                    any_success = True
                    self.logger.info(f"✅ Completed: {method} - Ri-{self.architecture} - Seed {seed}")

                except Exception as e:
                    self.logger.error(f"✗ Error in {method} - Seed {seed}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            self.results[method] = method_results

            if any_success:
                self.completed_methods.append(method)
                self.logger.info(f"\n✅ Completed experiments for {method} ({len(method_results)}/{len(seeds)} seeds)")
            else:
                self.logger.error(f"\n✗ All seeds failed for {method}")

            self.save_checkpoint()

        self.logger.info(f"\n{'='*60}")
        self.logger.info("ALL RI-CONDITIONED EXPERIMENTS COMPLETED")
        self.logger.info(f"Successful methods: {len(self.completed_methods)}/{len(methods)}")
        self.logger.info(f"{'='*60}\n")

        self.save_results_summary()
        self.plot_weight_evolution()

    def plot_weight_evolution(self):
        """Plot task weight evolution for all methods."""
        methods_to_plot = []

        for method in self.results.keys():
            if method != 'Manual' and len(self.results[method]) > 0:
                has_results = any(
                    'weight_history' in self.results[method][seed_key]
                    for seed_key in self.results[method].keys()
                )
                if has_results:
                    methods_to_plot.append(method)

        if not methods_to_plot:
            self.logger.warning("No methods with weight evolution to plot")
            return

        fig, axes = plt.subplots(len(methods_to_plot), 5, figsize=(20, 4*len(methods_to_plot)))
        if len(methods_to_plot) == 1:
            axes = axes.reshape(1, -1)

        tasks = ['viscosity', 'diffusivity', 'richardson', 'regime', 'positivity']
        task_titles = ['Viscosity Weight', 'Diffusivity Weight', 'Richardson Weight',
                      'Regime Weight', 'Positivity Weight']

        for i, method in enumerate(methods_to_plot):
            for seed_key in self.results[method].keys():
                if 'weight_history' not in self.results[method][seed_key]:
                    continue

                weight_history = self.results[method][seed_key]['weight_history']

                for j, (task, title) in enumerate(zip(tasks, task_titles)):
                    if len(methods_to_plot) > 1:
                        ax = axes[i, j]
                    else:
                        ax = axes[j]

                    weights = weight_history[task]
                    ax.plot(weights, label=seed_key, linewidth=2, alpha=0.7)

                    ax.axhline(y=1.0, color='blue', linestyle='--', label='Manual', alpha=0.5)

                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Task Weight')
                    ax.set_title(f'{title}\n(Ri-{method})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / 'weight_evolution_ri.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved weight evolution plot: {plot_path}")

    def save_results_summary(self):
        """Save summary of all results to CSV."""
        rows = []
        for method in self.results:
            if len(self.results[method]) == 0:
                continue

            for seed_key in self.results[method]:
                result = self.results[method][seed_key]

                if 'test_metrics' not in result:
                    continue

                test_metrics = result['test_metrics']
                final_weights = result.get('final_weights', {})

                row = {
                    'Method': method,
                    'Architecture': f'Ri-{self.architecture}',
                    'Seed': seed_key,
                    **test_metrics,
                    'Final_Weights': str(final_weights)
                }
                rows.append(row)

        if not rows:
            self.logger.error("No results to save")
            return

        df = pd.DataFrame(rows)
        csv_path = self.output_dir / 'results_summary_ri.csv'
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved results summary: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Task Weight Optimization for Ri-Conditioned Models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--methods', nargs='+',
                       choices=['Manual', 'Uncertainty', 'GradNorm', 'DWA'],
                       default=['Manual', 'Uncertainty', 'GradNorm', 'DWA'],
                       help='Methods to run')
    parser.add_argument('--architecture', type=str, required=True,
                       choices=['MLP', 'ResMLP', 'TabTransformer'],
                       help='Architecture to test')
    parser.add_argument('--seeds', nargs='+', type=int,
                       default=[42, 123, 456],
                       help='Random seeds')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_yaml = yaml.safe_load(f)

    data_dir = Path(config_yaml['paths']['data_dir'])
    scaler_dir = Path(config_yaml['paths']['scaler_dir'])
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"experiments/task_weights_ri/{args.architecture}")

    exp_config = {
        'epochs': config_yaml['training'].get('epochs', 300),
        'learning_rate': config_yaml['training'].get('learning_rate', 1e-3),
        'weight_decay': config_yaml['training'].get('weight_decay', 1e-5),
        'weight_learning_rate': config_yaml['training'].get('weight_learning_rate', 0.025),
        'batch_size': config_yaml['training'].get('batch_size', 8192),
        'early_stopping_patience': config_yaml['training'].get('early_stopping_patience', 20),
        'gradnorm_alpha': config_yaml['training'].get('gradnorm_alpha', 1.5),
        'dwa_temperature': config_yaml['training'].get('dwa_temperature', 2.0)
    }

    experiment = TaskWeightExperimentRiConditioned(args.architecture, exp_config, output_dir, resume=args.resume)
    experiment.run_all_experiments(data_dir, scaler_dir, args.methods, args.seeds)

    print("\n" + "="*60)
    print("RI-CONDITIONED EXPERIMENTS COMPLETED SUCCESSFULLY")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
