import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import json
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

@dataclass
class CIFARMetaState:
    agent_losses: torch.Tensor
    agent_accuracies: torch.Tensor
    meta_loss: torch.Tensor
    selected_samples: torch.Tensor
    metrics: Dict[str, float]

class MovingAverage:
    """Helper class to track moving averages."""
    def __init__(self, momentum=0.95):
        self.momentum = momentum
        self.value = None
        
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * new_value
        return self.value

class CIFARAgent(nn.Module):
    """Agent for scoring CIFAR samples."""
    def __init__(self, name: str, input_dim: int = 512):
        super().__init__()
        self.name = name
        self.score_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute scores for input features."""
        return self.score_net(features)

class WeightAnalyzer:
    def __init__(self, meta_agent):
        self.meta_agent = meta_agent
        self.fig_dir = os.path.join(meta_agent.checkpoint_dir, 'weight_analysis')
        os.makedirs(self.fig_dir, exist_ok=True)
        
    def plot_weight_evolution(self):
        """Plot the evolution of agent weights over time."""
        if not self.meta_agent.metrics_history['meta_metrics']:
            return  # Skip if no metrics available
            
        plt.figure(figsize=(12, 6))
        for name in self.meta_agent.agents.keys():
            weights = [metrics[f'{name}_weight'] for metrics in self.meta_agent.metrics_history['meta_metrics']]
            plt.plot(weights, label=name)
        
        plt.title('Agent Weight Evolution')
        plt.xlabel('Training Steps')
        plt.ylabel('Weight Value')
        plt.legend()
        plt.savefig(os.path.join(self.fig_dir, 'weight_evolution.png'))
        plt.close()
        
    def plot_temperature_adaptation(self):
        """Plot temperature parameter and exploration factor."""
        if not self.meta_agent.metrics_history['meta_metrics']:
            return  # Skip if no metrics available
            
        plt.figure(figsize=(12, 6))
        temps = [m['temperature'] for m in self.meta_agent.metrics_history['meta_metrics']]
        exp_factors = [m['exploration_factor'] for m in self.meta_agent.metrics_history['meta_metrics']]
        
        plt.plot(temps, label='Temperature')
        plt.plot(exp_factors, label='Exploration Factor')
        plt.title('Temperature and Exploration Evolution')
        plt.xlabel('Training Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(os.path.join(self.fig_dir, 'temperature_adaptation.png'))
        plt.close()

class CIFARBiLevelMetaAgent(nn.Module):
    def __init__(self, budget_percent: float = 0.1, patience: int = 5):
        super().__init__()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup checkpointing and logging directories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.checkpoint_dir = f'checkpoints/run_{timestamp}'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize metrics history
        self.metrics_history = {
            'agent_scores': defaultdict(list),
            'agent_losses': defaultdict(list),
            'agent_accuracies': defaultdict(list),
            'meta_metrics': [],
            'test_metrics': []
        }
        
        # Initialize WeightAnalyzer
        self.weight_analyzer = WeightAnalyzer(self)
        
        # Create plotting directory
        self.plot_dir = os.path.join(self.checkpoint_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Initialize datasets
        self.dataset = torchvision.datasets.CIFAR10(
            root='/Users/tanmoy/research/data', train=True, 
            download=True, transform=self.transform
        )
        self.test_dataset = torchvision.datasets.CIFAR10(
            root='/Users/tanmoy/research/data', train=False, 
            download=True, transform=self.transform
        )
        
        # Create test dataloader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4
        )
        
        # Initialize other components
        self.max_budget = int(len(self.dataset) * budget_percent)
        self.budget_schedule = {
            20: 0.2,   # At 20% epochs, load 20% of budget
            40: 0.4,   # At 40% epochs, load 40% of budget
            60: 0.6,   # At 60% epochs, load 60% of budget
            80: 0.8,   # At 80% epochs, load 80% of budget
            100: 1.0   # At 100% epochs, load full budget
        }
        self.current_budget_cap = 0
        self.total_epochs = 100
        self.current_indices = set()
        self.current_batch_offset = 0
        
        # Performance monitoring
        self.patience = patience
        self.best_accuracy = 0.0
        self.epochs_without_improvement = 0
        self.use_greedy = False
        
        # Performance history
        self.performance_history = {
            'accuracies': [],
            'epochs_since_improvement': 0,
            'best_accuracy': 0.0
        }
        
        # Create agents
        self.agents = nn.ModuleDict({
            'uncertainty': CIFARAgent('uncertainty'),
            'diversity': CIFARAgent('diversity'),
            'class_balance': CIFARAgent('class_balance'),
            'boundary': CIFARAgent('boundary')
        })
        
        # Initialize feature extractor
        self.feature_extractor = nn.Sequential(
            *list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1]
        ).to(self._device)
        self.feature_extractor.eval()  # Set to eval mode since we don't train it
        
        # Initialize classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).to(self._device)
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        self.classifier_criterion = nn.CrossEntropyLoss()
        
        # Move all components to device immediately
        self.feature_extractor = self.feature_extractor.to(self._device)
        self.classifier = self.classifier.to(self._device)
        
        # Move all agents to device
        for name, agent in self.agents.items():
            self.agents[name] = agent.to(self._device)
        
        # Move meta parameters to device
        self.coordination_weights = nn.Parameter(torch.randn(len(self.agents), device=self._device))
        self.agent_importances = nn.Parameter(torch.randn(len(self.agents), device=self._device))
        self.temperature = nn.Parameter(torch.ones(1, device=self._device) * 0.5)

        self.writer = None  # Add writer attribute

        # Initialize optimizers
        self.meta_optimizer = torch.optim.Adam([
            {'params': [self.coordination_weights, self.agent_importances, self.temperature]}
        ], lr=0.001)
        
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        self.agent_optimizers = {
            name: torch.optim.Adam(agent.parameters(), lr=0.001)
            for name, agent in self.agents.items()
        }

        # Initialize loss scaling factors
        self.loss_scales = {
            'performance': nn.Parameter(torch.ones(1, device=self._device)),
            'balance': nn.Parameter(torch.ones(1, device=self._device)),
            'entropy': nn.Parameter(torch.ones(1, device=self._device)),
            'temperature': nn.Parameter(torch.ones(1, device=self._device))
        }
        
        # Add loss scales to meta optimizer
        self.meta_optimizer = torch.optim.Adam([
            {'params': [self.coordination_weights, self.agent_importances, self.temperature]},
            {'params': self.loss_scales.values(), 'lr': 0.001}
        ])
        
        # Moving averages for loss terms
        self.loss_moving_avg = {
            'performance': MovingAverage(0.95),
            'balance': MovingAverage(0.95),
            'entropy': MovingAverage(0.95),
            'temperature': MovingAverage(0.95)
        }

    def update_budget_cap(self, epoch: int):
        """Update the current budget cap based on training progress."""
        progress_percent = (epoch / self.total_epochs) * 100
        for threshold, budget_fraction in sorted(self.budget_schedule.items()):
            if progress_percent <= threshold:
                self.current_budget_cap = int(self.max_budget * budget_fraction)
                break
        return self.current_budget_cap

    def compute_adaptive_weights(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute adaptive weights with temperature scaling."""
        images, labels = batch
        images = images.to(self._device)
        labels = labels.to(self._device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(images)
            features = features.squeeze(-1).squeeze(-1)

        # Get base scores from each agent
        scores = {}
        agent_outputs = {}
        agent_specific_losses = {}
        
        for name, agent in self.agents.items():
            # Get agent outputs (only features, no labels needed)
            agent_output = agent(features)
            agent_outputs[name] = agent_output
            
            # Compute agent-specific losses using features and labels
            if name == 'uncertainty':
                outputs = self.classifier(features)
                probs = F.softmax(outputs / self.temperature, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                agent_specific_losses[name] = F.mse_loss(agent_output.squeeze(), entropy)
            
            elif name == 'diversity':
                features_norm = F.normalize(features, p=2, dim=1)
                similarity = torch.matmul(features_norm, features_norm.t())
                diversity_target = 1 - similarity.mean(dim=1)
                agent_specific_losses[name] = F.mse_loss(agent_output.squeeze(), diversity_target)
            
            elif name == 'class_balance':
                class_counts = torch.bincount(labels, minlength=10).float()
                class_weights = 1.0 / (class_counts + 1)
                class_weights = F.softmax(class_weights / self.temperature, dim=0)
                balance_target = class_weights[labels]
                agent_specific_losses[name] = F.mse_loss(agent_output.squeeze(), balance_target)
            
            else:  # boundary
                outputs = self.classifier(features)
                top2_values, _ = torch.topk(outputs, k=2, dim=1)
                margins = top2_values[:, 0] - top2_values[:, 1]
                boundary_target = torch.sigmoid(-margins)
                agent_specific_losses[name] = F.mse_loss(agent_output.squeeze(), boundary_target)
            
            # Temperature-scaled normalization
            scores[name] = F.softmax(agent_output / self.temperature, dim=0)

        # Log agent scores
        for name, score in scores.items():
            self.metrics_history['agent_scores'][name].append(score.mean().item())

        return {
            'scores': scores,
            'agent_outputs': agent_outputs,
            'agent_losses': agent_specific_losses,
            'features': features,
            'labels': labels
        }

    def _compute_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute accuracy for given outputs and labels."""
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        total = labels.size(0)
        return 100. * correct / total

    def _get_exploration_factor(self) -> float:
        """Compute exploration factor based on current progress and performance."""
        progress = len(self.current_indices) / self.max_budget
        performance_factor = 1.0 - (self.best_accuracy / 100.0)  # Scale based on best accuracy
        return max(0.1, (1.0 - progress) * performance_factor)

    def compute_balanced_meta_loss(self, agent_losses, scores, weights):
        """Compute meta loss with adaptive balancing."""
        # Compute individual loss terms
        performance_loss = torch.mean(weights * agent_losses)
        balance_loss = torch.var(torch.tensor([s.mean().item() for s in scores.values()]))
        entropy_loss = -torch.sum(weights * torch.log(weights + 1e-10))
        temp_loss = torch.abs(self.temperature - (1.0 + self._get_exploration_factor()))
        
        # Update moving averages
        perf_avg = self.loss_moving_avg['performance'].update(performance_loss.item())
        balance_avg = self.loss_moving_avg['balance'].update(balance_loss.item())
        entropy_avg = self.loss_moving_avg['entropy'].update(entropy_loss.item())
        temp_avg = self.loss_moving_avg['temperature'].update(temp_loss.item())
        
        # Compute relative scales
        total_avg = perf_avg + balance_avg + entropy_avg + temp_avg
        if total_avg > 0:
            target_scales = {
                'performance': perf_avg / total_avg,
                'balance': balance_avg / total_avg,
                'entropy': entropy_avg / total_avg,
                'temperature': temp_avg / total_avg
            }
            
            # Update loss scales with gradient
            for name, scale in self.loss_scales.items():
                scale.data = scale.data * 0.95 + 0.05 * target_scales[name]
        
        # Compute balanced meta loss
        meta_loss = (
            self.loss_scales['performance'] * performance_loss +
            self.loss_scales['balance'] * balance_loss +
            self.loss_scales['entropy'] * entropy_loss +
            self.loss_scales['temperature'] * temp_loss
        )
        
        # Store current scales for logging
        scales_dict = {f'{k}_scale': v.item() for k, v in self.loss_scales.items()}
        
        return meta_loss, scales_dict

    def _greedy_selection(self, agent_outputs: Dict[str, torch.Tensor], remaining_budget: int) -> torch.Tensor:
        """Greedy selection strategy focusing on highest confidence samples."""
        # Get batch indices relative to the full dataset
        batch_start = self.current_batch_offset
        batch_size = list(agent_outputs.values())[0].size(0)
        global_indices = torch.arange(batch_start, batch_start + batch_size, device=self._device)
        
        # Filter out indices that are already selected
        mask = torch.tensor([idx not in self.current_indices for idx in global_indices], 
                          device=self._device)
        available_indices = global_indices[mask]
        
        if len(available_indices) == 0:
            return torch.tensor([], device=self._device, dtype=torch.long)

        # Combine agent outputs with learned weights
        weights = F.softmax(self.coordination_weights, dim=0)
        weighted_scores = torch.zeros_like(list(agent_outputs.values())[0].squeeze())[mask]
        
        for idx, (name, scores) in enumerate(agent_outputs.items()):
            weighted_scores += scores.squeeze()[mask] * weights[idx]
        
        # Select top-k samples within remaining budget
        k = min(remaining_budget, len(available_indices))
        if k <= 0:
            return torch.tensor([], device=self._device, dtype=torch.long)
        
        _, top_indices = torch.topk(weighted_scores, k=k)
        selected = available_indices[top_indices]
        
        # Update current indices
        self.current_indices.update(selected.cpu().numpy())
        
        return selected

    def _select_samples(self, agent_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Select samples with progressive budget loading."""
        current_epoch = self.current_batch_offset // len(self.test_loader)
        
        # Calculate target samples for current epoch
        progress_ratio = (current_epoch + 1) / self.total_epochs
        target_samples = min(
            self.current_budget_cap,
            int(self.max_budget * progress_ratio)
        )
        
        # Calculate remaining samples to select
        remaining_samples = target_samples - len(self.current_indices)
        
        if remaining_samples <= 0:
            return torch.tensor([], device=self._device, dtype=torch.long)
            
        # Get weighted scores
        first_scores = list(agent_outputs.values())[0]
        weighted_scores = torch.zeros_like(first_scores.squeeze(), device=self._device)
        weights = F.softmax(self.coordination_weights / self.temperature, dim=0)
        
        for idx, (name, scores) in enumerate(agent_outputs.items()):
            weighted_scores += scores.squeeze() * weights[idx]
        
        # Calculate samples to select in this batch
        batches_remaining = len(self.test_loader) - (self.current_batch_offset % len(self.test_loader))
        samples_per_batch = max(1, remaining_samples // max(1, batches_remaining))
        n_select = min(
            samples_per_batch,
            remaining_samples,
            weighted_scores.size(0)  # Add this to prevent index out of range
        )
        
        # Ensure n_select is valid
        if n_select <= 0:
            return torch.tensor([], device=self._device, dtype=torch.long)
        
        # Select top samples
        _, local_indices = torch.topk(weighted_scores, k=min(n_select, weighted_scores.size(0)))
        global_indices = local_indices + self.current_batch_offset
        
        # Update current indices
        self.current_indices.update(global_indices.cpu().numpy())
        
        # Log selection info if in debug mode
        if hasattr(self, 'debug') and self.debug:
            print(f"Current epoch: {current_epoch}")
            print(f"Target samples: {target_samples}")
            print(f"Remaining samples: {remaining_samples}")
            print(f"Samples per batch: {samples_per_batch}")
            print(f"Selected this batch: {n_select}")
            print(f"Total selected: {len(self.current_indices)}")
        
        return local_indices

    def meta_step(self, batch: Tuple[torch.Tensor, torch.Tensor], global_step: int) -> Optional[CIFARMetaState]:
        """Execute meta-learning step with budget management."""
        # Update batch offset and budget cap
        self.current_batch_offset = (global_step % len(self.test_loader)) * batch[0].size(0)
        current_epoch = global_step // len(self.test_loader)
        self.update_budget_cap(current_epoch)
        
        # Check if we've reached current budget cap
        if len(self.current_indices) >= self.current_budget_cap:
            return None

        # Zero all gradients
        self.meta_optimizer.zero_grad()
        for optimizer in self.agent_optimizers.values():
            optimizer.zero_grad()
        self.classifier_optimizer.zero_grad()

        # Compute adaptive weights and losses
        weight_info = self.compute_adaptive_weights(batch)
        scores = weight_info['scores']
        agent_outputs = weight_info['agent_outputs']
        agent_specific_losses = weight_info['agent_losses']

        # Stack losses for meta-optimization
        agent_losses = torch.stack([loss for loss in agent_specific_losses.values()])
        
        # Temperature-scaled weight computation
        weights = F.softmax(self.coordination_weights / self.temperature, dim=0)

        # Compute balanced meta loss
        meta_loss, loss_scales = self.compute_balanced_meta_loss(agent_losses, scores, weights)

        # Update all components if in training mode
        if self.training:
            meta_loss.backward()
            self.meta_optimizer.step()
            for optimizer in self.agent_optimizers.values():
                optimizer.step()
            self.classifier_optimizer.step()

        # Select samples using weighted scores
        selected_indices = self._select_samples(agent_outputs)
        
        # Store metrics
        metrics = {
            'meta_loss': meta_loss.item(),
            'temperature': self.temperature.item(),
            'exploration_factor': self._get_exploration_factor(),
            'total_selected': len(self.current_indices),
            'selected_this_batch': len(selected_indices),
            'progress': len(self.current_indices) / self.max_budget * 100,
            'current_budget_cap': self.current_budget_cap
        }
        
        # Add loss scales to metrics
        metrics.update(loss_scales)
        
        # Add agent-specific metrics
        for name, weight in zip(self.agents.keys(), weights):
            metrics[f'{name}_weight'] = weight.item()
            metrics[f'{name}_loss'] = agent_specific_losses[name].item()
        
        # Append to metrics history
        self.metrics_history['meta_metrics'].append(metrics)

        return CIFARMetaState(
            agent_losses=agent_losses.detach(),
            agent_accuracies=torch.tensor([self._compute_accuracy(scores[name], weight_info['labels']) for name in self.agents.keys()], device=self._device),
            meta_loss=meta_loss.detach(),
            selected_samples=selected_indices,
            metrics=metrics
        )

    def _compute_agent_loss(self, name: str, scores: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if name == 'uncertainty':
            return self._uncertainty_loss(scores, batch)
        elif name == 'diversity':
            return self._diversity_loss(scores, batch)
        elif name == 'class_balance':
            return self._class_balance_loss(scores, batch)
        else:  # boundary
            return self._boundary_loss(scores, batch)

    def _uncertainty_loss(self, scores: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            images = batch[0]
            features = self.feature_extractor(images)
            features = features.squeeze(-1).squeeze(-1)
            outputs = self.classifier(features)
            probs = F.softmax(outputs / self.temperature, dim=1)
            uncertainties = 1 - torch.max(probs, dim=1)[0]
            uncertainties_reshaped = uncertainties.reshape(-1, 1)
        return F.mse_loss(scores, uncertainties_reshaped) + 0.1 * torch.mean(scores)  # Added regularization

    def _diversity_loss(self, scores: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        images = batch[0]
        features = self.feature_extractor(images)
        features = features.squeeze(-1).squeeze(-1)
        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.matmul(features, features.t())
        scores_reshaped = scores.reshape(-1, 1)
        diversity_loss = -torch.mean(scores_reshaped * (1 - similarity_matrix))
        entropy_reg = -torch.mean(scores * torch.log(scores + 1e-10))  # Added entropy regularization
        return diversity_loss + 0.1 * entropy_reg

    def _boundary_loss(self, scores: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            images = batch[0]
            features = self.feature_extractor(images)
            features = features.squeeze(-1).squeeze(-1)
            outputs = self.classifier(features)
            top2_values, _ = torch.topk(outputs, k=2, dim=1)
            margins = top2_values[:, 0] - top2_values[:, 1]
            # Create new tensor instead of modifying in-place
            margins_reshaped = margins.reshape(-1, 1)
        return nn.MSELoss()(scores, torch.sigmoid(-margins_reshaped))

    def _class_balance_loss(self, scores: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        _, labels = batch
        class_counts = torch.bincount(labels, minlength=10).float()
        class_weights = 1.0 / (class_counts + 1)
        class_weights = F.softmax(class_weights / self.temperature, dim=0)
        balance_loss = -torch.mean(scores * class_weights[labels])
        variance_reg = torch.var(scores)  # Added variance regularization
        return balance_loss + 0.1 * variance_reg

    def _compute_weighted_scores(self, agent_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted scores for selection."""
        weights = F.softmax(self.coordination_weights / self.temperature, dim=0)
        weighted_scores = torch.zeros_like(list(agent_outputs.values())[0].squeeze())
        
        for idx, (name, scores) in enumerate(agent_outputs.items()):
            weighted_scores += scores.squeeze() * weights[idx]
        
        return weighted_scores

    def check_performance_and_adapt(self, current_accuracy: float) -> None:
        """Monitor performance and adapt strategy if needed."""
        self.performance_history['accuracies'].append(current_accuracy)
        
        # Update best accuracy and check for improvement
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        # Switch to greedy if no improvement for several epochs
        if self.epochs_without_improvement >= self.patience:
            if not self.use_greedy:
                print(f"\nSwitching to greedy selection after {self.patience} epochs without improvement")
                print(f"Best accuracy: {self.best_accuracy:.2f}%")
                self.use_greedy = True
        
        # Store monitoring metrics
        self.performance_history['epochs_since_improvement'] = self.epochs_without_improvement
        self.performance_history['best_accuracy'] = self.best_accuracy

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model and adapt selection strategy."""
        self.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self._device)
                labels = labels.to(self._device)
                
                features = self.feature_extractor(images)
                features = features.squeeze(-1).squeeze(-1)
                outputs = self.classifier(features)
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        
        # Check performance and adapt strategy
        self.check_performance_and_adapt(accuracy)
        
        metrics = {
            'test_accuracy': accuracy,
            'best_accuracy': self.best_accuracy,
            'epochs_without_improvement': self.epochs_without_improvement,
            'using_greedy': self.use_greedy
        }
        
        self.metrics_history['test_metrics'].append(metrics)
        
        # Generate analysis plots
        self.plot_all_metrics()
        self.plot_performance_history()
        
        return metrics

    def plot_performance_history(self):
        """Plot performance history and strategy adaptation."""
        plt.figure(figsize=(12, 6))
        
        # Plot accuracies
        plt.plot(self.performance_history['accuracies'], label='Accuracy')
        plt.axhline(y=self.best_accuracy, color='r', linestyle='--', label='Best Accuracy')
        
        # Mark strategy switch points
        switch_points = [i for i, acc in enumerate(self.performance_history['accuracies']) 
                        if i >= self.patience and 
                        all(acc >= x for x in self.performance_history['accuracies'][i-self.patience:i])]
        
        if switch_points:
            plt.vlines(switch_points, 0, 100, colors='g', linestyles=':', label='Strategy Switch')
        
        plt.title('Performance History and Strategy Adaptation')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, 'performance_history.png'))
        plt.close()

    def plot_all_metrics(self):
        """Generate comprehensive visualization of all metrics."""
        plot_dir = os.path.join(self.plot_dir, 'detailed_metrics')
        os.makedirs(plot_dir, exist_ok=True)
        
        # 1. Training Dynamics
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        self._plot_accuracy_history()
        
        plt.subplot(2, 2, 2)
        self._plot_loss_history()
        
        plt.subplot(2, 2, 3)
        self._plot_selection_progress()
        
        plt.subplot(2, 2, 4)
        self._plot_exploration_temperature()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'training_dynamics.png'))
        plt.close()
        
        # 2. Agent Performance
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        self._plot_agent_weights()
        
        plt.subplot(2, 2, 2)
        self._plot_agent_losses()
        
        plt.subplot(2, 2, 3)
        self._plot_agent_scores()
        
        plt.subplot(2, 2, 4)
        self._plot_agent_selection_rates()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'agent_performance.png'))
        plt.close()
        
        # 3. Selection Analysis
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        self._plot_class_distribution()
        
        plt.subplot(2, 2, 2)
        self._plot_selection_timing()
        
        plt.subplot(2, 2, 3)
        self._plot_budget_utilization()
        
        plt.subplot(2, 2, 4)
        self._plot_strategy_switches()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'selection_analysis.png'))
        plt.close()
        
        # 4. Loss Components
        plt.figure(figsize=(15, 10))
        self._plot_loss_components()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'loss_components.png'))
        plt.close()

    def _plot_accuracy_history(self):
        """Plot training and test accuracy."""
        accuracies = [m['test_accuracy'] for m in self.metrics_history['test_metrics']]
        plt.plot(accuracies, label='Test Accuracy')
        plt.axhline(y=self.best_accuracy, color='r', linestyle='--', label='Best Accuracy')
        plt.title('Accuracy History')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()

    def _plot_loss_history(self):
        """Plot various loss components."""
        meta_losses = [m['meta_loss'] for m in self.metrics_history['meta_metrics']]
        plt.plot(meta_losses, label='Meta Loss')
        plt.title('Loss History')
        plt.xlabel('Steps')
        plt.ylabel('Loss Value')
        plt.legend()

    def _plot_selection_progress(self):
        """Plot sample selection progress."""
        selected = [m['total_selected'] for m in self.metrics_history['meta_metrics']]
        plt.plot(selected, label='Selected Samples')
        plt.axhline(y=self.max_budget, color='r', linestyle='--', label='Budget Limit')
        plt.title('Selection Progress')
        plt.xlabel('Steps')
        plt.ylabel('Number of Samples')
        plt.legend()

    def _plot_exploration_temperature(self):
        """Plot temperature and exploration factor."""
        temps = [m['temperature'] for m in self.metrics_history['meta_metrics']]
        exp_factors = [m['exploration_factor'] for m in self.metrics_history['meta_metrics']]
        plt.plot(temps, label='Temperature')
        plt.plot(exp_factors, label='Exploration Factor')
        plt.title('Exploration Parameters')
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.legend()

    def _plot_agent_weights(self):
        """Plot evolution of agent weights."""
        for name in self.agents.keys():
            weights = [m[f'{name}_weight'] for m in self.metrics_history['meta_metrics']]
            plt.plot(weights, label=name)
        plt.title('Agent Weights Evolution')
        plt.xlabel('Steps')
        plt.ylabel('Weight Value')
        plt.legend()

    def _plot_agent_losses(self):
        """Plot agent-specific losses."""
        for name in self.agents.keys():
            losses = [m[f'{name}_loss'] for m in self.metrics_history['meta_metrics']]
            plt.plot(losses, label=f'{name} Loss')
        plt.title('Agent Losses')
        plt.xlabel('Steps')
        plt.ylabel('Loss Value')
        plt.legend()

    def _plot_agent_scores(self):
        """Plot agent selection scores."""
        for name, scores in self.metrics_history['agent_scores'].items():
            plt.plot(scores, label=name)
        plt.title('Agent Selection Scores')
        plt.xlabel('Steps')
        plt.ylabel('Score Value')
        plt.legend()

    def _plot_agent_selection_rates(self):
        """Plot agent selection rates."""
        selection_rates = defaultdict(list)
        for metrics in self.metrics_history['meta_metrics']:
            total = sum(metrics[f'{name}_weight'] for name in self.agents.keys())
            for name in self.agents.keys():
                selection_rates[name].append(metrics[f'{name}_weight'] / total)
        
        for name, rates in selection_rates.items():
            plt.plot(rates, label=name)
        plt.title('Agent Selection Rates')
        plt.xlabel('Steps')
        plt.ylabel('Selection Rate')
        plt.legend()

    def _plot_class_distribution(self):
        """Plot distribution of selected samples across classes."""
        if len(self.current_indices) > 0:
            selected_labels = [self.dataset[idx][1] for idx in self.current_indices]
            plt.hist(selected_labels, bins=10, alpha=0.7)
            plt.title('Class Distribution of Selected Samples')
            plt.xlabel('Class')
            plt.ylabel('Count')

    def _plot_selection_timing(self):
        """Plot sample selection timing."""
        selection_rates = np.diff([m['total_selected'] for m in self.metrics_history['meta_metrics']])
        plt.plot(selection_rates)
        plt.title('Sample Selection Rate')
        plt.xlabel('Steps')
        plt.ylabel('Samples/Step')

    def _plot_budget_utilization(self):
        """Plot budget utilization over time."""
        utilization = [m['total_selected']/self.max_budget * 100 
                      for m in self.metrics_history['meta_metrics']]
        plt.plot(utilization)
        plt.title('Budget Utilization')
        plt.xlabel('Steps')
        plt.ylabel('Utilization (%)')

    def _plot_strategy_switches(self):
        """Plot strategy adaptation points."""
        strategy = ['Greedy' if m.get('using_greedy', False) else 'Adaptive' 
                   for m in self.metrics_history['meta_metrics']]
        changes = [i for i in range(1, len(strategy)) if strategy[i] != strategy[i-1]]
        
        plt.plot(range(len(strategy)), [1 if s == 'Greedy' else 0 for s in strategy])
        plt.yticks([0, 1], ['Adaptive', 'Greedy'])
        plt.title('Selection Strategy')
        plt.xlabel('Steps')
        
        for change in changes:
            plt.axvline(x=change, color='r', linestyle='--', alpha=0.5)

    def _plot_loss_components(self):
        """Plot individual loss components and their scales."""
        for name, scale_history in self.loss_scales.items():
            scales = [m[f'{name}_scale'] for m in self.metrics_history['meta_metrics']]
            plt.plot(scales, label=f'{name} Scale')
        plt.title('Loss Component Scales')
        plt.xlabel('Steps')
        plt.ylabel('Scale Value')
        plt.legend()

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], path: str):
        """Save model checkpoint with all necessary state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'classifier_optimizer_state_dict': self.classifier_optimizer.state_dict(),
            'agent_optimizers_state_dict': {
                name: opt.state_dict() 
                for name, opt in self.agent_optimizers.items()
            },
            'metrics': metrics,
            'current_indices': list(self.current_indices),
            'current_batch_offset': self.current_batch_offset,
            'best_accuracy': self.best_accuracy,
            'epochs_without_improvement': self.epochs_without_improvement,
            'use_greedy': self.use_greedy,
            'metrics_history': self.metrics_history,
            'loss_scales': {k: v.data for k, v in self.loss_scales.items()},
            'loss_moving_avg': {
                k: v.value for k, v in self.loss_moving_avg.items() if v.value is not None
            },
            'temperature': self.temperature.data,
            'coordination_weights': self.coordination_weights.data,
            'agent_importances': self.agent_importances.data
        }
        
        # Save checkpoint
        try:
            torch.save(checkpoint, path)
            print(f"Checkpoint saved successfully to {path}")
        except Exception as e:
            print(f"Warning: Failed to save checkpoint to {path}: {str(e)}")
            # Try to save to a backup location
            backup_path = os.path.join(os.path.dirname(path), 'backup_checkpoint.pt')
            try:
                torch.save(checkpoint, backup_path)
                print(f"Backup checkpoint saved to {backup_path}")
            except Exception as e2:
                print(f"Critical: Failed to save backup checkpoint: {str(e2)}")

    def load_checkpoint(self, path: str) -> Dict[str, float]:
        """Load model checkpoint and restore state."""
        try:
            checkpoint = torch.load(path, map_location=self._device)
            
            # Load model states
            self.load_state_dict(checkpoint['model_state_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
            
            # Load optimizer states
            self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
            self.classifier_optimizer.load_state_dict(checkpoint['classifier_optimizer_state_dict'])
            
            # Load agent optimizer states
            for name, state_dict in checkpoint['agent_optimizers_state_dict'].items():
                if name in self.agent_optimizers:
                    self.agent_optimizers[name].load_state_dict(state_dict)
            
            # Load training state
            self.current_indices = set(checkpoint['current_indices'])
            self.current_batch_offset = checkpoint.get('current_batch_offset', 0)
            self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
            self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            self.use_greedy = checkpoint.get('use_greedy', False)
            
            # Load metrics history
            if 'metrics_history' in checkpoint:
                self.metrics_history = checkpoint['metrics_history']
            
            # Load loss scales and moving averages
            if 'loss_scales' in checkpoint:
                for k, v in checkpoint['loss_scales'].items():
                    self.loss_scales[k].data.copy_(v.to(self._device))
            
            if 'loss_moving_avg' in checkpoint:
                for k, v in checkpoint['loss_moving_avg'].items():
                    if v is not None:
                        self.loss_moving_avg[k].value = v
            
            # Load other parameters
            if 'temperature' in checkpoint:
                self.temperature.data.copy_(checkpoint['temperature'].to(self._device))
            if 'coordination_weights' in checkpoint:
                self.coordination_weights.data.copy_(checkpoint['coordination_weights'].to(self._device))
            if 'agent_importances' in checkpoint:
                self.agent_importances.data.copy_(checkpoint['agent_importances'].to(self._device))
            
            print(f"Checkpoint loaded successfully from {path}")
            return checkpoint.get('metrics', {})
            
        except Exception as e:
            print(f"Error loading checkpoint from {path}: {str(e)}")
            return {}

    def get_checkpoint_path(self, epoch: int, is_best: bool = False) -> str:
        """Generate checkpoint path."""
        if is_best:
            return os.path.join(self.checkpoint_dir, 'best_model.pt')
        return os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')

def train_meta_agent(meta_agent: CIFARBiLevelMetaAgent, epochs: int = 100, device: torch.device = None) -> Dict:
    """Train the meta-agent with progressive budget loading."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create train loader here
    train_loader = DataLoader(
        meta_agent.dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    meta_agent.total_epochs = epochs
    meta_agent.train_loader = train_loader
    experiment_metrics = defaultdict(list)
    best_accuracy = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        meta_agent.update_budget_cap(epoch)
        
        # First phase: Sample selection
        pbar = tqdm(train_loader, desc=f'Budget {meta_agent.max_budget} - Epoch {epoch+1}/{epochs} (Selection)')
        epoch_metrics = defaultdict(list)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            global_step = epoch * len(train_loader) + batch_idx
            
            state = meta_agent.meta_step((inputs, targets), global_step)
            
            if state is not None:
                epoch_metrics['meta_loss'].append(state.meta_loss.item())
                pbar.set_postfix({
                    'selected': f"{len(meta_agent.current_indices)}/{meta_agent.max_budget}",
                    'progress': f"{len(meta_agent.current_indices)/meta_agent.max_budget*100:.1f}%",
                    'loss': f"{state.meta_loss.item():.4f}"
                })
        
        # Second phase: Train classifier on selected samples
        if len(meta_agent.current_indices) > 0:
            # Create a loader for selected samples
            selected_dataset = Subset(meta_agent.dataset, list(meta_agent.current_indices))
            selected_loader = DataLoader(
                selected_dataset,
                batch_size=128,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            # Train classifier on selected samples
            pbar = tqdm(selected_loader, desc=f'Budget {meta_agent.max_budget} - Epoch {epoch+1}/{epochs} (Training)')
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero gradients
                meta_agent.classifier_optimizer.zero_grad()
                
                # Forward pass
                features = meta_agent.feature_extractor(inputs)
                features = features.squeeze(-1).squeeze(-1)
                outputs = meta_agent.classifier(features)
                loss = meta_agent.classifier_criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                meta_agent.classifier_optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': f"{train_loss/(batch_idx+1):.4f}",
                    'acc': f"{100.*correct/total:.2f}%"
                })
        
        # Evaluate and update best accuracy
        test_metrics = meta_agent.evaluate()
        current_accuracy = test_metrics.get('test_accuracy', 0.0)
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            epochs_without_improvement = 0
            best_checkpoint_path = os.path.join(meta_agent.checkpoint_dir, 'best_model.pt')
            meta_agent.save_checkpoint(epoch+1, test_metrics, best_checkpoint_path)
        else:
            epochs_without_improvement += 1
            
            if epochs_without_improvement >= meta_agent.patience:
                print(f"\nSwitching to greedy selection after {epochs_without_improvement} epochs without improvement")
                print(f"Best accuracy: {best_accuracy:.2f}%")
        
        # Store epoch metrics and print summary
        for k, v in epoch_metrics.items():
            if v:
                experiment_metrics[k].append(np.mean(v))
        experiment_metrics['test_accuracy'].append(current_accuracy)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Selected Samples: {len(meta_agent.current_indices)}/{meta_agent.max_budget}")
        print(f"Current Budget Cap: {meta_agent.current_budget_cap}")
        print(f"Train Accuracy: {100.*correct/total:.2f}%")
        print(f"Test Accuracy: {current_accuracy:.2f}%")
        print(f"Best Accuracy: {best_accuracy:.2f}%")
        
    return experiment_metrics

def run_multi_budget_experiment(
    budget_percentages: List[float] = [0.1, 0.2, 0.3, 0.4],
    epochs: int = 100,
    runs: int = 5
) -> Tuple[Dict, Dict]:
    """Run experiments with multiple budget percentages."""
    results = {}
    summary = {}
    
    for budget in budget_percentages:
        print(f"\n{'='*80}")
        print(f"Starting experiments with budget {budget*100}%")
        print(f"{'='*80}")
        
        budget_results = {
            'test_accuracy': [],
            'train_accuracy': [],
            'meta_loss': [],
            'selected_samples': []
        }
        
        for run in range(runs):
            print(f"\nRun {run+1}/{runs}")
            print("-" * 40)
            
            # Initialize meta agent
            meta_agent = CIFARBiLevelMetaAgent(budget_percent=budget)
            
            # Train the agent
            run_metrics = train_meta_agent(meta_agent, epochs=epochs)
            
            # Store final metrics for this run
            budget_results['test_accuracy'].append(run_metrics['test_accuracy'])
            budget_results['train_accuracy'].append(run_metrics.get('train_accuracy', []))
            budget_results['meta_loss'].append(run_metrics.get('meta_loss', []))
            budget_results['selected_samples'].append(run_metrics.get('selected_samples', []))
        
        # Compute summary statistics
        results[f'budget_{int(budget*100)}'] = {
            'test_accuracy': {
                'mean': np.mean([max(acc) for acc in budget_results['test_accuracy']]),
                'std': np.std([max(acc) for acc in budget_results['test_accuracy']]),
                'final_mean': np.mean([acc[-1] for acc in budget_results['test_accuracy']]),
                'final_std': np.std([acc[-1] for acc in budget_results['test_accuracy']])
            },
            'train_accuracy': {
                'mean': np.mean([max(acc) if acc else 0 for acc in budget_results['train_accuracy']]),
                'std': np.std([max(acc) if acc else 0 for acc in budget_results['train_accuracy']])
            },
            'convergence': {
                'epochs_to_best': np.mean([np.argmax(acc) for acc in budget_results['test_accuracy']]),
                'epochs_total': epochs
            }
        }
        
        # Update summary
        summary[f'budget_{int(budget*100)}'] = {
            'best_accuracy_mean': results[f'budget_{int(budget*100)}']['test_accuracy']['mean'],
            'best_accuracy_std': results[f'budget_{int(budget*100)}']['test_accuracy']['std'],
            'final_accuracy_mean': results[f'budget_{int(budget*100)}']['test_accuracy']['final_mean'],
            'final_accuracy_std': results[f'budget_{int(budget*100)}']['test_accuracy']['final_std'],
            'train_accuracy_mean': results[f'budget_{int(budget*100)}']['train_accuracy']['mean'],
            'train_accuracy_std': results[f'budget_{int(budget*100)}']['train_accuracy']['std'],
            'epochs_to_best': results[f'budget_{int(budget*100)}']['convergence']['epochs_to_best']
        }
        
        # Print current budget summary
        print(f"\nBudget {budget*100}% Summary:")
        print(f"Best Test Accuracy: {summary[f'budget_{int(budget*100)}']['best_accuracy_mean']:.2f}% ± {summary[f'budget_{int(budget*100)}']['best_accuracy_std']:.2f}%")
        print(f"Final Test Accuracy: {summary[f'budget_{int(budget*100)}']['final_accuracy_mean']:.2f}% ± {summary[f'budget_{int(budget*100)}']['final_accuracy_std']:.2f}%")
        print(f"Best Train Accuracy: {summary[f'budget_{int(budget*100)}']['train_accuracy_mean']:.2f}% ± {summary[f'budget_{int(budget*100)}']['train_accuracy_std']:.2f}%")
        print(f"Average Epochs to Best: {summary[f'budget_{int(budget*100)}']['epochs_to_best']:.1f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('experiments', f'multi_budget_run_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    return results, summary

if __name__ == "__main__":
    print("Starting multi-budget CIFAR-10 experiments with multiple runs...")
    
    # Define budget percentages to test
    budget_percentages = [0.1, 0.2, 0.3, 0.4]  # 10%, 20%, 30%, 40%
    
    # Run experiments with 5 runs per budget
    results, summary = run_multi_budget_experiment(
        budget_percentages=budget_percentages,
        epochs=100,
        runs=5
    )
    
    print("\nAll experiments completed!")