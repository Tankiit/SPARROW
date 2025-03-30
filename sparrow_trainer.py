import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
import csv
from typing import Dict, List, Tuple, Set, Optional
# Add TensorBoard imports
from torch.utils.tensorboard import SummaryWriter
# Add tqdm for progress bars
from tqdm import tqdm, trange

# Import the meta-agent for strategy selection
from meta_agent import SparrowMetaController, LearningState


class SparrowCoresetSelector:
    """
    SPARROW: Strategic Policy-based Active Re-weighting Workflow
    This class implements the coreset selection algorithm described in the paper
    """
    def __init__(self, 
                 full_dataset,
                 budget: int, 
                 initial_samples: int = 1000,
                 device=None):
        """
        Initialize the SPARROW coreset selector
        
        Args:
            full_dataset: The full dataset
            budget: Maximum number of samples to select
            initial_samples: Number of initial samples to randomly select
            device: Device to use for computation
        """
        self.full_dataset = full_dataset
        self.budget = budget
        self.initial_samples = initial_samples
        
        # Enhanced device selection logic with MPS support
        if device is None:
            # Try CUDA first
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            # Then try MPS (Apple Silicon)
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            # Fall back to CPU
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize strategy weights
        self.strategies = ["S_U", "S_D", "S_C", "S_B"]  # Uncertainty, Diversity, Class Balance, Boundary
        self.weights = {s: 0.25 for s in self.strategies}  # Equal initial weights
        
        # Temperature parameter for exploration vs exploitation
        self.temperature = 0.5
        
        # Initialize the selected indices with random samples
        print(f"Initializing with {initial_samples} samples...")
        self.current_indices = self._initial_sampling(initial_samples)
        
        # Initialize meta_controller to None
        self.meta_controller = None
        
        # Try to set up meta-controller if available
        try:
            # Check if the meta_agent module and SparrowMetaController class are available
            from meta_agent import SparrowMetaController
            
            # Create meta-controller instance
            self.meta_controller = SparrowMetaController(
                strategies=self.strategies,
                initial_temperature=0.5,
                total_epochs=200  # Assuming 200 epochs
            )
            print("Meta-controller initialized successfully")
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Meta-controller not available: {e}")
            print("Using basic strategy weighting")
        
        # Performance tracking
        self.performance_history = []
        self.weight_history = []
        
        # Current epoch tracking
        self.current_epoch = 0
        
        # Save initial weights
        self._save_weights()
    
    def _initial_sampling(self, n_samples: int) -> np.ndarray:
        """
        Perform initial sampling to select a diverse set of samples
        
        Args:
            n_samples: Number of samples to select
            
        Returns:
            Array of selected indices
        """
        # Use stratified sampling to ensure class balance
        class_indices = [[] for _ in range(10)]  # CIFAR-10 has 10 classes
        
        print("Scanning dataset for class distribution...")
        # Use tqdm to show progress
        for i in tqdm(range(len(self.full_dataset)), desc="Indexing dataset"):
            _, label = self.full_dataset[i]
            class_indices[label].append(i)
        
        # Select equal number of samples from each class
        samples_per_class = n_samples // 10
        selected_indices = []
        
        print(f"Selecting ~{samples_per_class} samples per class...")
        # Use tqdm for class selection
        for class_idx in tqdm(range(10), desc="Selecting samples by class"):
            indices = np.random.choice(class_indices[class_idx], 
                                    min(samples_per_class, len(class_indices[class_idx])), 
                                    replace=False)
            selected_indices.extend(indices)
        
        # If we don't have enough samples, fill the rest randomly
        if len(selected_indices) < n_samples:
            remaining = n_samples - len(selected_indices)
            print(f"Need {remaining} more samples to reach target, selecting randomly...")
            all_indices = set(range(len(self.full_dataset)))
            remaining_indices = list(all_indices - set(selected_indices))
            additional_indices = np.random.choice(remaining_indices, remaining, replace=False)
            selected_indices.extend(additional_indices)
        
        print(f"Selected {len(selected_indices)} initial samples")
        return np.array(selected_indices)
    
    def _save_weights(self):
        """Save the current weights to history"""
        self.weight_history.append(self.weights.copy())
    
    def compute_uncertainty_scores(self, model, batch_size=128) -> np.ndarray:
        """
        Compute uncertainty scores for all samples in the dataset
        
        Args:
            model: The trained model
            batch_size: Batch size for inference
            
        Returns:
            Array of uncertainty scores for all samples
        """
        model.eval()
        uncertainties = np.zeros(len(self.full_dataset))
        
        dataloader = DataLoader(
            self.full_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        with torch.no_grad():
            start_idx = 0
            # Add tqdm progress bar
            pbar = tqdm(dataloader, desc="Computing uncertainty scores", leave=False)
            for inputs, _ in pbar:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Calculate entropy as uncertainty measure
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
                
                batch_size = inputs.size(0)
                uncertainties[start_idx:start_idx+batch_size] = entropy.cpu().numpy()
                start_idx += batch_size
                
                # Update progress bar
                pbar.set_postfix({'processed': f'{start_idx}/{len(self.full_dataset)}'})
        
        return uncertainties
    
    def compute_diversity_scores(self, model, batch_size=128) -> np.ndarray:
        """
        Compute diversity scores based on feature space coverage
        
        Args:
            model: The trained model
            batch_size: Batch size for inference
            
        Returns:
            Array of diversity scores for all samples
        """
        model.eval()
        # Create a feature extractor from the model
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor.eval()
        
        # Get features for currently selected samples
        selected_features = []
        selected_subset = Subset(self.full_dataset, self.current_indices)
        selected_loader = DataLoader(
            selected_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        with torch.no_grad():
            # Add tqdm progress bar for feature extraction
            pbar_extract = tqdm(selected_loader, desc="Extracting features for selected samples", leave=False)
            for inputs, _ in pbar_extract:
                inputs = inputs.to(self.device)
                features = feature_extractor(inputs)
                selected_features.append(features.squeeze().cpu().numpy())
        
        if len(selected_features) > 0:
            selected_features = np.vstack(selected_features)
        else:
            selected_features = np.array([]).reshape(0, 512)  # ResNet18 feature dim is 512
        
        # Compute diversity scores for all samples
        diversity_scores = np.zeros(len(self.full_dataset))
        
        dataloader = DataLoader(
            self.full_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        with torch.no_grad():
            start_idx = 0
            # Add tqdm progress bar for diversity computation
            pbar_diversity = tqdm(dataloader, desc="Computing diversity scores", leave=False)
            for inputs, _ in pbar_diversity:
                inputs = inputs.to(self.device)
                features = feature_extractor(inputs)
                batch_features = features.squeeze().cpu().numpy()
                
                batch_size = inputs.size(0)
                
                if len(selected_features) > 0:
                    # For each sample, compute minimum distance to any selected sample
                    for i in range(batch_size):
                        if batch_features.ndim == 1:  # Handle batch of size 1
                            sample_features = batch_features.reshape(1, -1)
                        else:
                            sample_features = batch_features[i].reshape(1, -1)
                        
                        # Compute distances to all selected samples
                        distances = np.linalg.norm(selected_features - sample_features, axis=1)
                        min_distance = np.min(distances) if len(distances) > 0 else float('inf')
                        diversity_scores[start_idx + i] = min_distance
                else:
                    # If no samples are selected yet, assign high diversity to all
                    diversity_scores[start_idx:start_idx+batch_size] = 1.0
                
                start_idx += batch_size
                # Update progress bar
                pbar_diversity.set_postfix({'processed': f'{start_idx}/{len(self.full_dataset)}'})
        
        return diversity_scores
    
    def compute_class_balance_scores(self) -> np.ndarray:
        """
        Compute class balance scores to address imbalance
        
        Returns:
            Array of class balance scores for all samples
        """
        # Count current class distribution
        class_counts = Counter([self.full_dataset[i][1] for i in self.current_indices])
        
        # Compute inverse frequency for each class (higher score for underrepresented classes)
        total_samples = len(self.current_indices)
        class_weights = {c: total_samples / (count + 1) for c, count in class_counts.items()}
        
        # Normalize to [0, 1]
        max_weight = max(class_weights.values()) if class_weights else 1.0
        class_weights = {c: w / max_weight for c, w in class_weights.items()}
        
        # Fill in any missing classes
        for c in range(10):  # CIFAR-10 has 10 classes
            if c not in class_weights:
                class_weights[c] = 1.0  # Maximum weight for missing classes
        
        # Assign scores based on class
        balance_scores = np.zeros(len(self.full_dataset))
        for i in range(len(self.full_dataset)):
            _, label = self.full_dataset[i]
            balance_scores[i] = class_weights[label]
        
        return balance_scores
    
    def compute_boundary_scores(self, model, batch_size=128) -> np.ndarray:
        """
        Compute boundary scores for samples near decision boundaries
        
        Args:
            model: The trained model
            batch_size: Batch size for inference
            
        Returns:
            Array of boundary scores for all samples
        """
        model.eval()
        boundary_scores = np.zeros(len(self.full_dataset))
        
        dataloader = DataLoader(
            self.full_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        with torch.no_grad():
            start_idx = 0
            # Add tqdm progress bar
            pbar = tqdm(dataloader, desc="Computing boundary scores", leave=False)
            for inputs, _ in pbar:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get top-2 probabilities
                top_values, _ = torch.topk(probabilities, k=2, dim=1)
                
                # Boundary score = 1 - (p_max - p_second)
                margin = top_values[:, 0] - top_values[:, 1]
                score = 1.0 - margin
                
                batch_size = inputs.size(0)
                boundary_scores[start_idx:start_idx+batch_size] = score.cpu().numpy()
                start_idx += batch_size
                
                # Update progress bar
                pbar.set_postfix({'processed': f'{start_idx}/{len(self.full_dataset)}'})
        
        return boundary_scores
    
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to zero mean and unit variance
        
        Args:
            scores: Array of scores
            
        Returns:
            Normalized scores
        """
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return np.zeros_like(scores)
        return (scores - mean) / (std + 1e-8)
    
    def select_samples(self, model, n_samples: int, current_metrics: Optional[Dict] = None) -> np.ndarray:
        """
        Select the next batch of samples using the SPARROW algorithm with meta-agent guidance
        
        Args:
            model: The trained model
            n_samples: Number of samples to select
            current_metrics: Optional dictionary of current performance metrics
            
        Returns:
            Indices of selected samples
        """
        print(f"\nSelecting {n_samples} new samples...")
        
        # Compute scores for each strategy
        with tqdm(total=4, desc="Computing selection scores", leave=False) as pbar:
            pbar.set_postfix({'strategy': 'uncertainty'})
            uncertainty_scores = self.compute_uncertainty_scores(model)
            pbar.update(1)
            
            pbar.set_postfix({'strategy': 'diversity'})
            diversity_scores = self.compute_diversity_scores(model)
            pbar.update(1)
            
            pbar.set_postfix({'strategy': 'class balance'})
            class_balance_scores = self.compute_class_balance_scores()
            pbar.update(1)
            
            pbar.set_postfix({'strategy': 'boundary'})
            boundary_scores = self.compute_boundary_scores(model)
            pbar.update(1)
        
        # Normalize scores
        uncertainty_scores = self.normalize_scores(uncertainty_scores)
        diversity_scores = self.normalize_scores(diversity_scores)
        class_balance_scores = self.normalize_scores(class_balance_scores)
        boundary_scores = self.normalize_scores(boundary_scores)
        
        # Get class distribution
        class_distribution = Counter([self.full_dataset[i][1] for i in self.current_indices])
        
        # Evaluate per-strategy performance by selecting small batches with each strategy
        print("Evaluating strategy performance...")
        strategy_performances = self._evaluate_strategy_performances(
            model, 
            {
                "S_U": uncertainty_scores,
                "S_D": diversity_scores,
                "S_C": class_balance_scores,
                "S_B": boundary_scores
            }, 
            n_eval_samples=min(100, n_samples // 4)
        )
        
        # Extract feature statistics for additional guidance
        print("Computing feature statistics...")
        feature_statistics = self._compute_feature_statistics(model)
        
        # Update learning state
        accuracy = current_metrics.get("accuracy", 0.5) if current_metrics else 0.5
        loss = current_metrics.get("loss", 1.0) if current_metrics else 1.0
        
        # Fallback to basic weighted allocation by default
        print("Using weighted allocation based on current weights...")
        total_weight = sum(self.weights.values())
        allocation = {s: int(n_samples * (w / total_weight)) for s, w in self.weights.items()}
        
        # Ensure all samples are allocated (handle rounding errors)
        allocated = sum(allocation.values())
        if allocated < n_samples:
            # Add remaining samples to the strategy with highest weight
            dominant_strategy = max(self.weights.items(), key=lambda x: x[1])[0]
            allocation[dominant_strategy] += (n_samples - allocated)
            
        # Try to use meta-controller if available
        try:
            if hasattr(self, 'meta_controller') and self.meta_controller is not None:
                print("Using meta-controller for strategic allocation...")
                state = self.meta_controller.update_state(
                    epoch=self.current_epoch,
                    accuracy=accuracy,
                    loss=loss,
                    class_distribution=class_distribution,
                    dataset_size=len(self.current_indices),
                    strategy_performances=strategy_performances,
                    feature_statistics=feature_statistics
                )
                
                # Get sample allocation from meta-controller
                meta_allocation = self.meta_controller.get_sample_allocation(n_samples, state)
                
                # Update current weights with those from meta-controller
                self.weights = self.meta_controller.get_current_weights()
                self._save_weights()
                
                # Print strategy decision explanation (useful for debugging)
                explanation = self.meta_controller.get_latest_explanation()
                print(f"Strategy selection: {explanation}")
                
                # Use the meta-controller allocation
                allocation = meta_allocation
        except (AttributeError, NameError) as e:
            print(f"Meta-controller not available or error: {e}")
            print("Using fallback allocation based on current weights...")
        
        # Print and visualize allocation
        print(f"Sample allocation: {allocation}")
        
        # Create a visual representation of allocation
        visual = "\nSample allocation: ["
        for strategy, count in allocation.items():
            if count > 0:
                # Use first letter of strategy for the visual (U, D, C, B)
                visual += strategy[2] * (count * 50 // n_samples)
        visual += "]\n"
        print(visual)
        
        # Select samples using each strategy according to allocation
        selected_indices = []
        strategy_pbar = tqdm(allocation.items(), desc="Selecting samples by strategy", leave=False)
        
        for strategy, count in strategy_pbar:
            if count <= 0:
                continue
                
            strategy_pbar.set_postfix({'strategy': strategy, 'count': count})
            
            if strategy == "S_U":
                strategy_indices = self._select_by_scores(uncertainty_scores, count)
            elif strategy == "S_D":
                strategy_indices = self._select_by_scores(diversity_scores, count)
            elif strategy == "S_C":
                strategy_indices = self._select_by_scores(class_balance_scores, count)
            elif strategy == "S_B":
                strategy_indices = self._select_by_scores(boundary_scores, count)
            else:
                continue
                
            selected_indices.extend(strategy_indices)
        
        # Ensure we have no duplicates and exactly n_samples
        selected_indices = list(set(selected_indices))  # Remove duplicates
        
        # If we have too few, add more from the highest weighted strategy
        if len(selected_indices) < n_samples:
            print(f"Need {n_samples - len(selected_indices)} more samples to reach target")
            dominant_strategy = max(self.weights.items(), key=lambda x: x[1])[0]
            additional_needed = n_samples - len(selected_indices)
            
            if dominant_strategy == "S_U":
                additional_scores = uncertainty_scores
            elif dominant_strategy == "S_D":
                additional_scores = diversity_scores
            elif dominant_strategy == "S_C":
                additional_scores = class_balance_scores
            else:  # S_B
                additional_scores = boundary_scores
                
            additional_indices = self._select_by_scores(
                additional_scores, 
                additional_needed, 
                exclude=selected_indices
            )
            selected_indices.extend(additional_indices)
        
        # If we have too many, truncate
        if len(selected_indices) > n_samples:
            print(f"Truncating {len(selected_indices) - n_samples} excess samples")
            selected_indices = selected_indices[:n_samples]
        
        # Increment epoch counter
        self.current_epoch += 1
        
        print(f"Selected {len(selected_indices)} new samples")
        
        return np.array(selected_indices)
    
    def _select_by_scores(self, scores, count, exclude=None):
        """
        Select top scoring samples that aren't already in current_indices or exclude list
        """
        exclude_set = set(self.current_indices)
        if exclude:
            exclude_set.update(exclude)
            
        # Create mask for available indices
        mask = np.ones(len(self.full_dataset), dtype=bool)
        mask[list(exclude_set)] = False
        available_indices = np.arange(len(self.full_dataset))[mask]
        available_scores = scores[mask]
        
        # Select top-scoring samples
        if len(available_indices) <= count:
            return available_indices
        
        top_indices = np.argsort(available_scores)[-count:]
        selected = available_indices[top_indices]
        
        return selected
    
    def _evaluate_strategy_performances(self, model, strategy_scores, n_eval_samples=50):
        """
        Evaluate the performance contribution of each strategy
        
        Args:
            model: The current model
            strategy_scores: Dictionary of score arrays for each strategy
            n_eval_samples: Number of samples to use for evaluation
            
        Returns:
            Dictionary of performance metrics for each strategy
        """
        base_model = model
        performances = {}
        
        # Get a small validation set from test data (if available)
        try:
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            )
            val_indices = np.random.choice(len(testset), 1000, replace=False)
            val_subset = Subset(testset, val_indices)
            val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=2)
        except:
            # If no test data available, use a portion of current coreset
            if len(self.current_indices) >= 200:
                val_indices = np.random.choice(self.current_indices, 200, replace=False)
                val_subset = Subset(self.full_dataset, val_indices)
                val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=2)
            else:
                # Not enough data for validation, return empty performance dict
                return {}
        
        # Evaluate base model performance
        base_acc = self._quick_eval_model(base_model, val_loader)
        
        # For each strategy, select samples and train briefly
        strategy_pbar = tqdm(strategy_scores.items(), desc="Evaluating strategy performance", leave=False)
        for strategy_name, scores in strategy_pbar:
            strategy_pbar.set_postfix({'strategy': strategy_name})
            
            # Select samples using only this strategy
            strategy_indices = self._select_by_scores(scores, n_eval_samples)
            
            if len(strategy_indices) == 0:
                performances[strategy_name] = 0.0
                continue
            
            # Create a temporary dataset with these samples
            temp_indices = np.concatenate([self.current_indices, strategy_indices])
            temp_subset = Subset(self.full_dataset, temp_indices)
            temp_loader = DataLoader(
                temp_subset,
                batch_size=128,
                shuffle=True,
                num_workers=2
            )
            
            # Clone the model for this strategy evaluation
            import copy
            strategy_model = copy.deepcopy(base_model)
            
            # Train briefly
            strategy_model.train()
            optimizer = torch.optim.Adam(strategy_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            train_pbar = tqdm(range(3), desc=f"Training with {strategy_name}", leave=False)
            for _ in train_pbar:
                batch_losses = []
                for inputs, targets in temp_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = strategy_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                
                # Update progress bar
                train_pbar.set_postfix({'loss': f'{np.mean(batch_losses):.4f}'})
            
            # Evaluate performance
            strategy_acc = self._quick_eval_model(strategy_model, val_loader)
            improvement = strategy_acc - base_acc
            performances[strategy_name] = max(0, improvement)
            
            # Update strategy performance bar
            strategy_pbar.set_postfix({
                'strategy': strategy_name, 
                'acc': f'{strategy_acc:.4f}', 
                'impr': f'{improvement:.4f}'
            })
            
            # Clean up to avoid memory issues
            del strategy_model
            
        return performances
    
    def _quick_eval_model(self, model, data_loader):
        """Quick evaluation of model accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return correct / total
    
    def _compute_feature_statistics(self, model):
        """
        Compute statistics about the feature space
        
        Args:
            model: The current model
            
        Returns:
            Dictionary of feature statistics
        """
        # Create feature extractor
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor.to(self.device)
        feature_extractor.eval()
        
        # Sample a subset of current coreset for efficiency
        if len(self.current_indices) > 500:
            sample_indices = np.random.choice(self.current_indices, 500, replace=False)
        else:
            sample_indices = self.current_indices
            
        sample_subset = Subset(self.full_dataset, sample_indices)
        sample_loader = DataLoader(sample_subset, batch_size=128, shuffle=False, num_workers=2)
        
        # Extract features and labels
        features = []
        labels = []
        
        with torch.no_grad():
            # Add tqdm progress bar
            pbar = tqdm(sample_loader, desc="Computing feature statistics", leave=False)
            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                batch_features = feature_extractor(inputs).squeeze()
                features.append(batch_features.cpu().numpy())
                labels.extend(targets.numpy())
                pbar.set_postfix({'features': len(features)})
                
        if len(features) > 0:
            features = np.vstack(features)
            labels = np.array(labels)
            
            # Compute statistics
            # 1. Feature diversity - average distance between samples
            from sklearn.metrics import pairwise_distances
            # Add progress indication
            print("Computing pairwise distances...")
            if len(features) > 100:  # If too many samples, use a subset for efficiency
                subset_idx = np.random.choice(len(features), 100, replace=False)
                subset_features = features[subset_idx]
                distances = pairwise_distances(subset_features)
            else:
                distances = pairwise_distances(features)
                
            feature_diversity = np.mean(distances) / np.max(distances)
            
            # 2. Feature redundancy - correlation between dimensions
            print("Computing feature correlations...")
            if features.shape[1] > 1:  # Only if we have multiple feature dimensions
                corr_matrix = np.corrcoef(features.T)
                feature_redundancy = np.mean(np.abs(corr_matrix - np.eye(corr_matrix.shape[0])))
            else:
                feature_redundancy = 0.0
            
            # 3. Class separation - ratio of between-class to within-class distance
            print("Computing class separation...")
            class_ids = np.unique(labels)
            if len(class_ids) > 1:
                within_class_dist = []
                # Use tqdm for class iteration
                for c in tqdm(class_ids, desc="Within-class distances", leave=False):
                    class_features = features[labels == c]
                    if len(class_features) > 1:
                        class_distances = pairwise_distances(class_features)
                        within_class_dist.append(np.mean(class_distances))
                
                between_class_dist = []
                # Use tqdm for class pair iteration
                pairs = [(i, c1, c2) for i, c1 in enumerate(class_ids[:-1]) 
                        for c2 in class_ids[i+1:]]
                for _, c1, c2 in tqdm(pairs, desc="Between-class distances", leave=False):
                    class1_features = features[labels == c1]
                    class2_features = features[labels == c2]
                    if len(class1_features) > 0 and len(class2_features) > 0:
                        between_distances = pairwise_distances(class1_features, class2_features)
                        between_class_dist.append(np.mean(between_distances))
                
                if within_class_dist and between_class_dist:
                    class_separation = np.mean(between_class_dist) / np.mean(within_class_dist)
                else:
                    class_separation = 1.0
            else:
                class_separation = 1.0
            
            return {
                "feature_diversity": feature_diversity,
                "feature_redundancy": feature_redundancy,
                "class_separation": class_separation
            }
        
        # Default values if we couldn't compute statistics
        return {
            "feature_diversity": 0.5,
            "feature_redundancy": 0.5,
            "class_separation": 0.5
        }

    def update_weights(self, model, train_loader, val_loader) -> Dict[str, float]:
        """
        Update strategy weights based on performance and meta-agent guidance
        
        Args:
            model: The trained model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            
        Returns:
            Updated weights
        """
        print("\nUpdating strategy weights...")
        
        # Evaluate model performance
        model.eval()
        accuracy = self._quick_eval_model(model, val_loader)
        
        # Extract training loss
        model.train()
        train_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        loss_pbar = tqdm(train_loader, desc="Computing training loss", leave=False)
        for inputs, targets in loss_pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            loss_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        train_loss /= len(train_loader)
        
        # Get class distribution
        class_distribution = Counter([self.full_dataset[i][1] for i in self.current_indices])
        
        # Compute strategy performances
        with tqdm(total=4, desc="Computing strategy scores", leave=False) as pbar:
            pbar.set_postfix({'stage': 'uncertainty'})
            uncertainty_scores = self.compute_uncertainty_scores(model)
            pbar.update(1)
            
            pbar.set_postfix({'stage': 'diversity'})
            diversity_scores = self.compute_diversity_scores(model)
            pbar.update(1)
            
            pbar.set_postfix({'stage': 'class balance'})
            class_balance_scores = self.compute_class_balance_scores()
            pbar.update(1)
            
            pbar.set_postfix({'stage': 'boundary'})
            boundary_scores = self.compute_boundary_scores(model)
            pbar.update(1)
        
        # Evaluate strategy performances
        strategy_performances = self._evaluate_strategy_performances(
            model, 
            {
                "S_U": uncertainty_scores,
                "S_D": diversity_scores,
                "S_C": class_balance_scores,
                "S_B": boundary_scores
            }, 
            n_eval_samples=50
        )
        
        # Compute feature statistics
        print("Computing feature statistics...")
        feature_statistics = self._compute_feature_statistics(model)
        
        # Default fallback - use simple reward-based weight update
        print("Using reward-based weight update...")
        alpha_values = {s: 1.0 + 10.0 * strategy_performances.get(s, 0.0) for s in self.strategies}
        
        # Apply temperature to control exploration/exploitation
        new_weights = {}
        denominator = sum(np.exp(a / self.temperature) for a in alpha_values.values())
        for s in self.strategies:
            new_weights[s] = np.exp(alpha_values[s] / self.temperature) / denominator
        
        # Adjust temperature - decrease over time
        self.temperature = max(0.1, self.temperature * 0.95)
        
        # Blend with current weights for stability
        for s in self.strategies:
            self.weights[s] = 0.7 * self.weights[s] + 0.3 * new_weights[s]
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {s: w / total for s, w in self.weights.items()}
        
        # Try to use meta-controller if available
        try:
            if hasattr(self, 'meta_controller') and self.meta_controller is not None:
                print("Using meta-controller for weight update...")
                state = self.meta_controller.update_state(
                    epoch=self.current_epoch,
                    accuracy=accuracy,
                    loss=train_loss,
                    class_distribution=class_distribution,
                    dataset_size=len(self.current_indices),
                    strategy_performances=strategy_performances,
                    feature_statistics=feature_statistics
                )
                
                # Get new weights from meta-controller
                meta_weights = self.meta_controller.get_current_weights()
                
                # Print explanation of decision
                explanation = self.meta_controller.get_latest_explanation()
                print(f"Strategy update: {explanation}")
                
                # Use the meta-controller weights
                self.weights = meta_weights
        except (AttributeError, NameError) as e:
            print(f"Meta-controller not available or error: {e}")
            print("Using fallback weight update method (already applied)...")
        
        # Save updated weights
        self._save_weights()
        
        # Increment epoch counter
        self.current_epoch += 1
        
        # Print new weights
        print(f"New strategy weights: {self.weights}")
        
        return self.weights
        
    def get_coreset(self, model=None, n_additional=None, current_metrics=None) -> np.ndarray:
        """
        Get the current coreset (or expand it if model is provided)
        
        Args:
            model: Optional model to use for selecting additional samples
            n_additional: Optional number of additional samples to select
            current_metrics: Optional dictionary of current performance metrics
            
        Returns:
            Indices of the coreset
        """
        if model is not None and n_additional is not None and n_additional > 0:
            # Select additional samples
            try:
                # Try using the enhanced method with metrics
                new_indices = self.select_samples(
                    model,
                    min(n_additional, self.budget - len(self.current_indices)),
                    current_metrics
                )
            except TypeError:
                # Fallback to simpler method if we have a compatibility issue
                new_indices = self.select_samples(
                    model,
                    min(n_additional, self.budget - len(self.current_indices))
                )
            
            # Add to current indices
            self.current_indices = np.concatenate([self.current_indices, new_indices])
        
        return self.current_indices
    
    def get_coreset_subset(self) -> Subset:
        """
        Get a PyTorch Subset representing the coreset
        
        Returns:
            Subset containing the coreset
        """
        return Subset(self.full_dataset, self.current_indices)


class SPARROWTrainer:
    """Trainer class for SPARROW coreset selection"""
    def __init__(self, args):
        self.args = args
        
        # Enhanced device selection with MPS support
        if args.device:
            self.device = torch.device(args.device)
        else:
            # Try CUDA first
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            # Then try MPS (Apple Silicon)
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            # Fall back to CPU
            else:
                self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        
        # Set up the coreset selector
        self.coreset_selector = SparrowCoresetSelector(
            self.full_dataset, 
            budget=self.args.max_budget,
            initial_samples=self.args.initial_sample_size,
            device=self.device
        )
        
        self.current_indices = self.coreset_selector.current_indices
        self.update_train_loader()
        
        # Metrics tracking
        self.metrics = []
        self.best_accuracy = 0.0
        self.start_time = time.time()
        
        # Set up TensorBoard writer
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        if self.args.tensorboard_dir:
            tensorboard_dir = self.args.tensorboard_dir
        else:
            tensorboard_dir = os.path.join(self.args.save_path, 'tensorboard', current_time)
        
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_dir)
        print(f"TensorBoard logs will be saved to {tensorboard_dir}")
        
        # Add model graph to TensorBoard
        dummy_input = torch.randn(1, 3, 32, 32, device=self.device)  # CIFAR input size
        self.writer.add_graph(self.model, dummy_input)
        
        # Visualize the learning rate schedule
        # Log the planned learning rate schedule
        temp_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        temp_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(temp_optimizer, T_max=self.args.epochs)
        lrs = []
        for i in range(self.args.epochs):
            lrs.append(temp_optimizer.param_groups[0]['lr'])
            temp_scheduler.step()
        
        # Plot learning rate schedule
        fig, ax = plt.figure(figsize=(10, 5)), plt.subplot(111)
        ax.plot(range(len(lrs)), lrs)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True)
        self.writer.add_figure('Optimizer/LRSchedule', fig)
        plt.close(fig)

    def setup_data(self):
        """Set up datasets and initial data loaders"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Load CIFAR-10 dataset
        self.full_dataset = torchvision.datasets.CIFAR10(
            root='/Users/tanmoy/research/data', train=True, download=True, transform=transform_train
        )
        self.test_dataset = torchvision.datasets.CIFAR10(
            root='/Users/tanmoy/research/data', train=False, download=True, transform=transform_test
        )
        
        # Create test loader
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.args.workers
        )
        
        # For validation, we'll use 10% of the test set
        test_size = len(self.test_dataset)
        val_size = test_size // 10
        val_indices = np.random.choice(test_size, val_size, replace=False)
        self.val_loader = DataLoader(
            Subset(self.test_dataset, val_indices),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers
        )

    def setup_model(self):
        """Initialize the model"""
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # 10 classes for CIFAR-10
        self.model = self.model.to(self.device)

    def setup_optimizer(self):
        """Set up optimizer and loss function"""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        self.criterion = nn.CrossEntropyLoss()

    def update_train_loader(self):
        """Update the training data loader with current coreset"""
        subset = Subset(self.full_dataset, self.current_indices)
        self.train_loader = DataLoader(
            subset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=self.args.workers
        )

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Use tqdm for progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} [Train]', 
                   leave=False, dynamic_ncols=True)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Log training metrics per batch to TensorBoard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            
            # Update progress bar
            acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'acc': f'{acc:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            if batch_idx % self.args.print_freq == 0:
                print(f'Epoch: {self.current_epoch} '
                      f'[{batch_idx * len(inputs)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)] '
                      f'Loss: {loss.item():.6f}')

        return total_loss / len(self.train_loader), correct / total

    def evaluate(self):
        """Evaluate the model on the test set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        # Use tqdm for progress bar
        pbar = tqdm(self.test_loader, desc='Evaluation', leave=False, dynamic_ncols=True)
        
        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                current_correct = predicted.eq(targets).sum().item()
                correct += current_correct

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update progress bar
                acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}', 
                    'acc': f'{acc:.2f}%'
                })

        accuracy = correct / total
        f1 = f1_score(all_targets, all_preds, average='weighted')
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        # Log confusion matrix to TensorBoard (once every 10 epochs)
        if hasattr(self, 'current_epoch') and self.current_epoch % 10 == 0:
            fig, ax = plt.figure(figsize=(10, 10)), plt.subplot(111)
            cax = ax.matshow(conf_matrix)
            plt.title('Confusion Matrix')
            fig.colorbar(cax)
            ax.set_xticklabels([''] + [str(i) for i in range(10)])
            ax.set_yticklabels([''] + [str(i) for i in range(10)])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            self.writer.add_figure('Test/ConfusionMatrix', fig, self.current_epoch)

        return total_loss / len(self.test_loader), accuracy, f1, precision, recall, conf_matrix

    def run(self):
        """Run the complete training pipeline"""
        # Use trange for epoch progress
        epoch_iterator = trange(self.args.epochs, desc="Training Progress", dynamic_ncols=True)
        
        for self.current_epoch in epoch_iterator:
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Evaluate on test set
            test_loss, test_acc, f1, precision, recall, conf_matrix = self.evaluate()
            
            # Step the scheduler
            self.scheduler.step()
            
            # Log metrics to TensorBoard
            self.writer.add_scalar('Train/Loss', train_loss, self.current_epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, self.current_epoch)
            self.writer.add_scalar('Test/Loss', test_loss, self.current_epoch)
            self.writer.add_scalar('Test/Accuracy', test_acc, self.current_epoch)
            self.writer.add_scalar('Test/F1', f1, self.current_epoch)
            self.writer.add_scalar('Test/Precision', precision, self.current_epoch)
            self.writer.add_scalar('Test/Recall', recall, self.current_epoch)
            self.writer.add_scalar('Dataset/Size', len(self.current_indices), self.current_epoch)
            self.writer.add_scalar('Optimizer/LearningRate', self.optimizer.param_groups[0]['lr'], self.current_epoch)
            
            # Log strategy weights to TensorBoard
            for strategy, weight in self.coreset_selector.weights.items():
                self.writer.add_scalar(f'Strategy/Weight_{strategy}', weight, self.current_epoch)
            
            # Add histograms of model parameters (every 10 epochs to avoid overhead)
            if self.current_epoch % 10 == 0:
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(f'Parameters/{name}', param.clone().cpu().data.numpy(), self.current_epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f'Gradients/{name}', param.grad.clone().cpu().data.numpy(), self.current_epoch)
            
            # Add class distribution visualization
            if self.current_epoch % 20 == 0:
                class_counts = Counter([self.full_dataset[i][1] for i in self.current_indices])
                fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
                classes = sorted(class_counts.keys())
                counts = [class_counts[c] for c in classes]
                ax.bar(classes, counts)
                ax.set_xlabel('Class')
                ax.set_ylabel('Count')
                ax.set_title(f'Coreset Class Distribution at Epoch {self.current_epoch}')
                self.writer.add_figure('Coreset/ClassDistribution', fig, self.current_epoch)
                plt.close(fig)
            
            # Update strategy weights
            if self.current_epoch % 5 == 0:  # Update every 5 epochs to save computation
                strategy_weights = self.coreset_selector.update_weights(
                    self.model, 
                    self.train_loader, 
                    self.val_loader
                )
                print(f"Updated strategy weights: {strategy_weights}")
            
            # Expand the coreset if needed
            if len(self.current_indices) < self.args.max_budget:
                n_additional = min(
                    self.args.samples_per_epoch, 
                    self.args.max_budget - len(self.current_indices)
                )
                
                # Provide metrics to the meta-agent
                current_metrics = {
                    "accuracy": test_acc,
                    "loss": test_loss,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall
                }
                
                self.current_indices = self.coreset_selector.get_coreset(
                    model=self.model,
                    n_additional=n_additional,
                    current_metrics=current_metrics
                )
                
                # Update the training data loader
                self.update_train_loader()
                print(f"Coreset expanded to {len(self.current_indices)} samples")
            
            # Calculate time
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - self.start_time
            
            # Log time metrics to TensorBoard
            self.writer.add_scalar('Time/Epoch', epoch_time, self.current_epoch)
            self.writer.add_scalar('Time/Total', total_time, self.current_epoch)
            
            # Log metrics
            metrics = {
                "epoch": self.current_epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "dataset_size": len(self.current_indices),
                "strategy_weights": self.coreset_selector.weights,
                "epoch_time": epoch_time,
                "total_time": total_time
            }
            self.metrics.append(metrics)
            
            # Update progress bar with summary
            epoch_iterator.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'train_acc': f'{train_acc:.4f}',
                'test_acc': f'{test_acc:.4f}',
                'dataset': f'{len(self.current_indices)}/{self.args.max_budget}'
            })
            
            print(f"Epoch {self.current_epoch+1}/{self.args.epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
                  f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
                  f"Epoch Time: {epoch_time:.2f}s, Total Time: {total_time:.2f}s")
            
            # Save checkpoint if best accuracy
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                self.save_checkpoint({
                    'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_accuracy': self.best_accuracy,
                    'current_indices': self.current_indices,
                    'strategy_weights': self.coreset_selector.weights
                }, os.path.join(self.args.save_path, 'best_model.pth'))
                
                # Log best model to TensorBoard
                self.writer.add_scalar('Best/Accuracy', test_acc, self.current_epoch)
                self.writer.add_scalar('Best/Epoch', self.current_epoch, self.current_epoch)
        
        # Save final statistics
        self.save_final_statistics()
        
        # Print TensorBoard viewing instructions
        if self.args.tensorboard_dir:
            tensorboard_dir = self.args.tensorboard_dir
        else:
            current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
            tensorboard_dir = os.path.join(self.args.save_path, 'tensorboard', current_time)
        
        print(f"\nTraining completed! To view TensorBoard logs, run:")
        print(f"tensorboard --logdir={tensorboard_dir}")
        print("Then open http://localhost:6006 in your browser")
        
        # Close TensorBoard writer
        self.writer.close()
    
    def save_checkpoint(self, state, filename):
        """Save a checkpoint"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")
    
    def save_final_statistics(self):
        """Save final statistics after training"""
        os.makedirs(self.args.save_path, exist_ok=True)
        
        # Save metrics to CSV
        with open(os.path.join(self.args.save_path, 'metrics.csv'), 'w', newline='') as csvfile:
            # Extract all keys from all metrics
            fieldnames = set()
            for metric in self.metrics:
                fieldnames.update(k for k in metric.keys() if k != 'strategy_weights')
            fieldnames = list(fieldnames) + ['S_U', 'S_D', 'S_C', 'S_B']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for metric in self.metrics:
                row = {k: v for k, v in metric.items() if k != 'strategy_weights'}
                if 'strategy_weights' in metric:
                    for strategy, weight in metric['strategy_weights'].items():
                        row[strategy] = weight
                writer.writerow(row)
        
        # Save confusion matrix
        _, _, _, _, _, conf_matrix = self.evaluate()
        np.savetxt(os.path.join(self.args.save_path, 'confusion_matrix.csv'), conf_matrix, delimiter=",")
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(self.args.save_path, 'final_model.pth'))
        
        # Save coreset indices
        np.save(os.path.join(self.args.save_path, 'coreset_indices.npy'), self.current_indices)
        
        # Save weight history
        with open(os.path.join(self.args.save_path, 'weight_history.json'), 'w') as f:
            json.dump(self.coreset_selector.weight_history, f)
        
        # Save the complete coreset selector
        torch.save(self.coreset_selector, os.path.join(self.args.save_path, 'coreset_selector.pth'))
        
        # Plot weight evolution
        self.plot_weight_evolution()
        
        # Add final metrics and plots to TensorBoard
        # Final coreset distribution by class
        class_counts = Counter([self.full_dataset[i][1] for i in self.current_indices])
        fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        ax.bar(classes, counts)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Final Coreset Class Distribution')
        self.writer.add_figure('Coreset/ClassDistribution', fig)
        
        # Also add final embedding visualization if possible
        try:
            self.add_embedding_visualization()
        except Exception as e:
            print(f"Failed to create embedding visualization: {e}")
        
        print(f"Final statistics saved to {self.args.save_path}")
    
    def load_checkpoint(self, filename):
        """Load a checkpoint"""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.current_indices = checkpoint['current_indices']
        self.best_accuracy = checkpoint['best_accuracy']
        self.coreset_selector.weights = checkpoint['strategy_weights']
        self.update_train_loader()
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def plot_weight_evolution(self):
        """Plot the evolution of strategy weights during training"""
        plt.figure(figsize=(10, 6))
        
        weight_history = self.coreset_selector.weight_history
        epochs = list(range(len(weight_history)))
        
        for strategy in self.coreset_selector.strategies:
            weights = [w[strategy] for w in weight_history]
            plt.plot(epochs, weights, label=strategy)
        
        plt.xlabel('Epoch')
        plt.ylabel('Strategy Weight')
        plt.title('SPARROW Strategy Weight Evolution')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.args.save_path, 'weight_evolution.png'))
        plt.close()

    def add_embedding_visualization(self):
        """Add embedding visualization to TensorBoard"""
        # Get feature embeddings for the coreset
        feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        feature_extractor.eval()
        
        coreset_subset = Subset(self.full_dataset, self.current_indices)
        coreset_loader = DataLoader(
            coreset_subset,
            batch_size=128,
            shuffle=False,
            num_workers=4
        )
        
        all_features = []
        all_labels = []
        all_images = []
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(coreset_loader):
                if i == 0:  # Store only first batch of images to avoid memory issues
                    all_images = inputs
                
                inputs = inputs.to(self.device)
                features = feature_extractor(inputs)
                features = features.squeeze().cpu()
                
                all_features.append(features)
                all_labels.extend(targets.numpy())
        
        if all_features:
            all_features = torch.cat(all_features, 0)
            
            # Add embedding visualization
            self.writer.add_embedding(
                all_features,
                metadata=all_labels,
                label_img=all_images[:100] if len(all_images) > 0 else None,
                global_step=self.args.epochs,
                tag='coreset_embeddings'
            )


def main():
    """Main function to run SPARROW coreset selection"""
    parser = argparse.ArgumentParser(description='SPARROW: Strategic Policy-based Active Re-weighting Workflow')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                      help='dataset to use: cifar10 or cifar100 (default: cifar10)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128, 
                      help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200,
                      help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='learning rate (default: 0.001)')
    
    # Coreset parameters
    parser.add_argument('--max-budget', type=int, default=5000,
                      help='maximum coreset size (default: 5000)')
    parser.add_argument('--initial-sample-size', type=int, default=1000,
                      help='initial coreset size (default: 1000)')
    parser.add_argument('--samples-per-epoch', type=int, default=200,
                      help='number of samples to add per epoch (default: 200)')
    
    # System parameters
    parser.add_argument('--save-path', type=str, default='./results/sparrow',
                      help='path to save results (default: ./results/sparrow)')
    parser.add_argument('--resume', type=str, default='',
                      help='path to latest checkpoint (default: none)')
    parser.add_argument('--workers', type=int, default=4,
                      help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', type=int, default=10,
                      help='print frequency (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                      help='random seed (default: 42)')
    # Add tensorboard log directory option
    parser.add_argument('--tensorboard-dir', type=str, default=None,
                      help='tensorboard log directory (default: save_path/tensorboard/timestamp)')
    
    # Add device selection option
    parser.add_argument('--device', type=str, default=None,
                      help='device to use (cuda, mps, cpu, or None for auto-detection)')
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Save command line arguments
    with open(os.path.join(args.save_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Initialize trainer
    trainer = SPARROWTrainer(args)
    
    # Load checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Run training
    trainer.run()
    
    print("SPARROW coreset selection completed successfully!")


if __name__ == '__main__':
    main()