import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class LearningState:
    """
    Represents the current state of the learning process
    """
    accuracy: float
    loss: float
    class_distribution: Dict[int, int]
    dataset_size: int
    epoch: int
    total_epochs: int
    strategy_performances: Dict[str, float]
    performance_history: List[Dict[str, float]]
    feature_statistics: Optional[Dict[str, float]] = None

class MetaStrategyAgent:
    """
    Meta-agent for dynamic strategy selection in SPARROW
    """
    def __init__(
        self,
        strategies: List[str],
        initial_temperature: float = 0.5,
        learning_rate: float = 0.1,
        exploration_decay: float = 0.95,
        min_temperature: float = 0.1
    ):
        """
        Initialize the meta-agent for strategy selection
        
        Args:
            strategies: List of strategy names
            initial_temperature: Initial temperature for softmax normalization
            learning_rate: Learning rate for weight updates
            exploration_decay: Rate at which to decay temperature
            min_temperature: Minimum temperature value
        """
        self.strategies = strategies
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.exploration_decay = exploration_decay
        self.learning_rate = learning_rate
        
        # Initialize strategy parameters and weights
        self.alpha = {s: 1.0 for s in strategies}
        self.weights = {s: 1.0 / len(strategies) for s in strategies}
        
        # Performance tracking
        self.performance_history = []
        self.weight_history = []
        self.reward_history = []
        
        # Save initial state
        self._save_weights()
    
    def _save_weights(self):
        """Save current weights to history"""
        self.weight_history.append(self.weights.copy())
    
    def _softmax_normalized(self, values: Dict[str, float], temperature: float) -> Dict[str, float]:
        """
        Apply softmax with temperature to normalize values
        
        Args:
            values: Dictionary of values to normalize
            temperature: Temperature parameter for softmax
            
        Returns:
            Dictionary of normalized values (sums to 1)
        """
        exp_values = {k: np.exp(v / temperature) for k, v in values.items()}
        total = sum(exp_values.values())
        return {k: v / total for k, v in exp_values.items()}
    
    def get_weights(self) -> Dict[str, float]:
        """Get the current strategy weights"""
        return self.weights
    
    def adjust_temperature(self, state: LearningState):
        """
        Adjust temperature based on learning state to control exploration vs exploitation
        
        Args:
            state: Current learning state
        """
        # Basic decay schedule
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.exploration_decay
        )
        
        # Additional adaptive adjustments based on state
        curriculum_progress = state.epoch / state.total_epochs
        
        # Increase temperature if performance plateaus
        if len(state.performance_history) >= 3:
            recent_accs = [p["accuracy"] for p in state.performance_history[-3:]]
            if max(recent_accs) - min(recent_accs) < 0.005:  # Performance plateau
                self.temperature = min(1.0, self.temperature * 1.2)  # Increase temperature to explore more
        
        # Decrease temperature faster in later stages of training
        if curriculum_progress > 0.7:  # Late stage of training
            self.temperature = max(
                self.min_temperature,
                self.temperature * 0.9  # Accelerate cooling
            )
        
        logging.debug(f"Adjusted temperature to {self.temperature:.4f}")
    
    def compute_rewards(self, state: LearningState) -> Dict[str, float]:
        """
        Compute rewards for each strategy based on current learning state
        
        Args:
            state: Current learning state
            
        Returns:
            Dictionary of rewards for each strategy
        """
        rewards = {}
        
        # Direct performance rewards (if available)
        if state.strategy_performances:
            for strategy, performance in state.strategy_performances.items():
                rewards[strategy] = max(0, performance)
        
        # If no direct measurements, use heuristics based on current state
        else:
            # Default rewards based on the learning stage
            curriculum_progress = state.epoch / state.total_epochs
            
            # Early stage (first 30% of training)
            if curriculum_progress < 0.3:
                rewards = {
                    "S_U": 0.5,  # Uncertainty is moderately useful
                    "S_D": 0.3,  # Diversity is somewhat useful
                    "S_C": 0.9,  # Class balance is critical early on
                    "S_B": 0.4   # Boundary cases moderate importance
                }
            # Mid stage (30-70% of training)
            elif curriculum_progress < 0.7:
                rewards = {
                    "S_U": 0.7,  # Uncertainty becomes more important
                    "S_D": 0.8,  # Diversity becomes more important
                    "S_C": 0.4,  # Class balance less critical as distribution improves
                    "S_B": 0.7   # Boundary cases more important for refinement
                }
            # Late stage (last 30% of training)
            else:
                rewards = {
                    "S_U": 0.6,  # Uncertainty still important but less so
                    "S_D": 0.9,  # Diversity most important to avoid overfitting
                    "S_C": 0.3,  # Class balance least important at this stage
                    "S_B": 0.5   # Boundary cases decrease in importance
                }
            
            # Adjust rewards based on accuracy trend
            if len(state.performance_history) >= 2:
                acc_change = state.performance_history[-1]["accuracy"] - state.performance_history[-2]["accuracy"]
                
                # If accuracy is improving fast, reward current mixture
                if acc_change > 0.01:
                    current_max = max(self.weights.items(), key=lambda x: x[1])[0]
                    rewards[current_max] *= 1.2
                
                # If accuracy is stagnating, boost exploration
                elif acc_change < 0.001:
                    current_min = min(self.weights.items(), key=lambda x: x[1])[0]
                    rewards[current_min] *= 1.5
        
        # Record rewards for analysis
        self.reward_history.append(rewards.copy())
        
        return rewards
    
    def analyze_feature_statistics(self, state: LearningState) -> Dict[str, float]:
        """
        Analyze feature space statistics to guide strategy selection
        
        Args:
            state: Current learning state including feature statistics
            
        Returns:
            Dictionary of feature-based adjustments for each strategy
        """
        adjustments = {s: 0.0 for s in self.strategies}
        
        # Skip if no feature statistics are available
        if not state.feature_statistics:
            return adjustments
        
        # Example adjustments based on feature statistics
        feature_diversity = state.feature_statistics.get("feature_diversity", 0.5)
        feature_redundancy = state.feature_statistics.get("feature_redundancy", 0.5)
        class_separation = state.feature_statistics.get("class_separation", 0.5)
        
        # Adjust strategies based on feature space characteristics
        # Higher feature diversity → Less need for diversity sampling
        adjustments["S_D"] -= feature_diversity * 0.5
        
        # Higher feature redundancy → More need for diversity sampling
        adjustments["S_D"] += feature_redundancy * 0.5
        
        # Better class separation → Less need for boundary sampling
        adjustments["S_B"] -= class_separation * 0.4
        
        return adjustments
    
    def update_weights(self, state: LearningState) -> Dict[str, float]:
        """
        Update strategy weights based on learning state
        
        Args:
            state: Current learning state
            
        Returns:
            Updated strategy weights
        """
        # Compute rewards for each strategy
        rewards = self.compute_rewards(state)
        
        # Get feature-based adjustments
        feature_adjustments = self.analyze_feature_statistics(state)
        
        # Update alpha parameters
        for strategy in self.strategies:
            reward = rewards[strategy]
            adjustment = feature_adjustments[strategy]
            
            # Combined update with reward and feature adjustment
            self.alpha[strategy] = self.alpha[strategy] * (1 + self.learning_rate * (reward + adjustment))
        
        # Ensure alpha values are positive
        min_alpha = min(self.alpha.values())
        if min_alpha <= 0:
            offset = abs(min_alpha) + 0.1
            for strategy in self.strategies:
                self.alpha[strategy] += offset
        
        # Adjust temperature
        self.adjust_temperature(state)
        
        # Apply softmax normalization with temperature
        self.weights = self._softmax_normalized(self.alpha, self.temperature)
        
        # Save weights for history tracking
        self._save_weights()
        
        return self.weights
    
    def get_weight_history(self) -> List[Dict[str, float]]:
        """Get the history of strategy weights"""
        return self.weight_history
    
    def get_strategy_decision(self, state: LearningState) -> Tuple[Dict[str, float], str]:
        """
        Get strategy weights and explanation
        
        Args:
            state: Current learning state
            
        Returns:
            Tuple of (weights, explanation)
        """
        # Update weights
        weights = self.update_weights(state)
        
        # Generate explanation
        curriculum_progress = state.epoch / state.total_epochs
        stage = "early" if curriculum_progress < 0.3 else "middle" if curriculum_progress < 0.7 else "late"
        
        # Identify dominant strategy
        dominant = max(weights.items(), key=lambda x: x[1])
        
        explanation = (
            f"At {stage} training stage (progress: {curriculum_progress:.2f}), "
            f"prioritizing {dominant[0]} (weight: {dominant[1]:.2f}) "
            f"with temperature {self.temperature:.2f}. "
        )
        
        # Add trend analysis if possible
        if len(self.weight_history) >= 3:
            increasing = []
            decreasing = []
            
            for strategy in self.strategies:
                hist = [h[strategy] for h in self.weight_history[-3:]]
                if hist[-1] > hist[0] + 0.05:
                    increasing.append(strategy)
                elif hist[0] > hist[-1] + 0.05:
                    decreasing.append(strategy)
            
            if increasing:
                explanation += f"Increasing emphasis on {', '.join(increasing)}. "
            if decreasing:
                explanation += f"Decreasing emphasis on {', '.join(decreasing)}. "
        
        return weights, explanation

class SparrowMetaController:
    """
    Controller that integrates the meta-agent with SPARROW's coreset selection
    """
    def __init__(
        self,
        strategies: List[str] = ["S_U", "S_D", "S_C", "S_B"],
        initial_temperature: float = 0.5,
        total_epochs: int = 200
    ):
        """
        Initialize the SPARROW meta-controller
        
        Args:
            strategies: List of strategy names
            initial_temperature: Initial temperature for exploration
            total_epochs: Total number of epochs for training
        """
        self.meta_agent = MetaStrategyAgent(
            strategies=strategies,
            initial_temperature=initial_temperature
        )
        self.total_epochs = total_epochs
        self.performance_history = []
        self.sample_counts = {s: 0 for s in strategies}
        self.explanations = []
    
    def update_state(self, 
                     epoch: int,
                     accuracy: float,
                     loss: float,
                     class_distribution: Dict[int, int],
                     dataset_size: int,
                     strategy_performances: Optional[Dict[str, float]] = None,
                     feature_statistics: Optional[Dict[str, float]] = None
                    ) -> LearningState:
        """
        Update the learning state with current metrics
        
        Args:
            epoch: Current epoch number
            accuracy: Current model accuracy
            loss: Current model loss
            class_distribution: Distribution of classes in current coreset
            dataset_size: Size of current coreset
            strategy_performances: Optional direct performance metrics for each strategy
            feature_statistics: Optional feature space statistics
            
        Returns:
            Updated learning state
        """
        # Record performance
        performance = {
            "epoch": epoch,
            "accuracy": accuracy,
            "loss": loss,
            "dataset_size": dataset_size
        }
        self.performance_history.append(performance)
        
        # Create learning state
        state = LearningState(
            accuracy=accuracy,
            loss=loss,
            class_distribution=class_distribution,
            dataset_size=dataset_size,
            epoch=epoch,
            total_epochs=self.total_epochs,
            strategy_performances=strategy_performances or {},
            performance_history=self.performance_history,
            feature_statistics=feature_statistics
        )
        
        return state
    
    def get_sample_allocation(self, n_samples: int, state: LearningState) -> Dict[str, int]:
        """
        Determine how many samples to allocate to each strategy
        
        Args:
            n_samples: Total number of samples to select
            state: Current learning state
            
        Returns:
            Dictionary with number of samples for each strategy
        """
        # Get strategy weights and explanation
        weights, explanation = self.meta_agent.get_strategy_decision(state)
        self.explanations.append(explanation)
        
        # Allocate samples proportionally to weights
        allocation = {}
        remaining = n_samples
        
        # First pass - allocate integer samples
        for strategy, weight in weights.items():
            strategy_samples = int(n_samples * weight)
            allocation[strategy] = strategy_samples
            remaining -= strategy_samples
        
        # Distribute any remaining samples
        strategies_sorted = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for strategy, _ in strategies_sorted:
            if remaining <= 0:
                break
            allocation[strategy] += 1
            remaining -= 1
        
        # Update sample counts
        for strategy, count in allocation.items():
            self.sample_counts[strategy] += count
        
        return allocation
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        return self.meta_agent.get_weights()
    
    def get_weight_history(self) -> List[Dict[str, float]]:
        """Get history of strategy weights"""
        return self.meta_agent.get_weight_history()
    
    def get_sample_counts(self) -> Dict[str, int]:
        """Get total samples selected by each strategy"""
        return self.sample_counts
    
    def get_latest_explanation(self) -> str:
        """Get the latest decision explanation"""
        return self.explanations[-1] if self.explanations else ""
    
    def get_all_explanations(self) -> List[str]:
        """Get all decision explanations"""
        return self.explanations

# Example usage:
if __name__ == "__main__":
    # Initialize meta-controller
    controller = SparrowMetaController(total_epochs=100)
    
    # Simulate a training loop
    for epoch in range(100):
        # Simulate metrics
        accuracy = 0.5 + 0.3 * (1 - np.exp(-0.05 * epoch))
        loss = 1.0 - 0.6 * (1 - np.exp(-0.05 * epoch))
        class_distribution = {i: 100 + i*10 for i in range(10)}
        dataset_size = 1000 + 200 * min(epoch, 20)
        
        # Update state
        state = controller.update_state(
            epoch=epoch,
            accuracy=accuracy,
            loss=loss,
            class_distribution=class_distribution,
            dataset_size=dataset_size
        )
        
        # Get sample allocation
        n_samples = 100 if epoch < 50 else 50
        allocation = controller.get_sample_allocation(n_samples, state)
        
        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
            print(f"Strategy weights: {controller.get_current_weights()}")
            print(f"Sample allocation: {allocation}")
            print(f"Explanation: {controller.get_latest_explanation()}")
            print()