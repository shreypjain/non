"""
Multi-arm bandit environment simulator.

This module provides various multi-arm bandit environments with different
reward distributions for testing genetic algorithm strategies.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class MultiArmBandit(ABC):
    """
    Abstract base class for multi-arm bandit environments.

    A multi-arm bandit has multiple arms, each providing stochastic rewards
    when pulled. The agent's goal is to maximize cumulative reward over time.
    """

    def __init__(self, num_arms: int, seed: Optional[int] = None):
        """
        Initialize the bandit environment.

        Args:
            num_arms: Number of arms
            seed: Random seed for reproducibility
        """
        if num_arms <= 0:
            raise ValueError("Number of arms must be positive")

        self.num_arms = num_arms
        self.seed = seed
        self.pull_count = [0] * num_arms
        self.total_pulls = 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    @abstractmethod
    def pull(self, arm: int) -> float:
        """
        Pull an arm and receive a reward.

        Args:
            arm: Index of the arm to pull (0 to num_arms-1)

        Returns:
            Reward value

        Raises:
            ValueError: If arm index is invalid
        """
        pass

    @abstractmethod
    def get_optimal_arm(self) -> int:
        """
        Get the index of the optimal arm (highest expected reward).

        Returns:
            Index of the optimal arm
        """
        pass

    @abstractmethod
    def get_expected_reward(self, arm: int) -> float:
        """
        Get the expected reward for a specific arm.

        Args:
            arm: Index of the arm

        Returns:
            Expected reward value
        """
        pass

    def reset(self):
        """Reset the bandit state (pull counts)."""
        self.pull_count = [0] * self.num_arms
        self.total_pulls = 0

    def _validate_arm(self, arm: int):
        """Validate arm index."""
        if arm < 0 or arm >= self.num_arms:
            raise ValueError(f"Invalid arm index {arm}. Must be 0 to {self.num_arms-1}")

    def _record_pull(self, arm: int):
        """Record a pull for statistics."""
        self.pull_count[arm] += 1
        self.total_pulls += 1

    def get_regret(self) -> float:
        """
        Calculate cumulative regret (difference from optimal strategy).

        Returns:
            Total regret accumulated
        """
        optimal_arm = self.get_optimal_arm()
        optimal_reward = self.get_expected_reward(optimal_arm)

        regret = 0.0
        for arm, count in enumerate(self.pull_count):
            expected = self.get_expected_reward(arm)
            regret += count * (optimal_reward - expected)

        return regret


class BernoulliBandit(MultiArmBandit):
    """
    Bandit with arms providing binary rewards (0 or 1) with fixed probabilities.

    Each arm has a success probability, and pulling it returns 1 with that
    probability, 0 otherwise.
    """

    def __init__(self, probabilities: List[float], seed: Optional[int] = None):
        """
        Initialize Bernoulli bandit.

        Args:
            probabilities: Success probability for each arm (values in [0, 1])
            seed: Random seed for reproducibility

        Raises:
            ValueError: If probabilities are invalid
        """
        if not probabilities:
            raise ValueError("Must provide at least one probability")
        if not all(0 <= p <= 1 for p in probabilities):
            raise ValueError("All probabilities must be in [0, 1]")

        super().__init__(num_arms=len(probabilities), seed=seed)
        self.probabilities = probabilities

    def pull(self, arm: int) -> float:
        """Pull an arm and receive binary reward."""
        self._validate_arm(arm)
        self._record_pull(arm)

        return 1.0 if random.random() < self.probabilities[arm] else 0.0

    def get_optimal_arm(self) -> int:
        """Return arm with highest success probability."""
        return int(np.argmax(self.probabilities))

    def get_expected_reward(self, arm: int) -> float:
        """Return success probability for the arm."""
        self._validate_arm(arm)
        return self.probabilities[arm]


class GaussianBandit(MultiArmBandit):
    """
    Bandit with arms providing rewards from normal distributions.

    Each arm samples rewards from a Gaussian distribution with specified
    mean and standard deviation.
    """

    def __init__(
        self, means: List[float], stds: List[float], seed: Optional[int] = None
    ):
        """
        Initialize Gaussian bandit.

        Args:
            means: Mean reward for each arm
            stds: Standard deviation for each arm
            seed: Random seed for reproducibility

        Raises:
            ValueError: If parameters are invalid
        """
        if not means or not stds:
            raise ValueError("Must provide means and stds")
        if len(means) != len(stds):
            raise ValueError("Means and stds must have same length")
        if not all(s > 0 for s in stds):
            raise ValueError("All standard deviations must be positive")

        super().__init__(num_arms=len(means), seed=seed)
        self.means = means
        self.stds = stds

    def pull(self, arm: int) -> float:
        """Pull an arm and receive Gaussian reward."""
        self._validate_arm(arm)
        self._record_pull(arm)

        return np.random.normal(self.means[arm], self.stds[arm])

    def get_optimal_arm(self) -> int:
        """Return arm with highest mean reward."""
        return int(np.argmax(self.means))

    def get_expected_reward(self, arm: int) -> float:
        """Return mean reward for the arm."""
        self._validate_arm(arm)
        return self.means[arm]


class NonStationaryBandit(MultiArmBandit):
    """
    Bandit with time-varying reward distributions.

    Reward probabilities change over time according to specified patterns,
    testing adaptability of strategies.
    """

    def __init__(
        self,
        initial_probabilities: List[float],
        change_interval: int = 100,
        drift_amount: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize non-stationary bandit.

        Args:
            initial_probabilities: Starting success probabilities
            change_interval: Number of pulls between probability changes
            drift_amount: Amount to randomly drift probabilities
            seed: Random seed for reproducibility
        """
        super().__init__(num_arms=len(initial_probabilities), seed=seed)
        self.probabilities = initial_probabilities.copy()
        self.change_interval = change_interval
        self.drift_amount = drift_amount

    def pull(self, arm: int) -> float:
        """Pull an arm and receive binary reward, with drifting probabilities."""
        self._validate_arm(arm)
        self._record_pull(arm)

        # Check if it's time to drift probabilities
        if self.total_pulls % self.change_interval == 0 and self.total_pulls > 0:
            self._drift_probabilities()

        return 1.0 if random.random() < self.probabilities[arm] else 0.0

    def _drift_probabilities(self):
        """Apply random drift to all arm probabilities."""
        for i in range(self.num_arms):
            # Random drift in [-drift_amount, +drift_amount]
            drift = random.uniform(-self.drift_amount, self.drift_amount)
            self.probabilities[i] = max(0.0, min(1.0, self.probabilities[i] + drift))

    def get_optimal_arm(self) -> int:
        """Return current optimal arm (changes over time)."""
        return int(np.argmax(self.probabilities))

    def get_expected_reward(self, arm: int) -> float:
        """Return current expected reward for the arm."""
        self._validate_arm(arm)
        return self.probabilities[arm]


class ContextualBandit(MultiArmBandit):
    """
    Bandit where rewards depend on context features.

    Each arm's reward is a function of the current context vector,
    enabling more complex environment dynamics.
    """

    def __init__(
        self,
        num_arms: int,
        context_dim: int,
        arm_weights: Optional[List[np.ndarray]] = None,
        noise_std: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize contextual bandit.

        Args:
            num_arms: Number of arms
            context_dim: Dimension of context vectors
            arm_weights: Weight vectors for each arm (random if None)
            noise_std: Standard deviation of reward noise
            seed: Random seed for reproducibility
        """
        super().__init__(num_arms=num_arms, seed=seed)
        self.context_dim = context_dim
        self.noise_std = noise_std

        # Initialize arm weights
        if arm_weights is None:
            self.arm_weights = [
                np.random.randn(context_dim) for _ in range(num_arms)
            ]
        else:
            if len(arm_weights) != num_arms:
                raise ValueError("Must provide weight vector for each arm")
            self.arm_weights = arm_weights

        # Generate initial context
        self.current_context = self._sample_context()

    def _sample_context(self) -> np.ndarray:
        """Sample a random context vector."""
        return np.random.randn(self.context_dim)

    def pull(self, arm: int) -> float:
        """Pull an arm and receive context-dependent reward."""
        self._validate_arm(arm)
        self._record_pull(arm)

        # Compute reward as dot product + noise
        reward = np.dot(self.arm_weights[arm], self.current_context)
        reward += np.random.normal(0, self.noise_std)

        # Sample new context for next pull
        self.current_context = self._sample_context()

        return reward

    def get_optimal_arm(self) -> int:
        """Return optimal arm for current context."""
        expected_rewards = [
            np.dot(weights, self.current_context) for weights in self.arm_weights
        ]
        return int(np.argmax(expected_rewards))

    def get_expected_reward(self, arm: int) -> float:
        """Return expected reward for arm given current context."""
        self._validate_arm(arm)
        return np.dot(self.arm_weights[arm], self.current_context)

    def set_context(self, context: np.ndarray):
        """Manually set the current context."""
        if len(context) != self.context_dim:
            raise ValueError(f"Context must have dimension {self.context_dim}")
        self.current_context = context.copy()
