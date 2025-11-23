# Multi-Arm Bandit Genetic Algorithm Experiment

This experiment explores using genetic algorithms to solve multi-arm bandit problems, demonstrating how evolutionary computation can optimize decision-making strategies in environments with uncertain rewards.

## Problem Overview

The multi-arm bandit problem is a classic reinforcement learning challenge where an agent must choose between multiple options (arms), each providing uncertain rewards. The goal is to maximize cumulative reward over time while balancing exploration (trying different arms) and exploitation (choosing known good arms).

## Genetic Algorithm Approach

This implementation uses genetic algorithms to evolve effective arm-pulling strategies:

### Binary Encoding

Each strategy is encoded as a binary chromosome representing a sequence of arm choices over a fixed horizon. For example, with 4 arms and 10 time steps:

- Chromosome: `001011100110...` (20 bits for 10 pulls with 4 arms, 2 bits per choice)
- Decoding: `[0, 2, 3, 3, 0, 1, 2, ...]` (arm indices)

### Fitness Function

The fitness of a strategy is determined by simulating its performance against the bandit environment:

```
fitness(strategy) = average_cumulative_reward_over_multiple_trials
```

Multiple trials are averaged to account for the stochastic nature of rewards.

### Genetic Operators

1. Selection: Tournament selection to choose parents based on fitness
2. Crossover: Single-point or uniform crossover to combine parent strategies
3. Mutation: Bit-flip mutation to introduce exploration

### Evolution Process

1. Initialize random population of strategies
2. Evaluate fitness of each strategy
3. Select parents based on fitness
4. Create offspring through crossover and mutation
5. Replace population with offspring (generational or elitist)
6. Repeat until convergence or maximum generations reached

## Implementation Components

### Core Modules

- `genetic_algorithm.py`: Core GA implementation with selection, crossover, mutation
- `bandit_environment.py`: Multi-arm bandit simulator with various reward distributions
- `encoding.py`: Binary encoding and decoding utilities for strategies
- `fitness.py`: Fitness evaluation functions
- `benchmarks.py`: Standard benchmark problems
- `run_experiment.py`: Main experiment runner with analysis

### Binary Encoding Design

The encoding scheme maps bit strings to arm-pulling sequences:

- For N arms, we need ceil(log2(N)) bits per decision
- A strategy for T time steps requires T * ceil(log2(N)) bits
- Invalid encodings (bits representing arm indices >= N) are handled by modulo mapping

Example for 3 arms (requires 2 bits per choice):
```
Bits: 00 01 10 11 00 10
Arms:  0  1  2  3  0  2
         (3 mod 3 = 0, wraps to valid arm)
```

### Fitness Function Design

The fitness function evaluates strategy quality:

```python
def fitness(chromosome, bandit, num_trials=100, horizon=50):
    # Decode binary chromosome to arm sequence
    strategy = decode_chromosome(chromosome)

    # Run multiple trials
    total_reward = 0
    for trial in range(num_trials):
        bandit.reset()
        for t in range(horizon):
            arm = strategy[t % len(strategy)]
            reward = bandit.pull(arm)
            total_reward += reward

    return total_reward / num_trials
```

## Benchmark Problems

### 1. Bernoulli Bandit

Arms provide binary rewards (0 or 1) with fixed probabilities:
- 4 arms with success probabilities: [0.1, 0.5, 0.3, 0.7]
- Optimal strategy: always pull arm 3 (p=0.7)

### 2. Gaussian Bandit

Arms provide rewards from normal distributions:
- 4 arms with (mean, std): [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (1.5, 1.0)]
- Optimal strategy: always pull arm 3 (mean=1.5)

### 3. Non-Stationary Bandit

Arm reward distributions change over time, testing adaptability:
- Reward probabilities shift every 100 pulls
- Tests whether GA can find robust strategies

## Running the Experiment

Basic usage:
```bash
cd experiments/multi_arm_bandit
uv run python run_experiment.py
```

With custom parameters:
```bash
uv run python run_experiment.py --population-size 100 --generations 50 --mutation-rate 0.01
```

## Expected Results

The genetic algorithm should:

1. Converge to near-optimal strategies for stationary bandits
2. Show clear fitness improvement over generations
3. Discover exploitation-focused strategies for clear optimal arms
4. Maintain some exploration in uncertain environments

Performance metrics tracked:
- Best fitness per generation
- Average fitness per generation
- Diversity of population
- Convergence rate

## Analysis and Visualization

The experiment generates:

- Fitness evolution plots showing improvement over generations
- Strategy diversity metrics over time
- Comparison with baseline strategies (random, greedy, epsilon-greedy)
- Final best strategy visualization

## Theoretical Background

This experiment bridges evolutionary computation and reinforcement learning:

- Genetic algorithms provide a black-box optimization approach
- No gradient information or value function required
- Can handle non-differentiable reward functions
- Population-based search explores solution space broadly

Limitations compared to classic bandit algorithms:
- Requires fixed horizon (cannot adapt policy online)
- Computationally expensive (many fitness evaluations)
- Does not leverage problem structure (reward distributions)

However, GA advantages:
- Can optimize complex, non-standard objectives
- Naturally handles constraints on strategy structure
- Provides interpretable final strategies
- Robust to local optima through population diversity

## Future Extensions

Potential improvements and variations:

1. Co-evolution with dynamic bandit environments
2. Multi-objective optimization (reward vs. risk)
3. Hierarchical encoding for contextual bandits
4. Hybrid approaches combining GA with traditional bandit algorithms
5. Transfer learning across different bandit configurations

## References

Key concepts from:
- Reinforcement Learning: Multi-arm bandit problems
- Evolutionary Computation: Genetic algorithms, encoding schemes
- Optimization Theory: Exploration-exploitation tradeoffs
