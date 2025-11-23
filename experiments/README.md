# Experiments

This directory contains experimental implementations exploring advanced optimization and learning algorithms within the NoN framework.

## Purpose

The experiments directory serves as a research and development space for testing novel approaches to compound AI system optimization. Each experiment is self-contained and focuses on solving specific benchmark problems or exploring particular algorithmic strategies.

## Structure

Each experiment is organized in its own subdirectory with the following components:

- Implementation code for the algorithm
- Benchmark problems and test cases
- Documentation explaining the approach
- Results and analysis utilities

## Available Experiments

### 1. Multi-Arm Bandit Genetic Algorithm

Location: `multi_arm_bandit/`

An exploration of using genetic algorithms to solve multi-arm bandit problems. This experiment demonstrates how evolutionary computation can be applied to optimize decision-making strategies in uncertain environments.

Key features:
- Binary encoding of arm-pulling strategies
- Fitness evaluation based on cumulative rewards
- Genetic operators for strategy evolution
- Standard benchmark problems (Bernoulli, Gaussian bandits)

## Running Experiments

Each experiment includes its own runner script. Navigate to the experiment directory and follow the instructions in its README.

Example:
```bash
cd experiments/multi_arm_bandit
uv run python run_experiment.py
```

## Development Guidelines

When adding new experiments:

1. Create a new subdirectory with a descriptive name
2. Include a comprehensive README explaining the experiment
3. Implement clean, well-documented code following SOLID principles
4. Provide benchmark problems to validate the approach
5. Include analysis utilities for results visualization
6. Update this README with a brief description of the new experiment

## Research Focus Areas

Current and planned experimental directions:

- Evolutionary algorithms for AI system optimization
- Reinforcement learning integration with NoN networks
- Meta-learning approaches for operator composition
- Automated network architecture search
- Multi-objective optimization strategies

## Notes

Experiments are research-oriented and may not be production-ready. Code in this directory prioritizes exploratory flexibility over stability guarantees. For production use cases, refer to the main NoN package.
