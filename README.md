
# Bandit Algorithms Comparison
This Python code repository is designed to compare and visualize the performance of two popular multi-armed bandit algorithms: Thompson Sampling and Epsilon-Greedy. The code allows you to experiment with different bandit probabilities and hyperparameters and provides insightful visualizations to analyze the results.

## Overview
Multi-armed bandit algorithms are commonly used in scenarios where we need to make decisions under uncertainty. This repository contains two concrete implementations of bandit algorithms:

### Thompson Sampling: This algorithm leverages Bayesian methods to estimate bandit success probabilities and make decisions that balance exploration and exploitation.

### Epsilon-Greedy: Epsilon-Greedy is a simple yet effective algorithm that explores with a certain probability (ε) and exploits with the complementary probability (1-ε).

## Key Features
Configurable: Easily customize the number of bandit arms, bandit probabilities, and hyperparameters for each algorithm.

Automated Parameter Tuning: Experiment with different hyperparameters using automated tuning to optimize performance.

Visualization: Visualize bandit performance and compare cumulative rewards and regrets for both algorithms.

