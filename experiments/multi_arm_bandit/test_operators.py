#!/usr/bin/env python3
"""Quick test to verify operator registration fix."""

import sys
import os

# Add both experiments and project root to path
experiments_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(experiments_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, experiments_dir)

# Import to test operator registration
import nons.operators.base
import nons.operators.deterministic
from nons.core.network import NoN
from multi_arm_bandit import random_network_chromosome

print("Testing operator registration fix...")
print("=" * 60)

# Create a simple network
chromosome = random_network_chromosome(min_layers=2, max_layers=3)

print(f"\nGenerated network with {len(chromosome.genes)} layers:")
for i, gene in enumerate(chromosome.genes):
    ops = ", ".join(gene.operators)
    print(f"  Layer {i+1}: {ops}")

# Try to build the network
try:
    operator_spec = chromosome.to_operator_spec()
    print(f"\nOperator spec: {operator_spec}")

    network = NoN.from_operators(operator_spec)
    print(f"\n✓ SUCCESS: Network built successfully!")
    print(f"  Network has {len(network.layers)} layers")

except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
