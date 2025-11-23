"""
Quick script to inspect SuperGPQA dataset format.
"""

from datasets import load_dataset
import json

# Load dataset
print("Loading SuperGPQA dataset...")
dataset = load_dataset("m-a-p/SuperGPQA", split="train")

print(f"\nDataset size: {len(dataset)}")
print(f"\nDataset features: {dataset.features}")

# Show first example
print("\nFirst example:")
example = dataset[0]
print(json.dumps(example, indent=2, ensure_ascii=False))

# Show field names
print("\nField names:")
for key in example.keys():
    print(f"  - {key}: {type(example[key])}")
