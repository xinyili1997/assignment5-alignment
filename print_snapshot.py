#!/usr/bin/env python3
"""Print expected values from a snapshot file."""

import numpy as np

# Load the snapshot
snapshot_path = "tests/_snapshots/test_grpo_microbatch_train_step_grpo_clip_10_steps.npz"
expected_arrays = dict(np.load(snapshot_path))

print("Expected output from snapshot:")
print("=" * 60)
for key, value in expected_arrays.items():
    print(f"\n{key}:")
    print(f"  Shape: {value.shape}")
    print(f"  Dtype: {value.dtype}")
    print(f"  Values:\n{value}")
    if value.size > 20:
        print(f"  (showing full array)")
    print("-" * 60)
