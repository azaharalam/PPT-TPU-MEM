# tools/test_script.py

import glob
import os
import numpy as np
from ReuseDistance.reuse_distance_engine import ReuseDistanceEngine

def main():
    # 1) Instantiate with your TPU parameters (matching your Eyeriss config):
    engine = ReuseDistanceEngine(
        array_height=12,
        array_width=14,
        line_size=16,
        ub_kb=36,
        lambda_spatial=0.5,
        beta=5.0,
        delta_flow=0.8,   # Weight-Stationary, for example
        t_sram=2,
        t_dram=100,
        interface_bw=10
    )

    # 2) Find all layer directories under test_runs/eyeriss
    base_dir = "test_runs/eyeriss"
    layer_dirs = sorted(glob.glob(os.path.join(base_dir, "layer*")))

    if not layer_dirs:
        print(f"No layer directories found under {base_dir}")
        return

    # 3) For each layer, compute metrics using the 2-column .npy
    for layer in layer_dirs:
        npy_path = os.path.join(layer, "UNIFIED_TRACE.npy")
        if not os.path.isfile(npy_path):
            print(f"Missing .npy in {layer}, skipping.")
            continue
        
        
        print(f"\n=== Layer: {layer} ===")
        metrics = engine.compute_metrics(npy_path)
        # Pretty-print the results for this layer
        print(f"Total accesses : {metrics['total_accesses']}")
        print(f"Predicted hit-rate     : {metrics['hit_rate']:.4f}")
        print(f"Predicted miss_count   : {metrics['miss_count']:.1f}")
        print(f"Predicted DRAM cycles  : {metrics['bw_cycles']}")
        print(f"Predicted AMAT (cycles): {metrics['amat']:.2f}")

if __name__ == "__main__":
    main()
