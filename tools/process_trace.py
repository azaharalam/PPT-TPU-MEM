import os
import csv
import heapq
import glob
import sys
import numpy as np
from collections import defaultdict

class TraceProcessor:
    
    # Configuration constants (LOad from config(EYERISS)) ─────────────
    ARRAY_HEIGHT = 12        # number of PE rows
    ARRAY_WIDTH  = 14        # number of PE columns
    LINE_SIZE    = 16        # bytes per cache line
    LINE_BITS    = LINE_SIZE.bit_length() - 1  # = 4

    # Synthetic operand offsets from your Eyeriss config
    OFFSETS = {
        'I': 0,           # IfmapOffset
        'F': 10_000_000,  # FilterOffset
        'O': 20_000_000,  # OfmapOffset
    }

    def __init__(self, parent_dir):
        if not os.path.isdir(parent_dir):
            raise ValueError(f"Provided path is not a directory: {parent_dir}")
        self.parent_dir = parent_dir

    def map_pe(self, tag, idx):
        if tag == 'I':
            return idx, 0
        if tag == 'F':
            return 0, idx
        return divmod(idx, self.ARRAY_WIDTH)

    def read_trace(self, layer_dir, filename, tag, is_write):

        path = os.path.join(layer_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing expected trace: {path}")

        with open(path) as f:
            reader = csv.reader(f)
            for row_vals in reader:
                try:
                    cycle = int(float(row_vals[0]))
                except:
                    continue

                for idx, cell in enumerate(row_vals[1:]):
                    try:
                        addr = int(float(cell))
                        if addr < 0:continue
                        addr -= self.OFFSETS[tag] #Substracting the offset
                        if addr < 0:continue
                    except:
                        continue
          
                    line = addr >> self.LINE_BITS

                    # Map to (row, col) in the PE grid
                    r, c = self.map_pe(tag, idx)

                    yield (cycle, tag, r, c, line, is_write)

    def merge_traces_for_layer(self, layer_dir):
        old_csv = os.path.join(layer_dir, "UNIFIED_TRACE.csv")
        old_npy = os.path.join(layer_dir, "UNIFIED_TRACE.npy")
        if os.path.exists(old_csv):
            os.remove(old_csv)
        if os.path.exists(old_npy):
            os.remove(old_npy)
        out_csv = os.path.join(layer_dir, "UNIFIED_TRACE.csv")
        os.makedirs(layer_dir, exist_ok=True)

        streams = [
            self.read_trace(layer_dir, "IFMAP_DRAM_TRACE.csv",  'I', 0),
            self.read_trace(layer_dir, "FILTER_DRAM_TRACE.csv", 'F', 0),
            self.read_trace(layer_dir, "OFMAP_DRAM_TRACE.csv",  'O', 1),
        ]
        merged_iter = heapq.merge(*streams, key=lambda e: e[0])

        with open(out_csv, 'w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(['cycle', 'op', 'row', 'col', 'line', 'is_write'])
            for evt in merged_iter:
                writer.writerow(evt)

        print(f"Merged DRAM traces → {out_csv}")
        return out_csv
    
    def interleave_rows_for_layer(self, unified_csv):
        buckets = defaultdict(list)
        with open(unified_csv) as f:
            reader = csv.reader(f)
            next(reader)  
            for cycle, op, row, col, line, is_write in reader:
                line_int = int(line)
                if line_int < 0:
                    continue
                buckets[int(row)].append(line_int)

        idx = [0] * self.ARRAY_HEIGHT
        interleaved_pairs = []  # will hold (row_id, line_addr) tuples
        while any(idx[r] < len(buckets[r]) for r in range(self.ARRAY_HEIGHT)):
            for r in range(self.ARRAY_HEIGHT):
                if idx[r] < len(buckets[r]):
                    line_addr = buckets[r][idx[r]]
                    interleaved_pairs.append((r, line_addr))
                    idx[r] += 1

        # Convert to a 2D NumPy array of shape (N,2)
        arr = np.array(interleaved_pairs, dtype=np.uint32)  # now shape == (N, 2)
        out_npy = unified_csv.replace(".csv", ".npy")
        np.save(out_npy, arr)
        return out_npy

    def run(self):
        
        pattern = os.path.join(self.parent_dir, "layer*")
        layer_dirs = sorted(glob.glob(pattern))
        if not layer_dirs:
            print(f"No subdirectories matching pattern: {pattern}")
            return

        for layer_dir in layer_dirs:
            print(f"\nProcessing layer directory: {layer_dir} …")

            try:
                unified_csv = self.merge_traces_for_layer(layer_dir)
            except Exception as e:
                print(f" Failed merge for {layer_dir}: {e}")
                continue

            try:
                self.interleave_rows_for_layer(unified_csv)
            except Exception as e:
                print(f" Failed interleave for {layer_dir}: {e}")
                continue


# ────────────────────────────────────────────────────────────────────────────────
# Usage Example
# Todo: Incorporate with main simulation
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python trace_processor.py /path/to/parent_dir_containing_layerX_folders")
        sys.exit(1)

    parent_directory = sys.argv[1]
    processor = TraceProcessor(parent_directory)
    processor.run()
