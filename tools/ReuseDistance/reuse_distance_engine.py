import os
import math
import numpy as np
from collections import defaultdict


class ReuseDistanceEngine:

    def __init__(
        self,
        array_height: int,
        array_width: int,
        line_size: int,
        ub_kb: int,
        lambda_spatial: float,
        beta: float,
        delta_flow: float,
        t_sram: float,
        t_dram: float,
        interface_bw: float,
    ):
        self.array_height = array_height
        self.array_width = array_width

        self.line_size = line_size
        self.line_bits = int(math.log2(line_size))

        self.ub_kb = ub_kb
        self.ub_lines = (ub_kb * 1024) // line_size

        self.lambda_spatial = lambda_spatial  
        self.beta = beta                      
        self.delta_flow = delta_flow          

        self.t_sram = t_sram                  # on‐chip SRAM latency (cycles)
        self.t_dram = t_dram                  # DRAM latency (cycles)
        self.interface_bw = interface_bw      # words per cycle
        self.bus_width = interface_bw * line_size  # bytes per cycle

        self.D_max = 2 ** 20

    def compute_metrics(self, npy_path: str) -> dict:

        try:
            arr = np.load(npy_path)
        except Exception as e:
            raise IOError(f"Failed to load .npy file: {e}")

        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Input .npy must be of shape (N, 2), storing [row_id, line_addr] pairs.")

        N = arr.shape[0]
        rows = arr[:, 0].tolist()
        lines = arr[:, 1].tolist()

        
        prev = {}      
        nxt = {}       
        head = None    
        tail = None    

        rows_since = defaultdict(set)  
        hist_temp = [0] * (self.D_max + 1)

        sum_p_hit = 0.0
        sum_p_miss = 0.0

        for i in range(N):
            r_i = rows[i]
            a_i = lines[i]

            pos = 0
            cur = head
            found = False
            while cur is not None:
                if cur == a_i:
                    found = True
                    break
                cur = nxt[cur]
                pos += 1

            if not found:
                # Cold miss
                dist_temporal = self.D_max
                hist_temp[self.D_max] += 1
            else:
                dist_temporal = pos if pos < self.D_max else self.D_max
                hist_temp[dist_temporal] += 1

            if a_i not in rows_since or not rows_since[a_i]:
                dist_spatial = 0
            else:
                dist_spatial = len(rows_since[a_i])

            if dist_temporal == self.D_max:
                rd_eff = float("inf")
            else:
                rd_eff = (
                    (dist_temporal / (1.0 + self.lambda_spatial * dist_spatial))
                    * self.delta_flow
                )

            if rd_eff == float("inf"):
                p_hit = 0.0
            else:
                # z = −β (C_lines − rd_eff)
                z = -self.beta * (self.ub_lines - rd_eff)
                if z > 50:
                    p_hit = 1.0
                elif z < -50:
                    p_hit = 0.0
                else:
                    p_hit = 1.0 / (1.0 + math.exp(z))

            sum_p_hit += p_hit
            sum_p_miss += (1.0 - p_hit)

            if not found:
                prev[a_i] = None
                nxt[a_i] = head
                if head is not None:
                    prev[head] = a_i
                head = a_i
                if tail is None:
                    tail = a_i
            else:
                if head != a_i:
                    p = prev[a_i]
                    n = nxt[a_i]
                    if p is not None:
                        nxt[p] = n
                    if n is not None:
                        prev[n] = p
                    if tail == a_i:
                        tail = p
                    prev[a_i] = None
                    nxt[a_i] = head
                    if head is not None:
                        prev[head] = a_i
                    head = a_i

            rows_since[a_i].clear()
            rows_since[a_i].add(r_i)

            # (naïve O(S) approach)
            for ℓ in rows_since:
                if ℓ != a_i:
                    rows_since[ℓ].add(r_i)

        hit_rate = sum_p_hit / N
        miss_count = sum_p_miss

        bytes_to_dram = miss_count * self.line_size
        bw_cycles = int(math.ceil(bytes_to_dram / self.bus_width))

        amat = (
            hit_rate * self.t_sram
            + (1 - hit_rate) * self.t_dram
            + (bytes_to_dram / self.bus_width) / N
        )

        return {
            "total_accesses": int(N),
            "hit_rate": hit_rate,
            "miss_count": miss_count,
            "bw_cycles": bw_cycles,
            "amat": amat,
            "hist_temporal": hist_temp,  # optional debugging output
        }
    


