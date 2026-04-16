## Profiling with xprof

The script has built-in profiling support via `torch_xla.debug.profiler`. Enable it with `--profile`:

```bash
python qwen_inference.py --profile --profile-duration 20000
```

This starts a profiler server on port `9012` and captures a trace during each inference iteration. Traces are saved to `/tmp/data/profiler_out_tRiLlium_eXp/`.

### Viewing traces

Install and launch TensorBoard with the profile plugin:

```bash
pip install tensorboard tensorboard_plugin_profile
tensorboard --logdir /tmp/data/profiler_out_tRiLlium_eXp --port 6006
```

Or use [xprof](https://github.com/openxla/xprof) directly for the full analysis UI.

### Key xprof views

| View | What it shows |
|---|---|
| **Overview Page** | High-level summary: step time, device utilization, top ops |
| **Trace Viewer** | Interactive timeline of CPU and TPU execution. Use W/S to zoom, A/D to pan |
| **Memory Viewer** | HBM usage over time, peak allocation, fragmentation |
| **HLO Op Stats** | Per-operation timing and resource consumption |
| **Roofline Analysis** | Whether ops are compute-bound or memory-bound relative to hardware peak |
| **Graph Viewer** | Visualizes the HLO operation graph and op placement |

### Detecting performance bottlenecks

**TPU idle time:** In the Trace Viewer, look at the "XLA Ops" section. Gaps (white space) between colored blocks indicate the TPU is idle — usually caused by host-side Python overhead, synchronization waits, or data transfers.

**Host-device transfers:** Look for `TransferToDevice` / `TransferFromDevice` ops in the timeline. These block the TPU. Common causes: calling `.item()`, `.cpu()`, or `.nonzero()` mid-computation, which forces a sync.

**Compilation overhead:** The first run compiles XLA graphs (~95 min here). Check the metrics report for `CompileTime` vs `ExecuteTime`. After warmup, `CachedCompile` count should be high and `CompileTime` should be near zero.

**Slow operations:** Use HLO Op Stats to sort by total time. Ops with disproportionate time are optimization targets. Cross-reference with the Roofline view to understand if they're compute-bound or memory-bound.

### Detecting memory bottlenecks

**HBM pressure:** The Memory Viewer shows HBM allocation over time. If peak usage is near the chip's HBM capacity (16 GB per v6e chip), you're at risk of OOM. Look for:
- Allocation spikes during the forward pass
- Memory not being freed between steps (accumulation/leak)
- Fragmentation — gaps in the memory timeline where free memory exists but is unusable

**Memory-bound ops:** The Roofline chart plots each op's arithmetic intensity against throughput. Ops falling below the memory bandwidth roofline are memory-bound and may benefit from fusion, tiling, or reduced precision.

### Determining full TPU utilization

A fully utilized TPU shows:

1. **No idle gaps** in the Trace Viewer XLA Ops section — the device is continuously executing
2. **High MXU utilization** — check the Overview Page or HLO Op Stats. The MXU (Matrix Multiply Unit) is the main compute engine; low utilization means ops aren't matmul-heavy or are too small to saturate the unit
3. **Ops near the roofline** — in the Roofline view, ops close to the peak compute or peak bandwidth line are well-optimized
4. **Minimal transfer time** — `TransferToDeviceTime` and `TransferFromDeviceTime` should be negligible compared to `ExecuteTime`

### Metrics report

The script also saves a `torch_xla.debug.metrics` report to `/tmp/metrics_report.txt` after each run. Key counters to check:

```
CompileTime        — should be ~0 after first run (cached)
ExecuteTime        — actual device execution time
TransferToDeviceTime   — host-to-device data movement
TransferFromDeviceTime — device-to-host data movement
CachedCompile      — cache hits (should match step count after warmup)
```

For a quick check during development:

```python
import torch_xla.debug.metrics as met
print(met.short_metrics_report(
    counter_names=['CachedCompile', 'MarkStep'],
    metric_names=['CompileTime', 'ExecuteTime', 'TransferToDeviceTime']
))
```

### Example profiling results

Below is an example metrics report from a profiled run with `--num-inference-steps 5 --itters 3`:

**Compilation vs execution:**

| Metric | Value | Notes |
|---|---|---|
| `UncachedCompile` | 15 | Unique XLA graphs compiled |
| `CachedCompile` | 25 | Cache hits (graphs reused across steps) |
| `CompileTime` (total) | 12m 22s | One-time cost; median per-graph ~12s, largest ~2m 55s |
| `ExecuteReplicatedTime` (total) | 39.4s | Actual device execution across all steps |

Compilation dominates the first run. The 15 unique graphs correspond to different code paths (conditional/unconditional forward pass, VAE decode, scheduler step, etc.). After warmup, all executions are cache hits.

**Data transfer overhead:**

| Metric | Total time | Samples | Notes |
|---|---|---|---|
| `TransferToDeviceTime` | 569ms | 2178 | Host-to-device; median 96us per transfer — negligible |
| `TransferFromDeviceTime` | 3.4s | 20 | Device-to-host; mostly VAE output + image save |
| `OutboundData` | 117 GB | 2178 | Total data sent to device (weights + activations) |
| `InboundData` | 1.24 GB | 20 | Total data received from device |

Transfer-to-device is fast (sub-millisecond per op). Transfer-from-device is dominated by a few large transfers (p95 = 3.5s) corresponding to final image retrieval.

**Graph characteristics:**

| Metric | Value |
|---|---|
| `TensorsGraphSize` (median) | 44,136 ops per graph |
| `LazyTracing` (total) | 2m 46s across 940K trace points |
| `Pallas flash attention cache hits` | 2,398 |

The large graph size (~44K ops) reflects the full transformer forward pass fused into a single XLA graph. The high Pallas cache hit count confirms flash attention kernels are being reused efficiently.

**What to look for in the Trace Viewer:**

Once you launch TensorBoard (`tensorboard --logdir /tmp/data/profiler_out_tRiLlium_eXp --port 6006`), select the larger trace file (885MB) and look for:

1. **Continuous execution blocks** in the device timeline — gaps indicate TPU idle time
2. **`ExecuteReplicated` spans** — each one is a full graph execution (denoising step)
3. **`TransferFromDevice` events** — should only appear at the end (image retrieval), not mid-loop
4. **SPMD collectives** (all-gather, reduce-scatter) — overhead from the `(data, model)` mesh sharding