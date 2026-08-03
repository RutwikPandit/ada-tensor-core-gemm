# RTX 4060 Tensor Core and GEMM Pipeline Experiment Plan

## Goal

Use an RTX 4060 Laptop GPU to understand and measure the production GEMM
tradeoffs discussed for Hopper and Blackwell:

- asynchronous global-to-shared-memory movement
- multistage buffering
- Tensor Core issue and utilization
- shared-memory capacity versus CTA residency
- register pressure and accumulator ownership
- tile shape, problem shape, and GPU-wide load balance
- shared-memory bank conflicts and swizzled layouts
- fused epilogues

The final objective is not merely to produce a fast GEMM. It is to explain a
kernel's performance from its ownership, data movement, synchronization, and
resource usage, and then verify that explanation in Nsight Compute.

## Hardware Scope

The RTX 4060 Laptop GPU is an Ada Lovelace GPU:

```text
Architecture:         Ada Lovelace
Compute capability:   8.9 (SM89)
Tensor Core version:  Fourth generation
CUDA cores:           3,072
Memory:               8 GB GDDR6
Laptop power range:   OEM-dependent, nominally 35-115 W
```

Ada supports Tensor Core `mma.sync` operations and asynchronous
global-to-shared-memory copies using `cp.async`/LDGSTS.

It does not support Hopper-specific TMA, WGMMA, thread-block clusters, or
Blackwell TMEM. Therefore, this machine cannot reproduce those mechanisms
literally. It can reproduce the central pipeline tradeoff:

```text
Ada experiment:
GDDR6 -> cp.async -> staged shared memory -> mma.sync -> registers -> output

Hopper analogue:
HBM -> TMA -> staged shared memory -> WGMMA -> registers -> output
```

On Ada, a consumer waiting for TMA becomes a warp reaching
`cp.async.wait_group` before the required asynchronous copy group has finished.

## Success Definition

"100% utilization" must be defined carefully. We will track three different
quantities:

1. **Application throughput:** measured dense FP16/BF16 TFLOP/s.
2. **Local speed of light:** performance relative to cuBLASLt or the best
   applicable CUTLASS kernel on this exact laptop under the same conditions.
3. **Hardware activity:** Tensor Core and SM activity reported by Nsight
   Compute.

The primary target is:

```text
custom kernel throughput >= 90% of the best local cuBLASLt/CUTLASS result
high Tensor Core pipe activity on a favorable square GEMM
no register spills
stable clock, temperature, and power behavior
an understood reason for every major stall category
```

Reaching 100% of a marketing peak is not a reliable target on a laptop. GPU
boost clocks and power limits vary with cooling, OEM configuration, and the
rest of the system workload.

## Phase 0: Validate the Laptop

Run the following on the gaming laptop:

```powershell
nvidia-smi --query-gpu=name,compute_cap,power.limit,memory.total,driver_version --format=csv
nvidia-smi -q -d CLOCK,POWER
nvcc --version
ncu --version
```

Also build a small CUDA device-query program or run CUDA Samples `deviceQuery`
to record:

- SM count
- maximum shared memory per block and per SM
- register-file limits
- maximum active warps and CTAs per SM
- supported CUDA compute capability

Expected compilation target:

```text
-arch=sm_89
```

Before collecting results:

- connect AC power
- enable the laptop's maximum-performance cooling and power profile
- use discrete-GPU-only/MUX mode when available
- close games, browsers, overlays, and other GPU workloads
- warm up the kernel before timing
- record actual SM clock, power, and temperature during every experiment
- keep matrix dimensions, initialization, iteration count, and timing method
  identical across variants

## Phase 1: Establish the Local Speed of Light

Benchmark dense FP16 input with FP32 accumulation using:

1. cuBLASLt
2. CUTLASS Profiler or an equivalent known-good CUTLASS GEMM
3. the custom kernel developed for the stage experiments

Start with square, well-aligned problems:

```text
M = N = K = 4096
M = N = K = 8192
```

Use enough warm-up and measured iterations that kernel-launch and timing noise
are insignificant. Compute throughput as:

```text
TFLOP/s = 2 * M * N * K / elapsed_seconds / 1e12
```

The best library result under stable clocks becomes the local speed-of-light
baseline. Do not use NVIDIA's aggregate AI TOPS number as the FP16 dense GEMM
denominator.

## Phase 2: Pipeline Stage Sweep

Use one parameterized Tensor Core GEMM and hold all choices constant except the
number of shared-memory stages.

Initial candidate geometry:

```text
CTA output tile:  BM x BN = 128 x 128
K tile:           BK      = 32
Input type:       FP16
Accumulator:      FP32
```

Shared-memory use for operands is approximately:

```text
bytes_per_stage = sizeof(FP16) * (BM * BK + BK * BN)
                = 2 * (128 * 32 + 32 * 128)
                = 16 KiB
```

| Variant | Operand SMEM | Expected interpretation |
|---|---:|---|
| Synchronous baseline | 16 KiB | Load completes before compute; no useful overlap |
| 2 stages | 32 KiB | Basic producer/consumer double buffering |
| 3 stages | 48 KiB | More latency coverage at moderate SMEM cost |
| 4 stages | 64 KiB | Potentially fewer copy-related waits, lower residency |
| 5 stages | 80 KiB | Likely one CTA per SM; may provide no further benefit |

Ada permits roughly 99 KiB of shared memory per block, but implementation
metadata, padding, and epilogue storage must also fit. Exact compiled resource
usage must be checked rather than inferred from operand sizes alone.

### Hypothesis

```text
Too few stages:
    consumer reaches cp.async.wait_group before data is ready
    Tensor Cores are starved

Enough stages:
    future K tiles are copied while current K tiles execute
    copy latency is mostly hidden

Too many stages:
    shared-memory use increases
    active CTAs per SM can decrease
    synchronization and index bookkeeping increase
    throughput plateaus or falls
```

For every stage count, record:

- kernel time and TFLOP/s
- percentage of local library speed of light
- registers per thread
- static and dynamic shared memory per CTA
- achieved occupancy and active CTAs per SM
- Tensor Core pipe activity
- long-scoreboard/dependency stalls
- barrier and wait stalls
- shared-memory throughput and bank conflicts
- DRAM and L2 throughput
- SM clock, power, and temperature

## Phase 3: Tile-Shape Experiments

Hold datatype and problem size fixed while testing several CTA tiles:

```text
64 x 128
128 x 128
128 x 256
```

Questions to answer:

- Does the larger tile improve reuse of A and B?
- Does it require too many registers or too much shared memory?
- How many CTAs can reside on each SM?
- Are there enough output tiles to create multiple full waves across the GPU?
- Does a rectangular tile suit the matrix shape better than a square tile?

Repeat one problem with a tile count that divides the SM count cleanly and one
that leaves a partial final wave. This isolates GPU-wide load imbalance from
the efficiency of an individual CTA.

## Phase 4: Shared-Memory Layout

Compare:

1. a simple row-major shared-memory layout
2. padded layouts
3. an MMA-compatible XOR/swizzled layout

Hold the global access pattern and MMA shape constant. Measure:

- shared-memory bank conflicts
- shared-memory instruction count and throughput
- Tensor Core starvation caused by operand collection
- end-to-end GEMM throughput

The goal is to connect a logical layout change to physical bank selection and
then to MMA operand-delivery rate.

## Phase 5: Register and Accumulator Pressure

Change the amount of output owned by each warp or thread while keeping the CTA
tile as stable as possible.

Expected tradeoff:

```text
larger register tile
    -> more operand reuse and instruction-level parallelism
    -> more accumulator registers
    -> potentially fewer resident warps/CTAs
    -> possible spilling if pushed too far
```

Inspect compiled register count and local-memory traffic. Any local-memory
loads or stores caused by register spilling are considered a failed
configuration for the high-utilization target.

## Phase 6: Workload Shape

Sweep M while keeping N and K large:

```text
M = 1, 8, 32, 128, 512, 4096
```

This moves from decode-like GEMV/skinny GEMM toward a training- or prefill-like
square GEMM. It should demonstrate that an excellent square-GEMM kernel cannot
retain the same Tensor Core utilization when there is insufficient parallelism
or data reuse.

Also sweep K to observe pipeline startup and drain overhead:

```text
small K  -> prologue/epilogue and pipeline fill dominate
large K  -> steady-state mainloop dominates
```

## Phase 7: Alignment and Vectorization

Compare aligned vectorized global loads with deliberately weaker scalar or
misaligned accesses.

Questions to answer:

- Is global-memory transaction count increasing?
- Can `cp.async` use the intended transfer width?
- Does L2/DRAM traffic rise for the same useful bytes?
- Does the MMA pipeline become starved even though its computation is
  unchanged?

## Phase 8: Epilogue Fusion

Compare:

```text
GEMM -> write C -> separate bias/activation kernel
```

against:

```text
GEMM -> fused scaling/bias/activation -> write D
```

Measure launch count, intermediate memory traffic, total latency, and whether
the fused epilogue increases register pressure enough to hurt the GEMM
mainloop.

## Nsight Compute Collection

Start with section-based profiling because exact raw metric names can vary by
Nsight Compute and GPU version:

```powershell
ncu `
  --section SpeedOfLight `
  --section ComputeWorkloadAnalysis `
  --section MemoryWorkloadAnalysis `
  --section Occupancy `
  --section SchedulerStats `
  --section WarpStateStats `
  --section LaunchStats `
  --export stage_3 `
  .\gemm_bench.exe --stages 3 --m 8192 --n 8192 --k 8192
```

Use `--set full` only for selected finalists because metric replay can make a
complete sweep very slow.

For each report, answer these questions in order:

1. Is the kernel compute-bound or memory-bound?
2. Is the Tensor Core pipeline active enough to explain achieved TFLOP/s?
3. If Tensor Cores are idle, are operand copies, shared-memory delivery,
   synchronization, instruction issue, or insufficient parallelism responsible?
4. Did shared memory or register use reduce CTA residency?
5. Are there bank conflicts or register spills?
6. Is the final wave underfilling the GPU?
7. Did clock, power, or temperature change enough to invalidate the comparison?

## Result Table

Maintain one row per kernel configuration:

| Kernel | M/N/K | CTA tile | K tile | Stages | Reg/thread | SMEM/CTA | CTA/SM | TFLOP/s | % local SoL | Tensor activity | Dominant stall | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| cuBLASLt best | | | | | | | | | 100% | | | |
| CUTLASS best | | | | | | | | | | | | |
| Custom sync | | | | 1 | | | | | | | | |
| Custom async | | | | 2 | | | | | | | | |
| Custom async | | | | 3 | | | | | | | | |
| Custom async | | | | 4 | | | | | | | | |
| Custom async | | | | 5 | | | | | | | | |

## Deliverables

At completion, retain:

1. A reproducible build and run command for `sm_89`.
2. A correctness check against a CPU or cuBLAS reference.
3. A cuBLASLt/CUTLASS local speed-of-light result.
4. A parameterized multistage `cp.async` Tensor Core GEMM.
5. Nsight Compute reports for representative good and bad configurations.
6. Plots of stage count versus throughput, occupancy, and dominant stalls.
7. A short architecture explanation connecting each profiler result to the
   corresponding hardware mechanism.
8. A final note explaining what carries over to Hopper TMA/WGMMA and what must
   be reimplemented on SM90 hardware.

## Recommended Execution Order

```text
validate hardware and tools
    -> establish cuBLASLt/CUTLASS local SoL
    -> verify one correct Tensor Core GEMM
    -> sweep pipeline stages
    -> sweep CTA and register tiles
    -> test shared-memory layouts
    -> test square versus skinny workloads
    -> test epilogue fusion
    -> profile finalists deeply
    -> explain the results architecturally
```

## References

- [NVIDIA GeForce RTX 40 Series Laptop GPU comparison](https://www.nvidia.com/en-au/geforce/laptops/compare/)
- [NVIDIA Ada GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/)
- [CUDA Programming Guide: Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)
- [NVIDIA Ada Compatibility Guide](https://docs.nvidia.com/cuda/ada-compatibility-guide/)

