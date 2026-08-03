# RTX 4060 Laptop Tensor Core GEMM — Measured Results

Executed against `rtx4060_tensor_core_experiment_plan.md` on this machine.
Windows / PowerShell, CUDA 12.3, MSVC 14.44 (BuildTools 2022), Nsight Compute
2023.3.1. WSL has `ncu` but no `nvcc`, so everything runs natively on Windows.

```powershell
.\scripts\build.ps1             # -> bin\*.exe
.\bin\device_query.exe          # Phase 0 + measured mma and DRAM roofs
.\scripts\sweep.ps1 -M 4096 -N 4096 -K 4096
.\scripts\soak.ps1 -Seconds 240
.\scripts\profile.ps1 -Stages 2 -Layout 2 -Group 2 -Tag finalist
```

## Headline

| | 4096³ | 8192³ |
|---|---:|---:|
| cuBLAS (local SoL) | 31.22 TFLOP/s | 31.04 TFLOP/s |
| custom kernel | **29.50 TFLOP/s** | **29.26 TFLOP/s** |
| % of local SoL (cuBLAS) | **94.5%** | **94.2%** |
| dense tensor occupancy (clock-free) | **91.9–92.4%** | — |
| registers / thread | 116 | 116 |
| spill stores / loads | 0 / 0 | 0 / 0 |
| sustained over 240 s | 28.81 TFLOP/s | — |

The clock-free figure is the one to trust: busy tensor cycles / elapsed cycles =
`137,438,953,472 / (512 × 24) / sm__cycles_elapsed`, which gives **91.9–92.4%** for
our kernel across profiling runs and **97.9%** for cuBLAS. It needs no clock
estimate, so boost behaviour cannot distort it. Percent-of-roof figures depend on
which clock you divide by — see "Hardware speed of light".

> **Verification.** The analysis in this document was independently re-derived by
> eight adversarial review agents instructed to refute it. Every arithmetic result
> survived; four claims did not survive as *written* and are corrected here — the
> memory-bandwidth roof (was asserted as 307.2 GB/s, actually unresolved), the FP8
> rate tier (was 4×, actually 2× with FP32 accumulate), the FP16-accumulate power
> explanation (was "entirely power", actually ~half power / ~half broken latency
> hiding), and cuBLAS's distance from SoL (was 0.6%, actually ~2.1%). Where a
> figure is uncertain it is now given as a range.

Finalist geometry: `BM×BN×BK = 128×128×32`, 8 warps (256 threads), warp tile
64×32, `mma.sync.m16n8k16.row.col.f32.f16.f16.f32`, **2** `cp.async` stages, XOR
swizzled shared memory, CTA rasterization group 2. The plan's ≥90%-of-SoL target
is met at both sizes with no spills.

Two numbers that are easy to get wrong on this GPU: the FP32-accumulate tensor
roof is **512** FLOP/clk/SM (half the FP16-accumulate rate), and Nsight's headline
tensor metric reads **47%** where real dense occupancy is **92.4%**. A third,
memory bandwidth, this project could *not* settle — see Phase 0.

## Phase 0 — hardware, and fixing the denominator

```text
sm_89, 24 SMs, 8 GiB GDDR6 / 128-bit, L2 = 32 MiB
shared memory: 48 KiB default, 101,376 B per-block opt-in, 102,400 B per SM
registers: 65,536 per SM ; 1536 threads (48 warps) per SM ; 24 blocks per SM
SM clock: 3105 MHz "max clocks" table; 2625-2670 MHz sustained under load
          (soak average 2633.7 MHz); ~2596 MHz during the short timed runs
memory clock: 8001 MHz per both self-reported maxima, 9601 MHz observed under load
```

**Memory bandwidth: unresolved between ~260 and ~290 GB/s. Not 256, and not the
307.2 I initially asserted.** The sources genuinely conflict:

| Source | Implies |
|---|---|
| `cudaDeviceProp.memoryClockRate` = 8001 MHz | 256.03 GB/s |
| nvidia-smi "Max Clocks → Memory" = 8001 MHz | 256.03 GB/s |
| nvidia-smi *current* clock under load = 9601 MHz | 307.23 GB/s |
| third-party databases (TechPowerUp, Notebookcheck) | 16 Gbps → 256.0 GB/s |
| **measured 512 MiB float4 stream** | **259.5 GB/s achieved** |

The measured 259.5 GB/s **exceeds 256.03**, so 256 GB/s cannot be the roof —
nothing outruns a real ceiling. But 307.23 GB/s is not supportable either: it
needs 19.2 Gbps effective, which is not a JEDEC GDDR6 speed bin (14/16/18/20),
it contradicts both of the device's own declared maxima, and the "2400 MHz CK"
corroboration is circular (nvidia-smi reports data-rate/2 = 4×CK, so 9601/4 =
2400.25 is the disputed number restated, not independent evidence). Back-solving
from a normal 85–90% efficiency for a GDDR6 read+write stream puts the roof at
**271–287 GB/s (17–18 Gbps)**, which is consistent with the measurement; 259.5
would be 95% of 256 (above the practical ceiling once bus turnaround is counted)
and only 84% of 307.2 (low for a trivially streamable kernel).

**This does not affect any conclusion here**, because the GEMM is compute-bound
by a wide margin under *every* candidate roof — see the roofline in "Hardware
speed of light", which is stated in arithmetic-intensity terms precisely so it
does not depend on picking one.

The single most important correction to the plan. The `mma` FP16-in/**FP32**-
accumulate path on GeForce Ada issues at exactly **half** the rate of the
FP16-in/FP16-accumulate path. Measured with a register-resident `mma` loop timed
by `clock64()` (clock-invariant, so power throttling cannot distort it). All
figures **dense** — no `.sp` sparse variants are used anywhere in this project:

| dense `mma` on sm_89 | measured ops/clk/SM | ratio |
|---|---:|---:|
| `m16n8k16` fp16 in / **fp32** accumulate | 511.9 | 1.00x |
| `m16n8k16` fp16 in / fp16 accumulate | 1024.4 | 2.00x |
| `m16n8k32` int8 in / int32 accumulate | 2048.4 | 4.00x |
| `m16n8k32` fp8-e4m3 in / fp32 accumulate | needs CUDA >= 12.4 | (1024 = 2x, per whitepaper) |

These match NVIDIA's own published figures exactly. Ada whitepaper v2.02,
Appendix A, RTX 4090: "Peak FP16 Tensor TFLOPS with FP16 Accumulate 330.3/660.6"
and "with FP32 Accumulate 165.2/330.4" (dense/sparse). Dividing the dense values
by 128 SM × 2.520 GHz gives **1024.0** and **512.2** FLOP/clk/SM; our measured
1024.4 and 511.9 land within 0.04%.

**Scope, stated precisely: this is a product-SKU restriction, not a property of
compute capability 8.6/8.9.** On the same GA102 die, RTX A6000 and A40 publish
*identical* FP16-accumulate and FP32-accumulate rates (1023.8 FLOP/clk/SM both
rows) — unhalved at cc 8.6. Titan RTX is unhalved at cc 7.5 while the RTX 2080 Ti
on the same TU102 die is halved. A100 and H100 are unhalved (A100: 312 TFLOPS
dense on *both* rows). For datacenter Ada specifically, NVIDIA's own L40 and L4
tables are mutually inconsistent, so "GeForce-only" is proven for cc 7.5 and 8.6
but strictly unproven for 8.9.

The same halving applies to BF16, TF32 and FP8 with FP32 accumulate on GeForce
Ada — it is a general "FP32 accumulate at half rate" property of the datapath,
not something specific to FP16.

So the correct peak for an FP32-accumulating GEMM is
`512 × 24 SMs × f_sm` = **32.4 TFLOP/s at the observed 2.64 GHz**, not the
~46 TFLOP/s a naive `1024 FLOP/clk` calculation gives, and not the marketing
FP16 number (which corresponds to the fp16-accumulate row).

Two consequences:

- 29.50 TFLOP/s is ~91% of *hardware* peak, not the ~64% it would appear to be.
- **Nsight's `sm__pipe_tensor_op_hmma_cycles_active` tops out near 50% for this
  kernel class.** cuBLAS itself only reaches 49.68%. Use
  `sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off` instead, which reports
  **94.00%** for the same launch. See "Reading Tensor Core activity correctly".

Timing a wall-clock TFLOP/s "peak" instead would have been actively misleading:
a pure back-to-back `mma` loop is the densest possible power draw, so on a
35–115 W laptop part it clocks down and reports a "peak" of 22.4 TFLOP/s —
*below* the GEMM that is supposedly approaching it.

## Phase 1 — local speed of light

`cublasGemmEx(CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)` selects
`ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_tn`: same 128×128 CTA
tile, but only **128 threads and 234 registers/thread** — a much larger
per-warp register tile, trading occupancy for ILP. At 31.22 TFLOP/s it is at
~96% of the measured hardware peak, so it is a fair speed-of-light denominator.

The `_tn` suffix confirms the layout choice: A row-major, B column-major (both
K-contiguous) is what cuBLAS is fastest at, so the comparison is apples to
apples rather than flattered by a transpose cuBLAS has to absorb.

## Phase 2 — pipeline stage sweep (4096³, 128×128×32)

XOR-swizzled layout, 16 KiB of operands per stage:

| Stages | SMEM/CTA | CTA/SM | Reg/thr | TFLOP/s | Note |
|---:|---:|---:|---:|---:|---|
| 1 (sync) | 16,384 | 2 | 111 | 28.11 | no overlap |
| **2** | **32,768** | **2** | **115** | **29.17** | best |
| 3 | 49,152 | 2 | 116 | 28.68 | |
| 4 | 65,536 | 1 | 115 | 24.92 | residency halves |
| 5 | 81,920 | 1 | 116 | 24.91 | |
| 6 | 98,304 | 1 | 116 | 24.85 | |

Both halves of the plan's hypothesis reproduce, and Nsight names the mechanism
in each direction:

- **Too few stages.** Going 1 → 2 stages drops the long-scoreboard stall from
  **4.81 to 0.20** warps-per-issue (24×). That is the Ada analogue of a consumer
  arriving at `cp.async.wait_group` before the copy landed — exactly the
  TMA-wait stall the plan wanted to reproduce.
- **Too many stages.** 3 → 4 stages crosses 64 KiB/CTA, so only one CTA fits per
  SM (102,400 B per SM). `sm__warps_active` falls 32.9% → 16.7% and throughput
  drops 13%. Capacity, not the pipeline, is the limiter.

The knee is at 2 stages rather than the 3–4 the plan anticipated, because 24 SMs
× 2.64 GHz against 256 GB/s is a far lower bytes-per-flop ratio than a
datacenter part — one tile of lookahead already covers the DRAM latency here.

Note `pad` maxes out at 4 stages (5 would need 102,400 B > the 101,376 B
per-block opt-in limit), which is exactly the "metadata and padding must also
fit" caveat in the plan.

## Phase 3 — tile shape (4096³, best stage count per tile)

| CTA tile | Threads | Reg/thr | CTA/SM | TFLOP/s |
|---|---:|---:|---:|---:|
| 64×128×32 | 256 | 79 | 3 | 28.00 |
| **128×128×32** | **256** | **115** | **2** | **29.17** |
| 128×256×32 | 512 | 111 | 1 | 25.18 |

128×256 loses despite the better A/B reuse ratio: at 24 KiB/stage it can only
ever hold one CTA per SM, so there is nothing to cover its own barriers with.
64×128 keeps 3 CTAs/SM but halves the arithmetic intensity per byte staged.

## Phase 4 — shared-memory layout (4096³, 2 stages)

| Layout | SMEM/CTA | Shared-load bank conflicts | TFLOP/s |
|---|---:|---:|---:|
| row-major | 32,768 | 150,994,944 | 26.53 |
| padded (+8 halves) | 40,960 | **83,028** | 28.58 |
| **XOR swizzle** | **32,768** | 50,404,981 | **29.17** |

The row-major case is the predicted 4-way conflict: a `BK=32` half row is 64 B =
16 banks, so the 8 lanes feeding one `ldmatrix` quarter hit only two bank groups.

Two results worth keeping:

- **Padding removes essentially all conflicts** (1,819× fewer). A row stride of
  `BK+8 = 40` halves = 80 B = 20 banks makes `20r mod 32` distinct for r=0..7,
  and 80 B stays 16-B aligned so `cp.async.cg` and `ldmatrix` remain legal.
- **The swizzle wins anyway despite ~600× more conflicts than padding**, because
  it costs 8 KiB less per stage. Once conflicts are below the Tensor Core
  service rate they stop mattering, and shared-memory capacity — which sets
  CTA/SM — does. Optimizing the conflict counter alone would have picked wrong.

The swizzle is only 2-way here, not conflict-free: a 64 B row holds just 4
sixteen-byte chunks, so `chunk ^ (row & 3)` has only 4 distinct positions for 8
rows. `BK=64` (128 B rows, 8 chunks) would make it perfect, at 2× stage cost.

## L2-aware CTA rasterization — not in the plan, but decisive at 8192

The first 8192³ run was *slower* than 4096³ (22.94 vs 29.17 TFLOP/s) while
cuBLAS scaled fine. Nsight found it immediately:

| M=N=K | CTA order | DRAM read | DRAM throughput | TFLOP/s | % SoL |
|---|---|---:|---:|---:|---:|
| 4096 | row-major | 527 MB | 24.0% | 29.31 | 100.9% |
| 8192 | row-major | **9.16 GB** | **49.7%** | 22.94 | 75.2% |
| 8192 | group 2 | **4.70 GB** | 26.1% | **29.43** | **96.5%** |

Default row-major CTA order sweeps the entire B panel for every strip of A. This
GPU has an unusually large **32 MiB L2**, and at 4096 the whole B panel is
32 MiB — it fits, so the naive order is accidentally optimal. At 8192 the panel
is 128 MiB and every M-tile refetches it: 9.16 GB read against a 268 MB minimum,
34× amplification, and the kernel flips from Tensor-Core-bound to DRAM-bound.

Walking `group` M-tiles column-major inside a group shrinks the live footprint
to `group·BM·K + BN·K`. It is a pure CTA index remap — the mainloop is untouched
— and it recovers **+28%** at 8192, restoring `math_pipe_throttle` as the
dominant stall. Groups 2/4/8/16 are within 1.3% of each other; group 2 suffices
because only 48 CTAs are resident at once. At 4096 it changes nothing (29.31 vs
29.13), which is the right null result.

This is the plan's Phase 3 "enough output tiles to create multiple full waves"
question, but the binding constraint turned out to be L2 residency rather than
wave quantization.

## Phase 6 — workload shape (N = K = 4096)

| M | TFLOP/s | % of 4096³ result |
|---:|---:|---:|
| 1 | 0.10–0.15 | 0.4% |
| 8 | 0.83–1.17 | 3.4% |
| 32 | 3.30 | 11% |
| 128 | 13.1 | 44% |
| 512 | 18.2 | 62% |
| 2048 | 28.7 | 97% |
| 4096 | 29.5 | 100% |

An excellent square-GEMM kernel retains almost none of its Tensor Core
utilization on decode-shaped work, as the plan predicted. At M=1 a 128×128 CTA
tile computes one useful row out of 128 — 0.8% of the `mma` work it issues is
kept — and only 32 CTAs exist for 24 SMs. This is a tiling/algorithm problem
(split-K, or a GEMV kernel), not a pipeline-tuning problem.

## Phase 8 — epilogue fusion (4096³)

| Variant | ms | TFLOP/s | Reg/thr | Rel. error vs CPU |
|---|---:|---:|---:|---:|
| raw GEMM, no epilogue | 4.665 | 29.46 | 116 | 4.53e-04 |
| **fused** α·acc + bias + ReLU | 4.661 | 29.49 | 116 | 4.41e-04 |
| unfused: GEMM + separate kernel | 4.930 | 27.88 | 116 | 7.34e-04 |

Fusion is **free** — identical time and identical register count, because the
epilogue runs once per output element after the mainloop has released its
operand registers. The separate kernel costs 0.27 ms, which is exactly one extra
DRAM round trip: 2 × 4096² × 2 B = 67.1 MB at the measured 259.5 GB/s = 0.259 ms.

The unfused path is also *less accurate* (7.34e-04 vs 4.41e-04): it rounds to
FP16 twice, once storing the GEMM result and again after bias+ReLU. Fusion keeps
the intermediate in an FP32 register.

## Nsight Compute — the finalist, and how to read it

`reports\finalist_verified.ncu-rep`. 4096³, 128×128×32, 2 stages, swizzle:

| Metric | Custom | cuBLAS | Reading |
|---|---:|---:|---|
| **`..._src_fp16_dst_fp32_sparsity_off`** | **94.00%** | **99.38%** | **true dense tensor occupancy** |
| `sm__pipe_tensor_op_hmma_cycles_active` | 47.00% | 49.69% | artifact — see below |
| `sm__throughput` | 46.20% | 48.93% | compute-bound |
| `gpu__dram_throughput` | 20.71% | 10.33% | not memory-bound |
| `sm__warps_active` | 32.92% | 16.53% | we need 2× the occupancy cuBLAS does |
| stall: `math_pipe_throttle` | **16.66** | 13.79 | **dominant — Tensor Cores are the limiter** |
| stall: `long_scoreboard` | 0.17 | 0.04 | global latency fully hidden |
| stall: `barrier` | 2.78 | 1.17 | our 8 warps/CTA vs their 4 |
| local memory ld/st bytes | **0** | — | no spills |

Answering the plan's seven questions for this report:

1. **Compute-bound.** SM 46% vs DRAM 21%, L2 21%.
2. **Yes — 94.00% dense Tensor Core occupancy**, which accounts for the achieved
   29.5 TFLOP/s against the 32.4 TFLOP/s hardware peak. Do not read the 47%
   figure as "half the Tensor Cores are idle."
3. Not applicable — Tensor Cores are the limiter (`math_pipe_throttle` is the
   top stall, and it is back-pressure *from* the MMA pipe).
4. **Yes, and it is the main tuning lever.** 32 KiB/CTA is what keeps 2 CTAs/SM;
   anything above 51,200 B drops to 1 and costs ~13%.
5. Conflicts yes (2-way, `BK=32` limits the swizzle), spills no (0 bytes).
6. No, at 4096/8192. 1024 and 4096 CTAs over 24 SMs.
7. No — 2640–2670 MHz and 110 W across every run, 65 °C max.

The most instructive contrast is against cuBLAS: it reaches slightly *higher*
tensor activity at *half* our occupancy (16.53% vs 32.92%), using 234
registers/thread against our 116. It covers MMA latency with instruction-level
parallelism from a large per-warp register tile; we cover it with more resident
warps. That is the Phase 5 register/accumulator tradeoff, visible as two working
points on the same curve, and it is where the remaining 5.5% lives.

## Reading Tensor Core activity correctly

`scripts\profile.ps1` now reports both metrics. The one to trust is the second:

```text
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active              47.00 %   <- artifact
sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.avg.pct_of_peak_sustained_active 94.00 %   <- truth
```

**The argument that does not depend on trusting any metric.** The kernel delivers
29.48 TFLOP/s, which is `29.48e12 / (24 × 2.5962e9)` = **473 FLOP/clk/SM against a
measured hardware ceiling of 512** — 92.4%. A tensor pipe that were genuinely idle
53% of the time could not produce 92.4% of its own peak rate. So whatever the 47%
counter means, it is not "half the Tensor Cores are idle." Equivalently, and with
no clock in it at all:

```text
busy tensor cycles  = 137,438,953,472 FLOP / (512 FLOP/clk/SM x 24 SM) = 11,184,811
elapsed cycles      = 12,103,613     (sm__cycles_elapsed.avg)
dense occupancy     = 92.41%
```

**What the 47% counter actually is.** Not a denominator artifact, as I first
described it — a **numerator** one. One run of `reports/finalist_verified.ncu-rep`
collects every raw counter needed to prove it, so nothing has to be trusted:

```text
sm__inst_executed_pipe_tensor_op_hmma.sum                = 33,554,432
sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum   = 137,438,953,472
sm__cycles_elapsed.avg                                   = 12,166,938.67
sm__cycles_active.avg                                     = 11,909,536
```

Every reported percentage reproduces by hand from those four numbers:

| Quantity | Hand calculation | Result | Nsight reports |
|---|---|---:|---:|
| dense occupancy, elapsed | 11,184,810.67 / 12,166,938.67 | 91.93% | **91.93%** |
| dense occupancy, active | 11,184,810.67 / 11,909,536 | 93.93% | **93.91%** |
| generic `hmma_cycles_active` | (16 × 33,554,432 / 96) / 11,909,536 | 46.96% | **46.96%** |

where busy cycles = `137,438,953,472 / (512 × 24)` = 11,184,810.67. The first two
rows confirm Nsight's dense peak really is 512 ops/clk/SM. The third is decisive:
`sm__pipe_tensor_op_hmma_cycles_active` equals **exactly 16 cycles × HMMA count**,
to within 0.006%, regardless of accumulator width — a fixed weight, not a
measurement.

That also settles the one alternative reading worth taking seriously. NVIDIA
documents the `sm__pipe_*_cycles_active` family as derived from hardware
pipe-utilisation signals *because* "pipes can support sets of instructions with
different issue rates", which would argue against fixed weighting; if the GeForce
FP32-accumulate restriction were implemented as issue-slot throttling, the pipe
really would be idle ~50% of cycles and 47% would be truthful. But a genuinely
throttled pipe occupied 32 cycles per HMMA would have to report 11,184,811 busy
cycles (93.9%), not 5,592,405. It reports 5,592,405. Fixed weighting it is.

**Use the elapsed basis.** `_sparsity_off` on
`sum.pct_of_peak_sustained_elapsed` gives **91.93%** — the figure comparable to
wall-clock throughput. The `.avg.pct_of_peak_sustained_active` form normalises by
`sm__cycles_active`, excluding cycles where an SM has no resident warps
(11,909,536 / 12,166,938 = 97.9%), so it reads ~2% higher. Quote **~92%** for
speed-of-light arguments.

Dropping `_sparsity_off` costs exactly 2×, because the base metric's peak includes
the **sparse** rate: it read 46.60% on a run whose elapsed cycles were 12,001,495,
i.e. `137,438,953,472 / 12,001,495 / 24` = 477 ops/clk/SM and `477 / 0.4660` = 1024
= 2 × the dense 512. Nothing here uses sparsity, so `_sparsity_off` is required.
(The 2× relationship is inferred from this arithmetic; NVIDIA's Nsight
documentation does not state it.)

**Getting Nsight to *display* near 100%** requires a kernel whose accumulate
width matches the metric's expectation. Three ways, in increasing usefulness:

| Build | Nsight tensor activity | TFLOP/s | Error vs CPU |
|---|---:|---:|---:|
| FP16 accumulate (`ACC_F16=1`), generic metric | 82.99% | 46.32 | 3.78e-02 |
| FP32 accumulate, generic metric | 47.00% | 29.48 | 4.53e-04 |
| **FP32 accumulate, `_sparsity_off` (elapsed / active)** | **92.4% / 94.00%** | **29.48** | **4.53e-04** |
| cuBLAS, `_sparsity_off` (elapsed / active) | 97.9% / 99.38% | 31.22 | — |

Note the FP16-accumulate build is *less* tensor-bound (82.89% on its own dense
metric) than the FP32-accumulate build is (94.00%) — it consumes shared-memory
operands twice as fast, so delivery becomes the limiter. And its 3.78e-02
relative error makes it unusable for real work at K=4096: FP16 unit roundoff is
2⁻¹¹ = 4.88e-04, and a random-walk model over 4096 terms gives √4096 × u =
3.13e-02, so the measured 3.78e-02 is 1.21× the expected magnitude — exactly the
predicted loss, roughly 5 of 11 mantissa bits. By contrast the FP32-accumulate
build's 4.53e-04 is 0.93 u, i.e. purely the FP16 *output store* quantisation
floor with negligible accumulation error. The FP32-accumulate kernel is both the
better Tensor Core utiliser and the only numerically viable one.

## Sustained load: 240 s soak

`scripts\soak.ps1 -Seconds 240` — 50,272 back-to-back launches at 4096³ in 60 windows,
with nvidia-smi sampled every 250 ms on the same timeline (`data\soak_*.telemetry.csv`,
`data\soak_*.throughput.txt`).

| | FP32 accumulate (240 s) | FP16 accumulate (120 s) |
|---|---:|---:|
| throughput avg | **28.81 TFLOP/s** | **44.49 TFLOP/s** |
| max / min window | 29.41 / 27.81 (5.5% spread) | 46.08 / 43.67 (5.2%) |
| decay, first third → last third | −2.5% | −2.0% |
| SM clock avg (min–max) | **2633.7 MHz** (2625–2670) | **2305.2 MHz** (2250–2565) |
| memory clock | 9601 MHz, never varied | 9601 MHz, never varied |
| power avg / max | 113.8 W / 116.9 W | **124.5 W** / 126.8 W |
| temperature | 61.0 → 71.2 °C (max 72) | 62.4 → 71.5 °C (max 73) |
| `sw_power_cap` asserted | **0 / 765 samples** | **379 / 379 samples** |
| `hw_slowdown`, thermal | 0 / 765 | 0 / 765 |

The FP32-accumulate kernel is **not** power or thermally limited on this laptop:
it holds 2634 MHz at 114 W and 72 °C for four minutes with no throttle reason
ever asserted, and loses only 2.5% throughput. The 2640 MHz figure is the right
number to divide by.

The FP16-accumulate kernel *is* power limited — `sw_power_cap` asserted in
**379/379** loaded samples, 124.5 W against the board's ~125 W ceiling, and the GPU
surrenders 12.5% of SM clock (2305 vs 2634 MHz) to stay inside the budget.

But **power is only about half the story**, and my first reading of this was wrong.
The FP16-accumulate build is *not* 2× per clock — it is 1.76×
(804 vs 456 FLOP/clk/SM from the soaks; independently 12,103,613/6,798,058 = 1.78×
from Nsight cycles, which involves no clock at all). The 1.544× wall-clock ratio
decomposes almost exactly:

```text
1.544  =  2.000          x  0.875              x  0.882
          peak per clock    power-capped clock    tensor-pipe utilisation
```

The power cap accounts for ~51% of the 22.8% shortfall from 2×; the other ~49% is a
genuine loss of tensor-pipe utilisation, visible independently in Nsight as
82.89% versus 94.00% occupancy (82.89/94.00 = 0.8818, matching the 0.8822
efficiency ratio to 0.04%). Removing the power cap entirely would still only give
44.49/0.875 = 50.8 TFLOP/s = 1.76×.

The mechanism is worth keeping: **exposed non-tensor cycles grew in absolute
terms**, 1,205,653 versus 918,802 (+31%), so this is not fixed overhead becoming a
larger fraction of a shorter run — latency hiding genuinely breaks down. Same
geometry, same 33,554,432 HMMA, but the shadow cast by each `mma` halves from 32 to
16 cycles per sub-partition, while the `cp.async`, `ldmatrix`, barrier and address
work underneath it is unchanged. **The 128×128×32 / 2-stage tile is balanced for a
32-cycle `mma`**; halve the mma cost and the tile needs re-tuning (more stages, or
a larger register tile) to hide the same operand traffic.

One thing the FP16 build does win outright: perf/W is 44.49/124.5 = 0.357 versus
28.81/113.8 = 0.253 TFLOP/s/W, i.e. **1.41× more energy efficient** despite the
lower clock. Also note the two soaks are not duration-matched (120 s vs 240 s), so
the FP16 average is taken over a cooler stretch of the decay curve.

## Hardware speed of light, and where the missing ~6% goes

**Compute roof (dense, FP32 accumulate).** 512 FLOP/clk/SM is measured, not
assumed. But the roof is only as meaningful as the clock you pair with it, and
**different runs ran at different clocks** — so pairing a soak-average clock with
a short-run throughput is a mistake:

```text
512 x 24 x 2.6337 GHz = 32.36 TFLOP/s   240 s soak average clock
512 x 24 x 2.5962 GHz = 31.90 TFLOP/s   the short timed runs (see below)
512 x 24 x 3.105  GHz = 38.16 TFLOP/s   "max clocks" table -- never observed under load
```

The 2.5962 GHz figure is not a guess: `sm__cycles_elapsed.avg / kernel time` =
12,103,613 / 4.662 ms = **2596.2 MHz** for our kernel and 11,429,448 / 4.402 ms =
2596.1 MHz for cuBLAS. Both short benchmark runs sat 1.4% below the soak average,
because they are ~5 s long and never fully leave the clock ramp. (I earlier quoted
a roof "at 2640 MHz" — that clock appears nowhere in the measured data; the
observed load range is 2625–2670 MHz.)

**Memory roof.** Unresolved between ~260 and ~290 GB/s (Phase 0). Stating the
roofline in arithmetic-intensity terms makes the conclusion independent of that:
at 4096³ the kernel does 137.44 GFLOP against 234–527 MB of DRAM traffic → **AI =
261–586 FLOP/byte**, while the ridge point is `32.4e12 / BW` = 106–133 FLOP/byte
across every candidate roof (244–307 GB/s). Worst case the kernel still sits
**1.97× on the compute side** of the ridge; best case 5.6×. Put differently,
sustaining the full compute roof needs only 55–124 GB/s. Compute-bound under every
assumption.

(`gpu__dram_throughput` is *not* good evidence here and shouldn't be cited as
such: it reads 20.71% and 24.04% in two runs of the identical kernel and geometry,
and cuBLAS's 10.33% is irreconcilable with its own 234.44 MB at 4.402 ms — these
percentages come from runs in different clock states.)

**Prologue/epilogue, measured rather than estimated.** At fixed M and N the CTA
count is fixed, so kernel time against K separates fixed cost from steady state:
`ms = a + b·K`, where `a` is prologue + epilogue + launch and `b·K` is the
mainloop.

| Fit range | a (ms) | b (ms/K) | R² | fixed cost at K=4096 |
|---|---:|---:|---:|---:|
| all K (512…8192) | 0.2568 | 0.0010950 | 0.9979 | 5.41% |
| K ≥ 1536 | 0.0474 | — | 0.99998 | 1.01% |
| **K ≥ 2048** | **0.0409** | **0.0011329** | **0.999981** | **0.87%** |
| K ≥ 3072 | 0.0303 | — | — | 0.65% |

**The answer is "sub-1%", not "0.87%".** The intercept's standard error is
0.0129 ms, so its 95% confidence interval is 0.005–0.077 ms = **0.11%–1.64%**, and
the point estimate moves with the subset chosen (0.02%–1.01%). Using the *measured*
4.662 ms rather than the fitted value gives 0.47%. Any of these supports the
conclusion — the fixed cost is well under 1%, not the ~5% estimated — but 0.87%
is false precision.

Two things I got wrong in the first pass. **(a)** The small-K points lie *above*
the fitted time line (+0.32 ms at K=512, +0.41 ms at K=1024), not below; they are
below only in throughput. **(b)** The cause is not pipeline fill (worth ≤12% at
K=512) and not L2 residency (the 8–25 MB panels all fit the 32 MiB L2, which would
make small K *faster*). It is **incomplete SM clock ramp**: the implied clocks are
1632 MHz at K=512 and 1915 MHz at K=1024, and `data\sweep_results.csv` independently
logs 1890 MHz at sweep start. Short kernels never leave the ramp. The right cut is
K ≥ 1536 (where the break actually is), not K ≥ 2048.

Also label the intercept accurately: it is the **exposed, non-overlapped,
K-independent residue**, not "prologue + epilogue + launch cost". The epilogue's
real DRAM cost is ~0.137 ms — over three times the intercept — but with 1024 CTAs
over 48 resident slots roughly 70% of it hides under other CTAs' mainloops, and it
recurs per wave (21.33 waves), not once per kernel. The launch component is only
the ~2–3 µs kernel-to-kernel gap, since timing is total/iters over back-to-back
same-stream launches; a one-shot synchronous launch would add 10–20 µs more.

**The budget.** Each row is divided by the roof at *its own* clock, and the
clock-free column is the one to compare across rows:

| | TFLOP/s | clock | % of roof at that clock | dense occupancy (clock-free) |
|---|---:|---:|---:|---:|
| cuBLAS, single run | 31.22 | 2596 MHz | 97.9% | **97.86%** |
| custom, single run 4096³ | 29.48 | 2596 MHz | 92.4% | **92.41%** |
| custom, best soak window | 29.41 | ~2634 MHz | 90.9% | — |
| custom, 240 s sustained | 28.81 | 2634 MHz | ~89% ±0.5 | — |

Decomposing our **7.6%** deficit without double-counting: the 92.41% clock-free
occupancy is the *total* non-tensor-limited time. Of it, sub-1% is K-independent
(exposed epilogue tail, launch gap), ~2.3% is unavoidable wave quantisation
(1024 CTAs / 48 resident slots = 21.33 waves), and the remaining ~4% is per-k-tile
cost — one `__syncthreads()` per tile plus `ldmatrix` issue that is not fully
hidden. That matches the stall profile, where `barrier` is 2.78 warps per issue
against cuBLAS's 1.17.

**Is cuBLAS hardware SoL?** Close, but **not within 0.6% as I first said — it is
~2.1% short.** The 99.38% figure is the active-cycle form; on the elapsed basis
that is comparable to wall-clock throughput, cuBLAS reaches **97.86%**
(11,184,811 / 11,429,448). And essentially all of that 2.14% shortfall is
explained by wave quantisation: 1024 CTAs over 24 SMs is 42.67 CTAs/SM, so 16 SMs
take 43 CTAs and 8 take 42, costing ~2.3%. In other words cuBLAS is at the
*practical* ceiling for this problem shape, and the remaining gap is a tiling
artifact rather than anything cuBLAS could fix. It is a sound SoL proxy — just
quote it as ~98%, not ~99.4%.

Both kernels necessarily perform 11,184,811 busy tensor cycles per sub-partition
(that is `FLOP / (512 × 24)`, an identity for equal FLOPs, not a finding). They
differ only in elapsed cycles: 11,429,448 for cuBLAS versus 12,103,613 for ours,
**+5.90%**, matching the +5.91% kernel-time ratio (4.662/4.402) — though note that
agreement is arithmetically forced, since both ran at the same 2596 MHz, so it is
a consistency check rather than independent evidence. cuBLAS gets there with
`m16n8k8` (67.1 M instructions at 16 cycles each) against our `m16n8k16` (33.6 M
at 32 cycles), plus 234 registers/thread of accumulator tile at half our occupancy.

## FP8

The hardware supports it and the PTX is right, but the toolkit is one version too
old:

```text
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
  -> ptxas: "Unexpected instruction types specified for 'mma'"  (CUDA 12.3)
```

Ada SM89 FP8 tensor core support requires **CUDA 12.4 or newer**; this machine has
12.3 and no other toolkit installed. `src\device_query.cu` contains the FP8 probe
guarded by `__CUDACC_VER_MAJOR__/MINOR__`, so it activates automatically on a
newer toolkit and prints `UNAVAILABLE - needs CUDA >= 12.4 (have 12.3)` until then.

**The FP8 accumulate width matters as much as it does for FP16.** The Ada
whitepaper (v2.02, Appendix A) lists for the RTX 4090:

```text
Peak FP8 Tensor TFLOPS with FP16 Accumulate : 660.6 dense  -> 2048 FLOP/clk/SM
Peak FP8 Tensor TFLOPS with FP32 Accumulate : 330.3 dense  -> 1024 FLOP/clk/SM
```

Same exact 2:1 halving as FP16. So the instruction we actually want,
`...f32.e4m3.e4m3.f32`, sits in the **2× tier, not the 4× tier**:

```text
FP8 e4m3 with FP32 accumulate: 1024 FLOP/clk/SM x 24 SMs x 2.634 GHz = 64.7 TFLOP/s
FP8 e4m3 with FP16 accumulate: 2048 FLOP/clk/SM x 24 SMs x 2.634 GHz = 129.5 TFLOP/s
```

The 4× tier is real but only reachable with FP16 (or INT32) accumulation. Measured
confirmation of that tier today: `m16n8k32` INT8 compiles and runs on CUDA 12.3 at
**2048.4 ops/clk/SM**, exactly 4.00× the FP16/FP32-accumulate baseline, matching
the whitepaper's INT8 : FP16(FP32-acc) ratio of 660.6 : 165.2 = 4.00.

Neither FP8 roof will be reached in practice on this laptop. The FP16-accumulate
kernel already pins the ~125 W board limit at merely 2× the tensor rate and gives
back 12.5% of SM clock for it, *and* loses another ~12% to broken latency hiding
(below). A denser path makes both problems worse, so expect the power cap and
operand delivery — not the Tensor Cores — to set the result.

## Correctness

Two oracles, and the second one matters:

- vs cuBLAS: `0.00e+00` (bit-identical) at 4096³ and 8192³.
- vs a **double-precision CPU** dot product on a deterministic sample of output
  elements: `4.4e-04 – 4.9e-04` for every shape from M=1 to M=4096. That is
  exactly FP16 output rounding (2⁻¹¹ ≈ 4.9e-04), i.e. the kernel is as accurate
  as the output format allows.

cuBLAS turned out to be an unreliable oracle for skinny problems: at M=128 the
kernel-vs-cuBLAS delta varied between runs (2.34e-02 and 5.13e-03 on repeats)
because cuBLAS switches to split-K variants whose accumulation order — and so
whose FP16 rounding — changes. Against CPU ground truth our kernel is
4.85e-04 at M=128, so the drift was cuBLAS's, and a tolerance check against it
was comparing to a moving target. One M=1 run early on reported a large delta
(2.19e+03) that never reproduced in 90+ subsequent identical runs, all
bit-identical; the CPU check now covers that case directly and passes.

## What carries to Hopper, and what does not

Carries over as-is:

- The staging *structure* — prologue fill, steady-state issue-then-consume, one
  barrier per k-tile — is the same shape as a TMA/WGMMA mainloop.
- Shared-memory capacity vs CTA residency as the primary tile-size constraint.
- The swizzle-vs-padding reasoning, and the fact that conflicts stop mattering
  once they drop below the MMA service rate.
- L2-aware CTA rasterization. This gets *more* important on Hopper, not less.
- Fused epilogues being free, and the double-rounding argument for them.

Must be rewritten on SM90:

- `cp.async` / `cp.async.wait_group` → `cp.async.bulk.tensor` (TMA) with a
  descriptor built on the host, and mbarrier arrive/wait instead of group
  counting. Address generation moves out of the consumer threads entirely.
- `mma.sync.m16n8k16` (warp-wide, operands in registers via `ldmatrix`) →
  `wgmma.mma_async` (warpgroup-wide, A/B read directly from shared memory). The
  `ldmatrix` fragment-layout work in this kernel disappears.
- Warp specialization: a producer warpgroup issuing TMA while consumer
  warpgroups run WGMMA. Ada has no efficient equivalent, so the producer/
  consumer split here is implicit in one warp's instruction stream.
- Thread-block clusters / distributed shared memory have no Ada analogue.
- The FP32-accumulate half-rate penalty is a **GeForce** restriction. On H100 it
  does not apply, so the ~50% tensor-pipe ceiling seen here would not appear —
  budget for that before reusing these numbers as a baseline.

## Files

| File | Purpose |
|---|---|
| `src/device_query.cu` | Phase 0 limits, Tensor Core ceiling probe (4 dtypes), DRAM bandwidth probe |
| `src/gemm_bench.cu` | the parameterized kernel, cuBLAS baseline, both correctness oracles, soak mode |
| `scripts/build.ps1` | `sm_89` build; emits plain, fused-epilogue and FP16-accumulate binaries |
| `scripts/sweep.ps1` | stage × layout × tile sweep to CSV, with clock/power/temp logging |
| `scripts/profile.ps1` | Nsight section reports + the metric set used above |
| `scripts/soak.ps1` | sustained-load run with correlated nvidia-smi telemetry |
| `data/sweep_results.csv` | 45-row Phase 2–4 sweep |
| `data/soak_*.{telemetry.csv,throughput.txt}` | 240 s / 120 s soak timelines |
| `reports/*.ncu-rep` | gitignored; regenerate via `reports/README.md` |

Reproducing the headline numbers:

```powershell
.\scripts\build.ps1
.\bin\device_query.exe                                        # roofs: 512/1024/2048 ops/clk/SM
.\scripts\soak.ps1 -Seconds 240 -Window 4 -Tag soak_fp32acc   # sustained power/clock/throughput
.\scripts\profile.ps1 -Stages 2 -Layout 2 -Group 2 -Tag finalist_verified
ncu-ui .\reports\finalist_verified.ncu-rep                    # 91.93% dense tensor activity
```
