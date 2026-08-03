# Ada Tensor Core GEMM

A dense FP16 matrix-multiply kernel for NVIDIA Ada (`sm_89`), written from PTX
primitives, that reaches **94.5% of cuBLAS** on an RTX 4060 Laptop GPU — and, more
importantly, a full accounting of *why* it lands where it does.

Built as a learning exercise with a specific constraint: understand how modern
datacenter GEMM kernels get their performance, using a laptop.

---

## Why build this

If you read about GEMM performance on H100 or B200, the interesting content is
never the multiply. It's the **choreography around** the multiply:

- moving tiles from HBM into shared memory *asynchronously*, so the Tensor Cores
  never wait on memory
- **multistage buffering** — how many tiles to keep in flight, and what it costs
- the tension between **shared-memory capacity** and **how many CTAs fit per SM**
- **bank conflicts** and the swizzled layouts that avoid them
- **register pressure** vs. instruction-level parallelism
- keeping every SM busy, and keeping the data in cache

The mechanisms people discuss for that choreography — TMA, WGMMA, warp
specialization, thread-block clusters, TMEM — are Hopper and Blackwell features.
None of them exist on a consumer laptop GPU.

But the *tradeoffs* they exist to manage are architecture-independent. Ada has its
own async-copy and Tensor Core instructions, and the shape of the problem is the
same:

```text
Ada     :  GDDR6  ──cp.async (LDGSTS)──►  staged SMEM  ──mma.sync──►  regs  ──►  D
Hopper  :  HBM    ──cp.async.bulk.tensor (TMA)──►  staged SMEM  ──wgmma──►  regs  ──►  D
```

On Hopper, a consumer stalling on TMA arrival is a warpgroup waiting on an
mbarrier. On Ada, it's a warp reaching `cp.async.wait_group` before its copy
landed. **Same stall, same cure, different instruction.** That was the bet: if you
can build the Ada version and explain every profiler counter, you have learned the
transferable part.

The secondary goal turned out to matter just as much: **the measurement itself is
harder than the kernel.** Three of the numbers you'd naturally reach for on this
GPU are wrong or misleading, and finding that out was most of the work. See
[Three traps](#three-traps-that-will-mislead-you).

---

## What the kernel has to do

`D = A · B` where A is M×K, B is K×N, D is M×N, all FP16 with FP32 accumulation.

The whole problem is that a 4096³ multiply is 137 GFLOP against operands that are
only 34 MB each. Read naively, every output element re-reads a full row and
column, and you are hopelessly memory-bound. So you tile: each CTA owns a patch of
the output and streams the K dimension through shared memory, reusing each loaded
byte across the whole patch.

```text
     K                     N                      N
  ┌──────┐             ┌──────┐               ┌──────┐
M │  A   │      ·    K │  B   │       =     M │  D   │
  └──────┘             └──────┘               └──────┘

one CTA computes a 128×128 tile of D, streaming K in 32-wide slices:

  GDDR6  ──cp.async──►  shared memory  ──ldmatrix──►  registers  ──mma.sync──►  acc
          16 B/thread    2 stages           4×A + 2×B    m16n8k16      64 regs
                         16 KiB each        per k-step                 per thread
                         XOR swizzled
```

The point of `cp.async` is that the copy is *fire-and-forget*: a warp issues it and
keeps executing. So while the Tensor Cores chew on k-tile *i*, the copies for
k-tile *i+1* are already in flight:

```text
k-tile:      0        1        2        3
copy     [==0==][==1==][==2==][==3==]
compute         [==0==][==1==][==2==][==3==]
                ▲
                wait_group before consuming tile 0 — this is the Ada "TMA wait"
```

With one stage there is no overlap and the Tensor Cores starve. With too many
stages, shared memory runs out and fewer CTAs fit per SM. Finding that knee is
the central experiment.

---

## Design decisions, and what each one actually cost

Every choice below was measured, not assumed. Full data in
[docs/RESULTS.md](docs/RESULTS.md).

### Operand layout: A row-major, B column-major

Both operands **K-contiguous**. This looks like an odd restriction and is actually
the most load-bearing decision in the kernel:

- the global→shared copy code for A and B becomes *identical*, with no transpose
- it feeds `mma.sync...row.col` natively, so no shuffling between load and multiply
- it is what cuBLAS is fastest at — it selects a `..._tn` kernel

That last point matters for honesty: if we picked a layout cuBLAS had to absorb a
transpose for, the 94.5% would be flattery rather than a result.

### Pipeline depth: 2 stages

| stages | SMEM/CTA | CTAs/SM | TFLOP/s |
|---:|---:|---:|---:|
| 1 (synchronous) | 16 KiB | 2 | 28.11 |
| **2** | **32 KiB** | **2** | **29.17** |
| 3 | 48 KiB | 2 | 28.68 |
| 4 | 64 KiB | **1** | 24.92 |
| 6 | 96 KiB | 1 | 24.85 |

Both failure modes show up, and Nsight names each one:

- **Too few.** Going 1 → 2 stages collapses the long-scoreboard stall from **4.81
  to 0.20** warps per issue. That stall *is* "Tensor Cores waiting on memory".
- **Too many.** At 4 stages the tile crosses 64 KiB, only one CTA fits per SM,
  occupancy halves and throughput drops 13%. Capacity, not the pipeline, binds.

The knee at 2 is *shallower* than the 3–4 stages typical on datacenter parts —
24 SMs against ~260 GB/s is a much lower bytes-per-flop ratio, so one tile of
lookahead already covers the latency.

### Shared-memory layout: XOR swizzle

A `BK=32` half row is 64 B = 16 banks, so a plain row-major tile makes the eight
lanes feeding one `ldmatrix` hit only two bank groups — a 4-way conflict.

| layout | SMEM/CTA | bank conflicts | TFLOP/s |
|---|---:|---:|---:|
| row-major | 32 KiB | 150,994,944 | 26.53 |
| padded (+8 halves) | 40 KiB | **83,028** | 28.58 |
| **XOR swizzle** | **32 KiB** | 50,404,981 | **29.17** |

The interesting result is that **swizzle wins despite ~600× more conflicts than
padding**, because it costs 8 KiB less per stage — and shared-memory capacity is
what sets CTAs/SM. Once conflicts drop below the rate the Tensor Cores consume
operands, they stop mattering. Optimizing the conflict counter alone picks wrong.

### CTA rasterization: an L2-aware remap

This one was not in the original plan and turned out to be decisive. The first
8192³ run was *slower* than 4096³ while cuBLAS scaled fine. Nsight found it
immediately:

| M=N=K | CTA order | DRAM read | TFLOP/s |
|---|---|---:|---:|
| 4096 | row-major | 527 MB | 29.31 |
| 8192 | row-major | **9.16 GB** | 22.94 |
| 8192 | **grouped** | **4.70 GB** | **29.43** |

Default row-major CTA order re-sweeps the entire B panel for every strip of A.
This GPU has an unusually large **32 MiB L2**, and at 4096³ the whole B panel is
32 MiB — it fits, so the naive order is *accidentally optimal*. At 8192³ the panel
is 128 MiB and every tile refetches it: 34× the compulsory traffic, and the kernel
flips from compute-bound to DRAM-bound.

Walking a group of M-tiles column-major shrinks the live footprint. It is a pure
index remap — the mainloop is untouched — and it recovers **+28%**.

### Epilogue: fused

Bias + ReLU folded into the kernel is **free** — same time, same 116 registers —
because the epilogue runs after the mainloop has released its operand registers.
Run as a separate kernel it costs 0.27 ms, which is exactly one extra DRAM round
trip, and it's *less accurate*, because the intermediate gets rounded to FP16
twice instead of staying in an FP32 register.

---

## Results

| M=N=K | 4096 | 8192 |
|---|---:|---:|
| this kernel | **29.50 TFLOP/s** | **29.26 TFLOP/s** |
| cuBLAS | 31.22 | 31.04 |
| **% of cuBLAS** | **94.5%** | **94.2%** |
| dense Tensor Core occupancy | **91.9–92.4%** | — |
| registers / thread | 116 | 116 |
| register spills | **0** | **0** |

Final geometry: `128×128×32` CTA tile, 8 warps, warp tile 64×32,
`mma.sync.m16n8k16.row.col.f32.f16.f16.f32`, 2 `cp.async` stages, XOR-swizzled
shared memory, grouped CTA rasterization.

**Sustained**, over 240 s and 50,272 launches: 28.81 TFLOP/s at 113.8 W and
2634 MHz, 72 °C, with **no throttle reason ever asserted** and only 2.5% decay.

**Correctness** against two oracles: bit-identical to cuBLAS at both sizes, and
4.5e-04 from a double-precision CPU reference — which is exactly the FP16
output-store rounding floor, i.e. as accurate as the output format permits. The
CPU oracle exists because cuBLAS turned out to be a *non-deterministic* reference
for skinny matrices, where it switches to split-K and its rounding drifts between
runs.

Where the remaining ~7.6% goes: ~2.3% is unavoidable wave quantization (1024 CTAs
over 48 resident slots), sub-1% is prologue/epilogue, and ~4% is per-k-tile
barrier and `ldmatrix` cost not fully hidden. cuBLAS closes most of that with a
much larger register tile — 234 registers/thread at *half* our occupancy, buying
latency hiding with ILP instead of resident warps. That is the next thing to try.

---

## Three traps that will mislead you

The measurement was harder than the kernel. If you benchmark Tensor Cores on a
GeForce part, these will bite:

### 1. The FP32-accumulate Tensor Core rate is half the FP16-accumulate rate

On GeForce Ada, `mma` with FP32 accumulation issues at **512 FLOP/clk/SM** against
1024 for FP16 accumulation. Measured here as 511.9 and 1024.4 with a
register-resident probe, matching NVIDIA's Ada whitepaper to 0.04%.

Almost every FP16 TFLOPS figure quoted for these cards is *already* the halved
number, so dividing by the "obvious" peak understates your kernel by 2×. It is a
**product-SKU restriction, not a property of compute capability** — the RTX A6000
and A40, same silicon and same `sm_86`, are unhalved.

### 2. Nsight's headline Tensor Core metric caps near 50% here

`sm__pipe_tensor_op_hmma_cycles_active` reads **47%** for this kernel. It is not
telling you half the Tensor Cores are idle. The counter is a fixed
**16 cycles × HMMA instruction count**, independent of accumulator width, so an
FP32-accumulating kernel structurally cannot exceed ~50%.

Use `sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off` instead — **91.9%**. The
`_sparsity_off` suffix matters too: without it the denominator is the *sparse*
peak, costing another exact factor of 2. `scripts/profile.ps1` collects the raw
counters so you can verify all of it by hand rather than trusting the label.

### 3. Cache size can make one problem size accidentally optimal

The 32 MiB L2 is large enough to hold the entire B panel at 4096³ and not at
8192³. A kernel tuned only at 4096³ looks great and then falls apart one size up,
for reasons that have nothing to do with the code you were tuning. Always sweep
the problem size.

---

## What this repo could not settle

Stated plainly, because the analysis was independently reviewed and these did not
survive:

- **Memory bandwidth is unresolved between ~260 and ~290 GB/s.** `cudaDeviceProp`
  and nvidia-smi's own "max clocks" table both say 8001 MHz → 256 GB/s, but a
  measured stream hits 259.5 GB/s, which cannot exceed a real roof. The 9601 MHz
  clock reported under load would imply 307 GB/s, but that is not a JEDEC GDDR6
  speed bin. It changes no conclusion here — the roofline is stated in
  arithmetic-intensity terms so compute-bound holds under every candidate roof.
- **FP8 is written but untested.** `sm_89` has FP8 Tensor Cores and the PTX in
  `src/device_query.cu` is correct, but ptxas only assembles it from **CUDA 12.4**;
  this was built on 12.3. The probe is version-guarded so it activates on upgrade.
  Note FP8 with *FP32* accumulate is halved exactly like FP16 — 1024 FLOP/clk/SM,
  not 2048. INT8 `m16n8k32` works today and measures 2048.4 ops/clk/SM.
- **Not done:** plots of the sweep data, a CUTLASS cross-check, and a standalone
  register/accumulator-pressure sweep.

---

## Running it

| | |
|---|---|
| GPU | Ada `sm_89` (developed on RTX 4060 Laptop, 24 SMs, 8 GB) |
| CUDA | 12.3+ (FP8 paths need 12.4+) |
| Host compiler | MSVC, VS 2022 Build Tools |
| Profiler | Nsight Compute 2023.3+ |

WSL is not used — it ships `ncu` but no `nvcc`. Everything runs natively on
Windows/PowerShell. Newer MSVC than CUDA officially supports is handled in
`build.ps1`.

```powershell
.\scripts\build.ps1              # -> bin\*.exe
.\bin\device_query.exe           # hardware limits + measured mma and DRAM roofs

.\bin\gemm_bench.exe --m 4096 --n 4096 --k 4096 --tile 1 --layout 2 --stages 2 `
                     --group 2 --iters 100 --warmup 40 --check --cpu 512

.\scripts\sweep.ps1              # stage x layout x tile sweep -> data\sweep_results.csv
.\scripts\soak.ps1 -Seconds 240  # sustained load + power/clock/thermal telemetry
.\scripts\profile.ps1 -Stages 2 -Layout 2 -Group 2 -Tag finalist
```

`gemm_bench --help` lists everything. The knobs that matter:

| Flag | Meaning |
|---|---|
| `--tile 0\|1\|2` | 64×128×32 / **128×128×32** / 128×256×32 |
| `--stages 1..6` | pipeline depth; 1 = synchronous baseline |
| `--layout 0\|1\|2` | row-major / padded / **XOR swizzle** |
| `--group G` | L2 CTA rasterization; 1 = plain row-major order |
| `--sweep`, `--gsweep` | sweep stages, or rasterization group |
| `--check` / `--cpu S` | verify against cuBLAS / a double-precision CPU reference |
| `--soak SEC` | run continuously, report per-window throughput |
| `--unfused` | run the epilogue as a separate kernel |

---

## Repo layout

```text
src/device_query.cu   hardware limits; clock64-timed mma issue-rate probe for
                      fp16/fp32-acc, fp16/fp16-acc, int8 and fp8; DRAM bandwidth probe
src/gemm_bench.cu     the kernel (templated on tile/stages/layout), cuBLAS baseline,
                      both correctness oracles, soak mode
scripts/              build, sweep, soak, profile
docs/RESULTS.md       full findings, every derivation, and the corrections
docs/experiment-plan.md   the plan this works through
data/                 committed sweep and soak measurements
reports/              Nsight reports (gitignored; see reports/README.md)
```

`gemm_bench.cu` is deliberately one file — the pipeline, the fragment layouts and
the epilogue read top to bottom, which matters more here than modularity.

---

## What carries to Hopper

**Transfers directly:** the staging structure (prologue fill, steady-state
issue-then-consume, one barrier per k-tile); shared-memory capacity vs. CTA
residency as the primary tile-size constraint; the swizzle-versus-padding
reasoning; L2-aware rasterization (which matters *more* on bigger GPUs); fused
epilogues.

**Must be rewritten:** `cp.async`/`wait_group` → TMA with mbarriers, which moves
address generation out of the consumer threads entirely; `mma.sync` with
`ldmatrix` fragments → `wgmma`, which reads operands straight from shared memory
and makes this kernel's fragment-layout work disappear; warp specialization into
producer/consumer warpgroups, which has no efficient Ada analogue. Thread-block
clusters and distributed shared memory have no analogue at all.

**And note:** the FP32-accumulate halving is a GeForce restriction. It does not
apply on H100, so the ~50% metric ceiling seen here won't appear — budget for that
before reusing these numbers as a baseline.

---

## License

[MIT](LICENSE).
