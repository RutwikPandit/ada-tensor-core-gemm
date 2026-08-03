# Feeding the Tensor Cores

### What building a GEMM on a laptop GPU teaches you about every GPU generation

A dense FP16 matrix-multiply kernel for NVIDIA Ada (`sm_89`), written from PTX
primitives, that reaches **96.5% of cuBLAS** — and, more to the point, a full
account of *why* it lands where it does.

---

## The thing nobody tells you about Tensor Cores

The first time you read the spec sheet, GEMM performance looks like a solved
problem. The Tensor Cores do the math. They have a published peak. Write a kernel
that issues `mma` instructions, and you should approach it.

Then you write the kernel and get 30% of peak.

The uncomfortable truth is that **the multiply is the easy part**. A single
`mma.sync.m16n8k16` consumes a 16×16 tile of A and a 16×8 tile of B and retires in
32 cycles. To keep four Tensor Cores per SM busy across 24 SMs, you have to deliver
operands at a rate that a 128-bit memory bus cannot remotely sustain — off by
roughly two orders of magnitude. Everything interesting in a GEMM kernel exists to
close that gap: tiling for reuse, staging through shared memory, overlapping the
copy of the next tile with the math on the current one, and arranging bytes so the
Tensor Cores never stall waiting for an operand.

So the real question is never "how fast are the Tensor Cores." It is:

> **How does this architecture move bytes into shared memory, and how does it get
> them from shared memory into the Tensor Cores?**

Those two mechanisms — the async copy and the matrix-multiply issue — are precisely
what NVIDIA has been redesigning every generation. They are where the performance
lives, and they are the reason a kernel that flies on one generation is mediocre on
the next.

---

## Two mechanisms, four generations

Here is the same kernel skeleton across generations. Notice that the *structure*
never changes and the *instructions* change completely:

| | async global → shared | shared → Tensor Core | scope of the mma |
|---|---|---|---|
| Volta (7.0) | none — `ld.global` + `st.shared` | `ldmatrix` → registers | warp |
| Ampere (8.0) | **`cp.async`** | `ldmatrix` → registers | warp |
| **Ada (8.9)** | **`cp.async`** (LDGSTS) | `ldmatrix` → registers | warp |
| Hopper (9.0) | **TMA** (`cp.async.bulk.tensor`) | **direct from shared** | **warpgroup** (`wgmma`) |
| Blackwell (10.0) | TMA | direct, results in **TMEM** | warpgroup+ (`tcgen05`) |

Read that table as a story about removing threads from the data path.

**Volta** had no async copy at all. A warp issued loads, waited for them, then
stored to shared memory. The threads doing math were the same threads doing
address arithmetic and waiting on memory.

**Ampere introduced `cp.async`** — fire-and-forget global→shared copy. A warp
issues it and keeps executing; a later `cp.async.wait_group` blocks until the data
lands. This is what makes software pipelining possible: you can copy tile *i+1*
while multiplying tile *i*. Ada inherits this unchanged.

**Hopper's TMA** goes much further. Instead of every thread computing an address
for its own 16-byte chunk, *one* thread kicks off a whole multidimensional tile
copy against a descriptor built on the host. Address generation leaves the SM
entirely. Then `wgmma` reads A and B **straight out of shared memory** rather than
requiring you to stage them into registers with `ldmatrix` — and it operates at
warpgroup scope, so 128 threads issue one instruction cooperatively.

**Blackwell** takes the accumulator out of the register file too, into dedicated
TMEM.

Each generation moves more of the plumbing out of the code you write and into
hardware. Which means: **if you only ever learn one generation's plumbing, you
learn the wrong lesson.** What transfers is not `cp.async` or TMA — it's knowing
*which resource you're actually short of*, because that question has the same shape
on all of them.

---

## Why a laptop is a legitimate place to learn this

None of Hopper's mechanisms exist on a consumer laptop GPU. No TMA, no `wgmma`, no
thread-block clusters, no TMEM. So you cannot reproduce them.

But you can reproduce the *problem they solve*, because Ada has its own async copy
and its own Tensor Core issue path:

```text
Ada     :  GDDR6  ──cp.async (LDGSTS)──►  staged SMEM  ──ldmatrix──► regs ──mma.sync──►  acc
Hopper  :  HBM    ──cp.async.bulk.tensor (TMA)──►  staged SMEM  ──────wgmma──────────►  acc
```

On Hopper, a consumer stalling on TMA arrival is a warpgroup waiting on an
mbarrier. On Ada it's a warp reaching `cp.async.wait_group` before its copy landed.
**Same stall, same cure, different instruction.** The questions you ask the
profiler are identical: is the copy deep enough to hide the latency? Did staging
more tiles cost me so much shared memory that fewer CTAs fit? Are operands arriving
fast enough to keep the math pipe saturated?

That was the bet behind this project: build the Ada version, explain every counter,
and the transferable part comes for free.

The bet mostly paid off. What I did not expect was that **the measurement would be
harder than the kernel** — three of the numbers you'd naturally reach for on this
GPU are wrong or actively misleading, and untangling that was most of the work.

---

## What the kernel has to do

`D = A · B`, all FP16 in, FP32 accumulate. At 4096³ that's 137 GFLOP against
operands of only 34 MB each. Read naively — every output element pulling a full row
and column — you are hopelessly memory-bound. So you tile: each CTA owns a patch of
the output and streams K through shared memory, reusing every loaded byte across
the whole patch.

```text
     K                     N                      N
  ┌──────┐             ┌──────┐               ┌──────┐
M │  A   │      ·    K │  B   │       =     M │  D   │
  └──────┘             └──────┘               └──────┘

one CTA computes a 128×128 tile of D, streaming K in 32-wide slices:

  GDDR6  ──cp.async──►  shared memory  ──ldmatrix──►  registers  ──mma──►  acc
          16 B/thread    3 stages           4×A + 2×B    m16n8k8    128 regs
                         16 KiB each        per k-step              per thread
                         XOR swizzled
```

The whole point of `cp.async` is the overlap it buys:

```text
k-tile:      0        1        2        3
copy     [==0==][==1==][==2==][==3==]
compute         [==0==][==1==][==2==][==3==]
                ▲
                wait_group before consuming tile 0 — Ada's "TMA wait"
```

One stage means no overlap and starved Tensor Cores. Too many stages means shared
memory runs out and fewer CTAs fit per SM. Finding that knee, and understanding
*which* resource binds on either side of it, is the central experiment.

---

## The design, decision by decision

Every choice below was measured. Full data in [docs/RESULTS.md](docs/RESULTS.md).

### Make both operands K-contiguous

A row-major (M×K), B column-major (K×N). This looks like an arbitrary restriction
and is the most load-bearing decision in the kernel:

- the global→shared copy code for A and B becomes *identical*, with no transpose
- it feeds `mma.sync...row.col` natively — no shuffling between load and multiply
- it is the layout cuBLAS is fastest at; it selects a `..._tn` kernel

That last point is about honesty. Pick a layout cuBLAS has to absorb a transpose
for, and your "% of cuBLAS" is flattery rather than a result.

### Pipeline depth: the knee is shallower than you'd expect

| stages | SMEM/CTA | CTAs/SM | TFLOP/s |
|---:|---:|---:|---:|
| 1 (synchronous) | 16 KiB | 2 | 28.11 |
| **2** | **32 KiB** | **2** | **29.17** |
| 3 | 48 KiB | 2 | 28.68 |
| 4 | 64 KiB | **1** | 24.92 |
| 6 | 96 KiB | 1 | 24.85 |

Both failure modes appear, and the profiler names each:

- **Too shallow.** Going 1 → 2 stages collapses the long-scoreboard stall from
  **4.81 to 0.20** warps per issue. That stall *is* "Tensor Cores waiting on
  memory" — the exact counter you'd watch on Hopper for a TMA that hasn't landed.
- **Too deep.** At 4 stages the tile crosses 64 KiB, only one CTA fits per SM,
  occupancy halves, throughput drops 13%. Capacity binds, not latency.

Two stages beats the 3–4 that's typical on datacenter parts, and the reason is
instructive: 24 SMs against ~260 GB/s is a far lower bytes-per-FLOP ratio than an
H100. **The right pipeline depth is a property of the machine balance, not of the
algorithm** — which is exactly the kind of thing you'd get wrong by copying a
Hopper kernel's stage count onto Ada, or vice versa.

### Shared-memory layout: optimize the wrong counter and you lose

A `BK=32` half row is 64 B = 16 banks, so the eight lanes feeding one `ldmatrix`
hit only two bank groups — a 4-way conflict.

| layout | SMEM/CTA | bank conflicts | TFLOP/s |
|---|---:|---:|---:|
| row-major | 32 KiB | 150,994,944 | 26.53 |
| padded (+8 halves) | 40 KiB | **83,028** | 28.58 |
| **XOR swizzle** | **32 KiB** | 50,404,981 | **29.17** |

Padding removes essentially every conflict — 1,819× fewer. **And it loses.** The
swizzle keeps ~600× more conflicts than padding and still wins, because it costs
8 KiB less per stage, and shared-memory capacity is what sets CTAs/SM.

The lesson generalizes past this kernel: once conflicts drop below the rate the
Tensor Cores consume operands, they stop mattering. A kernel tuned to minimize the
bank-conflict counter picks the wrong layout.

### CTA rasterization: the bug that only appears at scale

This one wasn't in the plan. The first 8192³ run came out *slower* than 4096³,
while cuBLAS scaled fine.

| M=N=K | CTA order | DRAM read | TFLOP/s |
|---|---|---:|---:|
| 4096 | row-major | 527 MB | 29.31 |
| 8192 | row-major | **9.16 GB** | 22.94 |
| 8192 | **grouped** | **4.70 GB** | **29.43** |

Default row-major CTA order re-sweeps the entire B panel for every strip of A. This
GPU has an unusually large **32 MiB L2** — and at 4096³ the whole B panel is 32 MiB.
It *fits*. The naive order is accidentally optimal. At 8192³ the panel is 128 MiB,
every tile refetches it, DRAM traffic hits 34× the compulsory minimum, and the
kernel flips from compute-bound to memory-bound.

Walking a group of M-tiles column-major shrinks the live footprint. Pure index
remap, mainloop untouched, **+28%**.

The transferable warning: **cache capacity can make one problem size accidentally
optimal.** Tune at a single size and you learn nothing about the kernel.

### Fused epilogue: free

Bias + ReLU folded in costs nothing — same time, same registers — because the
epilogue runs after the mainloop has released its operand registers. As a separate
kernel it costs 0.27 ms, exactly one extra DRAM round trip, and it's *less
accurate*: the intermediate gets rounded to FP16 twice instead of staying in an
FP32 register.

---

## Where the profiler lied

If you benchmark Tensor Cores on a GeForce part, these three will bite you.

### 1. Your FP32-accumulate peak is half what you think

On GeForce Ada, `mma` with **FP32** accumulation issues at **512 FLOP/clk/SM**
against **1024** for FP16 accumulation. Measured here with a register-resident
probe at 511.9 and 1024.4 — matching NVIDIA's Ada whitepaper to 0.04%.

Nearly every FP16 TFLOPS figure quoted for these cards is *already* the halved
number, so dividing by the "obvious" peak understates your kernel by 2×. And it is
a **product-SKU restriction, not a compute-capability property**: the RTX A6000 and
A40, same silicon and same `sm_86`, publish identical rates for both accumulate
widths. A100 and H100 aren't halved either — so this particular trap does not exist
on the hardware you'd port to.

The full measured ladder, all dense (no sparsity anywhere in this project):

| dense `mma` on sm_89 | ops/clk/SM | ratio |
|---|---:|---:|
| `m16n8k16` fp16 → **fp32** | 511.9 | 1.00× |
| `m16n8k16` fp16 → fp16 | 1024.4 | 2.00× |
| `m16n8k32` int8 → int32 | 2048.4 | 4.00× |

### 2. Nsight's headline Tensor Core metric caps near 50% here

`sm__pipe_tensor_op_hmma_cycles_active` reads **47%** for this kernel. It is not
telling you half the Tensor Cores are idle.

The counter is a fixed **16 cycles × HMMA instruction count**, independent of
accumulator width — verifiable by hand from the raw counters, and it reproduces to
0.006%. An FP32-accumulating `mma` really occupies its sub-partition for 32 cycles,
so the counter undercounts by exactly 2× and structurally cannot exceed ~50%.

Use `sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off` instead: **92.7%**. The
`_sparsity_off` suffix matters too — without it the denominator is the *sparse*
peak, costing another exact factor of 2. `scripts/profile.ps1` collects the raw
counters so you can verify every percentage yourself rather than trusting a label.

The general habit this teaches: **a utilization metric is a ratio, and you have to
know its denominator.** Cross-check against achieved FLOP/clk/SM, which has no
denominator to get wrong.

### 3. Wall-clock "peak" measurements are a trap on power-limited parts

A back-to-back `mma` loop is the densest possible power draw. On a 35–115 W laptop
part it clocks down so hard it reports a *lower* peak than the real GEMM that's
supposedly approaching it. Every roof figure here is measured in **FLOP per clock
per SM** via in-kernel `clock64()`, which is clock-invariant, and only converted to
TFLOP/s at the end using the clock actually observed under load.

---

## Chasing cuBLAS

The 8-warp kernel above lands at 94.5% of cuBLAS. Diagnosing the rest was easier
than expected, because of an identity: **both kernels must spend the same number of
busy Tensor Core cycles.** Same FLOPs at the same rate means 11,184,811 busy cycles
per sub-partition, for anyone. Only the elapsed cycles wrapped around that work can
differ.

So the question was never "why are our Tensor Cores slower." It was "what are we
doing in the extra cycles." And cuBLAS answers by inspection: it runs **128 threads
with 234 registers each** and issues **`m16n8k8`** instead of `m16n8k16`. It hides
mma latency with instruction-level parallelism from a big register tile rather than
with resident warps — and a `__syncthreads()` over 4 warps costs half what it does
over 8.

Copying both choices works:

| | threads | reg/thread | warps_active | barrier stall | tensor occupancy | 4096³ |
|---|---:|---:|---:|---:|---:|---:|
| 8 warps, `m16n8k16` | 256 | 116 | 32.9% | 2.78 | 91.9% | 94.5% |
| **4 warps, 2×`m16n8k8`** | **128** | **220** | **16.5%** | **1.96** | **92.7%** | **95.3%** |
| cuBLAS | 128 | 234 | 16.5% | 1.17 | 97.9% | 100% |

Occupancy now matches cuBLAS to 0.04 pp and the HMMA count matches exactly. At
8192³ the gain is larger: 94.2% → **96.5%**.

Two findings the table hides:

- **The two changes only work together.** `m16n8k8` on the *8-warp* tile pushes
  registers 116 → 132, and 65,536/(256×2) = **128** is the ceiling for two
  256-thread CTAs per SM. Residency halves and it collapses to 78%. On the 4-warp
  tile the same registers are free, because halving the threads doubles the
  per-thread budget. Sharp cliff, crossable only in the right order.
- **Going further fails.** A 256×128 tile with 128 threads needs 256 accumulator
  registers, hits the 255 hard limit, spills 668 B, lands at 66%. Kept in the repo
  as `--tile 4`, a documented negative result.

It also needs **3** stages rather than 2: with only 8 warps per SM there's less to
hide memory latency with, so the pipeline must be deeper. Occupancy and pipeline
depth are coupled — you cannot tune them independently.

---

## Results

| M=N=K | 4096 | 8192 |
|---|---:|---:|
| **best kernel** | **29.62 TFLOP/s** | **29.95 TFLOP/s** |
| cuBLAS | 31.06 | 31.03 |
| **% of cuBLAS** | **95.3%** | **96.5%** |
| dense Tensor Core occupancy | 92.7% | — |
| registers / thread | 220 | 220 |
| register spills | **0** | **0** |

Best geometry: `128×128×32` CTA tile, 4 warps, warp tile 64×64, two
`mma.sync.m16n8k8` per k-step, 3 `cp.async` stages, XOR-swizzled shared memory,
grouped CTA rasterization. The 8-warp variant (`--tile 1`) is also kept, since most
of the analysis above was done on it.

**Sustained**, 240 s and 50,272 launches: 28.81 TFLOP/s at 113.8 W and 2634 MHz,
72 °C, **no throttle reason ever asserted**, 2.5% decay.

**Correct** against two oracles: bit-identical to cuBLAS, and 4.5e-04 from a
double-precision CPU reference — exactly the FP16 output-store rounding floor, i.e.
as accurate as the output format permits. That CPU oracle exists because cuBLAS
turned out to be a *non-deterministic* reference for skinny matrices, where it
switches to split-K and its rounding drifts run to run. Worth knowing before you
trust a library as ground truth.

An aside that makes the accumulate-width tradeoff concrete: the FP16-accumulate
build hits **46.32 TFLOP/s** — but with 3.78e-02 relative error, which is exactly
the √4096 × 2⁻¹¹ you'd predict for summing 4096 products with no guard digits. It's
also *power*-limited rather than compute-limited (`sw_power_cap` asserted in
379/379 samples, clock down 12.5%), and it's only 1.76× per clock rather than 2×,
because halving the mma shadow breaks the latency hiding the tile was balanced for.
Fast, unusable, and instructive.

---

## What actually transfers to Hopper

**The reasoning, not the code.**

Transfers directly: the staging structure (prologue fill, steady-state
issue-then-consume, one barrier per k-tile); shared-memory capacity versus CTA
residency as the primary tile-size constraint; the swizzle-versus-padding argument;
L2-aware rasterization, which matters *more* on bigger GPUs; fused epilogues;
occupancy and pipeline depth being coupled.

Must be rewritten: `cp.async`/`wait_group` → TMA with mbarriers, which moves
address generation off the SM entirely; `mma.sync` with `ldmatrix` fragments →
`wgmma`, which reads operands straight from shared memory and makes all of this
kernel's fragment-layout work simply disappear; warp specialization into
producer/consumer warpgroups, which has no efficient Ada analogue. Clusters and
distributed shared memory have no analogue at all.

And note the trap that *doesn't* port: the FP32-accumulate halving is a GeForce
restriction, so the ~50% metric ceiling seen here won't appear on H100. Budget for
that before reusing these numbers as a baseline.

---

## What this project could not settle

Stated plainly, because the analysis was independently reviewed and these didn't
survive:

- **Memory bandwidth is unresolved between ~260 and ~290 GB/s.**
  `cudaDeviceProp` and nvidia-smi's own "max clocks" table both say 8001 MHz →
  256 GB/s, but a measured stream hits 259.5 GB/s, which cannot exceed a real roof.
  The 9601 MHz clock reported under load would imply 307 GB/s, but that isn't a
  JEDEC GDDR6 speed bin. It changes no conclusion — the roofline here is stated in
  arithmetic-intensity terms, so compute-bound holds under every candidate roof.
- **FP8 is written but untested.** `sm_89` has FP8 Tensor Cores and the PTX in
  `src/device_query.cu` is correct, but ptxas only assembles it from **CUDA 12.4**;
  this was built on 12.3. The probe is version-guarded so it lights up on upgrade.
  Note FP8 with *FP32* accumulate is halved exactly like FP16 — 1024 FLOP/clk/SM,
  not 2048.
- **Biggest remaining gap to cuBLAS is shared-memory delivery**: 26,472 bank
  conflicts for cuBLAS against our 33.6 M. The XOR swizzle is only 2-way at
  `BK=32`, because a 64 B row holds just four 16 B chunks. `BK=64` would make it
  conflict-free. That's the next experiment.
- Not done: plots of the sweep data, and a CUTLASS cross-check.

---

## Reproducing it

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
.\bin\device_query.exe           # measured mma and DRAM roofs, not spec-sheet ones

# the best configuration, with both correctness oracles
.\bin\gemm_bench_k8.exe --m 4096 --n 4096 --k 4096 --tile 3 --layout 2 --stages 3 `
                        --group 2 --iters 100 --warmup 40 --check --cpu 512

.\scripts\sweep.ps1              # stage x layout x tile sweep -> data\sweep_results.csv
.\scripts\soak.ps1 -Seconds 240  # sustained load + power/clock/thermal telemetry
.\scripts\profile.ps1 -Exe bin\gemm_bench_k8.exe -Tile 3 -Stages 3 -Group 2 -Tag best
```

Four binaries get built, which is how the accumulate-width and mma-shape
comparisons stay honest — each is a separate compile, so register counts are
directly comparable:

| binary | what it is |
|---|---|
| `gemm_bench.exe` | baseline: FP32 accumulate, `m16n8k16` |
| `gemm_bench_k8.exe` | two `m16n8k8` per k-step — **the fastest** |
| `gemm_bench_f16acc.exe` | FP16 accumulate: 2× the tensor rate, unusable accuracy |
| `gemm_bench_fused.exe` | fused bias+ReLU epilogue |

`--help` lists every knob. The ones that matter: `--tile 0..4`, `--stages 1..6`,
`--layout 0|1|2` (row-major / padded / swizzle), `--group G` (L2 rasterization),
`--sweep`/`--gsweep`, `--check`/`--cpu S`, `--soak SEC`, `--unfused`.

---

## Layout

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

## License

[MIT](LICENSE).
