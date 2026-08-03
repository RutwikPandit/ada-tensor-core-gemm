# Ada Tensor Core GEMM — RTX 4060 Laptop (sm_89)

A multistage `cp.async` + `mma.sync` FP16 GEMM for Ada, built to reproduce the
production GEMM pipeline tradeoffs (async global→shared movement, multistage
buffering, shared-memory capacity vs CTA residency, bank conflicts and swizzling,
register/accumulator pressure, L2 rasterization, fused epilogues) on consumer
hardware, and then to **explain every profiler number** rather than just post a
throughput figure.

```text
Ada  :  GDDR6 -> cp.async(LDGSTS) -> staged SMEM -> mma.sync -> registers -> D
Hopper: HBM   -> TMA              -> staged SMEM -> wgmma    -> registers -> D
```

## Results

| M=N=K | 4096 | 8192 |
|---|---:|---:|
| custom kernel | **29.50 TFLOP/s** | **29.26 TFLOP/s** |
| cuBLAS (local speed of light) | 31.22 | 31.04 |
| **% of cuBLAS** | **94.5%** | **94.2%** |
| dense Tensor Core occupancy (clock-free) | **91.9–92.4%** | — |
| registers / thread, spills | 116, **0** | 116, 0 |
| sustained over 240 s | 28.81 TFLOP/s @ 113.8 W, no throttling | — |

Finalist: `128×128×32` CTA tile, 8 warps, warp tile 64×32,
`mma.sync.m16n8k16.row.col.f32.f16.f16.f32`, **2** `cp.async` stages, XOR-swizzled
shared memory, L2 CTA rasterization group 2. Verified against cuBLAS
(bit-identical) and a double-precision CPU oracle (4.5e-04 = the FP16 output-store
rounding floor).

Full analysis, with every number derived and cross-checked:
**[docs/RESULTS.md](docs/RESULTS.md)**.

## Three things that are easy to get wrong on this GPU

1. **The FP32-accumulate tensor roof is 512 FLOP/clk/SM, half the FP16-accumulate
   rate** — a GeForce restriction (A6000/A40 on the same cc 8.6 die are unhalved).
   Measured 511.9 / 1024.4, matching NVIDIA's Ada whitepaper to 0.04%. Widely
   quoted "FP16 TFLOPS" figures for this part are *already* the halved number.
2. **Nsight's `sm__pipe_tensor_op_hmma_cycles_active` caps near 50% here** — it is
   a fixed `16 cycles × HMMA count`, independent of accumulator width. Use
   `sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off` instead: 47% becomes 92%.
3. **The 32 MiB L2 makes 4096³ accidentally optimal.** Naive row-major CTA order
   works there because the whole B panel fits; at 8192³ it thrashes and DRAM reads
   go 527 MB → 9.16 GB. An L2-aware CTA remap (`--group 2`) recovers +28%.

## Requirements

| | |
|---|---|
| GPU | Ada, sm_89 (developed on RTX 4060 Laptop, 24 SMs, 8 GB) |
| CUDA | 12.3+ (FP8 paths need **12.4+**, see below) |
| Host compiler | MSVC (VS 2022 Build Tools). Newer MSVC than CUDA officially allows is handled via `-allow-unsupported-compiler` |
| Profiler | Nsight Compute 2023.3+ |

WSL was not used: it ships `ncu` but no `nvcc`. Everything runs natively on
Windows/PowerShell.

## Quick start

```powershell
.\scripts\build.ps1              # -> bin\{device_query,gemm_bench,gemm_bench_fused,gemm_bench_f16acc}.exe
.\bin\device_query.exe           # hardware limits + measured mma and DRAM roofs

# correctness + throughput against cuBLAS and a CPU oracle
.\bin\gemm_bench.exe --m 4096 --n 4096 --k 4096 --tile 1 --layout 2 --stages 2 --group 2 `
                     --iters 100 --warmup 40 --check --cpu 512

.\scripts\sweep.ps1              # stage x layout x tile sweep -> data\sweep_results.csv
.\scripts\soak.ps1 -Seconds 240  # sustained load + power/clock/thermal telemetry
.\scripts\profile.ps1 -Stages 2 -Layout 2 -Group 2 -Tag finalist   # Nsight report
```

`gemm_bench --help` lists every knob. The interesting ones:

| Flag | Meaning |
|---|---|
| `--tile 0\|1\|2` | 64×128×32 / **128×128×32** / 128×256×32 |
| `--stages 1..6` | `cp.async` pipeline depth; 1 = synchronous baseline |
| `--layout 0\|1\|2` | shared memory: row-major / padded / **XOR swizzle** |
| `--group G` | L2 CTA rasterization; 1 = plain row-major order |
| `--sweep`, `--gsweep` | sweep stages, or rasterization group |
| `--check` | compare against cuBLAS |
| `--cpu S` | verify S outputs against a double-precision CPU dot product |
| `--soak SEC` | run continuously, report per-window throughput |
| `--unfused` | run the epilogue as a separate kernel (Phase 8) |

## Layout

```text
src/device_query.cu   hardware limits; clock64-timed mma issue-rate probe for
                      fp16/fp32-acc, fp16/fp16-acc, int8 and fp8; DRAM bandwidth probe
src/gemm_bench.cu     the kernel (templated on tile/stages/layout), cuBLAS baseline,
                      cuBLAS + CPU correctness oracles, soak mode
scripts/build.ps1     sm_89 build; emits plain, fused-epilogue and FP16-accumulate binaries
scripts/sweep.ps1     stage x layout x tile sweep, logging clocks/power/temp per run
scripts/soak.ps1      sustained load with correlated nvidia-smi telemetry
scripts/profile.ps1   Nsight sections + the metric set used in RESULTS.md
docs/RESULTS.md       full findings, derivations and corrections
docs/experiment-plan.md  the original experiment plan this works through
data/                 measured sweep and soak timelines (committed)
reports/              Nsight reports (gitignored; see reports/README.md)
```

`gemm_bench.cu` is one file on purpose — the pipeline, the fragment layouts and the
epilogue read top-to-bottom, which matters more here than modularity.

## A note on layout choice

The GEMM is **A row-major (M×K), B column-major (K×N), D row-major** — both operands
K-contiguous. That is deliberate: it makes the global→shared copies for A and B
identical with no transpose, feeds `mma.row.col` natively, and is the layout cuBLAS
is fastest at (it selects `..._tn`), so the speed-of-light comparison is not
flattered by a transpose cuBLAS has to absorb.

## FP8

sm_89 has FP8 Tensor Cores and the PTX in `device_query.cu` is correct, but
**ptxas only assembles it from CUDA 12.4 onward**; on 12.3 the probe prints
`UNAVAILABLE - needs CUDA >= 12.4` and everything else still builds. Note also
that FP8 with **FP32** accumulate is halved on GeForce Ada exactly like FP16
(1024 FLOP/clk/SM, not 2048) — the 4× tier needs FP16 or INT32 accumulation.
INT8 `m16n8k32` works today and measures 2048.4 ops/clk/SM.

## Status

The experiment plan's phases 0–8 are complete and documented. Not done:
plots of the sweep data, a CUTLASS profiler cross-check, and the Phase 5
register/accumulator sweep as a standalone experiment (it is currently only
observed indirectly, via cuBLAS's 234 reg/thread at half our occupancy).
