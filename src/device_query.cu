// Phase 0: record the hardware limits the experiment plan depends on, and
// measure the real Tensor Core issue ceiling instead of trusting a spec sheet.
//
// This matters because the whole plan divides by a "peak". On GeForce Ada the
// mma FP16-in/FP32-accumulate path issues at HALF the rate of the
// FP16-in/FP16-accumulate path, so the widely quoted "TFLOPS" number is 2x the
// number a FP32-accumulating GEMM can ever reach. The probe below measures both.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Back-to-back mma with NACC independent accumulator chains and no memory
// traffic in the loop, so the only limit is the Tensor Core issue rate.
//
// The result is reported as FLOP per SM per clock, timed with clock64() inside
// the kernel. Wall-clock TFLOP/s would be useless here: a pure mma loop is the
// densest possible power draw, so a 35-115 W laptop part clocks down far below
// the frequency it sustains during a real GEMM, and the probe would report a
// "peak" lower than the GEMM that is supposedly approaching it.
// FP8 mma on sm_89 exists in hardware and the PTX below is the correct
// instruction, but ptxas only assembles it from CUDA 12.4 onward ("Support for
// Ada SM89 FP8 tensor cores requires CUDA 12.4 or newer"). Guarded so the probe
// lights up automatically on a newer toolkit instead of failing the build.
#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)
#define CUDA_AT_LEAST(maj, min) \
    (__CUDACC_VER_MAJOR__ > (maj) || (__CUDACC_VER_MAJOR__ == (maj) && __CUDACC_VER_MINOR__ >= (min)))
#if CUDA_AT_LEAST(12, 4)
#define HAVE_FP8_MMA 1
#else
#define HAVE_FP8_MMA 0
#endif

enum ProbeOp {
    OP_F16_ACC_F16 = 0,   // m16n8k16  fp16 x fp16 -> fp16
    OP_F16_ACC_F32 = 1,   // m16n8k16  fp16 x fp16 -> fp32
    OP_S8_ACC_S32  = 2,   // m16n8k32  int8 x int8 -> int32
    OP_E4M3_ACC_F32 = 3,  // m16n8k32  fp8  x fp8  -> fp32   (needs CUDA >= 12.4)
};

// ops per warp-level instruction = 2 * M * N * K
__host__ __device__ constexpr double ops_per_inst(int op) {
    return (op == OP_F16_ACC_F16 || op == OP_F16_ACC_F32) ? 2.0 * 16 * 8 * 16
                                                          : 2.0 * 16 * 8 * 32;
}

template <int NACC, int OP>
__global__ void mma_probe(float* sink, int iters, long long* cycles) {
    long long t_start = clock64();
    uint32_t a[4], b[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = 0x3c003c00u ^ (threadIdx.x + i);  // ~1.0h pairs
#pragma unroll
    for (int i = 0; i < 2; ++i) b[i] = 0x3c003c00u ^ (threadIdx.x + i);

    if (OP == OP_S8_ACC_S32) {
        int c[NACC][4];
#pragma unroll
        for (int j = 0; j < NACC; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) c[j][e] = 0;
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int j = 0; j < NACC; ++j)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                               "r"(b[0]), "r"(b[1]));
        }
        int s = 0;
#pragma unroll
        for (int j = 0; j < NACC; ++j)
            for (int e = 0; e < 4; ++e) s += c[j][e];
        if (s == 123456789) sink[0] = 1.f;
    } else if (OP == OP_E4M3_ACC_F32) {
#if HAVE_FP8_MMA
        float c[NACC][4];
#pragma unroll
        for (int j = 0; j < NACC; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) c[j][e] = 0.f;
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int j = 0; j < NACC; ++j)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                               "r"(b[0]), "r"(b[1]));
        }
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < NACC; ++j)
            for (int e = 0; e < 4; ++e) s += c[j][e];
        if (s == 1234.5f) sink[0] = s;
#endif
    } else if (OP == OP_F16_ACC_F32) {
        float c[NACC][4];
#pragma unroll
        for (int j = 0; j < NACC; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) c[j][e] = 0.f;
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int j = 0; j < NACC; ++j)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                               "r"(b[0]), "r"(b[1]));
        }
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < NACC; ++j)
            for (int e = 0; e < 4; ++e) s += c[j][e];
        if (s == 1234.5f) sink[0] = s;   // keep it live, never taken
    } else {
        uint32_t c[NACC][2];
#pragma unroll
        for (int j = 0; j < NACC; ++j) { c[j][0] = 0; c[j][1] = 0; }
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int j = 0; j < NACC; ++j)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                             "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                             : "+r"(c[j][0]), "+r"(c[j][1])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                               "r"(b[0]), "r"(b[1]));
        }
        uint32_t s = 0;
#pragma unroll
        for (int j = 0; j < NACC; ++j) s += c[j][0] + c[j][1];
        if (s == 0xdeadbeefu) sink[0] = 1.f;
    }
    if (threadIdx.x == 0) cycles[blockIdx.x] = clock64() - t_start;
}

// One (NACC, THREADS) point. NACC sets the independent mma chains per warp;
// too few and the measurement reports mma latency instead of issue rate.
template <int OP, int NACC, int THREADS>
static double probe_one(int sms) {
    constexpr int ITERS = 2048;
    // one block per SM so the measured cycles belong to a known warp population
    const int blocks = sms;
    float* sink; cudaMalloc(&sink, sizeof(float));
    long long* dCyc; cudaMalloc(&dCyc, blocks * sizeof(long long));
    for (int w = 0; w < 3; ++w) mma_probe<NACC, OP><<<blocks, THREADS>>>(sink, ITERS, dCyc);
    cudaDeviceSynchronize();
    mma_probe<NACC, OP><<<blocks, THREADS>>>(sink, ITERS, dCyc);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { cudaFree(sink); cudaFree(dCyc); return 0; }

    std::vector<long long> hCyc(blocks);
    cudaMemcpy(hCyc.data(), dCyc, blocks * sizeof(long long), cudaMemcpyDeviceToHost);
    long long best = hCyc[0];
    for (long long c : hCyc) if (c < best) best = c;   // least-perturbed SM
    cudaFree(sink); cudaFree(dCyc);

    double ops_per_sm = (double)(THREADS / 32) * ITERS * NACC * ops_per_inst(OP);
    return ops_per_sm / (double)best;
}

// Streaming read+write over a buffer far larger than the 32 MiB L2, to get the
// achievable DRAM roof. Needed because the two available clock sources disagree:
// cudaDeviceProp.memoryClockRate reports a lower default (8001 MHz -> 256 GB/s)
// while nvidia-smi clocks.mem reports the boosted 9601 MHz (-> 307.2 GB/s).
// A measurement settles it without having to pick.
__global__ void bw_kernel(const float4* __restrict__ src, float4* __restrict__ dst,
                          size_t n4) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) dst[i] = src[i];
}

static double probe_bandwidth(int sms) {
    const size_t bytes = 512ull << 20;          // 512 MiB per buffer
    const size_t n4 = bytes / sizeof(float4);
    float4 *a, *b;
    if (cudaMalloc(&a, bytes) != cudaSuccess) return 0;
    if (cudaMalloc(&b, bytes) != cudaSuccess) { cudaFree(a); return 0; }
    cudaMemset(a, 1, bytes);
    const int threads = 256, blocks = sms * 16;
    for (int w = 0; w < 3; ++w) bw_kernel<<<blocks, threads>>>(a, b, n4);
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int reps = 20;
    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r) bw_kernel<<<blocks, threads>>>(a, b, n4);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(a); cudaFree(b);
    // each element is read once and written once
    return 2.0 * bytes * reps / (ms * 1e-3) / 1e9;
}

// Sweep chains x warps and keep the best: the issue-rate ceiling is the
// asymptote, and which point reaches it differs per op (wider accumulators cost
// more registers, so each path needs a different chain/warp balance to stop
// being latency-bound rather than issue-bound).
template <int OP>
static double probe(int sms, const char* label, bool available = true) {
    if (!available) {
        printf("  %-42s: %s\n", label, "UNAVAILABLE - needs CUDA >= 12.4 (have "
               STRINGIFY(__CUDACC_VER_MAJOR__) "." STRINGIFY(__CUDACC_VER_MINOR__) ")");
        return 0;
    }
    double best = 0;
    const char* how = "";
    struct { double v; const char* tag; } pts[] = {
        {probe_one<OP,  4, 256>(sms), "4ch x  8warp"},
        {probe_one<OP,  8, 256>(sms), "8ch x  8warp"},
        {probe_one<OP, 16, 256>(sms), "16ch x  8warp"},
        {probe_one<OP,  4, 512>(sms), "4ch x 16warp"},
        {probe_one<OP,  8, 512>(sms), "8ch x 16warp"},
        {probe_one<OP, 16, 512>(sms), "16ch x 16warp"},
        {probe_one<OP,  8,1024>(sms), "8ch x 32warp"},
    };
    for (auto& p : pts) if (p.v > best) { best = p.v; how = p.tag; }
    printf("  %-42s: %6.1f ops/clk/SM   (best of 7, at %s)\n", label, best, how);
    return best;
}


int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        printf("no CUDA device\n");
        return 1;
    }
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p{};
        cudaGetDeviceProperties(&p, d);
        int smemPerSMOptin = 0, smemPerBlockOptin = 0, regsPerSM = 0;
        cudaDeviceGetAttribute(&smemPerSMOptin, cudaDevAttrMaxSharedMemoryPerMultiprocessor, d);
        cudaDeviceGetAttribute(&smemPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, d);
        cudaDeviceGetAttribute(&regsPerSM, cudaDevAttrMaxRegistersPerMultiprocessor, d);

        printf("device %d: %s\n", d, p.name);
        printf("  compute capability          : %d.%d  (-arch=sm_%d%d)\n",
               p.major, p.minor, p.major, p.minor);
        printf("  SM count                    : %d\n", p.multiProcessorCount);
        printf("  clock (max SM)              : %.0f MHz\n", p.clockRate / 1000.0);
        printf("  memory                      : %.2f GiB @ %.0f MHz, %d-bit\n",
               p.totalGlobalMem / 1073741824.0, p.memoryClockRate / 1000.0, p.memoryBusWidth);
        printf("  theoretical DRAM bandwidth  : %.1f GB/s\n",
               2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8) / 1e9);
        printf("  L2 cache                    : %d KiB\n", p.l2CacheSize / 1024);
        printf("  shared mem / block (default): %zu B\n", p.sharedMemPerBlock);
        printf("  shared mem / block (opt-in) : %d B  <-- cudaFuncAttributeMaxDynamicSharedMemorySize\n",
               smemPerBlockOptin);
        printf("  shared mem / SM             : %d B\n", smemPerSMOptin);
        printf("  regs / block                : %d\n", p.regsPerBlock);
        printf("  regs / SM                   : %d\n", regsPerSM);
        printf("  max threads / block         : %d\n", p.maxThreadsPerBlock);
        printf("  max threads / SM            : %d  (%d warps)\n",
               p.maxThreadsPerMultiProcessor, p.maxThreadsPerMultiProcessor / 32);
        printf("  max blocks / SM             : %d\n", p.maxBlocksPerMultiProcessor);
        printf("  warp size                   : %d\n", p.warpSize);
        printf("  async engines / concurrent  : %d / %d\n", p.asyncEngineCount, p.concurrentKernels);
    }

    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    printf("\nmeasured Tensor Core issue ceiling (register-resident mma, clock64-timed):\n");
    double f32 = probe<OP_F16_ACC_F32>(p.multiProcessorCount, "m16n8k16  fp16 in / fp32 accumulate");
    double f16 = probe<OP_F16_ACC_F16>(p.multiProcessorCount, "m16n8k16  fp16 in / fp16 accumulate");
    double s8  = probe<OP_S8_ACC_S32 >(p.multiProcessorCount, "m16n8k32  int8 in / int32 accumulate");
    double e4m3 = probe<OP_E4M3_ACC_F32>(p.multiProcessorCount,
                                         "m16n8k32  fp8-e4m3 in / fp32 accumulate", HAVE_FP8_MMA);
    printf("\n  relative to the fp16/fp32-accumulate baseline:\n");
    printf("    fp16 acc  %5.2fx    int8  %5.2fx", f16 / f32, s8 / f32);
    if (e4m3 > 0) printf("    fp8-e4m3  %5.2fx", e4m3 / f32);
    printf("\n  (dense mma only -- no .sp sparse variants used anywhere)\n");

    double bw = probe_bandwidth(p.multiProcessorCount);
    printf("\nmeasured DRAM bandwidth (512 MiB stream, read+write, L2-defeating):\n");
    printf("  %-42s: %6.1f GB/s\n", "achieved", bw);
    printf("  %-42s: %6.1f GB/s  (cudaDeviceProp %.0f MHz)\n", "theoretical, prop clock",
           2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8) / 1e9,
           p.memoryClockRate / 1000.0);
    printf("  %-42s: %6.1f GB/s  (nvidia-smi clocks.mem 9601 MHz = 4 x 2400)\n",
           "theoretical, boosted clock", 2.0 * 9601e6 * (p.memoryBusWidth / 8.0) / 1e9);
    printf("  => %.0f%% of the boosted-clock roof\n",
           100.0 * bw / (2.0 * 9601e6 * (p.memoryBusWidth / 8.0) / 1e9));

    // Report the denominator against the clock actually observed under load,
    // not the nominal boost clock: read clocks.sm from nvidia-smi during a run.
    printf("\n  peak for an FP32-accumulating GEMM = %.0f FLOP/clk/SM x %d SMs x f_sm\n",
           f32, p.multiProcessorCount);
    for (double ghz : {1.89, 2.40, 2.64}) {
        printf("      at %.2f GHz -> %5.1f TFLOP/s%s\n", ghz,
               f32 * p.multiProcessorCount * ghz * 1e9 / 1e12,
               ghz == 1.89 ? "   (nominal boost from cudaDeviceProp)" : "");
    }
    printf("  The marketing FP16 TFLOPS number corresponds to the fp16-accumulate row\n"
           "  and is not reachable by a kernel that accumulates in FP32.\n");
    return 0;
}
