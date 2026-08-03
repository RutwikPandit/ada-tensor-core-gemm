// ---------------------------------------------------------------------------
// Multistage cp.async + mma.sync FP16 GEMM for Ada (sm_89).
//
// Mirrors the Hopper pipeline shape with Ada mechanisms:
//     GDDR6 -> cp.async(LDGSTS) -> staged shared memory -> mma.sync -> regs -> D
//
// Problem:  D = A * B            (+ optional fused bias + ReLU)
//   A : M x K, row-major     (K contiguous)
//   B : K x N, column-major  (K contiguous)   == B^T stored N x K row-major
//   D : M x N, row-major
//
// The "TN" layout is deliberate: both operands are K-contiguous, so the global
// -> shared copies for A and B are the same code, need no transpose, and feed
// mma.sync.aligned.m16n8k16.row.col natively. This is also the layout cuBLAS
// is fastest at, which keeps the speed-of-light comparison honest.
//
// Build: see build.ps1   (-arch=sm_89)
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#ifndef FUSE_EPILOGUE
#define FUSE_EPILOGUE 0
#endif

// ACC_F16=1 accumulates in FP16 instead of FP32.
//
// On GeForce Ada the fp16-accumulate mma issues at 1024 FLOP/clk/SM, exactly 2x
// the fp32-accumulate path's 512 (measured in device_query.cu). That is the only
// way to drive Nsight's sm__pipe_tensor_op_hmma_cycles_active near 100%, because
// that metric's denominator is the *fastest* HMMA variant the SM can issue.
//
// It is a demonstration, not a better GEMM: summing K terms in FP16 has ~11 bits
// of mantissa and no guard digits, so accuracy degrades with K. The error is
// measured against the CPU oracle rather than asserted -- see RESULTS.md.
#ifndef ACC_F16
#define ACC_F16 0
#endif

#define CUDA_CHECK(x)                                                              \
    do {                                                                           \
        cudaError_t e_ = (x);                                                       \
        if (e_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
                    cudaGetErrorString(e_));                                        \
            exit(1);                                                                \
        }                                                                          \
    } while (0)

#define CUBLAS_CHECK(x)                                                            \
    do {                                                                           \
        cublasStatus_t s_ = (x);                                                    \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                          \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s_);\
            exit(1);                                                                \
        }                                                                          \
    } while (0)

// ---------------------------------------------------------------------------
// PTX primitives
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16-byte async global->shared copy. `pred == false` zero-fills instead of
// reading, which is how ragged M/N edges are handled without a branch.
__device__ __forceinline__ void cp_async_16B(uint32_t dst, const void* src, bool pred) {
    int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
                 "r"(dst), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Block until at most N copy groups are still outstanding.
// This is the Ada stand-in for a consumer waiting on TMA arrival.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// One ldmatrix.x4 fetches four 8x8 b16 tiles straight into MMA operand layout.
// lanes  0-7  -> d0, 8-15 -> d1, 16-23 -> d2, 24-31 -> d3
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&d)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(float (&c)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]));
}

// FP16-accumulate form: 2 result registers instead of 4 (each holds a packed
// half2), and it retires on the tensor pipe in half the cycles.
__device__ __forceinline__ void mma_m16n8k16(uint32_t (&c)[2], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]));
}

#if ACC_F16
using AccReg = uint32_t;             // packed half2 pair
static constexpr int ACC_N = 2;
#else
using AccReg = float;
static constexpr int ACC_N = 4;
#endif

// ---------------------------------------------------------------------------
// Shared-memory layouts (Phase 4)
//
// A BK=32 half row is 64 B = 16 banks, so a plain row-major tile makes the 8
// lanes of an ldmatrix quarter hit only 2 bank groups -> 4-way conflict.
//   LAYOUT_NONE : row * BK + k                      (conflicts, on purpose)
//   LAYOUT_PAD  : row * (BK+8) + k                  row stride 80 B = 20 words,
//                                                   20*r mod 32 is distinct for
//                                                   r=0..7 -> conflict free
//   LAYOUT_SWZ  : XOR the 16-B chunk index with the row -> conflict free with
//                 no extra shared memory once a row is >= 128 B (BK >= 64);
//                 at BK=32 only 4 chunks exist so it lands at 2-way.
// All three keep every access 16-B aligned, which cp.async.cg and ldmatrix need.
// ---------------------------------------------------------------------------
enum { LAYOUT_NONE = 0, LAYOUT_PAD = 1, LAYOUT_SWZ = 2 };

template <int BK, int LAYOUT>
struct SmemLayout {
    static constexpr int LDS = (LAYOUT == LAYOUT_PAD) ? (BK + 8) : BK;
    static constexpr int CHUNKS = BK / 8;
    __device__ __forceinline__ static int off(int row, int k) {
        if (LAYOUT == LAYOUT_SWZ) {
            int c = (k >> 3) ^ (row & (CHUNKS - 1));
            return row * LDS + (c << 3) + (k & 7);
        }
        return row * LDS + k;
    }
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES, int LAYOUT>
struct Traits {
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
    static constexpr int WM = BM / WARPS_M;      // warp output rows
    static constexpr int WN = BN / WARPS_N;      // warp output cols
    static constexpr int MI = WM / 16;           // m-fragments per warp
    static constexpr int NI = WN / 8;            // n-fragments per warp
    static constexpr int KI = BK / 16;           // mma k-steps per shared tile
    static constexpr int NG = NI / 2;            // ldmatrix.x4 groups for B

    using L = SmemLayout<BK, LAYOUT>;
    static constexpr int LDS = L::LDS;
    static constexpr int A_STAGE = BM * LDS;     // halves
    static constexpr int B_STAGE = BN * LDS;
    static constexpr int SMEM_BYTES = STAGES * (A_STAGE + B_STAGE) * (int)sizeof(half);

    static constexpr int CPR = BK / 8;                  // 16-B chunks per row
    static constexpr int RPI = THREADS / CPR;           // rows filled per pass
    static constexpr int A_PASS = (BM + RPI - 1) / RPI;
    static constexpr int B_PASS = (BN + RPI - 1) / RPI;

    static_assert(WM % 16 == 0 && WN % 16 == 0, "warp tile must be 16-aligned");
    static_assert(BK % 16 == 0, "BK must be a multiple of the mma K step");
    static_assert(THREADS % CPR == 0, "block must divide evenly into row chunks");
    static_assert(BM % RPI == 0 && BN % RPI == 0, "tile rows must divide evenly");
};

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES, int LAYOUT>
__global__ __launch_bounds__(Traits<BM, BN, BK, WARPS_M, WARPS_N, STAGES, LAYOUT>::THREADS)
void gemm_kernel(const half* __restrict__ A, const half* __restrict__ B,
                 half* __restrict__ D, const half* __restrict__ bias,
                 int M, int N, int K, float alpha, int group) {
    using T = Traits<BM, BN, BK, WARPS_M, WARPS_N, STAGES, LAYOUT>;
    using L = typename T::L;

    extern __shared__ __align__(16) char smem_raw[];
    half* sA = reinterpret_cast<half*>(smem_raw);
    half* sB = sA + STAGES * T::A_STAGE;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm   = warp / WARPS_N;
    const int wn   = warp % WARPS_N;

    // ---- CTA rasterization ------------------------------------------------
    // Default row-major CTA order sweeps all of B for every strip of A, so once
    // the B panel (N x K) outgrows L2 the same bytes are refetched from DRAM
    // once per M-tile. Walking `group` M-tiles column-major inside a group
    // shrinks the concurrent footprint to group*BM*K + BN*K. group == 1 is the
    // plain row-major order.
    int m_tile, n_tile;
    if (group <= 1) {
        m_tile = blockIdx.y;
        n_tile = blockIdx.x;
    } else {
        const int tilesN = gridDim.x, tilesM = gridDim.y;
        const int linear = blockIdx.y * tilesN + blockIdx.x;
        const int g      = linear / (group * tilesN);      // which group of M-tiles
        const int first  = g * group;
        const int rows   = min(group, tilesM - first);     // last group may be short
        const int idx    = linear - first * tilesN;
        m_tile = first + idx % rows;
        n_tile = idx / rows;
    }
    const int m_base = m_tile * BM;
    const int n_base = n_tile * BN;

    // global->shared assignment: one 8-half chunk per thread per pass
    const int ld_row = tid / T::CPR;
    const int ld_k   = (tid % T::CPR) * 8;

    const int ntiles = K / BK;

    AccReg acc[T::MI][T::NI][ACC_N];
#pragma unroll
    for (int i = 0; i < T::MI; ++i)
#pragma unroll
        for (int j = 0; j < T::NI; ++j)
#pragma unroll
            for (int e = 0; e < ACC_N; ++e) acc[i][j][e] = AccReg(0);

    // ---- issue one stage worth of async copies -----------------------------
    auto issue = [&](int stage, int ktile) {
        const int k0 = ktile * BK;
#pragma unroll
        for (int p = 0; p < T::A_PASS; ++p) {
            int r  = ld_row + p * T::RPI;
            int gm = m_base + r;
            bool ok = (gm < M);
            const half* src = A + (size_t)(ok ? gm : 0) * K + k0 + ld_k;
            cp_async_16B(smem_addr(&sA[stage * T::A_STAGE + L::off(r, ld_k)]), src, ok);
        }
#pragma unroll
        for (int p = 0; p < T::B_PASS; ++p) {
            int r  = ld_row + p * T::RPI;
            int gn = n_base + r;
            bool ok = (gn < N);
            const half* src = B + (size_t)(ok ? gn : 0) * K + k0 + ld_k;
            cp_async_16B(smem_addr(&sB[stage * T::B_STAGE + L::off(r, ld_k)]), src, ok);
        }
    };

    // ---- consume one stage: ldmatrix -> mma --------------------------------
    auto compute = [&](int stage) {
        const half* pA = sA + stage * T::A_STAGE;
        const half* pB = sB + stage * T::B_STAGE;
        const int lrow = lane & 15;          // row inside the 16x16 fragment
        const int lk8  = (lane >> 4) * 8;    // low / high half of the k16 step
#pragma unroll
        for (int ks = 0; ks < T::KI; ++ks) {
            uint32_t ra[T::MI][4];
            uint32_t rb[T::NG][4];
#pragma unroll
            for (int mi = 0; mi < T::MI; ++mi)
                ldmatrix_x4(ra[mi], smem_addr(&pA[L::off(wm * T::WM + mi * 16 + lrow,
                                                        ks * 16 + lk8)]));
#pragma unroll
            for (int g = 0; g < T::NG; ++g)
                ldmatrix_x4(rb[g], smem_addr(&pB[L::off(wn * T::WN + g * 16 + lrow,
                                                       ks * 16 + lk8)]));
#pragma unroll
            for (int mi = 0; mi < T::MI; ++mi) {
#pragma unroll
                for (int ni = 0; ni < T::NI; ++ni) {
                    // group g covers n-tiles 2g (lanes 0-7 rows) and 2g+1;
                    // d[h] is the k-low half, d[2+h] the k-high half.
                    const int g = ni >> 1, h = ni & 1;
                    uint32_t b2[2] = {rb[g][h], rb[g][2 + h]};
                    mma_m16n8k16(acc[mi][ni], ra[mi], b2);
                }
            }
        }
    };

    // ---- mainloop ----------------------------------------------------------
    if (STAGES == 1) {
        // Synchronous baseline: the copy for tile kt must fully land before any
        // mma on tile kt can issue. No producer/consumer overlap at all.
        for (int kt = 0; kt < ntiles; ++kt) {
            __syncthreads();
            issue(0, kt);
            cp_async_commit();
            cp_async_wait<0>();
            __syncthreads();
            compute(0);
        }
    } else {
        // Fill STAGES-1 groups, then keep exactly STAGES-1 in flight.
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < ntiles) issue(s, s);
            cp_async_commit();           // commit unconditionally: group counting
        }
        for (int kt = 0; kt < ntiles; ++kt) {
            cp_async_wait<STAGES - 2>(); // tile kt has arrived
            __syncthreads();             // also releases stage (kt-1)%STAGES
            int next = kt + STAGES - 1;
            if (next < ntiles) issue(next % STAGES, next);
            cp_async_commit();
            compute(kt % STAGES);        // overlaps the copies just issued
        }
    }

    // ---- epilogue ----------------------------------------------------------
    const int gid = lane >> 2;           // accumulator row within a 16x8 tile
    const int tig = (lane & 3) * 2;      // accumulator col pair
#pragma unroll
    for (int mi = 0; mi < T::MI; ++mi) {
#pragma unroll
        for (int ni = 0; ni < T::NI; ++ni) {
            int m0 = m_base + wm * T::WM + mi * 16 + gid;
            int n0 = n_base + wn * T::WN + ni * 8 + tig;
#pragma unroll
            for (int half_i = 0; half_i < 2; ++half_i) {
                int m = m0 + half_i * 8;
                if (m >= M) continue;
#if ACC_F16
                // one register holds the (row, col) and (row, col+1) pair
                __half2 pk = *reinterpret_cast<const __half2*>(&acc[mi][ni][half_i]);
                float v0 = __low2float(pk), v1 = __high2float(pk);
#else
                float v0 = acc[mi][ni][half_i * 2 + 0];
                float v1 = acc[mi][ni][half_i * 2 + 1];
#endif
#if FUSE_EPILOGUE
                v0 *= alpha; v1 *= alpha;
                if (bias) {
                    if (n0 + 0 < N) v0 += __half2float(bias[n0 + 0]);
                    if (n0 + 1 < N) v1 += __half2float(bias[n0 + 1]);
                }
                v0 = v0 > 0.f ? v0 : 0.f;
                v1 = v1 > 0.f ? v1 : 0.f;
#else
                (void)bias; (void)alpha;
#endif
                half* dst = D + (size_t)m * N + n0;
                if (n0 + 1 < N) {
                    *reinterpret_cast<__half2*>(dst) = __floats2half2_rn(v0, v1);
                } else if (n0 < N) {
                    dst[0] = __float2half(v0);
                }
            }
        }
    }
}

// Unfused reference epilogue for Phase 8.
__global__ void bias_relu_kernel(half* D, const half* bias, int M, int N, float alpha) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * N) return;
    float v = __half2float(D[i]) * alpha;
    if (bias) v += __half2float(bias[i % N]);
    D[i] = __float2half(v > 0.f ? v : 0.f);
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
template <int TILE> struct TileCfg;
// BM,  BN,  BK, WARPS_M, WARPS_N   -> threads, warp tile, acc regs
template <> struct TileCfg<0> { enum { BM =  64, BN = 128, BK = 32, WM = 2, WN = 4 }; }; // 256 thr, 32x32, 32
template <> struct TileCfg<1> { enum { BM = 128, BN = 128, BK = 32, WM = 2, WN = 4 }; }; // 256 thr, 64x32, 64
template <> struct TileCfg<2> { enum { BM = 128, BN = 256, BK = 32, WM = 2, WN = 8 }; }; // 512 thr, 64x32, 64

struct Result {
    bool  ok = false;
    const char* why = "";
    int   threads = 0, regs = 0, smem = 0, blocksPerSM = 0;
    double ms = 0, tflops = 0, maxerr = 0;
};

struct Args {
    int M = 4096, N = 4096, K = 4096;
    int tile = 1, stages = 3, layout = 1, group = 1;
    int iters = 50, warmup = 20;
    int cpu = 0;                 // number of output elements to verify on the CPU
    double soak = 0;             // seconds of back-to-back launches (0 = fixed iters)
    double window = 2.0;         // soak reporting window, seconds
    bool check = false, ref = true, sweep = false, unfused = false;
    bool gsweep = false;
};

static const half* g_A;
static const half* g_B;
static half* g_D;
static const half* g_bias;        // null unless built with FUSE_EPILOGUE=1
static const half* g_biasAlways;  // always valid, for the unfused epilogue kernel

template <int TILE, int STAGES, int LAYOUT>
static Result run_cfg(const Args& a, cudaStream_t stream) {
    using C = TileCfg<TILE>;
    constexpr int BM = C::BM, BN = C::BN, BK = C::BK, WM = C::WM, WN = C::WN;
    using T = Traits<BM, BN, BK, WM, WN, STAGES, LAYOUT>;
    auto kern = gemm_kernel<BM, BN, BK, WM, WN, STAGES, LAYOUT>;

    Result r;
    r.threads = T::THREADS;
    r.smem    = T::SMEM_BYTES;

    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (T::SMEM_BYTES > optin) { r.why = "smem over per-block opt-in limit"; return r; }
    CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    T::SMEM_BYTES));

    cudaFuncAttributes fa{};
    CUDA_CHECK(cudaFuncGetAttributes(&fa, kern));
    r.regs = fa.numRegs;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.blocksPerSM, kern,
                                                             T::THREADS, T::SMEM_BYTES));

    dim3 grid((a.N + BN - 1) / BN, (a.M + BM - 1) / BM);
    dim3 blk(T::THREADS);
    // In --unfused mode the separate bias+ReLU kernel must be inside the timed
    // region, otherwise the comparison silently credits the unfused path with
    // zero epilogue cost -- which is the entire quantity Phase 8 is measuring.
    const size_t nD = (size_t)a.M * a.N;
    auto launch = [&] {
        kern<<<grid, blk, T::SMEM_BYTES, stream>>>(g_A, g_B, g_D, g_bias,
                                                   a.M, a.N, a.K, 1.0f, a.group);
        if (a.unfused) {
            constexpr int t = 256;
            bias_relu_kernel<<<(unsigned)((nD + t - 1) / t), t, 0, stream>>>(
                g_D, g_biasAlways, a.M, a.N, 1.0f);
        }
    };

    for (int i = 0; i < a.warmup; ++i) launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { r.why = cudaGetErrorString(e); return r; }

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    const double flop = 2.0 * a.M * a.N * a.K;

    if (a.soak > 0) {
        // Sustained-load mode: keep the GPU saturated for `soak` seconds and
        // report throughput per window, so thermal/power decay is visible as a
        // trend rather than averaged away. Each window is bracketed by CUDA
        // events, and the host wall clock is printed alongside so an external
        // nvidia-smi sampler can be aligned to it.
        printf("# soak %.0f s, window %.1f s -- columns are parseable CSV after 'SOAK,'\n",
               a.soak, a.window);
        printf("SOAK,elapsed_s,window_ms,iters,tflops,ms_per_iter\n");
        fflush(stdout);

        const auto wall0 = std::chrono::steady_clock::now();
        double best = 0, worst = 1e30, sum_flop = 0, sum_s = 0;
        int nwin = 0;
        for (;;) {
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - wall0).count();
            if (elapsed >= a.soak) break;

            // launch until the window closes, then measure what actually ran
            int n = 0;
            CUDA_CHECK(cudaEventRecord(t0, stream));
            const auto wstart = std::chrono::steady_clock::now();
            do {
                launch();
                ++n;
                // throttle the host so the queue does not run away from the window
                if ((n & 7) == 0) CUDA_CHECK(cudaStreamSynchronize(stream));
            } while (std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - wstart).count() < a.window);
            CUDA_CHECK(cudaEventRecord(t1, stream));
            CUDA_CHECK(cudaEventSynchronize(t1));
            float wms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&wms, t0, t1));

            double tf = flop * n / (wms * 1e-3) / 1e12;
            double now = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - wall0).count();
            printf("SOAK,%.2f,%.2f,%d,%.3f,%.4f\n", now, wms, n, tf, wms / n);
            fflush(stdout);
            if (tf > best) best = tf;
            if (tf < worst) worst = tf;
            sum_flop += flop * n;
            sum_s += wms * 1e-3;
            ++nwin;
        }
        double avg = sum_flop / sum_s / 1e12;
        printf("# soak summary: windows=%d  avg=%.2f  max=%.2f  min=%.2f TFLOP/s"
               "  spread=%.1f%%\n",
               nwin, avg, best, worst, 100.0 * (best - worst) / best);
        r.ms = sum_s * 1e3 / (sum_flop / flop);
        r.tflops = avg;
        r.ok = true;
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        return r;
    }

    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < a.iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float tot = 0;
    CUDA_CHECK(cudaEventElapsedTime(&tot, t0, t1));
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    r.ms = tot / a.iters;
    r.tflops = flop / (r.ms * 1e-3) / 1e12;
    r.ok = true;
    return r;
}

// runtime -> template dispatch
using RunFn = Result (*)(const Args&, cudaStream_t);

template <int TILE, int LAYOUT>
static RunFn pick_stages(int s) {
    switch (s) {
        case 1: return &run_cfg<TILE, 1, LAYOUT>;
        case 2: return &run_cfg<TILE, 2, LAYOUT>;
        case 3: return &run_cfg<TILE, 3, LAYOUT>;
        case 4: return &run_cfg<TILE, 4, LAYOUT>;
        case 5: return &run_cfg<TILE, 5, LAYOUT>;
        case 6: return &run_cfg<TILE, 6, LAYOUT>;
    }
    return nullptr;
}
template <int TILE>
static RunFn pick_layout(int l, int s) {
    switch (l) {
        case 0: return pick_stages<TILE, LAYOUT_NONE>(s);
        case 1: return pick_stages<TILE, LAYOUT_PAD>(s);
        case 2: return pick_stages<TILE, LAYOUT_SWZ>(s);
    }
    return nullptr;
}
static RunFn pick(int tile, int layout, int stages) {
    switch (tile) {
        case 0: return pick_layout<0>(layout, stages);
        case 1: return pick_layout<1>(layout, stages);
        case 2: return pick_layout<2>(layout, stages);
    }
    return nullptr;
}

static const char* layout_name(int l) {
    return l == 0 ? "none" : l == 1 ? "pad" : "swizzle";
}
static void tile_dims(int t, int& bm, int& bn, int& bk) {
    switch (t) {
        case 0: bm = 64;  bn = 128; bk = 32; break;
        case 1: bm = 128; bn = 128; bk = 32; break;
        default: bm = 128; bn = 256; bk = 32; break;
    }
}

// cuBLAS reference / local speed of light.
//   row-major D(MxN) == col-major D^T(NxM) = B^T * A^T
//   B stored KxN col-major  -> op_T gives B^T
//   A stored MxK row-major  -> reads as KxM col-major == A^T -> op_N
static double cublas_gemm(cublasHandle_t h, const Args& a, half* out, int iters,
                          int warmup, cudaStream_t stream) {
    const float alpha = 1.f, beta = 0.f;
    auto launch = [&] {
        CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, a.N, a.M, a.K,
                                  &alpha, g_B, CUDA_R_16F, a.K,
                                  g_A, CUDA_R_16F, a.K,
                                  &beta, out, CUDA_R_16F, a.N,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float tot = 0; CUDA_CHECK(cudaEventElapsedTime(&tot, t0, t1));
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return tot / iters;
}

__global__ void max_rel_diff(const half* x, const half* y, size_t n, float* out) {
    __shared__ float red[256];
    float m = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)blockDim.x * gridDim.x) {
        float a = __half2float(x[i]), b = __half2float(y[i]);
        float d = fabsf(a - b) / fmaxf(1.f, fabsf(b));
        m = fmaxf(m, d);
    }
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax((int*)out, __float_as_int(red[0]));
}

// Deterministic ground truth. cuBLAS is a poor correctness oracle for skinny
// problems: it switches to split-K kernels whose accumulation order (and so
// whose FP16 rounding) varies between runs, which makes a tolerance check
// compare against a moving target. Instead spot-check a deterministic sample of
// output elements against a double-precision CPU dot product -- O(samples * K)
// regardless of problem size, so it works at 8192 as cheaply as at 512.
static double cpu_spot_check(const std::vector<half>& hA, const std::vector<half>& hB,
                             const std::vector<half>& hD, const std::vector<half>& hBias,
                             int M, int N, int K, int samples, bool fused, float alpha,
                             int* worst_m, int* worst_n) {
    double worst = 0.0;
    *worst_m = *worst_n = -1;
    // fixed stride walk so the sampled set is reproducible and spread out
    const long long total = (long long)M * N;
    const long long stride = (total / std::max(1, samples)) | 1;
    for (long long s = 0, idx = 0; s < samples && s < total; ++s, idx = (idx + stride) % total) {
        int m = (int)(idx / N), n = (int)(idx % N);
        double acc = 0.0;
        for (int k = 0; k < K; ++k)
            acc += (double)__half2float(hA[(size_t)m * K + k]) *
                   (double)__half2float(hB[(size_t)n * K + k]);
        if (fused) {
            acc = acc * alpha + (double)__half2float(hBias[n]);
            if (acc < 0) acc = 0;
        }
        double got = (double)__half2float(hD[(size_t)m * N + n]);
        double rel = fabs(got - acc) / std::max(1.0, fabs(acc));
        if (rel > worst) { worst = rel; *worst_m = m; *worst_n = n; }
    }
    return worst;
}

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto eq = [&](const char* s) { return strcmp(argv[i], s) == 0; };
        auto val = [&]() -> int { return (i + 1 < argc) ? atoi(argv[++i]) : 0; };
        auto dval = [&]() -> double { return (i + 1 < argc) ? atof(argv[++i]) : 0.0; };
        if      (eq("--m"))       a.M = val();
        else if (eq("--n"))       a.N = val();
        else if (eq("--k"))       a.K = val();
        else if (eq("--tile"))    a.tile = val();
        else if (eq("--stages"))  a.stages = val();
        else if (eq("--layout"))  a.layout = val();
        else if (eq("--iters"))   a.iters = val();
        else if (eq("--warmup"))  a.warmup = val();
        else if (eq("--group"))   a.group = val();
        else if (eq("--cpu"))     a.cpu = val();
        else if (eq("--soak"))    a.soak = dval();
        else if (eq("--window"))  a.window = dval();
        else if (eq("--check"))   a.check = true;
        else if (eq("--noref"))   a.ref = false;
        else if (eq("--sweep"))   a.sweep = true;
        else if (eq("--gsweep"))  a.gsweep = true;
        else if (eq("--unfused")) a.unfused = true;
        else if (eq("--help")) {
            printf("usage: %s [--m M --n N --k K] [--tile 0|1|2] [--stages 1..6]\n"
                   "       [--layout 0=none 1=pad 2=swizzle] [--group G] [--iters N]\n"
                   "       [--warmup N] [--check] [--cpu S] [--noref] [--sweep] [--gsweep] [--unfused]\n"
                   "  --check    compare against cuBLAS (fast, but a non-deterministic oracle\n"
                   "             for skinny M where cuBLAS switches to split-K)\n"
                   "  --cpu S    verify S output elements against a double-precision CPU dot\n"
                   "             product; deterministic ground truth, cost O(S*K)\n"
                   "  tile 0 = 64x128x32 (256thr)  1 = 128x128x32 (256thr)  2 = 128x256x32 (512thr)\n"
                   "  --group G  L2 CTA rasterization: G M-tiles walked column-major (1 = row-major)\n"
                   "  --sweep    stages 1..6 at fixed tile/layout/group\n"
                   "  --gsweep   group 1,2,4,8,16 at fixed tile/layout/stages\n",
                   argv[0]);
            return 0;
        }
        else {
            // silently ignoring an unrecognised flag once cost a whole soak run
            fprintf(stderr, "unknown option '%s' (try --help)\n", argv[i]);
            return 2;
        }
    }
    if (a.K % 32 != 0) { fprintf(stderr, "K must be a multiple of 32\n"); return 1; }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    size_t nA = (size_t)a.M * a.K, nB = (size_t)a.K * a.N, nD = (size_t)a.M * a.N;
    half *dA, *dB, *dD, *dRef, *dBias;
    CUDA_CHECK(cudaMalloc(&dA, nA * 2));
    CUDA_CHECK(cudaMalloc(&dB, nB * 2));
    CUDA_CHECK(cudaMalloc(&dD, nD * 2));
    CUDA_CHECK(cudaMalloc(&dRef, nD * 2));
    CUDA_CHECK(cudaMalloc(&dBias, (size_t)a.N * 2));

    std::vector<half> hA(nA), hB(nB), hBias(a.N);
    {   // deterministic host init so every variant sees identical data
        unsigned s = 12345;
        auto rnd = [&] { s = s * 1664525u + 1013904223u; return ((s >> 16) & 1023) / 1023.f - 0.5f; };
        for (size_t i = 0; i < nA; ++i) hA[i] = __float2half(rnd());
        for (size_t i = 0; i < nB; ++i) hB[i] = __float2half(rnd());
        for (int i = 0; i < a.N; ++i)  hBias[i] = __float2half(rnd());
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBias, hBias.data(), (size_t)a.N * 2, cudaMemcpyHostToDevice));
    }
    g_A = dA; g_B = dB; g_D = dD; g_biasAlways = dBias;
    g_bias = FUSE_EPILOGUE ? dBias : nullptr;

    cudaStream_t stream = 0;
    cublasHandle_t hcb = nullptr;
    double ref_ms = 0, ref_tflops = 0;
    if (a.ref || a.check) {
        CUBLAS_CHECK(cublasCreate(&hcb));
        CUBLAS_CHECK(cublasSetMathMode(hcb, CUBLAS_TENSOR_OP_MATH));
        ref_ms = cublas_gemm(hcb, a, dRef, a.iters, a.warmup, stream);
        ref_tflops = 2.0 * a.M * a.N * a.K / (ref_ms * 1e-3) / 1e12;
    }

    printf("# device        : %s  sm_%d%d  %d SMs\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("# problem       : M=%d N=%d K=%d  fp16 in / fp32 acc / fp16 out  (A row-major, B col-major)\n",
           a.M, a.N, a.K);
    printf("# accumulate    : %s\n", ACC_F16 ? "FP16 (1024 FLOP/clk/SM tensor rate)"
                                             : "FP32 (512 FLOP/clk/SM tensor rate)");
    printf("# epilogue      : %s\n", FUSE_EPILOGUE ? "FUSED alpha*acc + bias + ReLU" : "plain store");
    printf("# iters         : %d (warmup %d)\n", a.iters, a.warmup);
    if (a.ref) printf("# cuBLAS (SoL)  : %8.3f ms  %7.2f TFLOP/s\n", ref_ms, ref_tflops);
    printf("\n");
    printf("%-6s %-12s %-8s %6s %5s %5s %7s %6s %6s %10s %9s %8s %s\n",
           "tile", "CTA", "layout", "stages", "grp", "thr", "reg/thr", "smem", "CTA/SM",
           "ms", "TFLOP/s", "%%SoL", "maxrel");
    printf("%s\n", std::string(118, '-').c_str());

    struct Job { int tile, layout, stages, group; };
    std::vector<Job> jobs;
    if (a.sweep) {
        for (int s = 1; s <= 6; ++s) jobs.push_back({a.tile, a.layout, s, a.group});
    } else if (a.gsweep) {
        for (int g : {1, 2, 4, 8, 16}) jobs.push_back({a.tile, a.layout, a.stages, g});
    } else {
        jobs.push_back({a.tile, a.layout, a.stages, a.group});
    }

    float* dErr; CUDA_CHECK(cudaMalloc(&dErr, sizeof(float)));

    for (const Job& j : jobs) {
        RunFn fn = pick(j.tile, j.layout, j.stages);
        if (!fn) { fprintf(stderr, "bad config tile=%d layout=%d stages=%d\n",
                           j.tile, j.layout, j.stages); continue; }
        Args ja = a; ja.tile = j.tile; ja.layout = j.layout; ja.stages = j.stages;
        ja.group = j.group;
        CUDA_CHECK(cudaMemset(dD, 0, nD * 2));
        Result r = fn(ja, stream);

        double maxrel = -1, cpurel = -1;
        int wm = -1, wn = -1;
        // the cuBLAS oracle only applies to the raw-GEMM build
        if (r.ok && a.check && !a.unfused && !FUSE_EPILOGUE) {
            float zero = 0.f;
            CUDA_CHECK(cudaMemcpy(dErr, &zero, sizeof(float), cudaMemcpyHostToDevice));
            max_rel_diff<<<256, 256, 0, stream>>>(dD, dRef, nD, dErr);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            float e; CUDA_CHECK(cudaMemcpy(&e, dErr, sizeof(float), cudaMemcpyDeviceToHost));
            maxrel = e;
        }
        if (r.ok && a.cpu > 0) {
            std::vector<half> hD(nD);
            CUDA_CHECK(cudaMemcpy(hD.data(), dD, nD * 2, cudaMemcpyDeviceToHost));
            // both paths end up with bias+ReLU applied; only the plain build
            // without --unfused leaves the raw GEMM result
            const bool epilogue_applied = (FUSE_EPILOGUE != 0) || a.unfused;
            cpurel = cpu_spot_check(hA, hB, hD, hBias, a.M, a.N, a.K, a.cpu,
                                    epilogue_applied, 1.0f, &wm, &wn);
        }

        int bm, bn, bk; tile_dims(j.tile, bm, bn, bk);
        char cta[32]; snprintf(cta, sizeof cta, "%dx%dx%d", bm, bn, bk);
        if (!r.ok) {
            printf("%-6d %-12s %-8s %6d %5d %5d %7s %6s %6s %10s %9s %8s  SKIP: %s\n",
                   j.tile, cta, layout_name(j.layout), j.stages, j.group, r.threads,
                   "-", "-", "-", "-", "-", "-", r.why);
            continue;
        }
        char errbuf[24], solbuf[16];
        if (maxrel < 0) snprintf(errbuf, sizeof errbuf, "-");
        else            snprintf(errbuf, sizeof errbuf, "%.2e%s", maxrel,
                                 maxrel < 3e-2 ? " OK" : " FAIL");
        if (a.ref && ref_tflops > 0)
            snprintf(solbuf, sizeof solbuf, "%.1f%%", 100.0 * r.tflops / ref_tflops);
        else
            snprintf(solbuf, sizeof solbuf, "-");
        printf("%-6d %-12s %-8s %6d %5d %5d %7d %6d %6d %10.3f %9.2f %8s %s",
               j.tile, cta, layout_name(j.layout), j.stages, j.group, r.threads,
               r.regs, r.smem, r.blocksPerSM, r.ms, r.tflops, solbuf, errbuf);
        if (cpurel >= 0)
            printf("  cpu=%.2e%s(worst m=%d n=%d)", cpurel,
                   cpurel < 3e-2 ? " OK " : " FAIL ", wm, wn);
        printf("\n");
    }

    if (hcb) cublasDestroy(hcb);
    cudaFree(dA); cudaFree(dB); cudaFree(dD); cudaFree(dRef); cudaFree(dBias); cudaFree(dErr);
    return 0;
}
