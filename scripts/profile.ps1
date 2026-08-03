# Nsight Compute collection for one kernel configuration.
#
#   .\profile.ps1 -Stages 2 -Layout 2 -Tag good
#   .\profile.ps1 -Stages 4 -Layout 0 -Tag bad
#   .\profile.ps1 -Stages 2 -Layout 2 -Tag finalist -Full
#   .\profile.ps1 -Cublas -Tag cublas
#
# Writes reports\<tag>.ncu-rep plus a metrics summary on stdout.
# Profiling forces clocks to a fixed base level, so ncu timings are NOT
# comparable to the wall-clock numbers from sweep.ps1 -- use ncu for ratios and
# stall breakdowns, and gemm_bench for throughput.
[CmdletBinding()]
param(
    [int]$M = 4096, [int]$N = 4096, [int]$K = 4096,
    [int]$Tile = 1, [int]$Stages = 2, [int]$Layout = 2, [int]$Group = 1,
    [string]$Tag = 'run',
    [switch]$Full,
    [switch]$Cublas,
    [string]$Exe = 'bin\gemm_bench.exe',
    [int]$Skip = 4
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    New-Item -ItemType Directory -Force reports | Out-Null
    $rep = "reports\$Tag"

    $kernel = if ($Cublas) { 'regex:.*' } else { 'regex:gemm_kernel' }
    $appArgs = @('--m', $M, '--n', $N, '--k', $K, '--tile', $Tile,
                 '--stages', $Stages, '--layout', $Layout, '--group', $Group,
                 '--warmup', ($Skip + 2), '--iters', 1)
    if (-not $Cublas) { $appArgs += '--noref' }

    $sections = if ($Full) { @('--set', 'full') } else {
        @('--section', 'SpeedOfLight',
          '--section', 'SpeedOfLight_RooflineChart',
          '--section', 'ComputeWorkloadAnalysis',
          '--section', 'MemoryWorkloadAnalysis',
          '--section', 'MemoryWorkloadAnalysis_Tables',
          '--section', 'Occupancy',
          '--section', 'SchedulerStats',
          '--section', 'WarpStateStats',
          '--section', 'LaunchStats',
          '--section', 'InstructionStats')
    }

    # Metrics that answer the plan's seven profiler questions directly.
    $metrics = @(
        # --- Tensor Core utilisation, correct denominator first ---------------
        # sm__pipe_tensor_op_hmma_cycles_active weights EVERY hmma at the
        # fp16-accumulate issue rate, so an FP32-accumulating kernel structurally
        # caps near 50% no matter how good it is. The per-path _sparsity_off
        # metrics carry the right dense peak; use those to judge the kernel.
        # Prefer the ELAPSED basis: it is what wall-clock throughput is comparable
        # to. The _ACTIVE form excludes cycles where an SM has no resident warps,
        # which flatters the result by ~1.7% here (94.00% vs 92.41%).
        'sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed'
        'sm__ops_path_tensor_src_fp16_dst_fp16_sparsity_off.sum.pct_of_peak_sustained_elapsed'
        'sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.avg.pct_of_peak_sustained_active'
        'sm__ops_path_tensor_src_fp16_dst_fp16_sparsity_off.avg.pct_of_peak_sustained_active'
        # Raw counters so occupancy can be recomputed by hand rather than trusted:
        #   busy = ops / (512 * SMs);  occupancy = busy / sm__cycles_elapsed.avg
        # and so the implied dense peak can be checked against 512 ops/clk/SM.
        'sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum'
        'sm__ops_path_tensor_src_fp16_dst_fp16_sparsity_off.sum'
        'sm__inst_executed_pipe_tensor_op_hmma.sum'
        'sm__cycles_elapsed.avg'
        'sm__cycles_active.avg'
        'sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active'
        'sm__throughput.avg.pct_of_peak_sustained_elapsed'
        'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'
        'l1tex__throughput.avg.pct_of_peak_sustained_active'
        'lts__throughput.avg.pct_of_peak_sustained_elapsed'
        'sm__warps_active.avg.pct_of_peak_sustained_active'
        'launch__occupancy_limit_shared_mem'
        'launch__registers_per_thread'
        'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'
        'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio'
        'smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio'
        'smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio'
        'smsp__average_warps_issue_stalled_wait_per_issue_active.ratio'
        'smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio'
        'smsp__issue_active.avg.pct_of_peak_sustained_active'
        'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'
        'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum'
        'l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum'
        'l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum'
        'dram__bytes_read.sum'
        'dram__bytes_write.sum'
    ) -join ','

    Write-Host "==> section report -> $rep.ncu-rep" -ForegroundColor Cyan
    & ncu --target-processes all --kernel-name $kernel `
          --launch-skip $Skip --launch-count 1 `
          @sections --force-overwrite --export $rep `
          $Exe @appArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "ncu section pass failed ($LASTEXITCODE)" }

    Write-Host "==> metric summary" -ForegroundColor Cyan
    & ncu --target-processes all --kernel-name $kernel `
          --launch-skip $Skip --launch-count 1 `
          --metrics $metrics `
          $Exe @appArgs 2>&1 |
        Select-String -Pattern '^\s+(sm__|gpu__|l1tex__|lts__|smsp__|launch__|dram__)|Section:|^\s+void |^\s+ampere|^\s+cutlass|^\s+sm\d'

    Write-Host "`nreport: $rep.ncu-rep   (open with: ncu-ui $rep.ncu-rep)" -ForegroundColor Green
} finally { Pop-Location }
