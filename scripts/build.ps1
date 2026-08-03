# Build the sm_89 Tensor Core GEMM experiments.
#   .\build.ps1            # normal build
#   .\build.ps1 -Lineinfo  # add -lineinfo so Nsight Compute maps stalls to source
#   .\build.ps1 -Verbose   # print per-thread register / smem / spill counts
[CmdletBinding()]
param([switch]$Lineinfo = $true, [switch]$PtxAs)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$src  = Join-Path $repo 'src'
$bin  = Join-Path $repo 'bin'

# CUDA 12.3 rejects MSVC 19.44 by version check; the generated code is fine.
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

$flags = @(
    '-arch=sm_89'
    '-O3'
    '-std=c++17'
    '--use_fast_math'
    '-allow-unsupported-compiler'
    # MSVC 14.44's STL static_asserts on CUDA < 12.4 (STL1002); this is the
    # documented opt-out. Codegen is unaffected.
    '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'
    '-Xcompiler', '/wd4819'
)
if ($Lineinfo) { $flags += '-lineinfo' }
if ($PtxAs)    { $flags += @('-Xptxas', '-v') }

function Invoke-Nvcc {
    param([string[]]$NvccArgs, [string]$Label)
    $argline = ($NvccArgs | ForEach-Object { if ($_ -match '[\s]') { "`"$_`"" } else { $_ } }) -join ' '
    $cmd = "call `"$vcvars`" >nul 2>&1 && nvcc $argline"
    Write-Host "==> $Label" -ForegroundColor Cyan
    & cmd.exe /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "$Label failed (exit $LASTEXITCODE)" }
}

New-Item -ItemType Directory -Force $bin | Out-Null
Push-Location $bin      # nvcc drops .exp/.lib/.obj next to the output; keep them out of the tree
try {
    $dq = Join-Path $src 'device_query.cu'
    $gb = Join-Path $src 'gemm_bench.cu'

    Invoke-Nvcc ($flags + @($dq, '-o', 'device_query.exe')) 'device_query.exe'

    Invoke-Nvcc ($flags + @('-DFUSE_EPILOGUE=0', $gb,
                            '-o', 'gemm_bench.exe', '-lcublas')) 'gemm_bench.exe (plain epilogue)'

    Invoke-Nvcc ($flags + @('-DFUSE_EPILOGUE=1', $gb,
                            '-o', 'gemm_bench_fused.exe', '-lcublas')) 'gemm_bench_fused.exe (fused bias+ReLU)'

    # FP16-accumulate build: drives Nsight's generic tensor-pipe metric near 100%
    # on this GPU. Accuracy at large K is poor by construction -- see docs/RESULTS.md.
    Invoke-Nvcc ($flags + @('-DACC_F16=1', $gb,
                            '-o', 'gemm_bench_f16acc.exe', '-lcublas')) 'gemm_bench_f16acc.exe (FP16 accumulate)'

    # Two mma.m16n8k8 per k16 step instead of one m16n8k16 -- the shape cuBLAS
    # issues. Same FLOPs, 2x instructions at half the pipe occupancy each.
    Invoke-Nvcc ($flags + @('-DMMA_K8=1', $gb,
                            '-o', 'gemm_bench_k8.exe', '-lcublas')) 'gemm_bench_k8.exe (mma m16n8k8)'

    foreach ($ext in '*.exp', '*.lib', '*.obj') {
        Get-ChildItem $ext -ErrorAction SilentlyContinue | Remove-Item -Force
    }
    Write-Host "`nbuilt into $bin :" -ForegroundColor Green
    Get-ChildItem *.exe | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
} finally { Pop-Location }
