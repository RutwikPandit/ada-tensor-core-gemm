# Phases 2-4: sweep stages x layout x CTA tile, logging clocks/power/temperature
# around every run so a thermal drift can't be mistaken for a kernel effect.
#   .\sweep.ps1 -M 4096 -N 4096 -K 4096
[CmdletBinding()]
param(
    [int]$M = 4096, [int]$N = 4096, [int]$K = 4096,
    [int[]]$Tiles = @(0, 1, 2),
    [int[]]$Layouts = @(0, 1, 2),
    [int]$Iters = 50, [int]$Warmup = 30,
    [string]$Out = 'data\sweep_results.csv',
    [string]$Exe = 'bin\gemm_bench.exe'
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    New-Item -ItemType Directory -Force (Split-Path -Parent $Out) | Out-Null
    function Get-GpuState {
        $q = 'clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu'
        $r = (nvidia-smi --query-gpu=$q --format=csv,noheader,nounits) -split ',' |
             ForEach-Object { $_.Trim() }
        [pscustomobject]@{ SmMHz = $r[0]; MemMHz = $r[1]; PowerW = $r[2]; TempC = $r[3]; UtilPct = $r[4] }
    }

    $rows = @()
    foreach ($t in $Tiles) {
        foreach ($l in $Layouts) {
            $before = Get-GpuState
            $raw = & $Exe --m $M --n $N --k $K --tile $t --layout $l --sweep `
                          --iters $Iters --warmup $Warmup --check 2>&1
            $after = Get-GpuState
            $sol = ($raw | Select-String 'cuBLAS \(SoL\)').ToString() -replace '.*?([\d.]+) TFLOP/s.*', '$1'

            foreach ($line in $raw) {
                $f = ($line -split '\s+') | Where-Object { $_ -ne '' }
                if ($f.Count -lt 10 -or $f[0] -notmatch '^\d+$') { continue }
                if ($line -match 'SKIP') {
                    Write-Host ("  skip tile=$t layout=$($f[2]) stages=$($f[3]): " +
                                ($line -replace '.*SKIP: ', '')) -ForegroundColor DarkYellow
                    continue
                }
                $rows += [pscustomobject]@{
                    M = $M; N = $N; K = $K
                    tile = $f[0]; cta = $f[1]; layout = $f[2]; stages = [int]$f[3]
                    threads = [int]$f[4]; regs = [int]$f[5]
                    smem = [int]$f[6]; ctaPerSm = [int]$f[7]
                    ms = [double]$f[8]; tflops = [double]$f[9]
                    pctSoL = $f[10]; maxrel = $f[11]
                    solTflops = [double]$sol
                    smMHz0 = $before.SmMHz; smMHz1 = $after.SmMHz
                    powerW1 = $after.PowerW; tempC0 = $before.TempC; tempC1 = $after.TempC
                }
            }
            Write-Host "done tile=$t layout=$l  (SM $($after.SmMHz) MHz, $($after.PowerW) W, $($after.TempC) C)" -ForegroundColor Cyan
        }
    }
    $rows | Export-Csv -Path $Out -NoTypeInformation -Encoding utf8
    Write-Host "`nwrote $Out ($($rows.Count) rows)`n" -ForegroundColor Green
    $rows | Sort-Object -Property tflops -Descending |
        Select-Object cta, layout, stages, regs, smem, ctaPerSm, ms, tflops, pctSoL, maxrel |
        Format-Table -AutoSize
} finally { Pop-Location }
