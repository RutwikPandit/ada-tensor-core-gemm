# Sustained-load soak: run the best kernel back-to-back for -Seconds while
# sampling power / clocks / temperature / throttle reasons, then correlate the
# two timelines.
#
#   .\soak.ps1 -Seconds 180
#   .\soak.ps1 -Seconds 180 -Exe .\gemm_bench_f16acc.exe -Tag f16acc
#
# Nothing else may use the GPU while this runs, including other profiling.
[CmdletBinding()]
param(
    [int]$Seconds = 180,
    [double]$Window = 2.0,
    [int]$M = 4096, [int]$N = 4096, [int]$K = 4096,
    [int]$Tile = 1, [int]$Stages = 2, [int]$Layout = 2, [int]$Group = 2,
    [string]$Exe = 'bin\gemm_bench.exe',
    [string]$Tag = 'soak',
    [int]$SampleMs = 250
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    New-Item -ItemType Directory -Force 'data' | Out-Null
    # Start-Process resolves paths against its own cwd, not $PWD, so use absolute
    $stdout = Join-Path (Get-Location) "data\$Tag.throughput.txt"
    $smiCsv = Join-Path (Get-Location) "data\$Tag.telemetry.csv"
    $ExePath = Join-Path (Get-Location) $Exe
    if (-not (Test-Path $ExePath)) { throw "$ExePath not found - run scripts\build.ps1 first" }

    $appArgs = @('--m', $M, '--n', $N, '--k', $K, '--tile', $Tile, '--stages', $Stages,
                 '--layout', $Layout, '--group', $Group, '--noref',
                 '--warmup', 20, '--soak', $Seconds, '--window', $Window)

    Write-Host "soaking $Seconds s: $Exe $($appArgs -join ' ')" -ForegroundColor Cyan
    $t0 = Get-Date
    $proc = Start-Process -FilePath $ExePath -ArgumentList $appArgs -NoNewWindow -PassThru `
                          -RedirectStandardOutput $stdout

    $q = 'clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,' +
         'clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,' +
         'clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,' +
         'clocks_throttle_reasons.hw_power_brake_slowdown'
    $samples = New-Object System.Collections.Generic.List[object]
    while (-not $proc.HasExited) {
        $line = (nvidia-smi --query-gpu=$q --format=csv,noheader,nounits)
        $f = $line -split ',' | ForEach-Object { $_.Trim() }
        $samples.Add([pscustomobject]@{
            t        = [math]::Round(((Get-Date) - $t0).TotalSeconds, 2)
            smMHz    = [int]$f[0]
            memMHz   = [int]$f[1]
            powerW   = [double]$f[2]
            tempC    = [int]$f[3]
            utilPct  = [int]$f[4]
            swPowerCap    = $f[5]
            hwSlowdown    = $f[6]
            hwThermal     = $f[7]
            swThermal     = $f[8]
            hwPowerBrake  = $f[9]
        })
        Start-Sleep -Milliseconds $SampleMs
    }
    $proc.WaitForExit()
    $samples | Export-Csv -Path $smiCsv -NoTypeInformation -Encoding utf8

    $busy = $samples | Where-Object { $_.utilPct -ge 50 }
    Write-Host "`n=== telemetry over $($busy.Count) busy samples ($SampleMs ms apart) ===" -ForegroundColor Green
    foreach ($p in 'smMHz','memMHz','powerW','tempC') {
        $s = $busy | Measure-Object -Property $p -Average -Minimum -Maximum
        "{0,-8} avg {1,8:N1}   min {2,8:N1}   max {3,8:N1}" -f $p, $s.Average, $s.Minimum, $s.Maximum
    }
    # first vs last third: is anything drifting?
    $n = $busy.Count
    if ($n -ge 9) {
        $a = $busy[0..([int]($n/3) - 1)]
        $b = $busy[([int](2*$n/3))..($n-1)]
        "`nfirst third vs last third (drift check):"
        foreach ($p in 'smMHz','powerW','tempC') {
            $av = ($a | Measure-Object $p -Average).Average
            $bv = ($b | Measure-Object $p -Average).Average
            $d  = '{0:+0.0;-0.0;0.0}' -f ($bv - $av)
            "{0,-8} {1,8:N1} -> {2,8:N1}   delta {3,7}" -f $p, $av, $bv, $d
        }
    }
    "`nthrottle reasons asserted at any point while busy:"
    foreach ($p in 'swPowerCap','hwSlowdown','hwThermal','swThermal','hwPowerBrake') {
        $hits = ($busy | Where-Object { $_.$p -eq 'Active' }).Count
        "  {0,-14} {1,5} / {2} samples" -f $p, $hits, $busy.Count
    }

    Write-Host "`n=== throughput windows ===" -ForegroundColor Green
    $rows = Get-Content $stdout | Where-Object { $_ -like 'SOAK,*' -and $_ -notlike '*elapsed_s*' } |
            ForEach-Object { $f = $_ -split ','
                [pscustomobject]@{ t=[double]$f[1]; ms=[double]$f[2]; iters=[int]$f[3]
                                   tflops=[double]$f[4]; msPerIter=[double]$f[5] } }
    $rows | Format-Table -AutoSize
    Get-Content $stdout | Where-Object { $_ -like '#*' }
    if ($rows.Count -ge 6) {
        $k = [int]($rows.Count/3)
        $fa = ($rows[0..($k-1)] | Measure-Object tflops -Average).Average
        $la = ($rows[($rows.Count-$k)..($rows.Count-1)] | Measure-Object tflops -Average).Average
        "`nthroughput first third {0:N2} -> last third {1:N2} TFLOP/s  ({2:N2}%)" -f `
            $fa, $la, (100*($la-$fa)/$fa)
    }
    Write-Host "`nwrote $stdout and $smiCsv" -ForegroundColor Green
} finally { Pop-Location }
