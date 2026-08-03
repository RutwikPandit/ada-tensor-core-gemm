# Nsight Compute reports

`.ncu-rep` files are gitignored — roughly 3 MB each, and every one is
reproducible from source. Regenerate with `scripts/profile.ps1`:

```powershell
# the finalist, with the raw counters needed to hand-verify tensor occupancy
.\scripts\profile.ps1 -Stages 2 -Layout 2 -Group 2 -Tag finalist_verified

# contrast cases referenced in docs/RESULTS.md
.\scripts\profile.ps1 -Stages 1 -Layout 0 -Tag sync_s1_none      # no async pipelining
.\scripts\profile.ps1 -Stages 4 -Layout 0 -Tag bad_s4_none       # SMEM kills residency
.\scripts\profile.ps1 -Stages 2 -Layout 1 -Tag s2_pad            # padded vs swizzled
.\scripts\profile.ps1 -Cublas -Tag cublas_ref -Skip 6            # the SoL kernel

# L2 rasterization, before and after, at 8192
.\scripts\profile.ps1 -M 8192 -N 8192 -K 8192 -Group 1 -Tag s2_swz_8192
.\scripts\profile.ps1 -M 8192 -N 8192 -K 8192 -Group 2 -Tag s2_swz_8192_grp2

# full metric set for a finalist (slow: metric replay)
.\scripts\profile.ps1 -Stages 2 -Layout 2 -Group 2 -Tag finalist_full -Full
```

Open with `ncu-ui reports\<tag>.ncu-rep`.

Profiling locks clocks to a fixed level, so **times from `ncu` are not comparable
to `gemm_bench` wall-clock numbers**. Use `ncu` for ratios, cycle counts and stall
breakdowns; use `gemm_bench` / `soak.ps1` for throughput.
