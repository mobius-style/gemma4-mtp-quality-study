# Environment Manifest

Captured before the new benchmark run on 2026-08-09 (Asia/Tokyo).

- Host: Ubuntu 24.04.1 LTS, Linux 6.17.0-40-generic
- CPU: Intel Core i7-11700, 8 cores / 16 threads
- RAM: 125 GiB
- GPU: NVIDIA GeForce RTX 5070 Ti; llama.cpp reports 15,841 MiB available
- NVIDIA kernel module: 595.71.05
- llama.cpp commit: `7ba604f1cb61cd14898138e9abc0b4ff2601f180`
- `llama-server` SHA-256: `1590bb2b1f9f704ed204fec890b2bb8cfaceb93bc2ca08dc3f70ef8053a0824a`
- Build: Release, GCC 13.3.0, CUDA backend, SM 120, Flash Attention enabled, native CPU enabled
- Runtime: CUDA runtime 13.0.48 and cuBLAS 13.0.0.19 from the pinned local runtime directory
- Python: 3.10.14

The startup allocator reported 6,637.69 MiB of main-model device buffer plus
226.90 MiB of draft-model device buffer for 12B, and 13,755.42 MiB plus
225.21 MiB for 26B-A4B. These are model-buffer placement values from
`llama.cpp`, not independent peak process-VRAM measurements.

The main and MTP-draft model paths, sizes, and hashes are recorded in
`configs/environment.json`. Although the main GGUF files live in Ollama's blob
store, Ollama does not perform inference in this study; the pinned independent
`llama-server` binary opens those files directly.

## Telemetry limitation

`nvidia-smi` failed before measurement because the running kernel module
(595.71.05) and user-space NVML library (595.84) did not match. CUDA inference
remained operational, but clock, power, utilization, and temperature traces
could not be recorded. This is a measurement limitation, not silently imputed
data. Server logs and per-request timings remain available.
