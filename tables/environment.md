| Component | Frozen value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti; 15,841 MiB reported by llama.cpp |
| CPU | Intel Core i7-11700, 8 cores / 16 threads |
| RAM | 125 GiB |
| OS | Ubuntu 24.04.1 LTS, kernel 6.17.0-40-generic |
| Runtime | llama.cpp `7ba604f1cb61cd14898138e9abc0b4ff2601f180` |
| Server SHA-256 | `1590bb2b1f9f704ed204fec890b2bb8cfaceb93bc2ca08dc3f70ef8053a0824a` |
| CUDA/build | CUDA 13 runtime, SM120, Release, GCC 13.3, Flash Attention |
| Target models | Gemma 4 12B IT QAT Q4_0 dense; Gemma 4 26B-A4B IT QAT Q4_0 MoE |
| Inference | direct llama-server; Ollama not used |
| Common protocol | ctx 8192, parallel 1, temp 0, top-k 1, F16 KV, batch/ubatch 512 |
| GPU placement | 12B 49/49 + draft 5/5; 26B 31/31 + draft 5/5 |
| Reported device model buffers | 12B main 6,637.69 MiB + draft 226.90 MiB; 26B main 13,755.42 MiB + draft 225.21 MiB; not peak process VRAM |
| Telemetry | NVML unavailable due driver/library mismatch; temperature/power/clocks absent |
