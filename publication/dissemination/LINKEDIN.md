# Manual publication package — LinkedIn

Status: hold until the Zenodo DOI, GitHub release, and Hugging Face Dataset are public.

I have published an independent study of Gemma 4 multi-token prediction on one
RTX 5070 Ti using QAT Q4_0 12B and 26B-A4B models with a pinned CUDA
`llama.cpp` runtime.

In the fixed tested grid, N=6 raised aggregate decode throughput from 89.20 to
226.39 tokens/s for 12B and from 169.58 to 322.53 tokens/s for 26B-A4B. Exact
greedy-output identity was only 56% and 59%, while the paired objective-quality
pilot cleared its prospectively frozen -5-point non-inferiority margin.

The conclusion is deliberately scoped: MTP was substantially faster and did
not show degradation beyond that pilot margin on this stack, but it was not
behaviorally identical and N=6 is not claimed as a universal optimum.

Paper and archive: https://doi.org/10.5281/zenodo.21860461  
Source and release asset: https://github.com/mobius-style/gemma4-mtp-quality-study/releases/tag/v1.0.0
Sanitized processed Dataset: https://huggingface.co/datasets/moebiusT7/gemma4-mtp-quality-study

#Gemma #llamacpp #LLMInference #SpeculativeDecoding #ReproducibleResearch

