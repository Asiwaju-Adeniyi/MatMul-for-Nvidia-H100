# GPU MatMul From Scratch — Kernel Implementations Inspired by Aleksa Gordić

This repo contains my hands-on implementations of matmul kernels inspired by Aleksa Gordić’s post *“Inside NVIDIA GPUs: Anatomy of High-Performance MatMul Kernels.”*  
The purpose is simple: **practice writing GPU matmul kernels** — from basic to near-SOTA — without the surrounding architecture theory (these are available in the blog).

## What’s Inside
This repo focuses strictly on **CUDA kernel implementations**, including:

- **Naive matmul kernels**  
  (thread-per-output, row/column major variations)

- **Warp-tiled synchronous matmul kernels**  
  Register tiling, shared-memory blocking, and warp-level compute patterns.

- **Tensor Core–accelerated matmul kernels**  
  WMMA, fragment operations, and mixed-precision variants.

- **Asynchronous / pipelined kernels**  
  Double-buffering, load–compute overlap, and Hopper-style TMA-based patterns.

Each kernel is separated, documented, and benchmarked to show how performance evolves with each optimization.

## Goal
To build a clear, practical collection of matmul kernels — from beginner-friendly to advanced — that mirror the progression in Aleksa’s blog, but **focused entirely on code**, not architecture education.

## Future Additions
- More tensor-core variants  
- Hopper/Blackwell experimental kernels  
- Benchmark comparisons

## Acknowledgements
Inspired by Aleksa Gordić’s work.  
All kernels are my own re-implementations for learning and experimentation.
