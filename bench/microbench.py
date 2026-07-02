"""Ceiling probes: achievable GEMM TFLOPS and memory bandwidth on this device.

Run once per rented box; feed results to bench.validate --bw-ceiling-gbps /
--flops-ceiling-tflops. Physics demands fitted <= microbenched <= peak; a
fitted factor above its microbenched ceiling is a harness bug, caught before
it poisons the index.

Requires torch with device support (CUDA or ROCm build). UNTESTED in this
repo's CI (no accelerator); intentionally tiny so review substitutes for CI.

Usage: python -m bench.microbench [--dtype bf16] [--n 8192]
"""

import argparse
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=8192, help="GEMM dimension")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    import torch  # lazy: only needed on the GPU box
    if not torch.cuda.is_available():
        raise SystemExit("no accelerator visible (torch.cuda.is_available() is False)")
    dev = torch.device("cuda")
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # --- GEMM ceiling ---
    a = torch.randn(args.n, args.n, device=dev, dtype=dt)
    b = torch.randn(args.n, args.n, device=dev, dtype=dt)
    for _ in range(5):
        a @ b                       # warm-up: JIT/heuristic selection
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        a @ b
    torch.cuda.synchronize()
    dt_s = (time.perf_counter() - t0) / args.iters
    tflops = 2 * args.n ** 3 / dt_s / 1e12
    print(f"GEMM ceiling: {tflops:.0f} TFLOPS ({args.dtype}, n={args.n})")

    # --- bandwidth ceiling (device-to-device copy: read + write) ---
    x = torch.empty(args.n * args.n * 4, device=dev, dtype=torch.uint8)
    y = torch.empty_like(x)
    for _ in range(5):
        y.copy_(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        y.copy_(x)
    torch.cuda.synchronize()
    dt_s = (time.perf_counter() - t0) / args.iters
    gbps = 2 * x.numel() / dt_s / 1e9   # bytes read + written
    print(f"bandwidth ceiling: {gbps:.0f} GB/s (d2d copy)")
    print("\nfeed to validator:  python -m bench.validate traces.jsonl "
          f"--bw-ceiling-gbps {gbps:.0f} --flops-ceiling-tflops {tflops:.0f}")


if __name__ == "__main__":
    main()
