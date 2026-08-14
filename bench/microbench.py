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

    # Lazy: only needed on the accelerator box. Guarded because this script
    # is run by operators on whatever machine they have to hand, and an
    # unguarded import turned "no accelerator here" into a traceback. The
    # skip path is declared, documented and tested, so it has to be reachable
    # without torch installed as well as without a device visible.
    try:
        import torch
    except ImportError:
        print("torch is not installed, so there is nothing to measure. "
              "Install it on the machine with the accelerator and run this "
              "there; a ceiling measured anywhere else is not a ceiling.")
        return 2

    # Backend detection rather than assuming CUDA. ROCm presents itself
    # through torch.cuda, so AMD needs nothing special. TPU is XLA and
    # Trainium is Neuron, and both need a different device and a different
    # synchronise. Where a backend is present but unsupported here, the
    # honest outcome is to skip and say so rather than to report a number
    # produced some other way: a ceiling that was not measured must not be
    # labelled as though it was.
    dev = sync = backend = None
    if torch.cuda.is_available():
        backend, dev, sync = "cuda", torch.device("cuda"), torch.cuda.synchronize
    else:
        try:
            import torch_xla.core.xla_model as xm
            backend, dev = "xla", xm.xla_device()

            def sync():
                xm.mark_step()
                xm.wait_device_ops()
        except ImportError:
            pass
    if dev is None:
        try:
            import torch_neuronx  # noqa: F401
            backend = "neuron"
        except ImportError:
            pass

    if dev is None:
        print(f"no supported accelerator backend visible"
              f"{' (' + backend + ' found but not usable here)' if backend else ''}.")
        print("Skipping the microbenchmark. Pass the datasheet figures to the")
        print("validator instead, and label them CONFIG rather than MEASURED:")
        print("  python -m bench.validate traces.jsonl \\")
        print("      --bw-ceiling-gbps <datasheet> --flops-ceiling-tflops <datasheet>")
        print()
        print("A datasheet peak is not a microbenchmark. Effective bandwidth")
        print("computed against it is conventionally below 1.0; against a copy")
        print("benchmark it is not, and mixing the two reported a clean file as")
        print("contaminated once already. Record which one was used.")
        raise SystemExit(2)

    print(f"backend: {backend}")
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # --- GEMM ceiling ---
    a = torch.randn(args.n, args.n, device=dev, dtype=dt)
    b = torch.randn(args.n, args.n, device=dev, dtype=dt)
    for _ in range(5):
        a @ b                       # warm-up: JIT/heuristic selection
    sync()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        a @ b
    sync()
    dt_s = (time.perf_counter() - t0) / args.iters
    tflops = 2 * args.n ** 3 / dt_s / 1e12
    print(f"GEMM ceiling: {tflops:.0f} TFLOPS ({args.dtype}, n={args.n})")

    # --- bandwidth ceiling (device-to-device copy: read + write) ---
    x = torch.empty(args.n * args.n * 4, device=dev, dtype=torch.uint8)
    y = torch.empty_like(x)
    for _ in range(5):
        y.copy_(x)
    sync()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        y.copy_(x)
    sync()
    dt_s = (time.perf_counter() - t0) / args.iters
    gbps = 2 * x.numel() / dt_s / 1e9   # bytes read + written
    print(f"bandwidth ceiling: {gbps:.0f} GB/s (d2d copy)")
    print("\nfeed to validator:  python -m bench.validate traces.jsonl "
          f"--bw-ceiling-gbps {gbps:.0f} --flops-ceiling-tflops {tflops:.0f}")


if __name__ == "__main__":
    # The skip path returns 2 and it has to reach the shell, or a caller
    # cannot tell "nothing to measure" from "measured and found nothing".
    raise SystemExit(main() or 0)
