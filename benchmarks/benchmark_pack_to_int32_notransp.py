# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script comparing pack_to_int32 implementations with packed_dim=0.

Compares:
- pack_to_int32: Original PyTorch implementation (with transpose)
- pack_to_int32_accelerated: Triton-accelerated (no-transpose for dim=0)

Tests only packed_dim=0 scenarios with real-world weight matrix shapes.

Real-world scenarios for packed_dim=0 (packing along rows/output features):
- LLaMA-7B: weights like (4096, 4096), (11008, 4096), (4096, 11008)
- LLaMA-13B: weights like (5120, 5120), (13824, 5120)
- LLaMA-70B: weights like (8192, 8192), (28672, 8192)
- Mistral-7B: similar to LLaMA-7B
- Falcon-7B: (4544, 4544), (4544, 18176)
"""

import gc
import torch

from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    pack_to_int32_accelerated,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
N_RUNS = 100


# Real-world weight matrix shapes (rows, cols) for various LLMs
# When packed_dim=0, we pack along rows (output features)
REAL_WORLD_SHAPES = [
    # LLaMA-7B / Mistral-7B style
    (4096, 4096, "LLaMA-7B q/k/v/o_proj"),
    (11008, 4096, "LLaMA-7B gate/up_proj"),
    (4096, 11008, "LLaMA-7B down_proj"),
    # LLaMA-13B style
    (5120, 5120, "LLaMA-13B q/k/v/o_proj"),
    (13824, 5120, "LLaMA-13B gate/up_proj"),
    # LLaMA-70B style
    (8192, 8192, "LLaMA-70B q/k/v/o_proj"),
    (28672, 8192, "LLaMA-70B gate/up_proj"),
    # Falcon-7B style (non-power-of-2)
    (4544, 4544, "Falcon-7B dense"),
    (4544, 18176, "Falcon-7B dense_h_to_4h"),
]


def create_test_data(rows, cols, num_bits, target_device):
    """Create random int8 quantized weights."""
    # Simulate quantized weights in valid range for num_bits
    max_val = (1 << (num_bits - 1)) - 1
    min_val = -(1 << (num_bits - 1))
    
    # Create random int8 values in the valid quantized range
    x = torch.randint(
        min_val, max_val + 1, (rows, cols), dtype=torch.int8, device=target_device
    )
    return x


def benchmark_cuda(func, x, num_bits, packed_dim, name, warmup=False):
    """Benchmark a packing function on CUDA using CUDA events for accurate timing."""
    x = x.clone()

    # Warmup phase
    if warmup:
        print(f"  Warming up {name}...")
        for _ in range(20):
            _ = func(x, num_bits, packed_dim)
        torch.cuda.synchronize()
        print("  Warmup complete, starting benchmark...")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

    times = []

    for _ in range(N_RUNS):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = func(x, num_bits, packed_dim)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        times.append(elapsed_ms)

    # Remove outliers (top/bottom 10%)
    times.sort()
    n_trim = max(1, len(times) // 10)
    trimmed_times = times[n_trim:-n_trim] if len(times) > 2 * n_trim else times

    avg_ms = sum(trimmed_times) / len(trimmed_times)
    min_ms = min(trimmed_times)
    max_ms = max(trimmed_times)

    return avg_ms, min_ms, max_ms, result


def verify_correctness(x, num_bits, packed_dim):
    """Verify that all implementations produce the same result."""
    result_original = pack_to_int32(x.clone(), num_bits, packed_dim)
    result_accel = pack_to_int32_accelerated(x.clone(), num_bits, packed_dim)

    match_accel = torch.equal(result_original, result_accel)

    return match_accel


def run_benchmark_for_shape(rows, cols, num_bits, shape_name):
    """Run benchmark for a specific shape and num_bits configuration."""
    x = create_test_data(rows, cols, num_bits, device)

    packed_dim = 0  # Always pack along dim 0 for this benchmark

    # Verify correctness first
    match_accel = verify_correctness(x, num_bits, packed_dim)

    # Benchmark each implementation
    avg_orig, min_orig, max_orig, _ = benchmark_cuda(
        pack_to_int32, x, num_bits, packed_dim, "Original (PyTorch)", warmup=True
    )

    avg_accel, min_accel, max_accel, _ = benchmark_cuda(
        pack_to_int32_accelerated, x, num_bits, packed_dim, "Accelerated (Triton)", warmup=True
    )

    # Calculate speedup
    speedup_accel = avg_orig / avg_accel if avg_accel > 0 else float('inf')

    return {
        "shape": (rows, cols),
        "shape_name": shape_name,
        "num_bits": num_bits,
        "original_ms": avg_orig,
        "accelerated_ms": avg_accel,
        "speedup_accel": speedup_accel,
        "correct_accel": match_accel,
    }


def print_results_table(results):
    """Print results in a formatted table."""
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY (packed_dim=0)")
    print("=" * 90)
    print(
        f"{'Shape':<20} {'Name':<25} {'Bits':>4} "
        f"{'Orig (ms)':>10} {'Accel (ms)':>10} "
        f"{'Speedup':>10} {'Correct':>8}"
    )
    print("-" * 90)

    for r in results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
        correct_str = "✓" if r["correct_accel"] else "✗"
        print(
            f"{shape_str:<20} {r['shape_name']:<25} {r['num_bits']:>4} "
            f"{r['original_ms']:>10.3f} {r['accelerated_ms']:>10.3f} "
            f"{r['speedup_accel']:>10.2f}x {correct_str:>8}"
        )

    print("=" * 90)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, benchmark requires GPU")
        return

    from compressed_tensors.utils.triton import HAS_TRITON

    if not HAS_TRITON:
        print("Triton is not available, skipping benchmark")
        return

    print("Benchmark: pack_to_int32 with packed_dim=0 (Triton no-transpose)")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"N_RUNS: {N_RUNS}")
    print("\nComparing:")
    print("  - Original: PyTorch scatter_add with transpose")
    print("  - Accelerated: Triton kernel without transpose (direct memory layout)")

    results = []

    # Test common bit depths
    bit_depths = [4, 8]

    for num_bits in bit_depths:
        print(f"\n{'='*80}")
        print(f"Testing {num_bits}-bit quantization (packed_dim=0)")
        print("=" * 80)

        for rows, cols, shape_name in REAL_WORLD_SHAPES:
            print(f"\n  Shape: ({rows}, {cols}) - {shape_name}")
            result = run_benchmark_for_shape(rows, cols, num_bits, shape_name)
            results.append(result)

            # Print quick summary
            print(
                f"    Original: {result['original_ms']:.3f}ms, "
                f"Accelerated: {result['accelerated_ms']:.3f}ms ({result['speedup_accel']:.2f}x)"
            )

    # Print final summary table
    print_results_table(results)

    # Analysis
    print("\nANALYSIS:")
    print("-" * 80)

    # Calculate average speedup
    avg_speedup_accel = sum(r["speedup_accel"] for r in results) / len(results)

    print(f"Average speedup (Accelerated vs Original): {avg_speedup_accel:.2f}x")

    # Check correctness
    all_correct = all(r["correct_accel"] for r in results)
    if all_correct:
        print("\n✓ All implementations produce identical results")
    else:
        print("\n✗ WARNING: Some implementations produce different results!")
        for r in results:
            if not r["correct_accel"]:
                print(f"  - {r['shape']} {r['num_bits']}-bit: accel={'✗'}")


if __name__ == "__main__":
    main()
