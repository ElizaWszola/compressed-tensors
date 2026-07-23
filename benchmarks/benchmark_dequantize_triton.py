# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for _dequantize Triton implementation in forward_helpers.py.

Compares Triton kernel vs PyTorch ops, both on CUDA (apples to apples).

Based on benchmark_quantize_triton.py structure.
"""

import gc
import time
import torch

from compressed_tensors.quantization.lifecycle.forward_helpers import _dequantize
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationType,
    QuantizationStrategy,
)
from compressed_tensors.quantization.utils.helpers import calculate_range

SIZE = 4096 * 4096  # ~16.7M elements
device = "cuda:0" if torch.cuda.is_available() else "cpu"
N_RUNS = 200


def create_test_data(rows, cols, quant_type, num_bits, target_device):
    """Create quantized test data and dequantization parameters."""
    args = QuantizationArgs(
        num_bits=num_bits,
        type=quant_type,
        symmetric=True,
        strategy=QuantizationStrategy.TENSOR,
    )
    q_min, q_max = calculate_range(args, torch.device(target_device))

    # Create quantized values within the valid range
    x_q = torch.randint(
        int(q_min.item()),
        int(q_max.item()) + 1,
        (rows, cols),
        dtype=torch.float32,
        device=target_device,
    )
    scale = (torch.rand(1) * 0.01 + 0.001).to(target_device)
    zero_point = None  # symmetric quantization

    return x_q, scale, zero_point, args


def pytorch_dequantize_cuda(x_q, scale, zero_point):
    """PyTorch reference implementation on CUDA (no Triton)."""
    dequant_value = x_q.to(scale.dtype)
    if zero_point is not None:
        dequant_value = dequant_value - zero_point.to(scale.dtype)
    dequant_value = dequant_value * scale
    return dequant_value


def benchmark_cuda(func, x_q, scale, zero_point, name, warmup=False):
    """Benchmark a dequantization function on CUDA."""
    x_q = x_q.clone()
    if warmup:
        print(f"  Warming up {name}...")
        for _ in range(10):
            _ = func(x_q, scale, zero_point)
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        print(f"  Warmup complete, starting benchmark...")

    times = []
    peaks = []

    for _ in range(N_RUNS):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        baseline_mem = torch.cuda.memory_allocated(0)

        torch.cuda.synchronize()
        start = time.time()
        result = func(x_q, scale, zero_point)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        peak = (torch.cuda.max_memory_allocated(0) - baseline_mem) / 1e9

        times.append(elapsed)
        peaks.append(peak)

        del result
        torch.cuda.empty_cache()
        gc.collect()

    avg_time = sum(times) / N_RUNS
    avg_peak = sum(peaks) / N_RUNS

    return avg_time, avg_peak


def run_config(quant_type, num_bits, rows, cols):
    """Run benchmarks for a specific configuration."""
    type_str = "int" if quant_type == QuantizationType.INT else "fp"
    config_name = f"{type_str}{num_bits}"

    print(f"\n{'='*80}")
    print(f"Benchmarking {config_name} dequantization ({rows}x{cols} = {rows*cols/1e6:.1f}M elements)")
    print("=" * 80)

    # Create CUDA test data - both paths run on CUDA for fair comparison
    x_q_cuda, scale_cuda, zp_cuda, args = create_test_data(
        rows, cols, quant_type, num_bits, device
    )

    # PyTorch reference on CUDA (no Triton kernel, just PyTorch ops)
    print("\nRunning PyTorch reference (CUDA, no Triton)...")
    time_pytorch, peak_pytorch = benchmark_cuda(
        pytorch_dequantize_cuda, x_q_cuda, scale_cuda, zp_cuda, "pytorch_cuda", warmup=True
    )
    print(f"PyTorch (CUDA):")
    print(f"  Time: {time_pytorch*1000:.2f}ms")
    print(f"  Peak: {peak_pytorch:.3f} GB")

    # Triton kernel (CUDA path in _dequantize)
    print("\nRunning Triton kernel (CUDA)...")
    time_triton, peak_triton = benchmark_cuda(
        _dequantize, x_q_cuda, scale_cuda, zp_cuda, "triton", warmup=True
    )
    print(f"Triton (CUDA):")
    print(f"  Time: {time_triton*1000:.2f}ms")
    print(f"  Peak: {peak_triton:.3f} GB")

    # Verify correctness - compare Triton kernel vs PyTorch ops on same CUDA data
    x_q_test, scale_test, zp_test, _ = create_test_data(512, 1024, quant_type, num_bits, device)

    # PyTorch reference on CUDA
    pytorch_out = pytorch_dequantize_cuda(
        x_q=x_q_test.clone(),
        scale=scale_test.clone(),
        zero_point=zp_test,
    )

    # Triton kernel on CUDA
    triton_out = _dequantize(
        x_q=x_q_test.clone(),
        scale=scale_test.clone(),
        zero_point=zp_test,
    )

    # Dequantization is a simple multiply, should be very precise
    atol = 1e-5
    rtol = 1e-5

    diff = (pytorch_out - triton_out).abs()
    max_diff = diff.max().item()
    correct = torch.allclose(pytorch_out, triton_out, atol=atol, rtol=rtol)

    if not correct:
        max_idx = diff.argmax()
        row_idx = max_idx // 1024
        col_idx = max_idx % 1024
        print(f"\nWarning: outputs differ, max_diff={max_diff:.6f} (atol={atol})")
        print(f"  At index [{row_idx}, {col_idx}]:")
        print(f"    x_q={x_q_test[row_idx, col_idx].item():.6f}")
        print(f"    scale={scale_test.item():.15f}")
        print(f"    pytorch={pytorch_out[row_idx, col_idx].item():.15f}")
        print(f"    triton={triton_out[row_idx, col_idx].item():.15f}")

    del x_q_cuda, scale_cuda
    del x_q_test, scale_test, pytorch_out, triton_out
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "config": config_name,
        "rows": rows,
        "cols": cols,
        "pytorch_ms": time_pytorch * 1000,
        "triton_ms": time_triton * 1000,
        "triton_peak": peak_triton,
        "speedup": time_pytorch / time_triton if time_triton > 0 else 0,
        "correct": correct,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, Triton requires GPU")
        return

    print(f"Benchmarking _dequantize from forward_helpers.py")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"N_RUNS: {N_RUNS}")

    # Configurations to benchmark
    configs = [
        (QuantizationType.INT, 4),
        (QuantizationType.INT, 8),
        (QuantizationType.FLOAT, 4),
        (QuantizationType.FLOAT, 8),
    ]

    # Tensor sizes
    sizes = [
        (4096, 4096),
        (4096, 11008),  # LLaMA MLP
        (8192, 8192),
    ]

    results = []

    for quant_type, num_bits in configs:
        for rows, cols in sizes:
            result = run_config(quant_type, num_bits, rows, cols)
            results.append(result)

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY (both on CUDA - apples to apples)")
    print("=" * 100)
    print(f"{'Config':<8} {'Size':<15} {'PyTorch/CUDA (ms)':<18} "
          f"{'Triton/CUDA (ms)':<18} {'Speedup':<10} {'Correct':<8}")
    print("-" * 100)

    for r in results:
        size_str = f"{r['rows']}x{r['cols']}"
        correct_str = "Yes" if r["correct"] else "NO"
        print(f"{r['config']:<8} {size_str:<15} {r['pytorch_ms']:>14.2f} ms  "
              f"{r['triton_ms']:>14.2f} ms  "
              f"{r['speedup']:>6.2f}x    {correct_str:<8}")


if __name__ == "__main__":
    main()
