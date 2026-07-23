# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import ceil

import torch
import triton
import triton.language as tl

from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    round_to_quantized_type_args,
)
from compressed_tensors.quantization.utils import maybe_pad_tensor_for_block_quant


def _apply_quantize_op(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args: QuantizationArgs,
    dtype: torch.dtype | None,
    do_quantize: bool,
    do_dequantize: bool,
    global_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Dispatch to the appropriate quantization kernel."""
    if do_quantize and do_dequantize:
        return _quantize_dequantize(
            x=x,
            scale=scale,
            zero_point=zero_point,
            q_min=q_min,
            q_max=q_max,
            args=args,
            global_scale=global_scale,
        )
    elif do_quantize:
        return _quantize(
            x=x,
            scale=scale,
            zero_point=zero_point,
            q_min=q_min,
            q_max=q_max,
            args=args,
            dtype=dtype,
            global_scale=global_scale,
        )
    else:
        return _dequantize(
            x_q=x,
            scale=scale,
            zero_point=zero_point,
            global_scale=global_scale,
        )


def _process_block(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    dtype: torch.dtype | None,
    do_quantize: bool,
    do_dequantize: bool,
    global_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Blockwise quantization: pad, reshape into 2D blocks, quantize, restore."""
    original_shape = x.shape
    block_height, block_width = args.block_structure

    x = maybe_pad_tensor_for_block_quant(x, args.block_structure)
    padded_shape = x.shape

    # reshape into blocks and transpose to make each block contiguous
    num_rows_blocks = padded_shape[0] // block_height
    num_cols_blocks = padded_shape[1] // block_width
    x_blocks = x.reshape(
        num_rows_blocks,
        block_height,
        num_cols_blocks,
        block_width,
    ).transpose(1, 2)

    # expand scale/zero_point for block broadcasting
    sb = scale.unsqueeze(-1).unsqueeze(-1)
    zb = zero_point.unsqueeze(-1).unsqueeze(-1) if zero_point is not None else None

    x_blocks = _apply_quantize_op(
        x_blocks,
        sb,
        zb,
        q_min,
        q_max,
        args,
        dtype,
        do_quantize,
        do_dequantize,
        global_scale,
    )

    # restore padded shape
    output = x_blocks.transpose(1, 2).reshape(padded_shape)

    # truncate to original dimensions if padding was applied
    if original_shape != padded_shape:
        output = output[tuple([slice(v) for v in original_shape])]

    return output


def _process_group(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    args: QuantizationArgs,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    dtype: torch.dtype | None,
    do_quantize: bool,
    do_dequantize: bool,
    g_idx: torch.Tensor | None,
    global_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Group/tensor-group quantization: handle activation ordering, reshape
    into groups, quantize, restore."""
    group_size = args.group_size
    output_dtype = dtype if dtype is not None else x.dtype
    columns = x.shape[-1]

    while scale.ndim < 2:
        scale = scale.unsqueeze(1)
        zero_point = zero_point.unsqueeze(1) if zero_point is not None else None

    if columns >= group_size and columns % group_size != 0:
        raise ValueError(
            "tensor column shape must be divisble "
            f"by the given group_size {group_size} but got {columns}"
        )

    # support column-order (default) quantization as well as other orderings
    # such as activation ordering. Below checks if g_idx has been initialized
    is_column_order = g_idx is None or g_idx.device.type == "meta" or -1 in g_idx
    if not is_column_order:
        perm = torch.argsort(g_idx)
        x = x.index_select(-1, perm)

    # reshape last dim into (num_groups, group_size)
    reshaped_dims = (ceil(x.shape[-1] / group_size), group_size)
    x = x.unflatten(-1, reshaped_dims)

    output = _apply_quantize_op(
        x,
        scale.unsqueeze(-1),
        zero_point.unsqueeze(-1) if zero_point is not None else None,
        q_min,
        q_max,
        args,
        dtype,
        do_quantize,
        do_dequantize,
        global_scale,
    )

    output = output.flatten(start_dim=-2).to(output_dtype)

    if not is_column_order:
        inv_perm = torch.argsort(perm)
        output = output.index_select(-1, inv_perm)

    return output


@torch.no_grad()
def _quantize_dequantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args: QuantizationArgs,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused quantize-then-dequantize in a single pass, avoiding:
    - Double scale/global_scale division
    - Intermediate quantized dtype allocation
    """
    # compute effective scale once
    if global_scale is not None:
        scale = scale / global_scale

    scaled = x / scale

    if zero_point is not None:
        scaled += zero_point.to(x.dtype)

    # clamp and round (stays in float — no int8/fp8 intermediate)
    quantized = round_to_quantized_type_args(
        tensor=scaled, args=args, min=q_min, max=q_max
    )

    # dequantize: subtract zero_point and multiply by scale
    # cast to scale.dtype to match _dequantize behavior
    dequant = quantized.to(scale.dtype)
    if zero_point is not None:
        dequant = dequant - zero_point.to(scale.dtype)

    return dequant * scale


@triton.jit
def _dequantize_scalar_kernel(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    scale_ptr: tl.tensor,
    zero_point_ptr: tl.tensor,
    n_elements,
    HAS_ZERO_POINT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast Triton kernel for per-tensor dequantization with scalar scale.

    output = (x_q - zero_point) * scale

    Optimized for the common case where scale is a single scalar value.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x_q = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Load scalar scale (single value, broadcasted)
    scale = tl.load(scale_ptr)

    # Dequantize
    if HAS_ZERO_POINT:
        zero_point = tl.load(zero_point_ptr)
        output = (x_q - zero_point) * scale
    else:
        output = x_q * scale

    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _dequantize_kernel(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    scale_ptr: tl.tensor,
    zero_point_ptr: tl.tensor,
    global_scale_ptr: tl.tensor,
    num_rows,
    num_cols,
    group_size,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Triton kernel for dequantization: output = (x_q - zero_point) * scale

    Handles per-group scale/zero_point with configurable group_size.
    """
    # Set up the pids.
    pid_r = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    offsets_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offsets_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offsets = num_cols * offsets_r[:, None] + offsets_c[None, :]

    masks_r = offsets_r < num_rows
    masks_c = offsets_c < num_cols
    masks = masks_r[:, None] & masks_c[None, :]

    scale_offsets_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    scale_offsets_c = (pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)) // group_size
    scale_offsets = (num_cols // group_size) * scale_offsets_r[
        :, None
    ] + scale_offsets_c[None, :]
    scale_masks_r = scale_offsets_r < num_rows
    scale_masks_c = scale_offsets_c < num_cols // group_size
    scale_masks = scale_masks_r[:, None] & scale_masks_c[None, :]

    input = tl.load(input_ptr + offsets, masks, 0.0)
    scale = tl.load(scale_ptr + scale_offsets, scale_masks, 0.0)

    if global_scale_ptr is not None:
        global_scale = tl.load(global_scale_ptr)
        scale = scale / global_scale.to(scale.dtype)

    # Dequantize: (x_q - zero_point) * scale
    if zero_point_ptr is not None:
        zero_point = tl.load(zero_point_ptr + scale_offsets, scale_masks, 0.0)
        output = (input - zero_point) * scale
    else:
        output = input * scale

    tl.store(output_ptr + offsets, output, masks)


@torch.no_grad()
def _quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args: QuantizationArgs,
    dtype: torch.dtype | None = None,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    # if a global scale is optionally provided, use it
    # to further scale the local `scale` parameter
    if global_scale is not None:
        scale = scale / global_scale

    scaled = x / scale

    if zero_point is not None:
        scaled += zero_point.to(x.dtype)

    # clamp and round
    quantized_value = round_to_quantized_type_args(
        tensor=scaled, args=args, min=q_min, max=q_max
    )

    if dtype is not None:
        quantized_value = quantized_value.to(dtype)

    return quantized_value


@torch.no_grad()
def _dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:

    # Triton only works with CUDA and XPU tensors
    do_triton: bool = x_q.is_cuda or x_q.is_xpu

    if not do_triton:
        # CPU fallback
        if global_scale is not None:
            scale = scale / global_scale

        dequant_value = x_q.to(scale.dtype)

        if zero_point is not None:
            dequant_value = dequant_value - zero_point.to(scale.dtype)

        dequant_value = dequant_value * scale

        if dtype is not None:
            dequant_value = dequant_value.to(dtype)

        return dequant_value

    # Check if we can use the fast scalar path (per-tensor scale)
    is_scalar_scale = scale.numel() == 1
    is_scalar_zp = zero_point is None or zero_point.numel() == 1

    if is_scalar_scale and is_scalar_zp and global_scale is None:
        # Fast path: use optimized scalar kernel
        return _dequantize_scalar(x_q, scale, zero_point, dtype)

    # Slow path: use group-aware kernel for per-group scales
    return _dequantize_grouped(x_q, scale, zero_point, dtype, global_scale)


def _dequantize_scalar(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fast dequantization for per-tensor (scalar) scale."""
    original_shape = x_q.shape
    x_q_float = x_q.to(scale.dtype).flatten()

    n_elements = x_q_float.numel()
    # Use large block size to minimize kernel launch overhead
    # For simple element-wise ops, fewer blocks = less overhead
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    dequant_value = torch.empty_like(x_q_float)

    # Dummy pointer for zero_point if not provided
    zp_ptr = zero_point if zero_point is not None else scale

    _dequantize_scalar_kernel[grid](
        dequant_value,
        x_q_float,
        scale,
        zp_ptr,
        n_elements,
        HAS_ZERO_POINT=zero_point is not None,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    dequant_value = dequant_value.reshape(original_shape)

    if dtype is not None:
        dequant_value = dequant_value.to(dtype)

    return dequant_value


def _dequantize_grouped(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantization for per-group scales."""
    original_shape = x_q.shape

    # Convert to float for computation
    x_q = x_q.to(scale.dtype)

    if x_q.ndim == 4:
        n_rb, n_cb, bh, bw = x_q.shape
        group_size = bh * bw  # Each block is one "group"
        x_q = x_q.reshape(n_rb * n_cb, bh * bw)
        scale = scale.reshape(n_rb * n_cb, 1)
        if zero_point is not None:
            zero_point = zero_point.reshape(n_rb * n_cb, 1)
    elif x_q.ndim == 3:
        group_size = x_q.shape[2]
        x_q = x_q.reshape(x_q.shape[0], -1)
        scale = scale.reshape(scale.shape[0], -1)
        if zero_point is not None:
            zero_point = zero_point.reshape(zero_point.shape[0], -1)
    elif x_q.ndim == 2:
        group_size = x_q.shape[1]  # Entire row is one "group"
        num_rows = x_q.shape[0]
        if scale.ndim == 0:
            scale = scale.expand(num_rows, 1).contiguous()
        elif scale.ndim == 1:
            scale = scale.unsqueeze(1).expand(num_rows, 1).contiguous()
        elif scale.shape[0] == 1:
            scale = scale.expand(num_rows, -1).contiguous()
        if zero_point is not None:
            if zero_point.ndim == 0:
                zero_point = zero_point.expand(num_rows, 1).contiguous()
            elif zero_point.ndim == 1:
                zero_point = zero_point.unsqueeze(1).expand(num_rows, 1).contiguous()
            elif zero_point.shape[0] == 1:
                zero_point = zero_point.expand(num_rows, -1).contiguous()
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {x_q.ndim}D")

    block_size_r: int = 32
    block_size_c: int = 32
    num_rows = x_q.shape[0]
    num_cols = x_q.shape[1]

    def grid(META):
        return (
            triton.cdiv(num_rows, META["BLOCK_SIZE_R"]),
            triton.cdiv(num_cols, META["BLOCK_SIZE_C"]),
        )

    dequant_value = torch.empty_like(x_q)

    _dequantize_kernel[grid](
        dequant_value,
        x_q,
        scale,
        zero_point,
        global_scale,
        num_rows,
        num_cols,
        group_size,
        BLOCK_SIZE_R=block_size_r,
        BLOCK_SIZE_C=block_size_c,
    )

    dequant_value = dequant_value.reshape(original_shape)

    if dtype is not None:
        dequant_value = dequant_value.to(dtype)

    return dequant_value
