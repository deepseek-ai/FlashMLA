# SPDX-License-Identifier: Apache-2.0
"""Minimal CuTeDSL helpers required by the CSA Python kernels.

These helpers are derived from the Apache-2.0 vLLM CuTe utilities.  Keeping the
small PTX/MLIR surface here removes a runtime dependency on vLLM and Quack.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Callable

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def recast_val(x, dtype, *, loc=None, ip=None):
    return dtype(llvm.bitcast(dtype.mlir_type, x.ir_value(loc=loc, ip=ip)))


def _bf16x2_unary(asm: str, a: Uint32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip)],
        f"{asm}.bf16x2 $0, $1;",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


def _bf16x2_binary(
    asm: str,
    a: Uint32,
    b: Uint32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        f"{asm}.bf16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


@dsl_user_op
def bf16x2_abs(a: Uint32, *, loc=None, ip=None) -> Uint32:
    return _bf16x2_unary("abs", a, loc=loc, ip=ip)


@dsl_user_op
def bf16x2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return _bf16x2_binary("max", a, b, loc=loc, ip=ip)


@dsl_user_op
def bf16x2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return _bf16x2_binary("mul.rn", a, b, loc=loc, ip=ip)


@dsl_user_op
def fp32x2_to_bf16x2(
    a: Float32,
    b: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


@dsl_user_op
def bf16x2_to_fp32x2(data, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [data.ir_value(loc=loc, ip=ip)],
        "shl.b32 $0, $2, 16;\n\tand.b32 $1, $2, 0xFFFF0000;",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def fp8x4_to_bf16x4(x: Uint32, *, loc=None, ip=None) -> cute.TensorSSA:
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [x.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 x0, x1;\n\t"
        ".reg .b16 t00, t01, t10, t11;\n\t"
        "mov.b32 {x0, x1}, $2;\n\t"
        "cvt.rn.f16x2.e4m3x2 $0, x0;\n\t"
        "cvt.rn.f16x2.e4m3x2 $1, x1;\n\t"
        "mov.b32 {t00, t01}, $0;\n\t"
        "mov.b32 {t10, t11}, $1;\n\t"
        "cvt.rn.bf16.f16 t00, t00;\n\t"
        "cvt.rn.bf16.f16 t01, t01;\n\t"
        "cvt.rn.bf16.f16 t10, t10;\n\t"
        "cvt.rn.bf16.f16 t11, t11;\n\t"
        "mov.b32 $0, {t00, t01};\n\t"
        "mov.b32 $1, {t10, t11};\n\t"
        "}\n",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(2)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 2, Uint32)


@dsl_user_op
def fp32x8_to_fp4x8(
    vals: cute.Tensor,
    offset: cutlass.Constexpr[int],
    *,
    loc=None,
    ip=None,
) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [vals[offset + i].ir_value(loc=loc, ip=ip) for i in range(8)],
        "{\n\t"
        ".reg .b8 x0, x1, x2, x3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x0, $2, $1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x1, $4, $3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x2, $6, $5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x3, $8, $7;\n\t"
        "mov.b32 $0, {x0, x1, x2, x3};\n\t"
        "}\n",
        "=r,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


cvt = SimpleNamespace(
    bf16x2_to_fp32x2=bf16x2_to_fp32x2,
    fp32x2_to_bf16x2=fp32x2_to_bf16x2,
    fp32x8_to_fp4x8=fp32x8_to_fp4x8,
    fp8x4_to_bf16x4=fp8x4_to_bf16x4,
)


@cute.jit
def warp_reduce(
    value: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if const_expr(isinstance(value, cute.TensorSSA)):
        result = cute.make_fragment(value.shape, value.dtype)
        result.store(value)
        for i in cutlass.range_constexpr(cute.size(value.shape)):
            result[i] = warp_reduce(result[i], op, width)
        return result.load()
    for i in cutlass.range_constexpr(int(math.log2(width))):
        value = op(value, cute.arch.shuffle_sync_bfly(value, offset=1 << i))
    return value


# `nvvm.fmax` used to take the result type as its first positional argument;
# newer MLIR python bindings infer it from the operands and reject the extra
# one. Both forms build the same op and emit the same PTX, so accept whichever
# this `nvidia-cutlass-dsl` exposes rather than pinning the package from here.
# Resolved once, on first use.
_NVVM_FMAX_TAKES_RESULT_TYPE: bool | None = None


@dsl_user_op
def fmax(a, b, *, loc=None, ip=None) -> Float32:
    global _NVVM_FMAX_TAKES_RESULT_TYPE
    lhs = Float32(a).ir_value(loc=loc, ip=ip)
    rhs = Float32(b).ir_value(loc=loc, ip=ip)
    if _NVVM_FMAX_TAKES_RESULT_TYPE is None:
        try:
            value = nvvm.fmax(T.f32(), lhs, rhs, loc=loc, ip=ip)
        except TypeError:
            _NVVM_FMAX_TAKES_RESULT_TYPE = False
        else:
            _NVVM_FMAX_TAKES_RESULT_TYPE = True
            return Float32(value)
    if _NVVM_FMAX_TAKES_RESULT_TYPE:
        return Float32(nvvm.fmax(T.f32(), lhs, rhs, loc=loc, ip=ip))
    return Float32(nvvm.fmax(lhs, rhs, loc=loc, ip=ip))


@dsl_user_op
def shr_u32(
    value: Uint32,
    shift: Uint32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(value).ir_value(loc=loc, ip=ip),
                Uint32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shr.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


__all__ = [
    "bf16x2_abs",
    "bf16x2_max",
    "bf16x2_mul",
    "cvt",
    "fmax",
    "recast_val",
    "shr_u32",
    "warp_reduce",
]
