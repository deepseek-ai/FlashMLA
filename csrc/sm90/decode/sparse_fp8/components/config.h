#pragma once

#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cute/tensor.hpp>
#include "defines.h"

using namespace cute;

namespace sm90::decode::sparse_fp8 {

static constexpr int HEAD_DIM_K = 576;
static constexpr int HEAD_DIM_V = 512;
static constexpr int HEAD_DIM_NOPE = HEAD_DIM_V;
static constexpr int HEAD_DIM_ROPE = HEAD_DIM_K - HEAD_DIM_V;
static constexpr int QUANT_TILE_SIZE = 128;
static constexpr int NUM_SCALES = HEAD_DIM_NOPE / QUANT_TILE_SIZE;
static constexpr int NUM_BYTES_PER_TOKEN = HEAD_DIM_NOPE + NUM_SCALES*sizeof(float) + HEAD_DIM_ROPE*sizeof(bf16);
static constexpr int PAGE_BLOCK_SIZE = 64;

}