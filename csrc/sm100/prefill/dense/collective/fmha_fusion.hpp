/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once


#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

struct NoMask {
  // Sliding-window-attention (SWA): a trailing `int window_size = -1` is added to
  // every mask method so the templated mainloop/kernel can pass it uniformly to
  // ANY Mask type. window_size <= 0 means "disabled" (full attention / plain
  // causal) and every windowed term is a no-op. Only the causal masks honor it.
  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size,
      int window_size = -1) {

    return ceil_div(get<1>(problem_size), get<1>(tile_shape));
  }

  // SWA: index of the FIRST K-tile that any query in this Q-block can attend to
  // under the window. 0 unless a causal mask narrows it (overridden in CausalMask).
  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_trip_start(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size,
      int window_size = -1) {

    return 0;
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_masked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size,
      int window_size = -1) {

    return 0;
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_unmasked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size,
      int window_size = -1) {

    return get_trip_count(blk_coord, tile_shape, problem_size);
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void apply_mask(
      AccQK& acc_qk,
      IndexQK const& index_qk,
      ProblemSize const& problem_size,
      int window_size = -1) {

    return;
  }
};

struct ResidualMask : NoMask {

  using Base = NoMask;

  // SWA note: residual (non-causal) masks ignore window_size — sliding window is
  // only meaningful for causal attention. The param exists for signature parity.
  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_masked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size,
      int window_size = -1) {

    if (get<1>(problem_size) % get<1>(tile_shape) != 0) {
      return 1;
    }
    return 0;
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_unmasked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size,
      int window_size = -1) {

    // if the sequence length does not divide the tile size evenly
    if (get<1>(problem_size) % get<1>(tile_shape) != 0) {
      return get_trip_count(blk_coord, tile_shape, problem_size) - 1;
    }
    return get_trip_count(blk_coord, tile_shape, problem_size);
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void apply_mask(
      AccQK& acc_qk,
      IndexQK const& index_qk,
      ProblemSize const& problem_size,
      int window_size = -1) {

    // This is useful is seqlen_k % kBlockN != 0 since it masks
    // the remaining elements out from softmax.
    // d % kHeadDim != 0 or seqlen_q % kBlockM do not suffer from similar
    // issues as they are transparently taken care of by TMA and the
    // epilogue, if it is instantiated with predication support.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto pos = index_qk(i);
      if (get<1>(pos) >= get<1>(problem_size)) {
        acc_qk(i) = -INFINITY;
      }
    }
  }
};

struct ResidualMaskForBackward : NoMask {

  using Base = NoMask;

  // SWA note: residual (non-causal) masks ignore window_size (signature parity).
  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_masked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size,
      int window_size = -1) {

    if (get<1>(problem_size) % get<1>(tile_shape) != 0) {
      return 1;
    }
    return 0;
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_unmasked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size,
      int window_size = -1) {

    // if the sequence length does not divide the tile size evenly
    if (get<1>(problem_size) % get<1>(tile_shape) != 0) {
      return get_trip_count(blk_coord, tile_shape, problem_size) - 1;
    }
    return get_trip_count(blk_coord, tile_shape, problem_size);
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void apply_mask(
      AccQK& acc_qk,
      IndexQK const& index_qk,
      ProblemSize const& problem_size,
      int window_size = -1) {

    // This is useful is seqlen_k % kBlockN != 0 since it masks
    // the remaining elements out from softmax.
    // d % kHeadDim != 0 or seqlen_q % kBlockM do not suffer from similar
    // issues as they are transparently taken care of by TMA and the
    // epilogue, if it is instantiated with predication support.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto pos = index_qk(i);
      if (! elem_less(pos, select<0,1>(problem_size))) {
        acc_qk(i) = -INFINITY;
      }
    }
  }
};

// There are two ways to do causal if N_Q != N_K
// (1) The Q is at the beginning of the matrix
// (2) The Q is at the end of the matrix
template<bool kIsQBegin = true>
struct CausalMask : NoMask {

  using Base = NoMask;

  static constexpr bool IsQBegin = kIsQBegin;

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size,
      int window_size = -1) {

    // See note below on different ways to think about causal attention
    // Again, we'd add the offset_q into the max_blocks_q calculation
    //
    // SWA safe-baseline: window_size does NOT shrink the iterated K-tile range
    // here (we keep the full causal upper-bound count and rely on apply_mask to
    // zero out-of-window keys). Skipping leading out-of-window K-tiles is a
    // separate perf optimization that must shift the start of all four warp
    // loops together; not done in the baseline.
    int max_blocks_k = Base::get_trip_count(blk_coord, tile_shape, problem_size);
    if constexpr (IsQBegin) {
      int max_blocks_q = ceil_div((get<0>(blk_coord) + 1) * get<0>(tile_shape), get<1>(tile_shape));
      return std::min(max_blocks_k, max_blocks_q);
    } else {
      const int offset_q = get<1>(problem_size) - get<0>(problem_size);
      int max_blocks_q = ceil_div((get<0>(blk_coord) + 1) * get<0>(tile_shape) + offset_q, get<1>(tile_shape));
      return std::min(max_blocks_k, max_blocks_q);
    }
  }

  // SWA: first K-tile any query in this Q-block attends to under the window.
  // The smallest query in the block is q_lo (incl. offset_q); it attends to keys
  // k > q_lo - window_size, so K-tiles entirely below floor((q_lo-W+1)/N) hold no
  // in-window key for ANY row and can be skipped. Returns 0 when window disabled.
  // The forward mainloop subtracts this from the trip counts AND starts the K/V
  // load + softmax cS cursor here, so out-of-window leading tiles are never
  // loaded or computed. Backward/non-MLA paths don't call this (they keep the
  // mask-only safe baseline).
  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_trip_start(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size,
      int window_size = -1) {

    if (window_size <= 0) return 0;
    int offset_q = 0;
    if constexpr (!IsQBegin) {
      offset_q = get<1>(problem_size) - get<0>(problem_size);
    }
    int q_lo = get<0>(blk_coord) * get<0>(tile_shape) + offset_q;
    int k_lo = q_lo - window_size + 1;
    if (k_lo <= 0) return 0;
    return k_lo / get<1>(tile_shape);
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_masked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size,
      int window_size = -1) {

    // SWA: with a sliding window the masked set is TWO-sided (the upper causal
    // diagonal AND an interior lower window edge). Pure-causal counting only
    // covers the diagonal tail, so under it the leading window-edge tiles would
    // skip apply_mask and leave out-of-window keys unmasked. Safe baseline:
    // when the window is active, mark EVERY iterated tile as masked so apply_mask
    // runs on all of them (correct; optimize the count later).
    if (window_size > 0) {
      return get_trip_count(blk_coord, tile_shape, problem_size, window_size);
    }
    int trip_count = get_trip_count(blk_coord, tile_shape, problem_size, window_size);
    if constexpr (IsQBegin) {
      return std::min(trip_count, int(ceil_div(size<0>(tile_shape), size<1>(tile_shape))));
    } else {
      const int corner_count = int((get<1>(problem_size) % get<1>(tile_shape) || get<0>(problem_size) % get<0>(tile_shape))) ;
      return std::min(trip_count, int(ceil_div(get<0>(tile_shape), get<1>(tile_shape))) + corner_count);
    }
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_unmasked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size,
      int window_size = -1) {

    return get_trip_count(blk_coord, tile_shape, problem_size, window_size) - get_masked_trip_count(blk_coord, tile_shape, problem_size, window_size);
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void apply_mask(
      AccQK& acc_qk,
      IndexQK const& index_qk,
      ProblemSize const& problem_size,
      int window_size = -1) {

    // There are two ways to do causal if N_Q != N_K
    // (1) is to assume that the Q is at the beginning of the matrix
    //    - this is the default setting.
    // (2) is that it is at the end of the matrix
    //    - this is usually what we want for inference settings
    //      where we only compute the next row and use cache for the rest
    //    - if you'd like this, you only need to set kIsQBegin=false
    //
    // SWA: the extra `window_size > 0 && q - k >= window_size` term is the
    // causal sliding-window LOWER bound -- keep key k for query q iff
    // q - window_size < k <= q (i.e. only the last window_size keys). When
    // window_size <= 0 the term is a no-op, bit-identical to plain causal.

    if constexpr (IsQBegin) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(acc_qk); i++) {
        auto pos = index_qk(i);
        if ((get<0>(pos) < get<1>(pos)) || (get<1>(pos) >= get<1>(problem_size)) || (window_size > 0 && get<0>(pos) - get<1>(pos) >= window_size)) {
          acc_qk(i) = -INFINITY;
        }
      }
    } else {
      const auto offset_q = get<1>(problem_size) - get<0>(problem_size);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(acc_qk); i++) {
        auto pos = index_qk(i);
        if ((get<0>(pos) + offset_q < get<1>(pos)) || (get<1>(pos) >= get<1>(problem_size)) || (window_size > 0 && (get<0>(pos) + offset_q) - get<1>(pos) >= window_size)) {
          acc_qk(i) = -INFINITY;
        }
      }
    }
  }
};

template<bool kIsQBegin = true>
struct CausalForBackwardMask : CausalMask<kIsQBegin>, ResidualMaskForBackward {

  using Base = CausalMask<kIsQBegin>;

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void apply_mask(
      AccQK& acc_qk,
      IndexQK const& index_qk,
      ProblemSize const& problem_size,
      int window_size = -1) {

    // There are two ways to do causal if N_Q != N_K
    // (1) is to assume that the Q is at the beginning of the matrix
    //    - this is what we demonstrate here
    // (2) is that it is at the end of the matrix
    //    - this is usually what we want for inference settings
    //      where we only compute the next row and use cache for the rest
    //    - if you'd like this, you only need to add an offset like so:
    //      get<0>(pos) + offset_q < get<1>(pos)
    int offset_q = 0;
    if constexpr (!kIsQBegin) {
      offset_q = get<1>(problem_size) - get<0>(problem_size);
    }

    // SWA: the windowed term below MUST be character-identical to the forward
    // CausalMask<kIsQBegin>::apply_mask (!IsQBegin branch) so the backward masks
    // S exactly as the forward did -> dP/dS/dQ/dK/dV are gated consistently.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto pos = index_qk(i);
      bool masked = (get<0>(pos) + offset_q < get<1>(pos)) || !elem_less(pos, problem_size) || (window_size > 0 && (get<0>(pos) + offset_q) - get<1>(pos) >= window_size);
      if (masked) {
        acc_qk(i) = -INFINITY;
      }
    }
  }

};

struct VariableLength {
  int max_length;
  int* cumulative_length = nullptr;
  int total_length = -1;

  CUTE_HOST_DEVICE operator int() const {
    return max_length;
  }
};

template<class T> struct is_variable_length_impl : std::false_type {};
template<> struct is_variable_length_impl<VariableLength> : std::true_type {};
template<class T> constexpr bool is_variable_length_v = is_variable_length_impl<remove_cvref_t<T>>::value;

template<class Shape, class Idx>
CUTE_HOST_DEVICE
constexpr auto
apply_variable_length(Shape const& shape, Idx const& idx) {
  return transform_leaf(shape, [&](auto const& s) {
    if constexpr (is_variable_length_v<decltype(s)>) {
      return s.cumulative_length[idx+1] - s.cumulative_length[idx];
    }
    else {
      return s;
    }
  });
}

template<class Shape, class Coord, class Idx>
CUTE_HOST_DEVICE
constexpr auto
apply_variable_length(Shape const& shape, Coord const& coord, Idx const& idx) {
  auto new_shape = apply_variable_length(shape, idx);
  auto new_coord = transform_leaf(shape, coord, [&](auto const& s, auto const& c) {
    if constexpr (is_variable_length_v<decltype(s)>) {
      return cute::make_tuple(c, s.cumulative_length[idx]);
    }
    else {
      return c;
    }
  });
  return cute::make_tuple(new_shape, new_coord);
}

template<class Shape, class Coord>
CUTE_HOST_DEVICE
constexpr auto
apply_variable_length_offset(Shape const& shape, Coord const& coord) {
  auto idx = back(back(coord));
  auto result_shape = transform_leaf(shape, [&](auto const& s) {
    if constexpr (is_variable_length_v<decltype(s)>) {
      return s.cumulative_length[idx+1] - s.cumulative_length[idx];
    }
    else {
      return s;
    }
  });
  auto result_offset = transform_leaf(coord, shape, [&](auto const& c, auto const& s) {
    if constexpr (is_variable_length_v<decltype(s)>) {
      return s.cumulative_length[idx];
    }
    else {
      return _0{};
    }
  });
  return cute::make_tuple(result_shape, result_offset);
}

}  // namespace cutlass::fmha::collective

namespace cute {

template<>
struct is_integral<cutlass::fmha::collective::VariableLength> : true_type {};

CUTE_HOST_DEVICE
void print(cutlass::fmha::collective::VariableLength a) {
  printf("Varlen<%d, %p>", a.max_length, a.cumulative_length);
}

}
