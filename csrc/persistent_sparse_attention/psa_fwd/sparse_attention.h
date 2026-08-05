#pragma once

#include "../params.h"

namespace persistent_sparse_attention::sparse_attention {

// Launches the accepted SM100 head-128, D=512 sparse-prefill topology.
// Q and output are caller-owned BF16 buffers. q_ready may be null; otherwise
// each flag release-publishes q_ready_chunk_size consecutive raw-Q rows.
void launch_sparse_attention(const SparseAttnFwdParams& params);

// Stable production-lane entry points. The prefix launcher selects native
// problem-sized FlashMLA on 152-SM GB200 and an admission-safe bounded grid on
// 148-SM B200. The tail launcher retains the accepted bounded-grid topology.
// Historical schedule revision names intentionally do not cross this public
// boundary.
void launch_sparse_attention_prefix(const SparseAttnFwdParams& params);
void launch_sparse_attention_tail(const SparseAttnFwdParams& params);

}  // namespace persistent_sparse_attention::sparse_attention
