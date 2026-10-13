__version__ = "1.0.0"

# Attention kernels: sm86 "v2" (default), "legacy" / "legacy_ws" (the fork's sm80 kernels), bf16 / fp16 / fp8 / int8 KV.
from .flash_mla_interface import (
    get_mla_metadata,
    flash_mla_with_kvcache,
    get_mla_workspace_bytes,
    alloc_mla_workspace,
    sm86_config,
)

# Quantized KV-cache format (docs/kv_quant_format.md): pure-torch reference of the quantize-on-write / dequant math.
from .kv_quant import (
    quantize_kv,
    dequantize_kv,
    alloc_quant_cache,
    bytes_per_token,
)

# Fused decode prologue (RoPE + q concat + RMSNorm + paged-cache write, optional fp8/int8 quantize-on-write) and the
# DeepSeek-V2-Lite decode layer built on top of it. The CUDA part lives in the `flash_mla_fused_cuda` extension; the
# modules import without it (the torch references stay usable).
from .mla_prologue import (
    mla_decode_prologue,
    mla_decode_prologue_ref,
    make_rope_cache,
)
from .mla_decode_layer import (
    MLAConfig,
    MLADecodeLayer,
    DecodeState,
    alloc_cache,
    make_state,
    capture_layer_graph,
)
