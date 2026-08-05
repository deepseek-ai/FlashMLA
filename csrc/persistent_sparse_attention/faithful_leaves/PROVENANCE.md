# Faithful leaf provenance

Stable-name physical-source ports from the qualified 148-SM B200 CSA route.
The original SHA-256 values below identify the exact oracle inputs used for
the port; the copied sources are self-contained and never include those trees.

| Leaf | Stable source | Oracle source | Oracle SHA-256 | Port kind |
|---|---|---|---|---|
| K4 IQ1-Q, grid 104 | `index_query_quant.cu` | `deep_gemm/include/deep_gemm/impls/sm100_iq1_q_persistent.cuh` | `596c72c5fdafc9cef372852e698dd148f62ddce61ff4c66bf807f6092cb69fa0` | Device body preserved; CUTLASS annotations/named-barrier helper replaced with equivalent CUDA attributes/PTX; stable C launcher added. |
| K5 weight scale, grid 44 | `index_weight_scale.cu` | `source-oracle/K5-index-weight-scale` | `5e04d2730029835ea6704f2dd2f3a9760292c8987536fa10cdc86b849eba65e9` | Device body preserved; Torch stable-ABI binding removed; stable C launcher added. |
| K6 Index-K pack, grid 132 | `index_cache_pack.cu` | `source-oracle/K6-index-cache-pack` | `38c67fefd36f90b661d4d3241490aaa632d6a318e94de13d0d476615ead110a9` | Device body preserved; physical symbol renamed without an iteration label; stable C launcher added. |
| K8 DSv4 gather, grid 16 | `kv_cache_gather.cu` | `source-oracle/K8-kv-cache-gather` | `51d3255d04cbfef74aab89810878397216f9d88f3b83f7013e3a086005f1d471` | Device body preserved; Torch stable-ABI binding removed; stable C launcher added. |

The 152-SM GB200 route retains the existing portable adapters. The 148-SM
route alone selects these exact fixed-grid persistent leaves.
