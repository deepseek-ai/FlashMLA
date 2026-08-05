# Accepted source diff

The standalone implementation was flattened from the two-file accepted source
closure recorded by the CSA manifest.

| Input role | SHA-256 |
|---|---|
| Accepted canonical-tie implementation | `6d993c1ec46e518342ff181819fe8145c6df6dc0c21c3d2a242bf093df3efc01` |
| Required histogram/scan support implementation | `0c9a3d4a7ca212de65d2fcb1d9ddc1a24ea911d9f13679bf9c93514a0c64363c` |
| Accepted raw launch declaration | `5cde0e75d55ae78dbed87dc6a01032332d87d6aea5b3b865289fdea020f12774` |

Mechanical changes:

- flattened the inherited CUDA implementation into one translation unit;
- renamed namespaces, helpers, kernel, and launch API by operation semantics;
- retained both 512-thread teams, the 1024-thread CTA, named barriers, shared
  layout, four radix refinements, and cross-row work loop;
- accepts any positive runtime grid that passed the binding's physical-SM guard;
- retained score-descending/original-index-ascending insertion ordering;
- retained the deterministic, non-atomic cutoff overwrite after the fourth
  radix refinement;
- replaced the raw void launcher with a `cudaError_t` API and local argument
  validation.

There are no algorithmic changes to selection or tie resolution.
