# Figures

| File | Content |
|---|---|
| `fig1_layer_pipeline` | One decode layer: the seven kernels, what each streams from DRAM, time above the byte floor, and which optimization of which phase touches which kernel |
| `fig2_decode_kernel` | The v2 decode kernel: split-KV scheduler, one CTA from the inside, combine variants, the 8-bit KV variant |
| `fig3_gemm_and_l2_plan` | Phase 2: the skinny GEMM. Phase 3 (planned): DRAM and L2 on the time axis of one layer, before and after |
| `fig4_results` | The decode layer from the baseline to the current build, the decode kernel against other implementations, distance to the hardware ceilings |

Each figure exists as `.svg` (the source of truth) and as `.png` (rendered at 2× for viewers that lack the fonts the SVG
asks for). The Markdown documents embed the PNGs.

## Regenerating

The figures are drawn by plain Python scripts in `src/`. `sketch.py` is a small dependency-free generator of hand-drawn
style SVG (jittered double strokes, hachure fills, a seeded random generator so that the output is reproducible).

```bash
cd docs/optimization/figures/src
python fig1_layer_pipeline.py          # writes fig1_layer_pipeline.svg next to the script
python render.py fig1_layer_pipeline.svg --scale 2     # SVG -> PNG with headless Chrome or Edge, then palette compression
```

- `sketch.py` uses Pillow for text metrics when the fonts are present (it prints a warning when a label is wider than
  the space it was given); without Pillow it falls back to an estimate.
- `render.py` needs Pillow and a Chrome or Edge binary. The figures were rendered on Windows with Comic Sans MS; on a
  machine without that font the SVG falls back to another cursive face and some labels may no longer fit.
- Move the regenerated `.svg` and `.png` one directory up to replace the published ones.

Numbers in the figures are the ones in [../01-results.md](../01-results.md) and [../04-phase3-plan.md](../04-phase3-plan.md).
In Figure 3, panel B, the "before" bars are to scale; the "after" bars are illustrative, because only layer totals were
measured for the planned items.
