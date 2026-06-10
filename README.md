# thrml-toys

Toy examples using [THRML](https://github.com/extropic-ai/thrml) (Extropic's JAX simulator for block Gibbs / Ising machines).

Developed alongside the thrml-skill for correct usage patterns.

## denoise_e.py

Simpler grid-only version (no hidden units):
- Purely visible Ising EBM on 16x16 grid
- Neighbor-only interactions (bipartite blocks)
- Contrastive training using THRML's `estimate_moments` + `IsingSamplingProgram`
- Denoising via free block-Gibbs from noisy init

Generates comparison images in `results/`.

## denoise_e_hardcoded.py

No-training version with **completely homogeneous prior** (same params everywhere, no clean template reference):
- `GLOBAL_BIAS` identical for all pixels
- `J_NEIGHBOR` identical for all edges
- Explicit `compute_ising_energy` function shown
- Data term from noisy observation added only at inference
- Full parameter sweep over `J_NEIGHBOR` vs `DATA_STRENGTH` producing one combined `parameter_sweep.png`

Great for exploring the effect of the two main parameters of a uniform local smoothness prior.

## Results

Generated PNGs (comparison plots and the parameter sweep grid) are in the `results/` directory.

## Running

```bash
python denoise_e.py
python denoise_e_hardcoded.py   # runs the sweep automatically
```

Requires a Python with `thrml`, JAX, networkx, optax, matplotlib, numpy.