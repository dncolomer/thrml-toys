# thrml-toys

Toy examples using [THRML](https://github.com/extropic-ai/thrml) — Extropic's JAX-based simulator for block Gibbs sampling and Ising/Boltzmann machines.

These were developed with guidance from the [thrml-skill](https://github.com/extropic-ai/thrml-skill).

## denoise_e.py

Grid-only Ising EBM for denoising a noisy 16x16 letter "E".

- Purely visible model (no hidden/latent units)
- Only 4-connected neighbor interactions on the grid (bipartite blocks for parallel updates)
- Uses THRML contrastive training with `estimate_moments`, `IsingSamplingProgram`, single-chain inits, etc.
- Positive phase = exact clamped data moments from the pattern
- Negative phase = free model samples
- Denoising = free block-Gibbs relaxation starting from the noisy image

Produces comparison images in `denoise_results/`.

## denoise_e_hardcoded.py

**No-training** version with a completely homogeneous prior.

- Every pixel has the *exact same* bias (`GLOBAL_BIAS` = 0)
- Every neighbor edge has the *exact same* coupling (`J_NEIGHBOR`)
- No position-specific template from the clean "E" (parameters are identical across the image)
- The prior is just local smoothness (plus optional weak global bias)
- Denoising combines the prior with a data term pulled from the noisy observation (standard prior + likelihood)
- Includes a full **parameter sweep** over `J_NEIGHBOR` vs `DATA_STRENGTH`, producing one combined grid image `parameter_sweep.png` in `denoise_hardcoded_results/`

Great for exploring how smoothness strength vs observation trust affects recovery when the EBM has no knowledge of the specific letter shape.

## Running

Use the Python environment that has `thrml` installed (plus JAX, networkx, optax, matplotlib, numpy).

Example:

```bash
python denoise_e_hardcoded.py
```

The sweep will run 16 combinations and save `parameter_sweep.png`.

## Future

- Can be driven on accelerators / PrimeIntellect GPUs
- Easy to extend to multiple patterns, different graphs, or full training on datasets

See the docstrings and inline comments in the .py files for energy function details and THRML usage patterns (blocks of nodes, etc.).