# thrml-toys

Toy examples using [THRML](https://github.com/extropic-ai/thrml) — Extropic's JAX-based simulator for block Gibbs sampling and Ising/Boltzmann machines.

These were developed with guidance from the [thrml-skill](https://github.com/extropic-ai/thrml-skill).

## Scripts

### `denoise_e.py`

A grid-only Ising EBM denoiser for a noisy 16x16 letter "E".

- Purely visible model (no hidden/latent units).
- Only neighboring pixel interactions (4-connected grid, bipartite blocks for parallel updates).
- Uses THRML's high-level training primitives: `estimate_moments`, `IsingSamplingProgram`, single-chain `hinton_init`, etc.
- Positive phase uses exact data moments; negative phase uses free sampling.
- Denoising via free block-Gibbs starting from the noisy init.

Run:

```bash
python denoise_e.py
```

### `denoise_e_hardcoded.py`

A **no-training** example with a completely homogeneous (spatially uniform) prior.

- Every pixel has the *exact same* bias (`GLOBAL_BIAS`).
- Every neighbor edge has the *exact same* coupling (`J_NEIGHBOR`).
- No position-specific template derived from the clean "E" (parameters are identical across the image).
- The prior only encourages local smoothness (plus an optional weak global bias).
- Denoising combines the prior with a data term pulled from the noisy observation (standard prior + likelihood).
- Includes a full parameter sweep over `J_NEIGHBOR` vs `DATA_STRENGTH` (data term / trust in the noisy observation), producing one combined grid image (`parameter_sweep.png`).

This demonstrates using an explicit hardcoded energy function + THRML's sampler for denoising, without any learning.

Run:

```bash
python denoise_e_hardcoded.py
```

The sweep shows how different combinations of smoothness prior strength and observation trust affect recovery when the EBM has no knowledge of the specific letter shape.

## Running

Use the Python environment that has `thrml` installed (plus JAX, networkx, optax, matplotlib, numpy).

See the docstrings and comments inside each file for more details on the energy functions and THRML usage.

## Outputs

Generated images are in the `images/` directory (re-generate by running the scripts).