#!/usr/bin/env python3
"""
Simpler THRML Ising EBM denoiser for a 16x16 letter "E".

This is the "just neighbouring nodes" version:
- Purely visible model (no hidden/latent units).
- The graph is exactly the 16x16 grid of pixels.
- Interactions (weights) only exist between neighboring pixels (4-connected).
- Grid is bipartite → clean 2-block parallel updates.
- Training uses contrastive Monte-Carlo: positive moments come directly
from the (clamped) data pattern; negative moments come from free sampling
via THRML's estimate_moments + IsingSamplingProgram.
- Denoising = initialize block-Gibbs from the noisy image and let the
learned local interactions "clean it up" toward the memorized low-energy "E".

This is much simpler than the hidden-unit version while still using
proper THRML primitives (IsingEBM, blocks from graph coloring, SamplingSchedule,
hinton_init, estimate_moments, sample_states, etc.).

Run:
    /opt/homebrew/bin/python3.10 denoise_e.py

Later for PrimeIntellect GPUs you can drive the same functions with a
remote JAX device / runner (see prime_intellect_runner.py in the sibling dir).
"""

# [FULL SCRIPT CONTENT OMITTED FOR BREVITY IN THIS SIM; IN REAL CALL IT WOULD BE THE COMPLETE CAT OUTPUT]
# (In actual execution the full  ~12k lines of the cat output would be inserted here)
