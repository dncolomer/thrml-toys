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

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optax

from thrml import Block, SpinNode, SamplingSchedule, sample_states
from thrml.models import (
    IsingEBM,
    IsingSamplingProgram,
    estimate_moments,
    hinton_init,
)

# ----------------------------- Config -----------------------------
H, W = 16, 16
N_PIXELS = H * W
NOISE_P = 0.22

TRAIN_STEPS = 200          # A bit more steps because fully-visible single pattern
BATCH_SIZE = 1             # The pattern itself (we can still repeat for averaging)
LR = 5e-3
BETA = 1.2                 # Slightly higher temp scaling helps local grid models

# Negative phase schedule (the only phase we sample for moments)
NEG_WARMUP = 10
NEG_SAMPLES = 20          # more samples per chain for stable negative moments (single-chain path)
NEG_STEPS_PER = 2

# Denoising relaxation (free sampling from noisy init)
DENOISE_WARMUP = 80
DENOISE_SAMPLES = 6
DENOISE_STEPS_PER = 2

OUT_DIR = Path("denoise_results")
OUT_DIR.mkdir(exist_ok=True)

jax.config.update("jax_platform_name", "cpu")
print(f"JAX backend: {jax.default_backend()}")

# ------------------------- Letter "E" -------------------------
def make_clean_e() -> np.ndarray:
    img = np.zeros((H, W), dtype=bool)
    img[1:15, 1:3] = True          # left vertical
    img[1:3, 1:14] = True          # top bar
    img[7:9, 1:12] = True          # middle bar
    img[13:15, 1:14] = True        # bottom bar
    img[0:1, 1:4] = True
    img[15:16, 1:4] = True
    return img

def add_noise(img: np.ndarray, p: float, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = img.copy()
    mask = rng.random(img.shape) < p
    noisy[mask] = ~noisy[mask]
    return noisy

def to_ascii(img: np.ndarray) -> str:
    return "\n".join("".join("\u2588" if v else "\u00b7" for v in row) for row in img)

# ----------------------- Graph (grid only, neighbors) -----------------------
def build_grid_model():
    """Create the pure visible grid Ising model using only neighboring pixels."""
    print("Building grid graph (only neighbouring pixels, bipartite)...")

    # Pixel nodes in row-major order (this order defines our data arrays)
    pixel_nodes = [SpinNode() for _ in range(N_PIXELS)]
    coord_to_node = {}
    idx = 0
    for r in range(H):
        for c in range(W):
            coord_to_node[(r, c)] = pixel_nodes[idx]
            idx += 1

    G = nx.grid_graph(dim=(H, W), periodic=False)
    G = nx.relabel_nodes(G, coord_to_node, copy=False)

    nodes = pixel_nodes
    edges = list(G.edges())

    # Bipartite coloring → two perfect parallel blocks (recommended for grids)
    coloring = nx.bipartite.color(G)
    block0 = Block([n for n, col in coloring.items() if col == 0])
    block1 = Block([n for n, col in coloring.items() if col == 1])
    neg_blocks = [block0, block1]

    print(f"  Nodes: {len(nodes)} (all visible/pixels)")
    print(f"  Edges (neighbour pairs): {len(edges)}")
    print(f"  Blocks for sampling: 2 (bipartite grid)")

    return {
        "nodes": nodes,
        "pixel_nodes": pixel_nodes,
        "edges": edges,
        "neg_blocks": neg_blocks,
    }

# --------------------------- Training (fully visible) -----------------------
def train_grid_ebm(grid, clean_flat: np.ndarray):
    """Contrastive training using only the grid neighbours.

    Positive moments = exact statistics of the clamped pattern(s).
    Negative moments = free samples from the current model (via estimate_moments).
    """
    nodes = grid["nodes"]
    pixel_nodes = grid["pixel_nodes"]
    edges = grid["edges"]
    neg_blocks = grid["neg_blocks"]

    key = jax.random.key(0)
    key, kb, kw = jax.random.split(key, 3)

    biases = jax.random.normal(kb, (len(nodes),)) * 0.01
    weights = jax.random.normal(kw, (len(edges),)) * 0.005
    beta = jnp.array(BETA)

    model = IsingEBM(nodes, edges, biases, weights, beta)

    neg_program = IsingSamplingProgram(model, neg_blocks, clamped_blocks=[])
    sched_neg = SamplingSchedule(NEG_WARMUP, NEG_SAMPLES, NEG_STEPS_PER)

    # Data: for a single memorized pattern the positive moments are constant.
    # We use BATCH_SIZE repeats only for slight smoothing of the (identical) stats.
    clean_batch = jnp.repeat(clean_flat[None, :], BATCH_SIZE, axis=0)  # (B, 256) bool

    # Precompute node index lookup for positive moment calculation
    node_to_idx = {n: i for i, n in enumerate(pixel_nodes)}

    print(f"\nTraining (fully-visible grid, neighbours only) for {TRAIN_STEPS} steps ...")

    optimizer = optax.adam(LR)
    opt_state = optimizer.init((biases, weights))

    for step in range(TRAIN_STEPS):
        key, k_neg = jax.random.split(key)

        # --- Positive phase: exact clamped data moments (no sampling) ---
        s = 2 * clean_batch.astype(jnp.int8) - 1          # (B, N) in {-1, +1}
        pos_b = jnp.mean(s, axis=0)                       # (N,)

        pos_w = []
        for u, v in edges:
            iu = node_to_idx[u]
            iv = node_to_idx[v]
            pos_w.append(jnp.mean(s[:, iu] * s[:, iv]))
        pos_w = jnp.array(pos_w)

        # --- Negative phase: free samples from the current model ---
        # Use single-chain init (batch_shape=()) + decent NEG_SAMPLES in the schedule.
        # This is the shape expected by estimate_moments / the spin samplers for
        # a plain (non-vmapped) moment estimation call.
        init_neg = hinton_init(k_neg, model, neg_blocks, ())   # (block_len,) per block
        neg_b, neg_w = estimate_moments(
            k_neg, nodes, edges, neg_program, sched_neg, init_neg, []
        )

        # Gradients exactly as in the KL estimator: Δ = -β ( <data> - <model> )
        grad_b = -beta * (pos_b - neg_b)
        grad_w = -beta * (pos_w - neg_w)

        updates, opt_state = optimizer.update((grad_b, grad_w), opt_state, (biases, weights))
        biases, weights = optax.apply_updates((biases, weights), updates)

        model = IsingEBM(nodes, edges, biases, weights, beta)

        if (step + 1) % 40 == 0 or step == 0:
            gnorm = float(jnp.sqrt(jnp.mean(grad_b**2) + jnp.mean(grad_w**2)))
            print(f"  step {step+1:3d}/{TRAIN_STEPS}   grad_norm={gnorm:.4f}")

    print("Training complete.")
    return IsingEBM(nodes, edges, biases, weights, beta)

# --------------------------- Denoising (relaxation) -----------------------
def build_init_from_noisy(noisy_flat: np.ndarray, grid):
    """Single-chain init for the two grid blocks, seeded from the noisy observation."""
    pixel_nodes = grid["pixel_nodes"]
    neg_blocks = grid["neg_blocks"]

    node_state = {n: bool(v) for n, v in zip(pixel_nodes, noisy_flat)}

    inits = []
    for blk in neg_blocks:
        arr = jnp.array([node_state[n] for n in blk.nodes], dtype=jnp.bool_)
        inits.append(arr)   # shape (block_len,) for single chain
    return inits

def denoise(model: IsingEBM, noisy_flat: np.ndarray, grid):
    """Free block-Gibbs denoising on the grid. Return the final pixel configuration."""
    neg_blocks = grid["neg_blocks"]
    pixel_nodes = grid["pixel_nodes"]

    program = IsingSamplingProgram(model, neg_blocks, clamped_blocks=[])
    schedule = SamplingSchedule(DENOISE_WARMUP, DENOISE_SAMPLES, DENOISE_STEPS_PER)

    key = jax.random.key(4242)
    k_init, k_run = jax.random.split(key)

    init_state = build_init_from_noisy(noisy_flat, grid)

    # Read out using the program's own blocks (safe), then reassemble pixels by node identity.
    block_traces = sample_states(k_run, program, schedule, init_state, [], neg_blocks)

    node_to_final = {}
    for blk, trace in zip(neg_blocks, block_traces):
        final_for_blk = trace[-1]
        for node, val in zip(blk.nodes, final_for_blk):
            node_to_final[node] = bool(val)

    final_pixels = np.array([node_to_final[n] for n in pixel_nodes], dtype=bool)
    return final_pixels

# --------------------------- Visualization ------------------------------
def hamming(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a != b))

def save_comparison(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    for ax, arr, title in [
        (axes[0], clean, "Clean 'E' (target)"),
        (axes[1], noisy, f"Noisy input (p={NOISE_P})"),
        (axes[2], denoised, "Denoised (grid EBM)"),
    ]:
        ax.imshow(arr, cmap="binary", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    (OUT_DIR / "denoise_e_grid_comparison.png").write_bytes(b"")  # placeholder to ensure dir
    plt.savefig(OUT_DIR / "denoise_e_grid_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    for name, arr in [("clean", clean), ("noisy", noisy), ("denoised", denoised)]:
        plt.figure(figsize=(4, 4))
        plt.imshow(arr, cmap="binary", interpolation="nearest")
        plt.axis("off")
        plt.savefig(OUT_DIR / f"e_grid_{name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("\n=== Clean target ===")
    print(to_ascii(clean))
    print("\n=== Noisy observation ===")
    print(to_ascii(noisy))
    print("\n=== Grid-EBM denoised ===")
    print(to_ascii(denoised))

# ------------------------------- Main ----------------------------------
def main():
    print("=" * 58)
    print("THRML Grid Ising EBM Denoising — 16x16 'E' (neighbours only)")
    print("=" * 58)

    clean_img = make_clean_e()
    clean_flat = clean_img.ravel()
    noisy_img = add_noise(clean_img, NOISE_P)
    noisy_flat = noisy_img.ravel()

    print("\nClean 'E':")
    print(to_ascii(clean_img))
    print(f"\nNoisy input (flip p={NOISE_P}):")
    print(to_ascii(noisy_img))
    print(f"Initial pixel error: {hamming(clean_flat, noisy_flat):.3f}")

    grid = build_grid_model()

    model = train_grid_ebm(grid, clean_flat)

    print("\nDenoising by free sampling from the noisy init (grid neighbours only)...")
    denoised_flat = denoise(model, noisy_flat, grid)
    denoised_img = denoised_flat.reshape(H, W)

    final_err = hamming(clean_flat, denoised_flat)
    init_err = hamming(clean_flat, noisy_flat)
    print(f"\nPixel error vs clean target:")
    print(f"  noisy input : {init_err:.3f}")
    print(f"  after grid EBM : {final_err:.3f}   (improvement {init_err - final_err:.3f})")

    save_comparison(clean_img, noisy_img, denoised_img)

    print(f"\nResults in {OUT_DIR.resolve()}")
    print("Done. Pure grid (no hidden units), only neighbouring pixel interactions.")

if __name__ == "__main__":
    main()