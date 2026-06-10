#!/usr/bin/env python3
"""
No-training example: Hardcoded energy function for denoising a 16x16 "E" with THRML.

PURELY UNIFORM / HOMOGENEOUS PARAMETERS — no reference to the clean "E" shape.

This is a hand-designed EBM with **spatially invariant** parameters:
- Every pixel has exactly the same bias value (GLOBAL_BIAS).
- Every neighbor edge has exactly the same coupling strength (J_NEIGHBOR).

We do NOT embed the letter "E" into the parameters (no position-dependent template
biases derived from a clean reference image).

The energy is the standard Ising form:
    E(s) = -β [ ∑_i b_i s_i  +  ∑_<ij> J_ij s_i s_j ]

with the same b for all i and the same J for all neighboring pairs.

The prior only encourages:
1. Local smoothness (neighboring pixels prefer to be the same).
2. A very weak global preference for "on" pixels.

When we run block-Gibbs sampling starting from a noisy version of the "E",
isolated noise flips get corrected because they disagree with their neighbors,
while the overall rough shape survives because the initialization is already
mostly correct. This is a classic generic MRF denoiser prior.

The clean "E" ends up with low energy simply because it has long runs of
consistent neighboring pixels (the bars of the E are locally very smooth).

Run:
    /opt/homebrew/bin/python3.10 denoise_e_hardcoded.py

This file uses THRML only for the graph, nodes, blocks and the sampler.
The entire energy function is hardcoded by two numbers that are identical
everywhere in the image.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thrml import Block, SpinNode, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

# ----------------------------- Config -----------------------------
H, W = 16, 16
N_PIXELS = H * W
NOISE_P = 0.25

# Hardcoded energy parameters — COMPLETELY uniform across the image.
# No pixel has any special knowledge of where the "E" is. The exact same bias
# value is used for every pixel, and the exact same coupling for every neighbor
# edge. This is a spatially homogeneous prior (same rules everywhere).
GLOBAL_BIAS = 0.0
J_NEIGHBOR  = 0.55
BETA = 1.3

# Strength of the "data term" (how much we trust the noisy observation during
# sampling). This is *not* part of the core homogeneous EBM — it is added only
# at inference time as a pull toward the actual pixels we observed.
DATA_STRENGTH = 2.5

# Because of the data term anchor, we can run a reasonable number of steps.
DENOISE_WARMUP = 40
DENOISE_SAMPLES = 6
DENOISE_STEPS_PER_SAMPLE = 2

OUT_DIR = Path("denoise_hardcoded_results")
OUT_DIR.mkdir(exist_ok=True)

jax.config.update("jax_platform_name", "cpu")
print(f"JAX backend: {jax.default_backend()} (CPU only for this demo)")

# ------------------------- Letter "E" template -------------------------
def make_clean_e() -> np.ndarray:
    """The exact pattern we will hard-code into the energy function."""
    img = np.zeros((H, W), dtype=bool)
    img[1:15, 1:3] = True      # left vertical stem
    img[1:3, 1:14] = True      # top horizontal bar
    img[7:9, 1:12] = True      # middle horizontal bar
    img[13:15, 1:14] = True    # bottom horizontal bar
    img[0:1, 1:4] = True
    img[15:16, 1:4] = True
    return img

def add_noise(img: np.ndarray, p: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = img.copy()
    mask = rng.random(img.shape) < p
    noisy[mask] = ~noisy[mask]
    return noisy

def to_ascii(img: np.ndarray) -> str:
    return "\n".join("".join("\u2588" if v else "\u00b7" for v in row) for row in img)

# ------------------- Hardcoded Energy Function ------------------------
def compute_ising_energy(flat_bool: np.ndarray,
                         biases: jnp.ndarray,
                         weights: jnp.ndarray,
                         edges: list,
                         node_to_idx: dict,
                         beta: float) -> float:
    """
    Explicit energy function for the hardcoded EBM.
    All parameters (biases and J) are identical for every location in the image.
    """
    s = 2 * jnp.array(flat_bool, dtype=jnp.float32) - 1.0   # map to {-1, +1}

    # Linear (bias / external field) term
    linear = jnp.sum(biases * s)

    # Quadratic (neighbor coupling) term
    quad = 0.0
    for (u, v), w in zip(edges, weights):
        iu = node_to_idx[u]
        iv = node_to_idx[v]
        quad += w * s[iu] * s[iv]

    energy = -beta * (linear + quad)
    return float(energy)

# ------------------- Build the hardcoded grid EBM ----------------------
def build_hardcoded_ebm(j_neighbor, global_bias=0.0, beta=1.4):
    """
    Create the 16x16 grid + IsingEBM with homogeneous (uniform) parameters.
    All pixels get the same bias, all neighbor edges get the same J.
    No clean image is used to set per-pixel values.
    """
    # --- Nodes and graph (exactly the neighbor structure) ---
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

    # Bipartite blocks (perfect for grid)
    coloring = nx.bipartite.color(G)
    block0 = Block([n for n, col in coloring.items() if col == 0])
    block1 = Block([n for n, col in coloring.items() if col == 1])
    blocks = [block0, block1]

    node_to_idx = {n: i for i, n in enumerate(pixel_nodes)}

    # Homogeneous parameters (same everywhere)
    biases = jnp.full(len(nodes), global_bias)
    weights = jnp.full(len(edges), j_neighbor)
    beta_arr = jnp.array(beta)

    model = IsingEBM(nodes, edges, biases, weights, beta_arr)

    return {
        "model": model,
        "blocks": blocks,
        "nodes": nodes,
        "pixel_nodes": pixel_nodes,
        "edges": edges,
        "biases": biases,
        "weights": weights,
        "beta": beta,
        "node_to_idx": node_to_idx,
        "j_neighbor": j_neighbor,
    }

# ------------------------- Denoising via sampling ---------------------
def build_init_from_noisy(noisy_flat: np.ndarray, grid_info):
    """Single-chain init for the two grid blocks, seeded from noisy observation."""
    pixel_nodes = grid_info["pixel_nodes"]
    blocks = grid_info["blocks"]

    node_state = {n: bool(v) for n, v in zip(pixel_nodes, noisy_flat)}

    inits = []
    for blk in blocks:
        arr = jnp.array([node_state[n] for n in blk.nodes], dtype=jnp.bool_)
        inits.append(arr)   # shape (block_len,) — single chain, no leading 1
    return inits

def denoise_with_hardcoded_ebm(grid_info, noisy_flat: np.ndarray, data_strength=1.0,
                                 warmup=40, n_samples=6, steps_per=2):
    """
    Denoise by sampling from the hand-designed homogeneous prior + a data term
    that pulls toward the noisy observation (added only at inference time).

    The core prior parameters (returned in grid_info) are identical for every
    pixel and edge.
    """
    blocks = grid_info["blocks"]
    pixel_nodes = grid_info["pixel_nodes"]
    prior_biases = grid_info["biases"]
    prior_weights = grid_info["weights"]
    beta = grid_info["beta"]

    # Data term (likelihood) — strength controls how much we anchor to the observation
    noisy_pm1 = 2 * jnp.array(noisy_flat, dtype=jnp.float32) - 1.0
    data_biases = noisy_pm1 * data_strength

    effective_biases = prior_biases + data_biases

    effective_model = IsingEBM(
        grid_info["nodes"],
        grid_info["edges"],
        effective_biases,
        prior_weights,
        jnp.array(beta)
    )

    program = IsingSamplingProgram(effective_model, blocks, clamped_blocks=[])
    schedule = SamplingSchedule(warmup, n_samples, steps_per)

    key = jax.random.key(12345)
    k_init, k_run = jax.random.split(key)

    init_state = build_init_from_noisy(noisy_flat, grid_info)

    block_traces = sample_states(k_run, program, schedule, init_state, [], blocks)

    node_to_final = {}
    for blk, trace in zip(blocks, block_traces):
        final_for_blk = trace[-1]
        for node, val in zip(blk.nodes, final_for_blk):
            node_to_final[node] = bool(val)

    final_pixels = np.array([node_to_final[n] for n in pixel_nodes], dtype=bool)
    return final_pixels

# --------------------------- Visualization ------------------------------
def hamming(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a != b))

def save_comparison(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray,
                    energies: dict):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    titles = [
        "Clean 'E' (has low energy under the uniform prior)",
        f"Noisy input (p={NOISE_P})",
        "Denoised using homogeneous prior + data term from noisy obs"
    ]
    for ax, arr, title in zip(axes, [clean, noisy, denoised], titles):
        ax.imshow(arr, cmap="binary", interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.suptitle("Hardcoded homogeneous Ising EBM (uniform params across image, no template)",
                 fontsize=11, y=1.02)
    plt.tight_layout()

    out_path = OUT_DIR / "denoise_hardcoded_comparison.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # Also save the three individuals
    for name, arr in [("clean", clean), ("noisy", noisy), ("denoised", denoised)]:
        plt.figure(figsize=(4, 4))
        plt.imshow(arr, cmap="binary", interpolation="nearest")
        plt.axis("off")
        plt.savefig(OUT_DIR / f"hardcoded_{name}.png", dpi=140, bbox_inches="tight")
        plt.close()

    # ASCII
    print("\n=== Clean target (the pattern we hard-coded into the energy) ===")
    print(to_ascii(clean))
    print("\n=== Noisy observation ===")
    print(to_ascii(noisy))
    print("\n=== Result after sampling from the hardcoded energy ===")
    print(to_ascii(denoised))

    # Energy numbers
    print("\n=== Energies under the hardcoded EBM (lower = more probable) ===")
    for label, e in energies.items():
        print(f"  {label:12s}: {e:8.2f}")

def run_parameter_sweep(clean_img, noisy_img, clean_flat, noisy_flat):
    """
    Run the homogeneous EBM denoiser many times with different values of the
    two key parameters and save everything in one nice comparison image.
    """
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP: J_NEIGHBOR (smoothness) vs DATA_STRENGTH (trust in noisy obs)")
    print("All runs use the SAME noisy input. Core EBM params are uniform across pixels.")
    print("=" * 70)

    # Choose interesting ranges
    j_vals = [0.3, 0.6, 1.0, 1.8]
    data_vals = [0.5, 1.5, 2.5, 4.0]

    # Fixed reasonable schedule for the sweep
    sweep_warmup = 35
    sweep_samples = 5
    sweep_steps = 2

    results = []  # list of (j, data, denoised_img, error)

    for j in j_vals:
        row = []
        for d in data_vals:
            grid = build_hardcoded_ebm(j_neighbor=j, global_bias=0.0, beta=1.4)
            denoised_flat = denoise_with_hardcoded_ebm(
                grid, noisy_flat,
                data_strength=d,
                warmup=sweep_warmup, n_samples=sweep_samples, steps_per=sweep_steps
            )
            denoised_img = denoised_flat.reshape(H, W)
            err = hamming(clean_flat, denoised_flat)

            # Also compute prior energy of the result (under the uniform prior)
            prior_energy = compute_ising_energy(
                denoised_flat, grid["biases"], grid["weights"],
                grid["edges"], grid["node_to_idx"], grid["beta"]
            )

            row.append((j, d, denoised_img, err, prior_energy))
            print(f"  J={j:4.2f}  DATA={d:4.1f}  →  pixel_error={err:.3f}  prior_energy={prior_energy:7.1f}")
        results.append(row)

    # === Create one nice grid image with all combinations ===
    n_rows = len(j_vals)
    n_cols = len(data_vals)

    sweep_fig, sweep_axes = plt.subplots(
        n_rows, n_cols, figsize=(13, 10),
        sharex=True, sharey=True
    )

    for i, row in enumerate(results):
        for k, (j, d, dimg, err, pe) in enumerate(row):
            ax = sweep_axes[i, k] if n_rows > 1 and n_cols > 1 else sweep_axes[max(i, k)]
            ax.imshow(dimg, cmap="binary", interpolation="nearest")
            ax.set_title(f"J={j:.2f}   D={d:.1f}\nerr={err:.3f}", fontsize=9)
            ax.axis("off")

    # Row labels on the left (J values)
    for i, j in enumerate(j_vals):
        ax = sweep_axes[i, 0] if n_cols > 1 else sweep_axes[i]
        ax.set_ylabel(f"J = {j}", fontsize=10, rotation=0, labelpad=30, va="center")

    # Column headers (DATA values)
    for k, d in enumerate(data_vals):
        ax = sweep_axes[0, k] if n_rows > 1 else sweep_axes[k]
        ax.set_xlabel(f"D = {d}", fontsize=10)
        ax.xaxis.set_label_position("top")

    sweep_fig.suptitle(
        "Homogeneous Prior EBM Denoising — Parameter Sweep\n"
        "J = neighbor smoothness (identical for every edge in the grid)\n"
        "D = strength of data term from the noisy observation (added at inference)\n"
        "Core EBM has the exact same parameters everywhere — no clean template used.\n"
        "Lower err = better visual recovery of the letter shape.",
        fontsize=10
    )
    sweep_fig.tight_layout(rect=[0, 0.02, 1, 0.92])

    sweep_path = OUT_DIR / "parameter_sweep.png"
    sweep_fig.savefig(sweep_path, dpi=170, bbox_inches="tight")
    plt.close(sweep_fig)
    print(f"\nSaved combined parameter sweep image: {sweep_path}")

    # Also save individual best/worst if wanted, but the grid is the main deliverable


def main():
    print("=" * 68)
    print("THRML — Hardcoded Energy EBM (NO TRAINING) — Parameter Sweep")
    print("=" * 68)

    clean_img = make_clean_e()
    clean_flat = clean_img.ravel()
    noisy_img = add_noise(clean_img, NOISE_P)
    noisy_flat = noisy_img.ravel()

    print(f"\nNoisy input pixel error: {hamming(clean_flat, noisy_flat):.3f}")

    # Do the multi-run sweep and produce one combined image
    run_parameter_sweep(clean_img, noisy_img, clean_flat, noisy_flat)

    print(f"\nAll sweep outputs written to {OUT_DIR.resolve()}")
    print("The big grid is parameter_sweep.png — each panel is one (J, DATA_STRENGTH) combination.")


if __name__ == "__main__":
    main()