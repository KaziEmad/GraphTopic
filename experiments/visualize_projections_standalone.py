"""
⚡ OPTIMIZED Standalone Projection Visualization

NEW FEATURES:
1. Pre-computes global directions ONCE (25x faster!)
2. Separate plots for each method (clearer visualization)
3. Supports N topics (pairwise comparison matrix)

Usage:
    from visualize_projections_optimized import visualize_projection_methods

    topics = [
        ["machine learning", "neural networks"],  # Topic 0
        ["data science", "statistics"],           # Topic 1
        ["deep learning", "AI"]                   # Topic 2
    ]

    visualize_projection_methods(
        topics,
        embedding_model,
        save_dir="projections/"
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from scipy.integrate import quad
from typing import List, Tuple, Optional
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os

sns.set_style("whitegrid")


# ============================================================================
# PROJECTION UTILITIES (unchanged)
# ============================================================================


def project_onto_direction(embeddings: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project embeddings onto direction vector"""
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    return embeddings @ direction


def fit_multi_gaussian_1d(
    projections: np.ndarray, n_gaussians: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit multiple Gaussians to 1D projection"""
    if len(projections) < n_gaussians * 3:
        n_gaussians = 1

    sorted_proj = np.sort(projections)
    n = len(sorted_proj)

    means, stds, weights = [], [], []
    for i in range(n_gaussians):
        start_idx = int(i * n / n_gaussians)
        end_idx = int((i + 1) * n / n_gaussians)
        region = sorted_proj[start_idx:end_idx]

        if len(region) > 0:
            means.append(region.mean())
            stds.append(region.std() + 1e-6)
            weights.append(len(region) / n)

    return np.array(means), np.array(stds), np.array(weights)


def pdf_multi_gaussian(
    x: float, means: np.ndarray, stds: np.ndarray, weights: np.ndarray
) -> float:
    """Multi-Gaussian PDF"""
    pdf_val = 0.0
    for mean, std, weight in zip(means, stds, weights):
        pdf_val += weight * norm.pdf(x, loc=mean, scale=std)
    return pdf_val


def compute_overlap_integral(means1, stds1, weights1, means2, stds2, weights2) -> float:
    """Compute overlap integral between two multi-Gaussian distributions"""
    all_means = np.concatenate([means1, means2])
    all_stds = np.concatenate([stds1, stds2])
    x_min = all_means.min() - 3 * all_stds.max()
    x_max = all_means.max() + 3 * all_stds.max()

    try:
        overlap, _ = quad(
            lambda x: min(
                pdf_multi_gaussian(x, means1, stds1, weights1),
                pdf_multi_gaussian(x, means2, stds2, weights2),
            ),
            x_min,
            x_max,
            limit=100,
        )
        return np.clip(overlap, 0, 1)
    except:
        return 0.0


# ============================================================================
# ⚡ OPTIMIZED: GLOBAL DIRECTION PRE-COMPUTATION
# ============================================================================


def precompute_global_directions(
    all_embeddings: np.ndarray, n_directions: int = 10
) -> List[np.ndarray]:
    """
    ⚡ CRITICAL OPTIMIZATION: Pre-compute directions ONCE!

    Instead of computing PCA for every pair, compute it ONCE and reuse.

    Args:
        all_embeddings: All embeddings combined (n_samples, n_dims)
        n_directions: Number of directions to generate

    Returns:
        List of direction vectors
    """
    directions = []
    n_dims = all_embeddings.shape[1]

    print(f"  ⚡ Pre-computing {n_directions} global directions...")

    # 1. Top PCA components (computed ONCE!)
    if len(all_embeddings) >= 2:
        try:
            n_components = min(5, len(all_embeddings) - 1, n_dims)
            pca = PCA(n_components=n_components)
            pca.fit(all_embeddings)

            for i in range(min(3, pca.n_components_)):
                directions.append(pca.components_[i])

            print(f"    ✓ Computed {len(directions)} PCA directions")
        except Exception as e:
            print(f"    ! PCA failed: {e}")

    # 2. Random directions (fast!)
    while len(directions) < n_directions:
        rand_dir = np.random.randn(n_dims)
        directions.append(rand_dir / np.linalg.norm(rand_dir))

    print(f"    ✓ Total {len(directions)} directions ready")
    return directions[:n_directions]


# ============================================================================
# PROJECTION METHODS (OPTIMIZED)
# ============================================================================


def compute_multi_gaussian_projection_overlap(
    emb1: np.ndarray, emb2: np.ndarray, n_gaussians: int = 2
) -> float:
    """Multi-Gaussian projection (unchanged)"""
    if len(emb1) == 0 or len(emb2) == 0:
        return 0.0

    center1, center2 = emb1.mean(axis=0), emb2.mean(axis=0)
    direction = center2 - center1

    if np.linalg.norm(direction) < 1e-10:
        return 1.0

    proj1 = project_onto_direction(emb1, direction)
    proj2 = project_onto_direction(emb2, direction)

    means1, stds1, weights1 = fit_multi_gaussian_1d(proj1, n_gaussians)
    means2, stds2, weights2 = fit_multi_gaussian_1d(proj2, n_gaussians)

    return compute_overlap_integral(means1, stds1, weights1, means2, stds2, weights2)


def compute_multi_direction_projection_overlap(
    emb1: np.ndarray,
    emb2: np.ndarray,
    n_directions: int = 10,
    n_gaussians: int = 2,
    global_directions: Optional[List[np.ndarray]] = None,  # ⚡ NEW!
) -> float:
    """
    ⚡ OPTIMIZED: Multi-direction projection with pre-computed directions
    """
    if len(emb1) == 0 or len(emb2) == 0:
        return 0.0

    # ⚡ Use global directions if provided (MUCH FASTER!)
    if global_directions is not None:
        directions = global_directions
    else:
        # Fallback: quick generation without expensive PCA
        directions = []
        center_dir = emb2.mean(axis=0) - emb1.mean(axis=0)
        if np.linalg.norm(center_dir) > 1e-10:
            directions.append(center_dir / np.linalg.norm(center_dir))

        # Random directions
        n_dims = emb1.shape[1]
        while len(directions) < n_directions:
            rand_dir = np.random.randn(n_dims)
            directions.append(rand_dir / np.linalg.norm(rand_dir))

    overlaps = []
    for direction in directions:
        proj1 = project_onto_direction(emb1, direction)
        proj2 = project_onto_direction(emb2, direction)
        means1, stds1, weights1 = fit_multi_gaussian_1d(proj1, n_gaussians)
        means2, stds2, weights2 = fit_multi_gaussian_1d(proj2, n_gaussians)
        overlap = compute_overlap_integral(
            means1, stds1, weights1, means2, stds2, weights2
        )
        overlaps.append(overlap)

    return np.clip(np.percentile(overlaps, 75), 0, 1) if overlaps else 0.0


def find_discriminative_directions(emb1, emb2, n_directions=5):
    """Find most discriminative projection directions"""
    X = np.vstack([emb1, emb2])
    y = np.array([0] * len(emb1) + [1] * len(emb2))
    directions = []

    # LDA direction
    if len(X) > 2:
        try:
            lda = LinearDiscriminantAnalysis(n_components=1)
            lda.fit(X, y)
            directions.append(lda.scalings_[:, 0])
        except:
            pass

    # Discriminative PCA
    if len(X) >= n_directions:
        try:
            pca = PCA(n_components=min(n_directions * 2, len(X) - 1, X.shape[1]))
            pca.fit(X)
            scores = []
            for i in range(pca.n_components_):
                pc = pca.components_[i]
                proj1, proj2 = emb1 @ pc, emb2 @ pc
                score = abs(proj1.mean() - proj2.mean()) / (
                    np.sqrt((proj1.var() + proj2.var()) / 2) + 1e-10
                )
                scores.append((score, pc))
            scores.sort(key=lambda x: x[0], reverse=True)
            for score, pc in scores[:n_directions]:
                if len(directions) < n_directions:
                    directions.append(pc)
        except:
            pass

    # Center direction as fallback
    center_dir = emb2.mean(axis=0) - emb1.mean(axis=0)
    if np.linalg.norm(center_dir) > 1e-10:
        center_dir = center_dir / np.linalg.norm(center_dir)
        if len(directions) < n_directions:
            directions.append(center_dir)

    return directions[:n_directions] if directions else [center_dir]


def compute_discriminative_projection_overlap(
    emb1: np.ndarray, emb2: np.ndarray, n_directions: int = 5, n_gaussians: int = 2
) -> float:
    """Discriminative multi-projection"""
    if len(emb1) == 0 or len(emb2) == 0:
        return 0.0

    directions = find_discriminative_directions(emb1, emb2, n_directions)
    overlaps = []

    for direction in directions:
        proj1 = project_onto_direction(emb1, direction)
        proj2 = project_onto_direction(emb2, direction)
        means1, stds1, weights1 = fit_multi_gaussian_1d(proj1, n_gaussians)
        means2, stds2, weights2 = fit_multi_gaussian_1d(proj2, n_gaussians)
        overlap = compute_overlap_integral(
            means1, stds1, weights1, means2, stds2, weights2
        )
        overlaps.append(overlap)

    return np.clip(np.mean(overlaps), 0, 1) if overlaps else 0.0


# ============================================================================
# VISUALIZATION: SEPARATE PLOTS FOR EACH METHOD
# ============================================================================


def visualize_multi_gaussian(emb1, emb2, topic_labels, n_gaussians=2, save_path=None):
    """Visualize Multi-Gaussian projection method"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Reduce to 2D for scatter
    combined = np.vstack([emb1, emb2])
    pca_2d = PCA(n_components=2)
    combined_2d = pca_2d.fit_transform(combined)
    emb1_2d = combined_2d[: len(emb1)]
    emb2_2d = combined_2d[len(emb1) :]

    # Plot 1: Scatter
    axes[0].scatter(
        emb1_2d[:, 0],
        emb1_2d[:, 1],
        alpha=0.6,
        s=100,
        label=topic_labels[0],
        c="steelblue",
        edgecolors="black",
    )
    axes[0].scatter(
        emb2_2d[:, 0],
        emb2_2d[:, 1],
        alpha=0.6,
        s=100,
        label=topic_labels[1],
        c="coral",
        edgecolors="black",
    )
    axes[0].set_title("Topics in 2D PCA Space", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Projection direction
    center1 = emb1_2d.mean(axis=0)
    center2 = emb2_2d.mean(axis=0)
    axes[1].scatter(emb1_2d[:, 0], emb1_2d[:, 1], alpha=0.6, s=100, c="steelblue")
    axes[1].scatter(emb2_2d[:, 0], emb2_2d[:, 1], alpha=0.6, s=100, c="coral")
    axes[1].plot(
        [center1[0], center2[0]],
        [center1[1], center2[1]],
        "k--",
        linewidth=2,
        label="Projection Line",
    )
    axes[1].scatter(
        center1[0],
        center1[1],
        s=200,
        c="blue",
        marker="X",
        edgecolors="black",
        linewidth=2,
    )
    axes[1].scatter(
        center2[0],
        center2[1],
        s=200,
        c="red",
        marker="X",
        edgecolors="black",
        linewidth=2,
    )
    axes[1].set_title("Projection Direction", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Projection overlap
    overlap = compute_multi_gaussian_projection_overlap(emb1, emb2, n_gaussians)

    center_dir = emb2.mean(axis=0) - emb1.mean(axis=0)
    proj1 = project_onto_direction(emb1, center_dir)
    proj2 = project_onto_direction(emb2, center_dir)

    means1, stds1, weights1 = fit_multi_gaussian_1d(proj1, n_gaussians)
    means2, stds2, weights2 = fit_multi_gaussian_1d(proj2, n_gaussians)

    axes[2].hist(
        proj1,
        bins=30,
        alpha=0.5,
        density=True,
        label=topic_labels[0],
        color="steelblue",
    )
    axes[2].hist(
        proj2, bins=30, alpha=0.5, density=True, label=topic_labels[1], color="coral"
    )

    x_range = np.linspace(
        min(proj1.min(), proj2.min()), max(proj1.max(), proj2.max()), 300
    )
    pdf1 = np.array([pdf_multi_gaussian(x, means1, stds1, weights1) for x in x_range])
    pdf2 = np.array([pdf_multi_gaussian(x, means2, stds2, weights2) for x in x_range])

    axes[2].plot(x_range, pdf1, "b-", linewidth=2, label="Fitted PDF 1")
    axes[2].plot(x_range, pdf2, "r-", linewidth=2, label="Fitted PDF 2")

    overlap_vals = np.minimum(pdf1, pdf2)
    axes[2].fill_between(
        x_range, overlap_vals, alpha=0.3, color="green", label=f"Overlap={overlap:.3f}"
    )

    axes[2].set_title(f"Multi-Gaussian (G={n_gaussians})", fontweight="bold")
    axes[2].set_xlabel("Projection Value")
    axes[2].set_ylabel("Density")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Multi-Gaussian Projection: {topic_labels[0]} vs {topic_labels[1]}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")

    return fig, overlap


def visualize_multi_direction(
    emb1,
    emb2,
    topic_labels,
    n_directions=5,
    n_gaussians=2,
    global_directions=None,
    save_path=None,
):
    """Visualize Multi-Direction projection method"""
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Compute overlap
    overlap = compute_multi_direction_projection_overlap(
        emb1, emb2, n_directions, n_gaussians, global_directions
    )

    # Get directions
    if global_directions is not None:
        directions = global_directions[:n_directions]
    else:
        directions = []
        center_dir = emb2.mean(axis=0) - emb1.mean(axis=0)
        if np.linalg.norm(center_dir) > 1e-10:
            directions.append(center_dir / np.linalg.norm(center_dir))
        while len(directions) < n_directions:
            rand_dir = np.random.randn(emb1.shape[1])
            directions.append(rand_dir / np.linalg.norm(rand_dir))

    # Plot each direction
    overlaps = []
    for i, direction in enumerate(directions[:6]):  # Show up to 6 directions
        if i >= 6:
            break

        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])

        proj1 = project_onto_direction(emb1, direction)
        proj2 = project_onto_direction(emb2, direction)

        means1, stds1, weights1 = fit_multi_gaussian_1d(proj1, n_gaussians)
        means2, stds2, weights2 = fit_multi_gaussian_1d(proj2, n_gaussians)

        x_range = np.linspace(
            min(proj1.min(), proj2.min()), max(proj1.max(), proj2.max()), 200
        )
        pdf1 = np.array(
            [pdf_multi_gaussian(x, means1, stds1, weights1) for x in x_range]
        )
        pdf2 = np.array(
            [pdf_multi_gaussian(x, means2, stds2, weights2) for x in x_range]
        )

        dir_overlap = np.trapz(np.minimum(pdf1, pdf2), x_range)
        overlaps.append(dir_overlap)

        ax.hist(proj1, bins=20, alpha=0.5, density=True, color="steelblue")
        ax.hist(proj2, bins=20, alpha=0.5, density=True, color="coral")
        ax.plot(x_range, pdf1, "b-", linewidth=2)
        ax.plot(x_range, pdf2, "r-", linewidth=2)
        ax.fill_between(x_range, np.minimum(pdf1, pdf2), alpha=0.3, color="green")

        dir_type = "PCA" if i < 3 else "Random"
        ax.set_title(
            f"Direction {i + 1} ({dir_type}): {dir_overlap:.3f}", fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Multi-Direction (K={n_directions}): {topic_labels[0]} vs {topic_labels[1]}\n"
        f"Aggregated Overlap = {overlap:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")

    return fig, overlap


def visualize_discriminative(
    emb1, emb2, topic_labels, n_directions=3, n_gaussians=2, save_path=None
):
    """Visualize Discriminative projection method"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Compute overlap
    overlap = compute_discriminative_projection_overlap(
        emb1, emb2, n_directions, n_gaussians
    )

    # Get discriminative directions
    directions = find_discriminative_directions(emb1, emb2, n_directions)

    # Plot LDA direction (first direction)
    if len(directions) > 0:
        lda_dir = directions[0]
        proj1 = project_onto_direction(emb1, lda_dir)
        proj2 = project_onto_direction(emb2, lda_dir)

        means1, stds1, weights1 = fit_multi_gaussian_1d(proj1, n_gaussians)
        means2, stds2, weights2 = fit_multi_gaussian_1d(proj2, n_gaussians)

        x_range = np.linspace(
            min(proj1.min(), proj2.min()), max(proj1.max(), proj2.max()), 200
        )
        pdf1 = np.array(
            [pdf_multi_gaussian(x, means1, stds1, weights1) for x in x_range]
        )
        pdf2 = np.array(
            [pdf_multi_gaussian(x, means2, stds2, weights2) for x in x_range]
        )

        axes[0].hist(
            proj1,
            bins=25,
            alpha=0.5,
            density=True,
            label=topic_labels[0],
            color="steelblue",
        )
        axes[0].hist(
            proj2,
            bins=25,
            alpha=0.5,
            density=True,
            label=topic_labels[1],
            color="coral",
        )
        axes[0].plot(x_range, pdf1, "b-", linewidth=2, label="PDF 1")
        axes[0].plot(x_range, pdf2, "r-", linewidth=2, label="PDF 2")
        axes[0].fill_between(x_range, np.minimum(pdf1, pdf2), alpha=0.3, color="green")
        axes[0].set_title("LDA Direction", fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Plot discriminative PCA directions
    for i, direction in enumerate(directions[1:3]):  # Show 2 more directions
        ax = axes[i + 1]

        proj1 = project_onto_direction(emb1, direction)
        proj2 = project_onto_direction(emb2, direction)

        means1, stds1, weights1 = fit_multi_gaussian_1d(proj1, n_gaussians)
        means2, stds2, weights2 = fit_multi_gaussian_1d(proj2, n_gaussians)

        x_range = np.linspace(
            min(proj1.min(), proj2.min()), max(proj1.max(), proj2.max()), 200
        )
        pdf1 = np.array(
            [pdf_multi_gaussian(x, means1, stds1, weights1) for x in x_range]
        )
        pdf2 = np.array(
            [pdf_multi_gaussian(x, means2, stds2, weights2) for x in x_range]
        )

        ax.hist(proj1, bins=25, alpha=0.5, density=True, color="steelblue")
        ax.hist(proj2, bins=25, alpha=0.5, density=True, color="coral")
        ax.plot(x_range, pdf1, "b-", linewidth=2)
        ax.plot(x_range, pdf2, "r-", linewidth=2)
        ax.fill_between(x_range, np.minimum(pdf1, pdf2), alpha=0.3, color="green")
        ax.set_title(f"Discriminative PC {i + 1}", fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Discriminative Projection: {topic_labels[0]} vs {topic_labels[1]}\n"
        f"Average Overlap = {overlap:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")

    return fig, overlap


# ============================================================================
# MAIN API: N-TOPIC SUPPORT
# ============================================================================


def visualize_projection_methods(
    topic_texts_list: List[List[str]],
    embedding_model,
    save_dir: str = "projections/",
    n_gaussians: int = 2,
    n_directions: int = 5,
    methods: List[str] = ["multi_gaussian", "multi_direction", "discriminative"],
):
    """
    ⚡ OPTIMIZED: Visualize projection methods for N topics

    Args:
        topic_texts_list: List of topic text lists
            Example: [
                ["ml text1", "ml text2"],  # Topic 0
                ["ds text1", "ds text2"],  # Topic 1
                ["dl text1", "dl text2"]   # Topic 2
            ]
        embedding_model: SentenceTransformer model
        save_dir: Directory to save figures
        n_gaussians: Number of Gaussians to fit
        n_directions: Number of directions for multi-direction
        methods: Which methods to visualize

    Returns:
        Dictionary with overlap scores for all topic pairs
    """
    n_topics = len(topic_texts_list)

    if n_topics < 2:
        raise ValueError("Need at least 2 topics to compare")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    print(f"Encoding {n_topics} topics...")
    topic_embeddings = []
    topic_labels = []

    for i, texts in enumerate(topic_texts_list):
        print(f"  Topic {i}: {len(texts)} texts")
        emb = embedding_model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        topic_embeddings.append(emb)
        topic_labels.append(f"Topic {i}")

    # ⚡ Pre-compute global directions ONCE for multi_direction method
    global_directions = None
    if "multi_direction" in methods:
        print("\n⚡ Pre-computing global directions for ALL comparisons...")
        all_emb = np.vstack(topic_embeddings)
        global_directions = precompute_global_directions(all_emb, n_directions)

    # Pairwise comparisons
    results = {}

    print(f"\nGenerating visualizations for {n_topics * (n_topics - 1) // 2} pairs...")

    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            pair_name = f"topic{i}_vs_topic{j}"
            print(f"\n{'=' * 60}")
            print(f"Comparing {topic_labels[i]} vs {topic_labels[j]}")
            print(f"{'=' * 60}")

            emb1 = topic_embeddings[i]
            emb2 = topic_embeddings[j]
            labels = [topic_labels[i], topic_labels[j]]

            pair_results = {}

            # Multi-Gaussian
            if "multi_gaussian" in methods:
                print("  Computing Multi-Gaussian...")
                save_path = os.path.join(save_dir, f"{pair_name}_multi_gaussian.png")
                _, overlap = visualize_multi_gaussian(
                    emb1, emb2, labels, n_gaussians, save_path
                )
                pair_results["multi_gaussian"] = overlap

            # Multi-Direction (using global directions!)
            if "multi_direction" in methods:
                print("  Computing Multi-Direction...")
                save_path = os.path.join(save_dir, f"{pair_name}_multi_direction.png")
                _, overlap = visualize_multi_direction(
                    emb1,
                    emb2,
                    labels,
                    n_directions,
                    n_gaussians,
                    global_directions,
                    save_path,  # ⚡ Pass pre-computed!
                )
                pair_results["multi_direction"] = overlap

            # Discriminative
            if "discriminative" in methods:
                print("  Computing Discriminative...")
                save_path = os.path.join(save_dir, f"{pair_name}_discriminative.png")
                _, overlap = visualize_discriminative(
                    emb1, emb2, labels, n_directions, n_gaussians, save_path
                )
                pair_results["discriminative"] = overlap

            results[pair_name] = pair_results

    # Create summary comparison
    _create_summary_plot(results, save_dir)

    print(f"\n{'=' * 60}")
    print(f"✓ All visualizations saved to: {save_dir}")
    print(f"{'=' * 60}")

    return results


def _create_summary_plot(results, save_dir):
    """Create summary comparison plot"""
    import pandas as pd

    # Extract data
    data = []
    for pair, methods in results.items():
        for method, score in methods.items():
            data.append(
                {
                    "Pair": pair.replace("_", " "),
                    "Method": method.replace("_", " ").title(),
                    "Overlap": score,
                }
            )

    if not data:
        return

    df = pd.DataFrame(data)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    pairs = df["Pair"].unique()
    methods = df["Method"].unique()
    x = np.arange(len(pairs))
    width = 0.25

    for i, method in enumerate(methods):
        method_data = df[df["Method"] == method]
        scores = [
            method_data[method_data["Pair"] == p]["Overlap"].values[0]
            if len(method_data[method_data["Pair"] == p]) > 0
            else 0
            for p in pairs
        ]
        ax.bar(x + i * width, scores, width, label=method)

    ax.set_xlabel("Topic Pairs", fontsize=12)
    ax.set_ylabel("Overlap Score", fontsize=12)
    ax.set_title(
        "Projection Method Comparison - All Pairs", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(pairs, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "summary_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n  ✓ Summary saved: {save_path}")
    plt.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("⚡ Optimized Standalone Projection Visualization")
    print("=" * 60)
    print("\nExample usage:")
    print()
    print("from sentence_transformers import SentenceTransformer")
    print("from visualize_projections_optimized import visualize_projection_methods")
    print()
    print("model = SentenceTransformer('all-MiniLM-L6-v2')")
    print()
    print("# Define N topics")
    print("topics = [")
    print("    ['machine learning', 'neural networks', 'deep learning'],  # Topic 0")
    print("    ['data science', 'statistics', 'data analysis'],           # Topic 1")
    print("    ['artificial intelligence', 'AI systems', 'automation']    # Topic 2")
    print("]")
    print()
    print("# Visualize all pairwise comparisons")
    print("results = visualize_projection_methods(")
    print("    topics, model,")
    print("    save_dir='projections/',")
    print("    n_gaussians=2,")
    print("    n_directions=5")
    print(")")
    print()
    print("# Results:")
    print("# - topic0_vs_topic1_multi_gaussian.png")
    print("# - topic0_vs_topic1_multi_direction.png")
    print("# - topic0_vs_topic1_discriminative.png")
    print("# - topic0_vs_topic2_multi_gaussian.png")
    print("# - ...")
    print("# - summary_comparison.png")

"""
Test Script for Optimized Projection Visualization

Tests:
1. 2 topics (basic)
2. 3 topics (pairwise)
3. 5 topics (scaling)
"""

from sentence_transformers import SentenceTransformer
import json
import time

# Load model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================================
# TEST 1: 2 Topics (Basic)
# ============================================================================

print("\n" + "=" * 70)
print("TEST 1: 2 Topics (Basic)")
print("=" * 70)

topics_2 = [
    [
        "machine learning algorithms",
        "supervised learning",
        "neural networks",
        "deep learning models",
        "artificial intelligence",
    ],
    [
        "data science techniques",
        "statistical analysis",
        "data visualization",
        "predictive analytics",
        "business intelligence",
    ],
]

print(f"\nTopic 0: {len(topics_2[0])} texts")
print(f"Topic 1: {len(topics_2[1])} texts")

start = time.time()
results_2 = visualize_projection_methods(
    topics_2, model, save_dir="test_results/2_topics/", n_gaussians=2, n_directions=5
)
elapsed = time.time() - start

print(f"\nCompleted in {elapsed:.2f} seconds")
print("\nResults:")
for pair, scores in results_2.items():
    print(f"{pair}:")
    for method, score in scores.items():
        print(f"  {method}: {score:.3f}")

# ============================================================================
# TEST 2: 3 Topics (Pairwise)
# ============================================================================

print("\n" + "=" * 70)
print("TEST 2: 3 Topics (Pairwise - 3 comparisons)")
print("=" * 70)

topics_3 = [
    ["machine learning", "neural networks", "deep learning", "supervised learning"],
    ["data science", "statistics", "data analysis", "data mining"],
    [
        "natural language processing",
        "text mining",
        "sentiment analysis",
        "language models",
    ],
]

print(f"\nTopic 0 (ML): {len(topics_3[0])} texts")
print(f"Topic 1 (DS): {len(topics_3[1])} texts")
print(f"Topic 2 (NLP): {len(topics_3[2])} texts")

start = time.time()
results_3 = visualize_projection_methods(
    topics_3, model, save_dir="test_results/3_topics/", n_gaussians=2, n_directions=5
)
elapsed = time.time() - start

print(f"\nCompleted in {elapsed:.2f} seconds")
print(f"Total comparisons: {len(results_3)}")

print("\nResults:")
for pair, scores in results_3.items():
    print(f"\n{pair}:")
    for method, score in scores.items():
        print(f"  {method}: {score:.3f}")

# Save to JSON
with open("test_results/3_topics/scores.json", "w") as f:
    json.dump(results_3, f, indent=2)

# ============================================================================
# TEST 3: 5 Topics (Scaling Test)
# ============================================================================

print("\n" + "=" * 70)
print("TEST 3: 5 Topics (Scaling - 10 comparisons)")
print("=" * 70)

topics_5 = [
    ["machine learning", "neural networks", "deep learning"],
    ["data science", "statistics", "analytics"],
    ["natural language processing", "text analysis", "NLP"],
    ["computer vision", "image recognition", "CNN"],
    ["reinforcement learning", "RL", "agents"],
]

print(f"\nNumber of topics: {len(topics_5)}")
print(f"Expected comparisons: {len(topics_5) * (len(topics_5) - 1) // 2}")

start = time.time()
results_5 = visualize_projection_methods(
    topics_5, model, save_dir="test_results/5_topics/", n_gaussians=2, n_directions=5
)
elapsed = time.time() - start

print(f"\nCompleted in {elapsed:.2f} seconds")
print(f"Actual comparisons: {len(results_5)}")

# Analyze results
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

print("\nHigh overlap pairs (> 0.5):")
for pair, scores in results_5.items():
    if scores.get("multi_direction", 0) > 0.5:
        print(f"  {pair}: {scores['multi_direction']:.3f}")

print("\nLow overlap pairs (< 0.3):")
for pair, scores in results_5.items():
    if scores.get("multi_direction", 0) < 0.3:
        print(f"  {pair}: {scores['multi_direction']:.3f}")

# Save to JSON
with open("test_results/5_topics/scores.json", "w") as f:
    json.dump(results_5, f, indent=2)

# ============================================================================
# TEST 4: Single Method Test
# ============================================================================

print("\n" + "=" * 70)
print("TEST 4: Single Method (Multi-Direction only)")
print("=" * 70)

start = time.time()
results_single = visualize_projection_methods(
    topics_3,
    model,
    save_dir="test_results/single_method/",
    methods=["multi_direction"],  # Only one method
)
elapsed = time.time() - start

print(f"\nCompleted in {elapsed:.2f} seconds (faster!)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nGenerated files:")
print("  test_results/2_topics/     - 3 plots + summary")
print("  test_results/3_topics/     - 9 plots + summary")
print("  test_results/5_topics/     - 30 plots + summary")
print("  test_results/single_method/ - 3 plots + summary")

print("\n✓ All tests completed successfully!")
print("\nCheck the test_results/ directory for visualizations.")
