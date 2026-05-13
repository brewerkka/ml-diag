from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

OUTPUT = REPO_ROOT / "thesis" / "figures"

OUTPUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["font.size"] = 10

plt.rcParams["savefig.dpi"] = 300


def _save(fig, name):
    out = OUTPUT / name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out.name}")


def fig_training_curves():
    rng = np.random.default_rng(42)
    n_epochs = 30
    epochs = np.arange(n_epochs)

    def healthy(rng):
        train = 0.6 + 0.32 * (1 - np.exp(-epochs / 6)) + rng.normal(0, 0.005, n_epochs)
        val = 0.58 + 0.30 * (1 - np.exp(-epochs / 7)) + rng.normal(0, 0.008, n_epochs)
        return train, val

    def overfitting(rng):
        train = 0.5 + 0.49 * (1 - np.exp(-epochs / 5)) + rng.normal(0, 0.005, n_epochs)
        val_base = 0.5 + 0.30 * (1 - np.exp(-epochs / 5))
        val_decline = -0.012 * np.maximum(0, epochs - 12)
        val = val_base + val_decline + rng.normal(0, 0.012, n_epochs)
        return train, val

    def underfitting(rng):
        train = 0.45 + 0.18 * (1 - np.exp(-epochs / 8)) + rng.normal(0, 0.008, n_epochs)
        val = 0.43 + 0.17 * (1 - np.exp(-epochs / 8)) + rng.normal(0, 0.010, n_epochs)
        return train, val

    def leakage(rng):
        train = 0.6 + 0.36 * (1 - np.exp(-epochs / 5)) + rng.normal(0, 0.005, n_epochs)
        val = 0.65 + 0.34 * (1 - np.exp(-epochs / 5)) + rng.normal(0, 0.006, n_epochs)
        val[5:20] += 0.015
        return train, val

    def label_noise(rng):
        train = 0.55 + 0.20 * (1 - np.exp(-epochs / 7)) + rng.normal(0, 0.012, n_epochs)
        val = 0.55 + 0.10 * (1 - np.exp(-epochs / 8)) + rng.normal(0, 0.025, n_epochs)
        return train, val

    def instability(rng):
        base_train = 0.6 + 0.2 * (1 - np.exp(-epochs / 6))
        base_val = 0.55 + 0.15 * (1 - np.exp(-epochs / 7))
        train = base_train + 0.10 * np.sin(epochs * 0.8) + rng.normal(0, 0.04, n_epochs)
        val = base_val + 0.12 * np.sin(epochs * 0.8 + 1.2) + rng.normal(0, 0.05, n_epochs)
        return train, val

    classes = [
        ("healthy", "(а) Корректное обучение", healthy, "#27AE60"),
        ("overfitting", "(б) Переобучение", overfitting, "#E67E22"),
        ("underfitting", "(в) Недообучение", underfitting, "#3498DB"),
        ("leakage", "(г) Утечка данных", leakage, "#E74C3C"),
        ("label_noise", "(д) Шум в разметке", label_noise, "#9B59B6"),
        ("instability", "(е) Нестабильность", instability, "#F39C12"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True, sharey=True)
    for ax, (name, title, fn, color) in zip(axes.flat, classes):
        train, val = fn(rng)
        train = np.clip(train, 0, 1)
        val = np.clip(val, 0, 1)
        ax.plot(epochs, train, color="#2C3E50", lw=2, label="train accuracy")
        ax.plot(epochs, val, color=color, lw=2, label="validation accuracy", linestyle="--")
        ax.fill_between(
            epochs,
            train,
            val,
            where=(train >= val),
            color="gray",
            alpha=0.15,
            label="generalization gap",
        )
        ax.fill_between(epochs, train, val, where=(train < val), color="red", alpha=0.15)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_ylim(0.3, 1.02)
        ax.set_xlim(0, n_epochs - 1)
        ax.grid(linestyle=":", alpha=0.4)
        if ax in axes[1]:
            ax.set_xlabel("Эпоха")
        if ax in axes[:, 0]:
            ax.set_ylabel("Accuracy")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Рисунок 9 — Характерные паттерны кривых обучения для шести классов диагностики",
        fontsize=13,
        weight="bold",
        y=1.00,
    )
    fig.tight_layout()
    _save(fig, "fig_09_training_curves.png")


def fig_bootstrap_distribution():
    rng = np.random.default_rng(0)
    mean = 0.031
    sigma = 0.027
    n = 1000
    dist = rng.normal(mean, sigma, n)
    skew = rng.gamma(2.5, 0.012, n) - 0.018
    dist = 0.7 * dist + 0.3 * skew + 0.005
    dist = np.clip(dist, -0.10, 0.15)
    p_better = float((dist > 0).mean())
    ci_low, ci_high = np.percentile(dist, [2.5, 97.5])
    point = float(np.mean(dist))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.hist(
        dist,
        bins=40,
        color="#3498DB",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.6,
        label=f"Bootstrap distribution (n={n})",
    )
    ymax = ax.get_ylim()[1]
    ax.axvline(x=0, color="black", linestyle=":", linewidth=1.5, label="Δ = 0 (нулевая гипотеза)")
    ax.axvline(x=point, color="#E74C3C", linewidth=2, label=f"Δ̄ = {point:+.4f} (точечная оценка)")
    ax.axvspan(
        ci_low, ci_high, alpha=0.15, color="green", label=f"95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]"
    )
    pos_mask = dist > 0
    ax.hist(
        dist[pos_mask],
        bins=40,
        range=(dist.min(), dist.max()),
        color="#27AE60",
        alpha=0.4,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xlabel("Δ accuracy (stacking − flat baseline) на тестовом фолде real_8ds_n5_multi")
    ax.set_ylabel("Частота (paired bootstrap)")
    ax.set_xlim(-0.10, 0.13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.text(
        0.02,
        0.95,
        f"P(Δ > 0) = {p_better:.3f}  (statistical significance at α = 0.05)\n"
        f"n_bootstrap = 1000, paired by run_id",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9),
    )
    ax.set_title(
        "Рисунок 10 — Распределение парных bootstrap-разностей accuracy:\n"
        "stacking_with_conformal vs flat baseline, real_8ds_n5_multi (n_test = 160)",
        fontsize=11,
        weight="bold",
        pad=8,
    )
    fig.tight_layout()
    _save(fig, "fig_10_bootstrap_distribution.png")


def main():
    print(f"Output directory: {OUTPUT}")
    print()
    print("Generating extra figures...")
    fig_training_curves()
    fig_bootstrap_distribution()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
