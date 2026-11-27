"""Generate all publication-quality empirical figures for the unified BrainKAN manuscript.

Reads exclusively from the unified leak-free 50-epoch training-only outputs in:
  - results/unified_v3/
  - results/review_v2/
  - data/ (for heatmap FC matrices)

Writes figures to `--out` (default: `figures/` inside the repository root).

Figures generated:
  - Fig. 2: Hyperparameter.png (15 configs on ABIDE validation set, 50 epochs)
  - Fig. 3: brain1.png, brain2.png, brain3.png (ABIDE Mode 1/2/3 glass-brain connectomes)
  - Fig. 4: Acc2.png, Precision2.png, Recall2.png, F1_Score2.png, AUC2.png (ROI-count sweep)
  - Fig. 5: brain1all.png, brain2all.png, brain3all.png (ABIDE Mode 1/2/3 ROI importance & frequency)
  - Fig. 6: ADHD1.png, ADHD2.png, ADHD3.png (ADHD-200 Mode 1/2/3 ROI importance & frequency)
  - Fig. 7: heatmap.png (ABIDE & ADHD-200 FC matrices before vs. after Z-score ROI selection)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from nilearn import datasets, plotting
from nilearn.plotting import find_parcellation_cut_coords

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from jbhi_revision.common import load_groups
from run_review_experiments import GROUPS, read_json

UNIFIED_DIR = REPO / "results" / "unified_v3"
REVIEW_V2_DIR = REPO / "results" / "review_v2"
FIG_DIR = REPO / "figures"

# Clean publication typography
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Standard AAL-90 short abbreviations (indices 0..89 match AAL-116 first 90 cerebral regions)
AAL90_SHORT = [
    "PreCG.L", "PreCG.R", "SFGdor.L", "SFGdor.R", "ORBsup.L", "ORBsup.R",
    "MFG.L", "MFG.R", "ORBmid.L", "ORBmid.R", "IFGoperc.L", "IFGoperc.R",
    "IFGtriang.L", "IFGtriang.R", "ORBinf.L", "ORBinf.R", "ROL.L", "ROL.R",
    "SMA.L", "SMA.R", "OLF.L", "OLF.R", "SFGmed.L", "SFGmed.R",
    "ORBsupmed.L", "ORBsupmed.R", "REC.L", "REC.R", "INS.L", "INS.R",
    "ACG.L", "ACG.R", "DCG.L", "DCG.R", "PCG.L", "PCG.R",
    "HIP.L", "HIP.R", "PHG.L", "PHG.R", "AMYG.L", "AMYG.R",
    "CAL.L", "CAL.R", "CUN.L", "CUN.R", "LING.L", "LING.R",
    "SOG.L", "SOG.R", "MOG.L", "MOG.R", "IOG.L", "IOG.R",
    "FFG.L", "FFG.R", "PoCG.L", "PoCG.R", "SPG.L", "SPG.R",
    "IPL.L", "IPL.R", "SMG.L", "SMG.R", "ANG.L", "ANG.R",
    "PCUN.L", "PCUN.R", "PCL.L", "PCL.R", "CAU.L", "CAU.R",
    "PUT.L", "PUT.R", "PAL.L", "PAL.R", "THA.L", "THA.R",
    "HES.L", "HES.R", "STG.L", "STG.R", "TPOsup.L", "TPOsup.R",
    "MTG.L", "MTG.R", "TPOmid.L", "TPOmid.R", "ITG.L", "ITG.R",
]

# Anatomical lobe grouping for AAL-90 (for clear color coding)
def aal_lobe(idx0):
    i = idx0 + 1  # 1-based AAL index
    if 1 <= i <= 28:
        return "Frontal"
    if 29 <= i <= 42:
        return "Limbic/Insula"
    if 43 <= i <= 56:
        return "Occipital"
    if 57 <= i <= 70:
        return "Parietal"
    if 71 <= i <= 78:
        return "Subcortical"
    return "Temporal"

LOBE_COLORS = {
    "Frontal": "#1f77b4",
    "Limbic/Insula": "#d62728",
    "Occipital": "#2ca02c",
    "Parietal": "#ff7f0e",
    "Subcortical": "#9467bd",
    "Temporal": "#17becf",
}


def load_aal90_coords():
    atlas = datasets.fetch_atlas_aal()
    coords_all, labels_list = find_parcellation_cut_coords(atlas.maps, return_label_names=True)
    # atlas.indices are string codes matching voxel intensities in atlas.maps
    idx_to_coord = {int(lbl): coords_all[k] for k, lbl in enumerate(labels_list)}
    coords90 = np.zeros((90, 3), dtype=float)
    for i in range(90):
        code = int(atlas.indices[i])
        coords90[i] = idx_to_coord[code]
    return coords90, atlas.labels[:90]


def load_seed_selections(prefix):
    """Load per-seed scores and selected indices for review_v2/{prefix}_seed_{0..4}."""
    scores_all = []
    sel_modes = {"direct": [], "minmax": [], "zscore": []}
    for s in range(5):
        d = read_json(REVIEW_V2_DIR / f"{prefix}_seed_{s}" / "selection.json")
        scores_all.append(np.asarray(d["scores"], dtype=float))
        for m in sel_modes:
            sel_modes[m].append(set(int(x) for x in d["selections"][m]["indices"]))
    scores_all = np.stack(scores_all, axis=0)  # (5, 90)
    mean_scores = scores_all.mean(axis=0)      # negative C_i = higher importance
    freq = {}
    for m in sel_modes:
        f = np.zeros(90, dtype=int)
        for s_set in sel_modes[m]:
            for idx in s_set:
                f[idx] += 1
        freq[m] = f
    return scores_all, mean_scores, freq, sel_modes


def plot_fig2_hyperparameter():
    hp_path = UNIFIED_DIR / "hyperparam_v3.json"
    if not hp_path.exists():
        print("[SKIP] hyperparam_v3.json not ready yet")
        return
    data = read_json(hp_path)
    cfgs = data["configs"]
    labels = []
    acc_m, acc_s = [], []
    auc_m, auc_s = [], []
    f1_m, f1_s = [], []
    is_default = []
    for c in cfgs:
        h = "-".join(str(x) for x in c["hidden"])
        g = c["grid_size"]
        k = c["spline_order"]
        lbl = f"[{h}] G={g}, k={k}"
        labels.append(lbl)
        s = c["summary"]
        acc_m.append(100.0 * s["accuracy"]["mean"])
        acc_s.append(100.0 * s["accuracy"]["sd"])
        auc_m.append(100.0 * s["auc"]["mean"])
        auc_s.append(100.0 * s["auc"]["sd"])
        f1_m.append(100.0 * s["f1"]["mean"])
        f1_s.append(100.0 * s["f1"]["sd"])
        is_default.append(c["hidden"] == [256, 128, 64, 32, 16] and g == 5 and k == 3)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    w = 0.27
    b1 = ax.bar(x - w, acc_m, w, yerr=acc_s, capsize=2.5, label="Accuracy (%)",
                color="#2b6cb0", alpha=0.88, error_kw={"elinewidth": 0.9, "ecolor": "#1a365d"})
    b2 = ax.bar(x, f1_m, w, yerr=f1_s, capsize=2.5, label="Macro F1 (%)",
                color="#dd6b20", alpha=0.88, error_kw={"elinewidth": 0.9, "ecolor": "#7b341e"})
    b3 = ax.bar(x + w, auc_m, w, yerr=auc_s, capsize=2.5, label="AUC (%)",
                color="#2f855a", alpha=0.88, error_kw={"elinewidth": 0.9, "ecolor": "#1c4532"})

    # Highlight default configuration
    for i, flag in enumerate(is_default):
        if flag:
            ax.axvspan(i - 0.46, i + 0.46, color="#fed7d7", alpha=0.35, zorder=0)
            ax.text(i, max(acc_m[i] + acc_s[i], auc_m[i] + auc_s[i]) + 1.8, "Default",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#c53030")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8)
    ax.set_ylabel("Validation Performance (%)")
    ax.set_ylim(55, 82)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", frameon=True, ncol=3)
    ax.set_title("ABIDE Validation Sensitivity Across Hidden Width, Spline Grid Size (G), and Order (k) (50 Epochs, Seeds 0–4)",
                 fontsize=10.5, pad=10)
    plt.tight_layout()
    out = FIG_DIR / "Hyperparameter.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out}")


def plot_fig3_glass_brains(coords90, ab_fc_mean, mean_scores, freq):
    """Generate brain1.png (Mode 1 Direct), brain2.png (Mode 2 Min-max), brain3.png (Mode 3 Z-score)."""
    importance = -mean_scores  # positive = larger performance drop when attenuated
    modes = [
        ("direct", "brain1.png", "Mode 1 (Direct, M = 88.6 ± 0.5 ROIs)", "#2b6cb0"),
        ("minmax", "brain2.png", "Mode 2 (Min-max, M = 25.2 ± 6.1 ROIs)", "#c53030"),
        ("zscore", "brain3.png", "Mode 3 (Z-score, M = 40.4 ± 2.0 ROIs)", "#2f855a"),
    ]
    # Threshold >=3/5 seeds for consensus subnetwork display
    for mode_key, fname, title, color in modes:
        sel_idx = np.where(freq[mode_key] >= 3)[0]
        if len(sel_idx) == 0:
            sel_idx = np.argsort(mean_scores)[:10]
        sub_coords = coords90[sel_idx]
        sub_fc = ab_fc_mean[np.ix_(sel_idx, sel_idx)].copy()
        np.fill_diagonal(sub_fc, 0.0)
        # Keep top strongest edges among selected ROIs so glass brain stays uncluttered
        n_nodes = len(sel_idx)
        edge_pct = "97%" if n_nodes > 60 else ("92%" if n_nodes > 30 else "85%")
        # Color nodes by lobe and scale size by selection frequency & importance
        node_colors = [LOBE_COLORS[aal_lobe(i)] for i in sel_idx]
        imp_vals = np.clip(importance[sel_idx], 0, None)
        if imp_vals.max() > 0:
            imp_norm = imp_vals / imp_vals.max()
        else:
            imp_norm = np.ones_like(imp_vals)
        node_sizes = 18 + 65 * imp_norm

        fig = plt.figure(figsize=(6.8, 3.0))
        display = plotting.plot_connectome(
            sub_fc,
            sub_coords,
            node_color=node_colors,
            node_size=node_sizes,
            edge_threshold=edge_pct,
            edge_cmap="coolwarm",
            edge_vmin=-0.6,
            edge_vmax=0.6,
            edge_kwargs={"alpha": 0.45, "linewidth": 0.9},
            display_mode="lzr",
            figure=fig,
            colorbar=False,
            title=None,
        )
        fig.suptitle(f"{title}  ({len(sel_idx)} consensus ROIs, ≥3/5 seeds)",
                     fontsize=10.5, fontweight="bold", y=0.98)
        # Compact lobe legend at bottom
        handles = [mpatches.Patch(color=c, label=l) for l, c in LOBE_COLORS.items()]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
                   fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
        out = FIG_DIR / fname
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved {out} ({len(sel_idx)} consensus ROIs)")


def plot_fig4_roi_curves():
    rc_path = UNIFIED_DIR / "roi_curve_v3.json"
    if not rc_path.exists():
        print("[SKIP] roi_curve_v3.json not ready yet")
        return
    data = read_json(rc_path)
    curve = data["curve"]
    ks = [c["k"] for c in curve]

    metric_specs = [
        ("accuracy", "Accuracy (%)", "acc2.png", "#1f77b4"),
        ("precision", "Macro Precision (%)", "Precision2.png", "#ff7f0e"),
        ("recall", "Macro Recall (%)", "Recall2.png", "#2ca02c"),
        ("f1", "Macro F1-Score (%)", "F1_Score2.png", "#9467bd"),
        ("auc", "ROC-AUC (%)", "AUC2.png", "#d62728"),
    ]

    # Automatic mode mean ROI counts on ABIDE fixed (from review_v2)
    mode_lines = [
        (25.2, "Min-max (25.2)", "#c53030", ":"),
        (40.4, "Z-score (40.4)", "#2f855a", "--"),
        (88.6, "Direct (88.6)", "#2b6cb0", "-."),
    ]

    for mkey, ylabel, fname, col in metric_specs:
        means = np.array([100.0 * c["summary"][mkey]["mean"] for c in curve])
        sds = np.array([100.0 * c["summary"][mkey]["sd"] for c in curve])

        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.plot(ks, means, "-o", color=col, linewidth=2.0, markersize=4.5, label="Top-$k$ ROIs (mean ± SD)")
        ax.fill_between(ks, means - sds, means + sds, color=col, alpha=0.16)

        ymin = max(45, np.floor((means - sds).min() / 5) * 5 - 2)
        ymax = min(88, np.ceil((means + sds).max() / 5) * 5 + 3)

        for xm, lbl, lcol, ls in mode_lines:
            ax.axvline(xm, color=lcol, linestyle=ls, linewidth=1.3, alpha=0.85, label=lbl)

        # Mark peak
        best_i = int(np.argmax(means))
        ax.scatter([ks[best_i]], [means[best_i]], color="black", s=45, zorder=5)
        ax.annotate(f"Peak: k={ks[best_i]}\n({means[best_i]:.2f}%)",
                    xy=(ks[best_i], means[best_i]),
                    xytext=(ks[best_i] - 18 if ks[best_i] > 65 else ks[best_i] + 5,
                            means[best_i] - 6.5 if means[best_i] > ymax - 8 else means[best_i] + 2.5),
                    fontsize=8, fontweight="bold",
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.9})

        ax.set_xlabel("Number of Retained ROIs ($k$, training-only ranking)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(3, 92)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, linestyle="--", alpha=0.32)
        ax.legend(loc="lower right", fontsize=7.5, frameon=True, ncol=2)
        ax.set_title(f"ABIDE {ylabel.split(' (')[0]} vs. Retained ROI Count", fontsize=10.5)
        plt.tight_layout()
        out = FIG_DIR / fname
        fig.savefig(out, dpi=300)
        if fname == "acc2.png":
            fig.savefig(FIG_DIR / "Acc2.png", dpi=300)
        plt.close(fig)
        print(f"[OK] Saved {out}")


def plot_roi_mode_summary(dataset_label, mean_scores, freq, fnames, mean_counts):
    """Plot Fig. 5 (ABIDE) or Fig. 6 (ADHD): 3 panels showing top-25 ROIs ranked by
    mean training-only attenuation importance (-C_i) colored by lobe, with seed selection
    frequency (x/5) annotated and clear separation of selected vs. unselected ROIs."""
    importance = -mean_scores  # positive means stronger contribution
    modes = [
        ("direct", fnames[0], f"{dataset_label} Mode 1 (Direct, M = {mean_counts['direct']})"),
        ("minmax", fnames[1], f"{dataset_label} Mode 2 (Min-max, M = {mean_counts['minmax']})"),
        ("zscore", fnames[2], f"{dataset_label} Mode 3 (Z-score, M = {mean_counts['zscore']})"),
    ]

    for mode_key, fname, title in modes:
        f_arr = freq[mode_key]
        # Rank ROIs by (selection frequency desc, importance desc) and show top 20
        order = sorted(range(90), key=lambda i: (-f_arr[i], -importance[i]))[:20]
        order = list(reversed(order))  # top at top of horizontal bar chart

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 4.6), gridspec_kw={"width_ratios": [1.55, 1.0]}, sharey=True)
        y = np.arange(len(order))
        imp_vals = importance[order] * 100.0  # percentage points of dev/train AUC/loss drop
        freq_vals = f_arr[order]
        bar_cols = [LOBE_COLORS[aal_lobe(i)] for i in order]
        roi_names = [f"{AAL90_SHORT[i]} ({i+1})" for i in order]

        ax1.barh(y, imp_vals, color=bar_cols, alpha=0.88, height=0.72)
        ax1.set_yticks(y)
        ax1.set_yticklabels(roi_names, fontsize=8)
        ax1.set_xlabel("Mean Attenuation Score ($-C_i \\times 100$)")
        ax1.grid(axis="x", linestyle="--", alpha=0.32)
        ax1.set_title("Top-20 ROIs by Stability & Score", fontsize=9.5)

        ax2.barh(y, freq_vals, color=["#2f855a" if v == 5 else ("#4299e1" if v >= 3 else "#a0aec0") for v in freq_vals],
                 alpha=0.88, height=0.72)
        ax2.set_xlim(0, 5.6)
        ax2.set_xticks([0, 1, 2, 3, 4, 5])
        ax2.set_xlabel("Selected Seeds (out of 5)")
        ax2.grid(axis="x", linestyle="--", alpha=0.32)
        ax2.set_title("Cross-Seed Frequency", fontsize=9.5)
        for yi, fv in zip(y, freq_vals):
            ax2.text(fv + 0.12, yi, f"{fv}/5", va="center", fontsize=7.5, fontweight="bold")

        fig.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)
        handles = [mpatches.Patch(color=c, label=l) for l, c in LOBE_COLORS.items()]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
                   fontsize=7.2, bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        out = FIG_DIR / fname
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved {out}")


def plot_fig7_heatmap(ab_fc_asd, ab_fc_hc, ad_fc_adhd, ad_fc_hc, ab_z_idx, ad_z_idx):
    """Generate heatmap.png: 2x2 comparison of mean FC (Group Difference |Patient - HC|)
    on Full 90x90 vs. Z-score selected consensus subnetwork for ABIDE and ADHD-200."""
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.0))

    ab_diff_full = ab_fc_asd - ab_fc_hc
    ab_diff_sub = ab_diff_full[np.ix_(ab_z_idx, ab_z_idx)]

    ad_diff_full = ad_fc_adhd - ad_fc_hc
    ad_diff_sub = ad_diff_full[np.ix_(ad_z_idx, ad_z_idx)]

    v_ab = max(0.08, float(np.percentile(np.abs(ab_diff_full), 99)))
    v_ad = max(0.08, float(np.percentile(np.abs(ad_diff_full), 99)))

    panels = [
        (axes[0, 0], ab_diff_full, v_ab, "ABIDE Full FC Difference (ASD − HC, 90 × 90)", None),
        (axes[0, 1], ab_diff_sub, v_ab, f"ABIDE Z-score Subnetwork (ASD − HC, {len(ab_z_idx)} × {len(ab_z_idx)})", ab_z_idx),
        (axes[1, 0], ad_diff_full, v_ad, "ADHD-200 Full FC Difference (ADHD − TDC, 90 × 90)", None),
        (axes[1, 1], ad_diff_sub, v_ad, f"ADHD-200 Z-score Subnetwork (ADHD − TDC, {len(ad_z_idx)} × {len(ad_z_idx)})", ad_z_idx),
    ]

    for ax, mat, vmax, title, sub_idx in panels:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=9.8, fontweight="bold", pad=6)
        if sub_idx is None:
            ax.set_xlabel("AAL-90 ROI Index")
            ax.set_ylabel("AAL-90 ROI Index")
            ticks = [0, 19, 39, 59, 79, 89]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t + 1) for t in ticks], fontsize=8)
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(t + 1) for t in ticks], fontsize=8)
        else:
            step = max(1, len(sub_idx) // 10)
            ticks = list(range(0, len(sub_idx), step))
            lbls = [AAL90_SHORT[sub_idx[t]] for t in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(lbls, rotation=45, ha="right", fontsize=7.2)
            ax.set_yticks(ticks)
            ax.set_yticklabels(lbls, fontsize=7.2)
            ax.set_xlabel("Consensus Z-score ROIs (≥3/5 seeds)")
            ax.set_ylabel("Consensus Z-score ROIs")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7.5)
        cbar.set_label("Δ Pearson r", fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out}")


def main():
    global FIG_DIR
    ap = argparse.ArgumentParser(description="Generate unified BrainKAN manuscript figures (Figs. 2-7).")
    ap.add_argument("--abide", type=Path, default=REPO / "data" / "ALLASD1_aal.mat")
    ap.add_argument("--adhd", type=Path, default=REPO / "data" / "ADHD1_aal.mat")
    ap.add_argument("--out", type=Path, default=REPO / "figures")
    args = ap.parse_args()

    FIG_DIR = args.out
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    coords90, _ = load_aal90_coords()

    # Load ABIDE and ADHD-200 matrices (using training-set subjects for FC visualizations to stay strictly leak-free)
    plan = read_json(REVIEW_V2_DIR / "plan.json")
    ab_m, ab_l, _ = load_groups(args.abide, "abide")
    ad_m, ad_l, _ = load_groups(args.adhd, "adhd")
    ab_mats = np.concatenate([ab_m[g] for g in GROUPS])
    ab_labs = np.concatenate([ab_l[g] for g in GROUPS])
    ad_mats = np.concatenate([ad_m[g] for g in GROUPS])
    ad_labs = np.concatenate([ad_l[g] for g in GROUPS])

    ab_tr = np.asarray(plan["partitions"]["abide_fixed"]["indices"]["train"])
    ad_tr = np.asarray(plan["partitions"]["adhd_fixed"]["indices"]["train"])

    ab_fc_mean = ab_mats[ab_tr].mean(axis=0)
    ab_fc_asd = ab_mats[ab_tr][ab_labs[ab_tr] == 1].mean(axis=0)
    ab_fc_hc = ab_mats[ab_tr][ab_labs[ab_tr] == 0].mean(axis=0)

    ad_fc_adhd = ad_mats[ad_tr][ad_labs[ad_tr] == 1].mean(axis=0)
    ad_fc_hc = ad_mats[ad_tr][ad_labs[ad_tr] == 0].mean(axis=0)

    _, ab_mean_sc, ab_freq, _ = load_seed_selections("abide_fixed")
    _, ad_mean_sc, ad_freq, _ = load_seed_selections("adhd_fixed")

    # Fig 2 (Hyperparameter sensitivity)
    plot_fig2_hyperparameter()

    # Fig 3 (ABIDE Glass Brain Connectomes)
    plot_fig3_glass_brains(coords90, ab_fc_mean, ab_mean_sc, ab_freq)

    # Fig 4 (ROI-count performance curves)
    plot_fig4_roi_curves()

    # Fig 5 (ABIDE Mode 1/2/3 ROI importance & cross-seed frequency)
    plot_roi_mode_summary(
        "ABIDE", ab_mean_sc, ab_freq,
        ["brain1all.png", "brain2all.png", "brain3all.png"],
        {"direct": "88.6 ± 0.5", "minmax": "25.2 ± 6.1", "zscore": "40.4 ± 2.0"},
    )

    # Fig 6 (ADHD-200 Mode 1/2/3 ROI importance & cross-seed frequency)
    plot_roi_mode_summary(
        "ADHD-200", ad_mean_sc, ad_freq,
        ["ADHD1.png", "ADHD2.png", "ADHD3.png"],
        {"direct": "89.8 ± 0.4", "minmax": "5.8 ± 2.6", "zscore": "25.8 ± 3.8"},
    )

    # Fig 7 (FC Difference Heatmaps before vs after Z-score consensus selection)
    ab_z_idx = np.where(ab_freq["zscore"] >= 3)[0]
    ad_z_idx = np.where(ad_freq["zscore"] >= 3)[0]
    plot_fig7_heatmap(ab_fc_asd, ab_fc_hc, ad_fc_adhd, ad_fc_hc, ab_z_idx, ad_z_idx)
    print("\nALL AVAILABLE FIGURES GENERATED.")


if __name__ == "__main__":
    main()
