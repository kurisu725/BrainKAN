"""Generate all Route-A v4 publication-quality figures (Figs. 1-7) aligned with
development-recommended thresholds under the leak-free 50-epoch training-only pipeline.

Reads exclusively from:
  - results/unified_v4/ (threshold_and_roi_stability_v4.json)
  - results/unified_v3/ (hyperparam_v3.json, roi_curve_v3.json)
  - results/review_v2/  (per-seed training-only selection.json scores)
  - data/ or ../BrainKAN-data/ (MAT FC matrices for training-cohort visualizations)

Writes all figures to `--out` (default: results/unified_v4/figures/).
"""
import argparse
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

V3_DIR = REPO / "results" / "unified_v3"
V4_DIR = REPO / "results" / "unified_v4"
REVIEW_V2_DIR = REPO / "results" / "review_v2"
FIG_DIR = V4_DIR / "figures"

REC_THRESHOLDS = {
    "abide": {"direct": ("direct", 0.0), "minmax": ("minmax", 0.90), "zscore": ("zscore", 1.25)},
    "adhd":  {"direct": ("direct", 0.0), "minmax": ("minmax", 0.95), "zscore": ("zscore", 0.75)},
}

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


def aal_lobe(idx0):
    i = idx0 + 1
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


def find_mat(name):
    for root in (REPO / "data", REPO.parent / "BrainKAN-data"):
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(name)


def select_rois(scores, mode, tau):
    sc = np.asarray(scores, dtype=float)
    if mode == "direct":
        mask = sc < tau * float(sc.std(ddof=0))
    elif mode == "minmax":
        v = (sc - sc.min()) / (sc.max() - sc.min())
        mask = v < tau
    elif mode == "zscore":
        z = (sc - sc.mean()) / sc.std(ddof=0)
        mask = z < tau
    else:
        raise ValueError(mode)
    return set(np.flatnonzero(mask).tolist())


def load_aal90_coords():
    atlas = datasets.fetch_atlas_aal()
    coords_all, labels_list = find_parcellation_cut_coords(atlas.maps, return_label_names=True)
    idx_to_coord = {int(lbl): coords_all[k] for k, lbl in enumerate(labels_list)}
    coords90 = np.zeros((90, 3), dtype=float)
    for i in range(90):
        coords90[i] = idx_to_coord[int(atlas.indices[i])]
    return coords90, atlas.labels[:90]


def load_seed_selections_v4(ds):
    prefix = f"{ds}_fixed"
    scores_all = []
    sel_modes = {"direct": [], "minmax": [], "zscore": []}
    for s in range(5):
        d = read_json(REVIEW_V2_DIR / f"{prefix}_seed_{s}" / "selection.json")
        sc = np.asarray(d["scores"], dtype=float)
        scores_all.append(sc)
        for m in sel_modes:
            rm, tau = REC_THRESHOLDS[ds][m]
            sel_modes[m].append(select_rois(sc, rm, tau))
    scores_all = np.stack(scores_all, axis=0)
    mean_scores = scores_all.mean(axis=0)
    freq = {}
    counts = {}
    for m in sel_modes:
        f = np.zeros(90, dtype=int)
        for s_set in sel_modes[m]:
            for idx in s_set:
                f[idx] += 1
        freq[m] = f
        lens = [len(s_set) for s_set in sel_modes[m]]
        counts[m] = (float(np.mean(lens)), float(np.std(lens, ddof=1)))
    return scores_all, mean_scores, freq, counts


def plot_fig2_hyperparameter():
    hp_path = V3_DIR / "hyperparam_v3.json"
    data = read_json(hp_path)
    cfgs = data["configs"]
    labels, acc_m, acc_s, auc_m, auc_s, f1_m, f1_s, is_default = [], [], [], [], [], [], [], []
    for c in cfgs:
        h = "-".join(str(x) for x in c["hidden"])
        g, k = c["grid_size"], c["spline_order"]
        labels.append(f"[{h}] G={g}, k={k}")
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
    ax.bar(x - w, acc_m, w, yerr=acc_s, capsize=2.5, label="Accuracy (%)",
           color="#2b6cb0", alpha=0.88, error_kw={"elinewidth": 0.9, "ecolor": "#1a365d"})
    ax.bar(x, f1_m, w, yerr=f1_s, capsize=2.5, label="Macro F1 (%)",
           color="#dd6b20", alpha=0.88, error_kw={"elinewidth": 0.9, "ecolor": "#7b341e"})
    ax.bar(x + w, auc_m, w, yerr=auc_s, capsize=2.5, label="AUC (%)",
           color="#2f855a", alpha=0.88, error_kw={"elinewidth": 0.9, "ecolor": "#1c4532"})

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


def plot_fig3_glass_brains(coords90, ab_fc_mean, mean_scores, freq, counts):
    importance = -mean_scores
    modes = [
        ("direct", "brain1.png", f"Mode 1 (Direct, $C_i<0$; $M={counts['direct'][0]:.1f}\\pm{counts['direct'][1]:.1f}$)"),
        ("minmax", "brain2.png", f"Mode 2 (Min-max, $V_i<0.90$; $M={counts['minmax'][0]:.1f}\\pm{counts['minmax'][1]:.1f}$)"),
        ("zscore", "brain3.png", f"Mode 3 (Z-score, $Z_i<1.25$; $M={counts['zscore'][0]:.1f}\\pm{counts['zscore'][1]:.1f}$)"),
    ]
    for mode_key, fname, title in modes:
        sel_idx = np.where(freq[mode_key] >= 3)[0]
        sub_coords = coords90[sel_idx]
        sub_fc = ab_fc_mean[np.ix_(sel_idx, sel_idx)].copy()
        np.fill_diagonal(sub_fc, 0.0)
        n_nodes = len(sel_idx)
        edge_pct = "97.5%" if n_nodes > 70 else ("94%" if n_nodes > 40 else "88%")
        node_colors = [LOBE_COLORS[aal_lobe(i)] for i in sel_idx]
        imp_vals = np.clip(importance[sel_idx], 0, None)
        imp_norm = imp_vals / imp_vals.max() if imp_vals.max() > 0 else np.ones_like(imp_vals)
        node_sizes = 16 + 68 * imp_norm

        fig = plt.figure(figsize=(6.8, 3.0))
        plotting.plot_connectome(
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
                     fontsize=9.8, fontweight="bold", y=0.98)
        handles = [mpatches.Patch(color=c, label=l) for l, c in LOBE_COLORS.items()]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
                   fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
        out = FIG_DIR / fname
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved {out} ({len(sel_idx)} consensus ROIs)")


def plot_fig4_roi_curves(ab_counts):
    rc_path = V4_DIR / "roi_curve_v4.json"
    if not rc_path.exists():
        rc_path = V3_DIR / "roi_curve_v3.json"
    data = read_json(rc_path)
    curve = data["curve"]
    ks = [c["k"] for c in curve]

    metric_specs = [
        ("accuracy", "Accuracy (%)", "Acc2.png", "#1f77b4"),
        ("precision", "Macro Precision (%)", "Precision2.png", "#ff7f0e"),
        ("recall", "Macro Recall (%)", "Recall2.png", "#2ca02c"),
        ("f1", "Macro F1-Score (%)", "F1_Score2.png", "#9467bd"),
        ("auc", "ROC-AUC (%)", "AUC2.png", "#d62728"),
    ]

    # Route-A v4 dev-recommended mode mean ROI counts on ABIDE fixed
    mode_lines = [
        (ab_counts["zscore"][0], f"Z-score ($Z_i<1.25$: {ab_counts['zscore'][0]:.1f})", "#2f855a", "--"),
        (ab_counts["minmax"][0], f"Min-max ($V_i<0.90$: {ab_counts['minmax'][0]:.1f})", "#c53030", ":"),
        (ab_counts["direct"][0], f"Direct ($C_i<0$: {ab_counts['direct'][0]:.1f})", "#2b6cb0", "-."),
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
            ax.axvline(xm, color=lcol, linestyle=ls, linewidth=1.35, alpha=0.88, label=lbl)

        best_i = int(np.argmax(means))
        ax.scatter([ks[best_i]], [means[best_i]], color="black", s=45, zorder=5)
        ax.annotate(f"Peak: k={ks[best_i]}\n({means[best_i]:.2f}%)",
                    xy=(ks[best_i], means[best_i]),
                    xytext=(ks[best_i] - 24 if ks[best_i] > 65 else ks[best_i] + 5,
                            means[best_i] - 7.5 if means[best_i] > ymax - 8 else means[best_i] + 2.5),
                    fontsize=8, fontweight="bold",
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.9})

        ax.set_xlabel("Number of Retained ROIs ($k$, training-only ranking)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(3, 92)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, linestyle="--", alpha=0.32)
        ax.legend(loc="lower right", fontsize=7.0, frameon=True, ncol=1)
        ax.set_title(f"ABIDE {ylabel.split(' (')[0]} vs. Retained ROI Count", fontsize=10.5)
        plt.tight_layout()
        out = FIG_DIR / fname
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"[OK] Saved {out}")


def plot_roi_mode_summary(dataset_label, ds, mean_scores, freq, fnames, counts):
    importance = -mean_scores
    thr_txt = {
        "abide": {"direct": "$C_i<0$", "minmax": "$V_i<0.90$", "zscore": "$Z_i<1.25$"},
        "adhd":  {"direct": "$C_i<0$", "minmax": "$V_i<0.95$", "zscore": "$Z_i<0.75$"},
    }[ds]
    modes = [
        ("direct", fnames[0], f"{dataset_label} Mode 1 (Direct, {thr_txt['direct']}; $M={counts['direct'][0]:.1f}\\pm{counts['direct'][1]:.1f}$)"),
        ("minmax", fnames[1], f"{dataset_label} Mode 2 (Min-max, {thr_txt['minmax']}; $M={counts['minmax'][0]:.1f}\\pm{counts['minmax'][1]:.1f}$)"),
        ("zscore", fnames[2], f"{dataset_label} Mode 3 (Z-score, {thr_txt['zscore']}; $M={counts['zscore'][0]:.1f}\\pm{counts['zscore'][1]:.1f}$)"),
    ]

    for mode_key, fname, title in modes:
        f_arr = freq[mode_key]
        order = sorted(range(90), key=lambda i: (-f_arr[i], -importance[i]))[:20]
        order = list(reversed(order))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 4.6), gridspec_kw={"width_ratios": [1.55, 1.0]}, sharey=True)
        y = np.arange(len(order))
        imp_vals = importance[order] * 100.0
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

        fig.suptitle(title, fontsize=9.6, fontweight="bold", y=0.99)
        handles = [mpatches.Patch(color=c, label=l) for l, c in LOBE_COLORS.items()]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
                   fontsize=7.2, bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        out = FIG_DIR / fname
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved {out}")


def plot_fig7_heatmap(ab_fc_asd, ab_fc_hc, ad_fc_adhd, ad_fc_hc, ab_sel_idx, ad_sel_idx, ab_mean_sc, ad_mean_sc):
    """2x2 comparison of training-cohort group-difference FC (Patient - HC):
    Full 90x90 (AAL order) vs. v4 consensus subnetwork ordered by training-only attenuation importance (-C_i)."""
    # Sort selected indices by descending importance (-C_i) so highest-ranked ROIs appear top-left
    ab_ord = ab_sel_idx[np.argsort(ab_mean_sc[ab_sel_idx])]
    ad_ord = ad_sel_idx[np.argsort(ad_mean_sc[ad_sel_idx])]

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.2))
    ab_diff_full = ab_fc_asd - ab_fc_hc
    ab_diff_sub = ab_diff_full[np.ix_(ab_ord, ab_ord)]
    ad_diff_full = ad_fc_adhd - ad_fc_hc
    ad_diff_sub = ad_diff_full[np.ix_(ad_ord, ad_ord)]

    v_ab = max(0.08, float(np.percentile(np.abs(ab_diff_full), 99)))
    v_ad = max(0.08, float(np.percentile(np.abs(ad_diff_full), 99)))

    panels = [
        (axes[0, 0], ab_diff_full, v_ab, "ABIDE Full FC Difference (ASD − HC, 90 × 90)", None, ""),
        (axes[0, 1], ab_diff_sub, v_ab, f"ABIDE Z-score Consensus ($Z_i<1.25$, {len(ab_ord)} × {len(ab_ord)}, ranked)", ab_ord, "Ranked Z-score ROIs (≥3/5 seeds)"),
        (axes[1, 0], ad_diff_full, v_ad, "ADHD-200 Full FC Difference (ADHD − TDC, 90 × 90)", None, ""),
        (axes[1, 1], ad_diff_sub, v_ad, f"ADHD-200 Min-max Consensus ($V_i<0.95$, {len(ad_ord)} × {len(ad_ord)}, ranked)", ad_ord, "Ranked Min-max ROIs (≥3/5 seeds)"),
    ]

    for ax, mat, vmax, title, sub_idx, sub_lbl in panels:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=9.4, fontweight="bold", pad=6)
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
            ax.set_xticklabels(lbls, rotation=45, ha="right", fontsize=7.0)
            ax.set_yticks(ticks)
            ax.set_yticklabels(lbls, fontsize=7.0)
            ax.set_xlabel(sub_lbl)
            ax.set_ylabel(sub_lbl)
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
    ap = argparse.ArgumentParser(description="Generate Route-A v4 BrainKAN manuscript figures (Figs. 2-7).")
    ap.add_argument("--abide", type=Path, default=None)
    ap.add_argument("--adhd", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=V4_DIR / "figures")
    args = ap.parse_args()

    abide_mat = args.abide or find_mat("ALLASD1_aal.mat")
    adhd_mat = args.adhd or find_mat("ADHD1_aal.mat")
    FIG_DIR = args.out
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    coords90, _ = load_aal90_coords()
    plan = read_json(REVIEW_V2_DIR / "plan.json")
    ab_m, ab_l, _ = load_groups(abide_mat, "abide")
    ad_m, ad_l, _ = load_groups(adhd_mat, "adhd")
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

    _, ab_mean_sc, ab_freq, ab_counts = load_seed_selections_v4("abide")
    _, ad_mean_sc, ad_freq, ad_counts = load_seed_selections_v4("adhd")

    plot_fig2_hyperparameter()
    plot_fig3_glass_brains(coords90, ab_fc_mean, ab_mean_sc, ab_freq, ab_counts)
    plot_fig4_roi_curves(ab_counts)
    plot_roi_mode_summary("ABIDE", "abide", ab_mean_sc, ab_freq, ["brain1all.png", "brain2all.png", "brain3all.png"], ab_counts)
    plot_roi_mode_summary("ADHD-200", "adhd", ad_mean_sc, ad_freq, ["ADHD1.png", "ADHD2.png", "ADHD3.png"], ad_counts)

    # Fig 7: most compact consensus subnetwork per dataset (ABIDE Z-score 86x86, ADHD-200 Min-max 69x69)
    ab_sub = np.where(ab_freq["zscore"] >= 3)[0]
    ad_sub = np.where(ad_freq["minmax"] >= 3)[0]
    plot_fig7_heatmap(ab_fc_asd, ab_fc_hc, ad_fc_adhd, ad_fc_hc, ab_sub, ad_sub, ab_mean_sc, ad_mean_sc)
    print("\nALL ROUTE-A V4 FIGURES GENERATED.")


if __name__ == "__main__":
    main()
