from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, FancyArrowPatch, Rectangle

# Publication typography & vector PDF settings
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})

# Palette (harmonized with workflow_dense_opus.pdf)
C_INK        = "#25313c"
C_MUTED      = "#5e6a72"
C_BORDER     = "#bcc5cb"

C_TEAL       = "#3f7f6e"
C_TEAL_BG    = "#eef5f2"
C_TEAL_BD    = "#b8d4ca"

C_ORG        = "#b55d1c"
C_ORG_BRIGHT = "#d9822b"
C_ORG_SOFT   = "#f0bc86"
C_ORG_BG     = "#fbf3ea"
C_ORG_BD     = "#e5c39f"

C_PURP       = "#6a5596"
C_PURP_MID   = "#8b78b3"
C_PURP_BG    = "#f3f0f8"
C_PURP_BD    = "#cec4e2"

C_BLUE       = "#2d6a9f"
C_BLUE_MID   = "#548ec0"
C_BLUE_BG    = "#eaf2f9"
C_BLUE_BD    = "#b9d3e8"


def make_fc_matrix(n=8, seed=42):
    r = np.random.default_rng(seed)
    a = r.uniform(0.18, 0.88, (n, n))
    m = 0.5 * (a + a.T)
    np.fill_diagonal(m, 1.0)
    return m


def draw_fc_grid(ax, x, y, w, h, mat, base_cmap="teal",
                 highlight_idx=None, highlight_indices=None,
                 show_diag_dot=False, alpha_att=False, zorder=4):
    """Draw a crisp vector FC heatmap grid."""
    n = mat.shape[0]
    cw = w / n
    ch = h / n

    if base_cmap == "teal":
        lo = np.array([238, 245, 242]) / 255.0
        hi = np.array([63, 127, 110]) / 255.0
        border_c = C_TEAL
    elif base_cmap == "purple":
        lo = np.array([243, 240, 248]) / 255.0
        hi = np.array([106, 85, 150]) / 255.0
        border_c = C_PURP
    elif base_cmap == "blue":
        lo = np.array([234, 242, 249]) / 255.0
        hi = np.array([45, 106, 159]) / 255.0
        border_c = C_BLUE
    else:
        lo = np.array([240, 242, 244]) / 255.0
        hi = np.array([94, 106, 114]) / 255.0
        border_c = C_MUTED

    org_lo = np.array([251, 240, 226]) / 255.0
    org_hi = np.array([217, 130, 43]) / 255.0

    for i in range(n):
        for j in range(n):
            val = float(mat[i, j])
            is_hi = False
            if highlight_idx is not None and (i == highlight_idx or j == highlight_idx):
                is_hi = True
            if highlight_indices is not None and (i in highlight_indices or j in highlight_indices):
                is_hi = True

            if is_hi and not (alpha_att and i == highlight_idx and j == highlight_idx):
                t = 0.25 + 0.55 * val if not alpha_att else 0.32 + 0.38 * val
                col = (1 - t) * org_lo + t * org_hi
            else:
                t = 0.15 + 0.78 * val
                col = (1 - t) * lo + t * hi

            ax.add_patch(Rectangle(
                (x + j * cw, y + h - (i + 1) * ch), cw, ch,
                facecolor=col, edgecolor="white", linewidth=0.25, zorder=zorder
            ))

    ax.add_patch(Rectangle(
        (x, y), w, h, facecolor="none", edgecolor=border_c, linewidth=0.85, zorder=zorder + 2
    ))

    if highlight_idx is not None and alpha_att:
        ry = y + h - (highlight_idx + 1) * ch
        cx = x + highlight_idx * cw
        ax.add_patch(Rectangle((x, ry), w, ch, facecolor="none",
                               edgecolor=C_ORG, linewidth=0.95, zorder=zorder + 3))
        ax.add_patch(Rectangle((cx, y), cw, h, facecolor="none",
                               edgecolor=C_ORG, linewidth=0.95, zorder=zorder + 3))
        if show_diag_dot:
            dot_x = cx + cw * 0.5
            dot_y = ry + ch * 0.5
            r_dot = max(cw, ch) * 0.75
            ax.add_patch(Circle((dot_x, dot_y), r_dot, facecolor=C_TEAL,
                                edgecolor="white", linewidth=0.75, zorder=zorder + 5))
            ax.text(dot_x, dot_y - 0.1, "1", color="white", fontsize=6.0, fontweight="bold",
                    ha="center", va="center", zorder=zorder + 6)


def draw_stacked_fc(ax, x, y, w, h, mat, base_cmap="teal", n_stack=3, dx=2.4, dy=2.4, zorder=4):
    bg_map = {"teal": ("#d7e7e1", C_TEAL), "blue": ("#d2e4f3", C_BLUE), "purple": ("#dfd8ec", C_PURP)}
    fc_back, ec_back = bg_map[base_cmap]
    for k in range(n_stack - 1, 0, -1):
        ax.add_patch(Rectangle(
            (x + k * dx, y + k * dy), w, h,
            facecolor=fc_back, edgecolor=ec_back, linewidth=0.6, alpha=0.85, zorder=zorder - k
        ))
    draw_fc_grid(ax, x, y, w, h, mat, base_cmap=base_cmap, zorder=zorder)


def draw_spline_curve(ax, p0, p1, amp=2.2, phase=0.0, color=C_ORG_BRIGHT, lw=0.75, alpha=0.85, zorder=5):
    x0, y0 = p0
    x1, y1 = p1
    t = np.linspace(0, 1, 28)
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy) + 1e-6
    nx, ny = -dy / L, dx / L
    wave = np.sin(2 * np.pi * t + phase) * np.sin(np.pi * t)
    xs = x0 + dx * t + nx * amp * wave
    ys = y0 + dy * t + ny * amp * wave
    ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, solid_capstyle="round", zorder=zorder)


def draw_kan(ax, x, y, w, h, layers=(4, 3, 2), node_color=C_TEAL,
             edge_color=C_ORG_BRIGHT, frozen=False, zorder=5):
    if frozen:
        ax.add_patch(FancyBboxPatch(
            (x - 3.5, y - 3.0), w + 7.0, h + 6.0,
            boxstyle="round,pad=0.25,rounding_size=3.2",
            facecolor="white", edgecolor=node_color, linewidth=0.95,
            linestyle=(0, (2.5, 1.8)), zorder=zorder - 2
        ))

    xs = np.linspace(x + 3.5, x + w - 3.5, len(layers))
    coords = []
    for li, n_nodes in enumerate(layers):
        if n_nodes == 1:
            ys = [y + h * 0.5]
        else:
            pad = max(3.5, h * (0.14 if n_nodes >= 4 else 0.22))
            ys = np.linspace(y + h - pad, y + pad, n_nodes)
        coords.append([(xs[li], yy) for yy in ys])

    for li in range(len(layers) - 1):
        for i, p0 in enumerate(coords[li]):
            for j, p1 in enumerate(coords[li + 1]):
                strong = ((i + j) % 2 == 0)
                amp = (2.1 if strong else 1.15) * (1 if (i + j) % 3 != 0 else -1)
                col = edge_color if strong else C_ORG_SOFT
                lw = 0.85 if strong else 0.55
                alp = 0.92 if strong else 0.65
                draw_spline_curve(ax, p0, p1, amp=amp, phase=0.35 * (i - j),
                                  color=col, lw=lw, alpha=alp, zorder=zorder)

    r_node = 2.45 if h >= 32 else 2.1
    for pts in coords:
        for px, py in pts:
            ax.add_patch(Circle(
                (px, py), r_node,
                facecolor="white" if frozen else node_color,
                edgecolor=node_color, linewidth=1.05, zorder=zorder + 2
            ))
            if frozen:
                ax.add_patch(Circle(
                    (px, py), r_node * 0.45,
                    facecolor=node_color, edgecolor="none", zorder=zorder + 3
                ))
    return coords


def draw_brain_subnetwork(ax, cx, cy, rx=16.5, ry=14.5, zorder=5):
    """Draw a clean top-down axial brain outline with selected ROI subnetwork."""
    def hemi(sign):
        ang = np.linspace(-np.pi / 2, np.pi / 2, 80)
        x_out = sign * rx * (0.96 * np.cos(ang) * (1.0 - 0.14 * np.sin(ang)))
        y_out = ry * np.sin(ang)
        x_in = sign * (0.07 * rx * np.ones_like(ang))
        y_in = ry * 0.92 * np.sin(ang[::-1])
        xs = cx + np.concatenate([x_out, x_in])
        ys = cy + np.concatenate([y_out, y_in])
        return np.column_stack([xs, ys])

    for sign in (-1, 1):
        ax.add_patch(Polygon(hemi(sign), closed=True, facecolor="#f7f5fb",
                             edgecolor="#b9aecf", linewidth=0.9, zorder=zorder))

    nodes = [
        (cx - 0.50 * rx, cy + 0.42 * ry),
        (cx + 0.48 * rx, cy + 0.38 * ry),
        (cx - 0.28 * rx, cy + 0.08 * ry),
        (cx + 0.30 * rx, cy + 0.05 * ry),
        (cx - 0.56 * rx, cy - 0.30 * ry),
        (cx + 0.54 * rx, cy - 0.34 * ry),
        (cx - 0.24 * rx, cy - 0.56 * ry),
        (cx + 0.26 * rx, cy - 0.52 * ry),
    ]
    edges = [(0, 2), (0, 1), (1, 3), (2, 3), (2, 4), (3, 5), (4, 6), (5, 7), (4, 5), (2, 5)]
    for u, v in edges:
        x0, y0 = nodes[u]
        x1, y1 = nodes[v]
        ax.plot([x0, x1], [y0, y1], color=C_PURP_MID, linewidth=0.85, alpha=0.75, zorder=zorder + 1)
    for k, (nx, ny) in enumerate(nodes):
        hi = k in (2, 3, 4, 5)
        ax.add_patch(Circle(
            (nx, ny), 2.15 if hi else 1.75,
            facecolor=C_ORG_BRIGHT if hi else C_PURP,
            edgecolor="white", linewidth=0.55, zorder=zorder + 2
        ))


def arrow(ax, p0, p1, color=C_MUTED, lw=1.0, style="-|>", ms=7.0, ls="-", rad=0.0, zorder=6):
    a = FancyArrowPatch(
        p0, p1,
        arrowstyle=style, mutation_scale=ms,
        color=color, linewidth=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=zorder
    )
    ax.add_patch(a)


def pill(ax, x, y, w, h, text, fc="white", ec=C_BORDER, tc=C_INK,
         fs=7.0, bold=False, lw=0.85, r=3.0, zorder=7, subtext=None, subfs=6.3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.2,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder
    ))
    if subtext is None:
        ax.text(x + w * 0.5, y + h * 0.5, text, color=tc, fontsize=fs,
                fontweight="bold" if bold else "normal",
                ha="center", va="center", zorder=zorder + 1)
    else:
        ax.text(x + w * 0.5, y + h * 0.68, text, color=tc, fontsize=fs,
                fontweight="bold" if bold else "normal",
                ha="center", va="center", zorder=zorder + 1)
        ax.text(x + w * 0.5, y + h * 0.27, subtext, color=C_MUTED, fontsize=subfs,
                ha="center", va="center", zorder=zorder + 1)


def main():
    # Exact IEEE JBHI double-column textwidth: 7.16 in = 515.52 pt
    # Height: 4.10 in = 295.2 pt for generous breathing room in Panel (b) and margins
    W, H = 515.52, 295.2
    fig = plt.figure(figsize=(W / 72.0, H / 72.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # =========================================================================
    # 1. OUTER TRAINING CONTAINER (y in [103, 282], top margin 13.2 pt)
    # =========================================================================
    ax.add_patch(FancyBboxPatch(
        (6.0, 103.0), W - 12.0, 179.0,
        boxstyle="round,pad=0.3,rounding_size=5.0",
        facecolor="#fbfcfd", edgecolor=C_BORDER, linewidth=0.95,
        linestyle=(0, (4.0, 2.5)), zorder=0
    ))
    pill(ax, W * 0.5 - 84, 275.2, 168, 13.8,
         "Training set  (no test data used)",
         fc="#eef2f4", ec="#9aa6ac", tc=C_INK, fs=7.5, bold=True, lw=0.85, r=6.8, zorder=9)

    # =========================================================================
    # 2. THREE TRAINING PANELS: (a), (b), (c)
    # =========================================================================
    panel_y0, panel_h = 109.5, 160.5  # [109.5, 270.0]
    pa_x, pa_w = 12.0, 126.0          # [12, 138]
    pb_x, pb_w = 145.0, 214.0         # [145, 359]
    pc_x, pc_w = 366.0, 137.5         # [366, 503.5]

    for px, pw, bg, bd, title, tcol in [
        (pa_x, pa_w, C_TEAL_BG, C_TEAL_BD, "(a) Full-input KAN", C_TEAL),
        (pb_x, pb_w, C_ORG_BG,  C_ORG_BD,  "(b) Training-only ROI scoring & selection", C_ORG),
        (pc_x, pc_w, C_PURP_BG, C_PURP_BD, "(c) Reduced KAN", C_PURP),
    ]:
        ax.add_patch(FancyBboxPatch(
            (px, panel_y0), pw, panel_h,
            boxstyle="round,pad=0.2,rounding_size=4.5",
            facecolor=bg, edgecolor=bd, linewidth=0.9, zorder=1
        ))
        ax.text(px + pw * 0.5, panel_y0 + panel_h - 9.0, title,
                color=tcol, fontsize=8.2, fontweight="bold", ha="center", va="center", zorder=6)

    # =========================================================================
    # PANEL (a): Full-input KAN
    # =========================================================================
    mat_full = make_fc_matrix(8, seed=11)
    mx_a, my_a, ms_a = 19.5, 181.0, 35.0
    draw_stacked_fc(ax, mx_a, my_a, ms_a, ms_a, mat_full, base_cmap="teal", n_stack=3, dx=2.4, dy=2.4, zorder=4)
    ax.text(mx_a + ms_a * 0.5 + 2.2, my_a + ms_a + 12.0, r"$90 \times 90$",
            color=C_TEAL, fontsize=7.0, ha="center", va="center", zorder=6)
    ax.text(mx_a + ms_a * 0.5, my_a - 9.5, r"Train FC $X_{\mathrm{tr}}$",
            color=C_INK, fontsize=7.2, ha="center", va="center", zorder=6)

    arrow(ax, (mx_a + ms_a + 6.5, my_a + ms_a * 0.5), (84.0, my_a + ms_a * 0.5),
          color=C_TEAL, lw=1.05, ms=7.2)
    ax.text(72.5, my_a + ms_a * 0.5 + 9.2, "LN +\nflatten",
            color=C_TEAL, fontsize=6.4, ha="center", va="center", linespacing=1.08, zorder=6)

    kan_a_x, kan_a_y, kan_a_w, kan_a_h = 85.0, 175.0, 44.5, 47.0
    draw_kan(ax, kan_a_x, kan_a_y, kan_a_w, kan_a_h, layers=(4, 3, 2),
             node_color=C_TEAL, edge_color=C_ORG_BRIGHT, frozen=False, zorder=5)
    ax.text(kan_a_x + kan_a_w * 0.5 - 3.0, kan_a_y - 9.5, r"Fit $f_\theta$",
            color=C_INK, fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=6)

    ax.text(kan_a_x + kan_a_w * 0.5, 240.0, r"Labels $y_{\mathrm{tr}}$",
            color=C_TEAL, fontsize=7.0, ha="center", va="center", zorder=6)
    arrow(ax, (kan_a_x + kan_a_w * 0.5, 234.8), (kan_a_x + kan_a_w * 0.5, kan_a_y + kan_a_h + 1.5),
          color=C_TEAL, lw=0.95, ms=6.8)

    # Boxed KAN edge-spline legend at bottom of (a)
    leg_x, leg_y, leg_w, leg_h = 18.0, 118.0, 114.0, 27.5
    ax.add_patch(FancyBboxPatch(
        (leg_x, leg_y), leg_w, leg_h,
        boxstyle="round,pad=0.2,rounding_size=3.0",
        facecolor="white", edgecolor=C_TEAL_BD, linewidth=0.8, linestyle=(0, (2.2, 1.5)), zorder=3
    ))
    ax.add_patch(Circle((leg_x + 8.5, leg_y + 13.8), 2.1, facecolor=C_TEAL, edgecolor="none", zorder=5))
    ax.add_patch(Circle((leg_x + 31.5, leg_y + 13.8), 2.1, facecolor=C_TEAL, edgecolor="none", zorder=5))
    draw_spline_curve(ax, (leg_x + 10.6, leg_y + 13.8), (leg_x + 29.4, leg_y + 13.8),
                      amp=4.8, phase=0.0, color=C_ORG_BRIGHT, lw=1.25, alpha=1.0, zorder=5)
    ax.text(leg_x + 20.0, leg_y + 21.8, r"$\phi_{l,q,p}$", color=C_ORG, fontsize=6.7,
            ha="center", va="center", zorder=6)
    ax.text(leg_x + 37.5, leg_y + 13.8, "Learnable edge\nB-spline activation",
            color=C_INK, fontsize=6.5, ha="left", va="center", linespacing=1.18, zorder=6)
    # Clean pointer to right-lower edge of KAN, completely avoiding "Fit f_theta" text
    arrow(ax, (124.5, leg_y + leg_h + 0.5), (121.0, kan_a_y + 7.5),
          color=C_ORG_SOFT, lw=0.85, style="-", ls="--", zorder=4)

    # =========================================================================
    # PANEL (b): Training-only ROI scoring & selection
    # =========================================================================
    # Orthogonal "Freeze f_theta" route cleanly above X_tr (pill well right of X_tr!)
    kan_b_x, kan_b_y, kan_b_w, kan_b_h = 213.0, 185.0, 34.0, 35.0
    kan_b_cx = kan_b_x + kan_b_w * 0.5  # 230.0
    ax.plot([kan_a_x + kan_a_w + 0.5, 134.5, 134.5, kan_b_cx],
            [215.0, 215.0, 250.0, 250.0], color=C_TEAL, lw=1.0, zorder=5)
    arrow(ax, (kan_b_cx, 250.0), (kan_b_cx, kan_b_y + kan_b_h + 13.5),
          color=C_TEAL, lw=1.0, ms=6.8)
    pill(ax, 179.0, 244.8, 44.0, 10.5, r"Freeze $f_\theta$",
         fc="white", ec=C_TEAL_BD, tc=C_TEAL, fs=6.4, bold=True, lw=0.75, r=4.5, zorder=7)

    # Raw X_tr (top-left of b)
    mx_b1, my_b1, ms_b = 152.0, 208.5, 25.0
    draw_fc_grid(ax, mx_b1, my_b1, ms_b, ms_b, mat_full, base_cmap="teal", zorder=4)
    ax.text(mx_b1 + ms_b * 0.5, my_b1 + ms_b + 5.2, r"$X_{\mathrm{tr}}$",
            color=C_INK, fontsize=7.1, ha="center", va="center", zorder=6)

    # Attenuated X'_{tr,i} (mid-left of b)
    mx_b2, my_b2 = 152.0, 169.5
    draw_fc_grid(ax, mx_b2, my_b2, ms_b, ms_b, mat_full, base_cmap="teal",
                 highlight_idx=3, show_diag_dot=True, alpha_att=True, zorder=4)
    ax.text(mx_b2 + ms_b * 0.5, my_b2 - 6.5, r"$X'_{\mathrm{tr},i}$",
            color=C_INK, fontsize=7.1, ha="center", va="center", zorder=6)

    # Down arrow X_tr -> X'_{tr,i}
    arrow(ax, (mx_b1 + ms_b * 0.5, my_b1 - 1.2), (mx_b2 + ms_b * 0.5, my_b2 + ms_b + 1.4),
          color=C_ORG, lw=1.0, ms=6.8)
    ax.text(mx_b1 + ms_b * 0.5 - 2.0, 201.5, r"$\alpha=0.5$",
            color=C_ORG, fontsize=6.2, fontweight="bold", ha="right", va="center", zorder=6)

    # Frozen f_theta
    draw_kan(ax, kan_b_x, kan_b_y, kan_b_w, kan_b_h, layers=(4, 3, 2),
             node_color=C_TEAL, edge_color=C_ORG_BRIGHT, frozen=True, zorder=5)
    ax.text(kan_b_cx, kan_b_y + kan_b_h + 8.0, r"Frozen $f_\theta$",
            color=C_TEAL, fontsize=7.1, fontweight="bold", ha="center", va="center", zorder=6)

    # Converging arrows X_tr, X'_{tr,i} -> Frozen f_theta; "LN + flatten" cleanly centered in gap
    arrow(ax, (mx_b1 + ms_b + 1.5, my_b1 + ms_b * 0.45), (kan_b_x - 4.2, kan_b_y + kan_b_h * 0.78),
          color=C_TEAL, lw=0.95, ms=6.5)
    arrow(ax, (mx_b2 + ms_b + 1.5, my_b2 + ms_b * 0.52), (kan_b_x - 4.2, kan_b_y + kan_b_h * 0.24),
          color=C_ORG, lw=0.95, ms=6.5)
    ax.text(196.0, 203.0, "LN +\nflatten", color=C_MUTED, fontsize=5.7,
            ha="center", va="center", linespacing=1.04, zorder=6)

    # Delta L_CE box
    dce_x, dce_y = 269.5, 202.5
    ax.add_patch(FancyBboxPatch(
        (dce_x - 11.5, dce_y - 10.0), 23.0, 20.0,
        boxstyle="round,pad=0.2,rounding_size=3.2",
        facecolor="white", edgecolor=C_ORG, linewidth=0.95, zorder=5
    ))
    ax.text(dce_x, dce_y, r"$\Delta\mathcal{L}_{\mathrm{CE}}$",
            color=C_ORG, fontsize=7.2, fontweight="bold", ha="center", va="center", zorder=6)

    arrow(ax, (kan_b_x + kan_b_w + 4.0, kan_b_y + kan_b_h * 0.5), (dce_x - 12.0, dce_y),
          color=C_INK, lw=0.95, ms=6.5)

    ax.text(dce_x, 232.0, r"$y_{\mathrm{tr}}$", color=C_INK, fontsize=7.2,
            ha="center", va="center", zorder=6)
    arrow(ax, (dce_x, 227.2), (dce_x, dce_y + 10.8), color=C_INK, lw=0.9, ms=6.4)

    # Explicit formulas neatly stacked in open space BELOW Frozen f_theta & Delta CE (y = 171 and 160)
    ax.text(239.0, 171.5,
            r"$C_i = \mathcal{L}_{\mathrm{CE}}(X_{\mathrm{tr}}) - \mathcal{L}_{\mathrm{CE}}(X'_{\mathrm{tr},i})$",
            color=C_INK, fontsize=6.5, ha="center", va="center", zorder=6)
    ax.text(239.0, 160.5,
            r"$\forall i$: row & col $i \times \alpha$, diag$=1$ (PSD)",
            color=C_ORG, fontsize=6.2, ha="center", va="center", zorder=6)

    # ROI score C_i bar chart (right side of b, explicit 0 baseline)
    bar_x0, bar_y0, bar_w, bar_h = 291.5, 163.5, 61.0, 77.5
    ax.add_patch(FancyBboxPatch(
        (bar_x0, bar_y0), bar_w, bar_h,
        boxstyle="round,pad=0.2,rounding_size=3.0",
        facecolor="white", edgecolor=C_ORG_BD, linewidth=0.85, zorder=3
    ))
    ax.text(bar_x0 + bar_w * 0.5, bar_y0 + bar_h - 7.0, r"ROI score $C_i$",
            color=C_INK, fontsize=7.1, fontweight="bold", ha="center", va="center", zorder=6)

    zero_x = bar_x0 + 38.5
    ax.plot([zero_x, zero_x], [bar_y0 + 12.0, bar_y0 + bar_h - 14.0],
            color=C_MUTED, linewidth=0.75, linestyle="--", zorder=4)
    ax.text(zero_x, bar_y0 + 6.0, "0", color=C_MUTED, fontsize=6.0, ha="center", va="center", zorder=6)
    ax.text(bar_x0 + 18.0, bar_y0 + 6.0, r"$C_i<0$", color=C_ORG, fontsize=6.1,
            fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(bar_x0 + 50.5, bar_y0 + 6.0, r"$>0$", color=C_MUTED, fontsize=5.9,
            ha="center", va="center", zorder=6)

    scores = [-0.92, -0.79, -0.66, -0.52, -0.37, -0.22, -0.09, 0.14, 0.29]
    ys_bar = np.linspace(bar_y0 + bar_h - 18.2, bar_y0 + 15.8, len(scores))
    bh = 3.6
    for idx, sc in enumerate(scores):
        yy = ys_bar[idx] - bh * 0.5
        if sc < 0:
            bw = abs(sc) * 32.0
            fc = C_ORG_BRIGHT if idx < 5 else "#e9c9a8"
            ax.add_patch(Rectangle((zero_x - bw, yy), bw, bh, facecolor=fc, edgecolor="none", zorder=5))
        else:
            bw = sc * 17.5 / 0.35
            ax.add_patch(Rectangle((zero_x, yy), bw, bh, facecolor="#c8d1d6", edgecolor="none", zorder=5))

    arrow(ax, (dce_x + 12.0, dce_y), (bar_x0 - 1.2, dce_y), color=C_ORG, lw=1.0, ms=6.8)

    # Subtle separator line between scoring (top) and threshold rules (bottom)
    ax.plot([pb_x + 7.0, pb_x + pb_w - 7.0], [152.5, 152.5],
            color=C_ORG_BD, linewidth=0.65, linestyle=":", zorder=2)

    # Threshold rule row (bottom of panel b, generous vertical clearance!)
    rule_y, rule_h, rule_w = 122.0, 21.0, 58.0
    ax.text(229.0, 147.8, "Threshold rule (choose one mode):",
            color=C_ORG, fontsize=6.5, fontweight="bold", ha="center", va="center", zorder=6)

    rules = [
        (155.0, "Direct",  r"$C_i < 0$"),
        (221.0, "Min-max", r"$V_i < \tau_{\mathrm{mm}}$"),
        (287.0, "Z-score", r"$Z_i < \tau_z$"),
    ]
    for rx, title, cond in rules:
        pill(ax, rx, rule_y, rule_w, rule_h, title, subtext=cond,
             fc="white", ec=C_ORG_BRIGHT, tc=C_ORG, fs=6.8, subfs=6.3, bold=True, lw=0.85, r=3.2, zorder=6)

    # Arrow from bar chart down into threshold row (x = 322, well right of "Threshold rule" text)
    arrow(ax, (bar_x0 + bar_w * 0.5, bar_y0 - 0.8), (bar_x0 + bar_w * 0.5, rule_y + rule_h + 1.2),
          color=C_ORG, lw=0.95, ms=6.8)

    # Selected ROI subset badge Omega_m cleanly centered inside bottom of Training Box (y = 104..116.5)
    om_x, om_y, om_w, om_h = 212.0, 102.8, 76.0, 13.0
    for rx, _, _ in rules:
        cx_r = rx + rule_w * 0.5
        ax.plot([cx_r, cx_r], [rule_y - 0.5, om_y + om_h + 2.8], color=C_ORG_BRIGHT, lw=0.85, zorder=4)
    ax.plot([155.0 + rule_w * 0.5, 287.0 + rule_w * 0.5],
            [om_y + om_h + 2.8, om_y + om_h + 2.8], color=C_ORG_BRIGHT, lw=0.85, zorder=4)
    ax.plot([250.0, 250.0], [om_y + om_h + 2.8, om_y + om_h], color=C_ORG_BRIGHT, lw=0.85, zorder=4)

    pill(ax, om_x, om_y, om_w, om_h, r"Subset $\Omega_m$ ($M \leq N$)",
         fc="#fff6eb", ec=C_ORG, tc=C_ORG, fs=6.9, bold=True, lw=1.0, r=6.2, zorder=8)

    # =========================================================================
    # PANEL (c): Reduced KAN & ROI Subnetwork
    # =========================================================================
    mx_c1, my_c1, ms_c1 = 372.5, 200.0, 25.0
    draw_fc_grid(ax, mx_c1, my_c1, ms_c1, ms_c1, mat_full, base_cmap="teal",
                 highlight_indices={1, 3, 5, 6}, zorder=4)
    ax.text(mx_c1 + ms_c1 * 0.5, my_c1 + ms_c1 + 6.0, r"$X_{\mathrm{tr}}$",
            color=C_INK, fontsize=7.0, ha="center", va="center", zorder=6)

    mat_sub = make_fc_matrix(5, seed=29)
    mx_c2, my_c2, ms_c2 = 413.5, 202.5, 20.0
    draw_fc_grid(ax, mx_c2, my_c2, ms_c2, ms_c2, mat_sub, base_cmap="purple", zorder=4)
    ax.text(mx_c2 + ms_c2 * 0.5 - 2.0, my_c2 + ms_c2 + 8.5, r"$X_{\mathrm{tr}}[\Omega_m,\Omega_m]$",
            color=C_PURP, fontsize=6.6, fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(mx_c2 + ms_c2 * 0.5, my_c2 - 7.5, r"$M \times M$",
            color=C_PURP, fontsize=6.6, ha="center", va="center", zorder=6)

    arrow(ax, (mx_c1 + ms_c1 + 1.5, my_c1 + ms_c1 * 0.5), (mx_c2 - 1.5, my_c2 + ms_c2 * 0.5),
          color=C_PURP, lw=0.95, ms=6.8)

    # Routing from right of Omega_m badge -> up along x=405.5 into slice junction in (c)
    ax.plot([om_x + om_w + 0.5, 405.5, 405.5],
            [om_y + om_h * 0.5, om_y + om_h * 0.5, my_c2 + ms_c2 * 0.5 - 2.5],
            color=C_ORG, lw=1.0, zorder=5)
    arrow(ax, (405.5, my_c2 + ms_c2 * 0.5 - 6.0), (405.5, my_c2 + ms_c2 * 0.5 - 1.2),
          color=C_ORG, lw=1.0, ms=6.8)

    # Arrow submatrix -> Fit g_psi
    kan_c_x, kan_c_y, kan_c_w, kan_c_h = 459.0, 192.5, 38.0, 40.0
    arrow(ax, (mx_c2 + ms_c2 + 1.5, my_c2 + ms_c2 * 0.5), (kan_c_x - 1.2, kan_c_y + kan_c_h * 0.5),
          color=C_PURP, lw=1.0, ms=6.8)
    ax.text(446.8, my_c2 + ms_c2 * 0.5 - 9.5, "LN +\nflatten",
            color=C_PURP, fontsize=6.1, ha="center", va="center", linespacing=1.05, zorder=6)

    draw_kan(ax, kan_c_x, kan_c_y, kan_c_w, kan_c_h, layers=(3, 2, 2),
             node_color=C_PURP, edge_color=C_ORG_BRIGHT, frozen=False, zorder=5)
    ax.text(kan_c_x + kan_c_w * 0.5, kan_c_y - 8.5, r"Fit $g_\psi$",
            color=C_INK, fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=6)

    ax.text(kan_c_x + kan_c_w * 0.5, 244.5, r"Labels $y_{\mathrm{tr}}$",
            color=C_PURP, fontsize=6.9, ha="center", va="center", zorder=6)
    arrow(ax, (kan_c_x + kan_c_w * 0.5, 239.8), (kan_c_x + kan_c_w * 0.5, kan_c_y + kan_c_h + 1.2),
          color=C_PURP, lw=0.9, ms=6.5)

    # ROI Subnetwork interpretability card (bottom of c)
    ax.add_patch(FancyBboxPatch(
        (411.5, 117.5), 86.5, 46.5,
        boxstyle="round,pad=0.2,rounding_size=3.5",
        facecolor="white", edgecolor=C_PURP_BD, linewidth=0.8, zorder=3
    ))
    draw_brain_subnetwork(ax, 433.5, 140.8, rx=16.0, ry=14.4, zorder=5)
    ax.text(452.0, 147.0, "ROI\nsubnetwork", color=C_PURP, fontsize=6.5,
            fontweight="bold", ha="left", va="center", linespacing=1.12, zorder=6)
    ax.text(452.0, 128.5, "ROI-level\ninterpretation", color=C_MUTED, fontsize=5.7,
            ha="left", va="center", linespacing=1.12, zorder=6)

    arrow(ax, (mx_c2 + ms_c2 * 0.5, my_c2 - 13.0), (mx_c2 + ms_c2 * 0.5, 165.0),
          color=C_PURP_MID, lw=0.9, ms=6.5, ls="--")

    # =========================================================================
    # 3. ISOLATION BARRIER & CLEAN HORIZONTAL BUSES (y = 95.5 bus, y = 84.0 barrier)
    # =========================================================================
    bus_y = 95.5  # halfway between Training Box bottom (103) and barrier (84) -> ZERO border collision!
    div_y = 84.0
    ax.plot([10.0, W - 10.0], [div_y, div_y], color=C_BLUE, linewidth=1.0,
            linestyle=(0, (4.5, 2.5)), alpha=0.75, zorder=2)
    pill(ax, 134.0, div_y - 6.6, 226.0, 13.2,
         r"Freeze $\Omega_m$ & $g_\psi$  |  Single held-out test evaluation",
         fc="white", ec=C_BLUE, tc=C_BLUE, fs=6.9, bold=True, lw=0.85, r=6.6, zorder=9)

    # =========================================================================
    # 4. BOTTOM BAND: Independent test set
    # =========================================================================
    ax.add_patch(FancyBboxPatch(
        (6.0, 8.5), W - 12.0, 67.0,
        boxstyle="round,pad=0.3,rounding_size=5.0",
        facecolor=C_BLUE_BG, edgecolor=C_BLUE_BD, linewidth=0.95, zorder=1
    ))
    pill(ax, 13.0, 61.0, 92.0, 12.5, "Independent test set",
         fc="white", ec=C_BLUE_BD, tc=C_BLUE, fs=7.2, bold=True, lw=0.8, r=6.0, zorder=6)

    ty_mid = 39.2
    ty_lbl = 15.0  # 15 pt margin above page bottom

    # (1) Test FC X_test
    tx1, ts1 = 22.0, 29.0
    mat_test = make_fc_matrix(8, seed=77)
    draw_fc_grid(ax, tx1, ty_mid - ts1 * 0.5, ts1, ts1, mat_test, base_cmap="blue", zorder=4)
    ax.text(tx1 + ts1 * 0.5, ty_lbl, r"Test FC $X_{\mathrm{test}}$",
            color=C_INK, fontsize=7.1, ha="center", va="center", zorder=6)

    # (2) Fixed Omega_m slice (center x = 109.0 — balanced spacing with Test FC)
    sl_cx = 109.0
    arrow(ax, (tx1 + ts1 + 2.5, ty_mid), (sl_cx - 22.5, ty_mid), color=C_BLUE, lw=1.05, ms=7.2)
    pill(ax, sl_cx - 21.5, ty_mid - 10.5, 43.0, 21.0, r"Fixed $\Omega_m$",
         fc="#fff6eb", ec=C_ORG, tc=C_ORG, fs=7.0, bold=True, lw=0.95, r=3.5, zorder=6)
    ax.text(sl_cx, ty_lbl, "Index slice", color=C_ORG, fontsize=6.6, ha="center", va="center", zorder=6)

    # Clean dashed orange drop from bottom-center of Omega_m -> bus_y -> x=109 -> straight down
    ax.plot([250.0, 250.0, sl_cx, sl_cx],
            [om_y - 0.3, bus_y, bus_y, ty_mid + 11.5],
            color=C_ORG, lw=1.05, ls=(0, (3.2, 2.0)), zorder=5)
    arrow(ax, (sl_cx, ty_mid + 16.5), (sl_cx, ty_mid + 11.2), color=C_ORG, lw=1.05, ms=7.0)

    # (3) Sliced test submatrix X_test[Omega_m, Omega_m]
    tx2, ts2 = 171.0, 21.0
    arrow(ax, (sl_cx + 22.0, ty_mid), (tx2 - 2.0, ty_mid), color=C_BLUE, lw=1.05, ms=7.2)
    draw_fc_grid(ax, tx2, ty_mid - ts2 * 0.5, ts2, ts2, mat_sub, base_cmap="blue", zorder=4)
    ax.text(tx2 + ts2 * 0.5, ty_lbl, r"$X_{\mathrm{test}}[\Omega_m,\Omega_m]$",
            color=C_INK, fontsize=7.0, ha="center", va="center", zorder=6)

    # (4) LN + flatten (balanced spacing between X_test[Omega_m] and Frozen g_psi)
    ln_cx = 264.0
    arrow(ax, (tx2 + ts2 + 2.5, ty_mid), (ln_cx - 28.5, ty_mid), color=C_BLUE, lw=1.05, ms=7.2)
    pill(ax, ln_cx - 27.5, ty_mid - 10.5, 55.0, 21.0, "LN + flatten",
         fc="white", ec=C_BLUE_MID, tc=C_BLUE, fs=7.0, bold=False, lw=0.85, r=3.5, zorder=6)
    ax.text(ln_cx, ty_lbl, "Same preprocessing", color=C_MUTED, fontsize=6.4, ha="center", va="center", zorder=6)

    # (5) Frozen g_psi (center x = 374.0)
    tkan_cx = 374.0
    tkan_w, tkan_h = 40.0, 30.0
    tkan_x = tkan_cx - tkan_w * 0.5
    arrow(ax, (ln_cx + 28.0, ty_mid), (tkan_x - 4.5, ty_mid), color=C_BLUE, lw=1.05, ms=7.2)
    draw_kan(ax, tkan_x, ty_mid - tkan_h * 0.5, tkan_w, tkan_h, layers=(3, 2, 2),
             node_color=C_PURP, edge_color=C_ORG_BRIGHT, frozen=True, zorder=5)
    ax.text(tkan_cx, ty_lbl, r"Frozen $g_\psi$",
            color=C_PURP, fontsize=7.3, fontweight="bold", ha="center", va="center", zorder=6)

    # Clean dashed purple drop from Fit g_psi right -> x=505.8 -> bus_y -> x=374 -> straight down
    ax.plot([kan_c_x + kan_c_w + 0.5, 505.8, 505.8, tkan_cx, tkan_cx],
            [212.5, 212.5, bus_y, bus_y, ty_mid + tkan_h * 0.5 + 3.8],
            color=C_PURP, lw=1.05, ls=(0, (3.2, 2.0)), zorder=5)
    arrow(ax, (tkan_cx, ty_mid + tkan_h * 0.5 + 9.0), (tkan_cx, ty_mid + tkan_h * 0.5 + 3.6),
          color=C_PURP, lw=1.05, ms=7.0)

    # (6) Prediction y_hat
    yhat_cx = 425.5
    arrow(ax, (tkan_x + tkan_w + 4.0, ty_mid), (yhat_cx - 10.5, ty_mid), color=C_BLUE, lw=1.05, ms=7.2)
    ax.add_patch(Circle((yhat_cx, ty_mid), 9.5, facecolor="white", edgecolor=C_BLUE, linewidth=1.0, zorder=5))
    ax.text(yhat_cx, ty_mid - 0.3, r"$\hat{y}$", color=C_BLUE, fontsize=9.2,
            ha="center", va="center", zorder=6)
    ax.text(yhat_cx, ty_lbl, "Prediction", color=C_MUTED, fontsize=6.5, ha="center", va="center", zorder=6)

    # (7) Output classes: Patient / Control
    out_x, out_w, out_h = 456.0, 46.0, 15.5
    pill(ax, out_x, ty_mid + 4.5, out_w, out_h, "Patient",
         fc="white", ec=C_ORG_BRIGHT, tc=C_ORG, fs=7.2, bold=True, lw=0.95, r=3.5, zorder=6)
    pill(ax, out_x, ty_mid - 20.0, out_w, out_h, "Control",
         fc="white", ec=C_BLUE, tc=C_BLUE, fs=7.2, bold=True, lw=0.95, r=3.5, zorder=6)

    arrow(ax, (yhat_cx + 9.8, ty_mid + 3.0), (out_x - 1.2, ty_mid + 12.2), color=C_ORG_BRIGHT, lw=1.0, ms=6.8)
    arrow(ax, (yhat_cx + 9.8, ty_mid - 3.0), (out_x - 1.2, ty_mid - 12.2), color=C_BLUE, lw=1.0, ms=6.8)

    # Save vector PDF and high-res PNG previews
    out_dir = Path(__file__).resolve().parent / "results" / "unified_v4" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "workflow_v4.pdf"
    png_path = out_dir / "workflow_v4.png"
    fig.savefig(pdf_path, format="pdf", dpi=300)
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    print("Saved:", pdf_path, "and", png_path)


if __name__ == "__main__":
    main()
