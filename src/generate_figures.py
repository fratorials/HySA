#!/usr/bin/env python3
"""
Generate Figure 3: Semantic Hypersphere Geometry, Unified Architecture, and Pythia-70M Boundary Probing.
High-precision typographical redesign matching NeurIPS / Nature / ICLR standards.
Features:
  - Panel (a): Concentric Topological Boundary T_t and Factual Domain Radius R_dom.
  - Panel (b): Unified Macro-Architecture with Intermediate Layer l Probing, Early-Exit, Semantic Fault, and Fluid Flow.
  - Panel (c): Pythia-70M Empirical Probing across Inode Basin, Semantic Fault Band, and Out-of-Domain Creative Discourse.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Publication-grade typographical settings with embedded TrueType fonts & minus signs
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "figure.titlesize": 12.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False
})

fig = plt.figure(figsize=(17.5, 5.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.38, 1.20], wspace=0.25)

# ==============================================================================
# PANEL (A): SEMANTIC HYPERSPHERE CONCENTRIC DOMAINS (T_t and R_dom)
# ==============================================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_aspect("equal")

center = np.array([0.0, 0.0])
T_t = 0.90      # Topological Temperature (Inode Core)
R_dom = 1.38    # Factual Domain Radius (Ontological Boundary, ~0.65/0.46 * 0.90)

# 1. Background: Outer Fluid Space (d > R_dom)
bg_outer = patches.Rectangle((-2.05, -2.05), 4.10, 4.10, facecolor="#f8fafc", zorder=0)
ax1.add_patch(bg_outer)

# 2. Middle Ring: Semantic Fault Band (T_t < d <= R_dom)
c_dom = patches.Circle(center, R_dom, edgecolor="#ea580c", facecolor="#fff7ed", alpha=0.90, lw=1.8, ls="--", zorder=1)
ax1.add_patch(c_dom)

# 3. Inner Disk: Confirmed Inode Basin (d <= T_t)
c_basin = patches.Circle(center, T_t, edgecolor="#2563eb", facecolor="#eff6ff", alpha=0.95, lw=2.2, zorder=2)
ax1.add_patch(c_basin)

# Subtle coordinate axes
ax1.axhline(0, color="#cbd5e1", lw=0.9, ls=":", zorder=2)
ax1.axvline(0, color="#cbd5e1", lw=0.9, ls=":", zorder=2)
ax1.text(1.85, 0.04, r"$z_1$", fontsize=8.5, color="#64748b", fontweight="bold")
ax1.text(0.04, 1.85, r"$z_2$", fontsize=8.5, color="#64748b", fontweight="bold")

# Radius callouts
theta_tt = np.pi * 0.74
pt_tt = np.array([T_t * np.cos(theta_tt), T_t * np.sin(theta_tt)])
ax1.annotate("", xy=pt_tt, xytext=center, arrowprops=dict(arrowstyle="<->", color="#1d4ed8", lw=1.6))
ax1.text(-0.52, 0.38, r"$T_t$", fontsize=8.5, color="#1e40af", fontweight="bold", rotation=48)

theta_rdom = np.pi * 0.38
pt_rdom = np.array([R_dom * np.cos(theta_rdom), R_dom * np.sin(theta_rdom)])
ax1.annotate("", xy=pt_rdom, xytext=center, arrowprops=dict(arrowstyle="<->", color="#c2410c", lw=1.4))
ax1.text(0.24, 0.62, r"$R_{\mathrm{dom}}$", fontsize=8.5, color="#9a3412", fontweight="bold", rotation=40)

# Inode Centroid mu_cat
ax1.scatter([0], [0], color="#1e3a8a", s=180, zorder=7, edgecolors="white", lw=2.0)
ax1.scatter([0], [0], color="none", s=280, zorder=6, edgecolors="#1e3a8a", lw=1.2, ls="--")
ax1.text(0.16, -0.02, r"$\mathbf{Inode}\;\mu_{\mathrm{cat}}$",
         ha="left", va="center", fontsize=8.2, fontweight="bold", color="#1e3a8a",
         bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#93c5fd", alpha=0.92, lw=0.7))

# Query 1: Valid Inode Hit (inside T_t)
q_valid = np.array([0.45, -0.35])
d_valid = np.linalg.norm(q_valid)
ax1.plot([0, q_valid[0]], [0, q_valid[1]], color="#16a34a", lw=1.4, ls=":", zorder=5)
ax1.scatter([q_valid[0]], [q_valid[1]], color="#16a34a", marker="*", s=210, zorder=8, edgecolors="black", lw=0.8)
ax1.annotate(r"$q_{\mathrm{valid}}$: $\mathbf{CONFIRMED}\;(\mathrm{Early\text{-}Exit})$",
             xy=q_valid, xytext=(0.05, -0.85),
             arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.4), fontsize=7.8, color="#15803d", fontweight="bold",
             bbox=dict(boxstyle="square,pad=0.20", facecolor="#f0fdf4", edgecolor="#16a34a", lw=0.8))

# Query 2: Factual Drift / Unindexed Query (inside R_dom, outside T_t)
q_drift = np.array([-0.82, -0.82])
d_drift = np.linalg.norm(q_drift)
ax1.plot([0, q_drift[0]], [0, q_drift[1]], color="#dc2626", lw=1.4, ls=":", zorder=5)
ax1.scatter([q_drift[0]], [q_drift[1]], color="#dc2626", marker="X", s=150, zorder=8, edgecolors="black", lw=0.8)
ax1.annotate(r"$q_{\mathrm{unindexed}}$: $\mathbf{SEMANTIC\;FAULT}$",
             xy=q_drift, xytext=(-0.75, -1.35), ha="center",
             arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.4), fontsize=7.8, color="#b91c1c", fontweight="bold",
             bbox=dict(boxstyle="square,pad=0.20", facecolor="#fef2f2", edgecolor="#dc2626", lw=0.8))

# Query 3: Creative Discourse (outside R_dom, Fluid Np)
q_creative = np.array([-1.25, 1.15])
ax1.scatter([q_creative[0]], [q_creative[1]], color="#0284c7", marker="o", s=110, zorder=8, edgecolors="white", lw=1.0)
ax1.annotate(r"$q_{\mathrm{creative}}$: $\mathbf{FLUID}\;N_p\;(d > R_{\mathrm{dom}})$",
             xy=q_creative, xytext=(-0.95, 1.62), ha="center",
             arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.4), fontsize=7.8, color="#0369a1", fontweight="bold",
             bbox=dict(boxstyle="square,pad=0.20", facecolor="#f0f9ff", edgecolor="#0284c7", lw=0.8))

# Zone labels at top corners
ax1.text(1.95, 1.95, "Outer Fluid Space\n($d^* > R_{\\mathrm{dom}}$)", ha="right", va="top", fontsize=7.2, color="#475569")
ax1.text(-1.95, 1.95, "Fault Buffer Band\n($T_t < d^* \\leq R_{\\mathrm{dom}}$)", ha="left", va="top", fontsize=7.2, color="#c2410c")

for spine in ax1.spines.values():
    spine.set_color("#94a3b8")
    spine.set_linewidth(1.0)

ax1.set_xlim(-2.05, 2.05)
ax1.set_ylim(-2.05, 2.05)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title(r"(a) Concentric Manifolds: $T_t$ Basin & $R_{\mathrm{dom}}$ Domain")


# ==============================================================================
# PANEL (B): UNIFIED MACRO-ARCHITECTURE (Intermediate Layer l Early-Exit Flow)
# ==============================================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis("off")
ax2.set_title("(b) Architecture: Layer $l$ Gating & Early-Exit Flow")

# 1. Top Card: Fluid Reasoning Backbone (Layers 1..l)
box_p = patches.FancyBboxPatch((0.5, 6.75), 9.0, 2.85, boxstyle="round,pad=0.22",
                               facecolor="#f0f9ff", edgecolor="#0284c7", lw=1.8)
ax2.add_patch(box_p)
ax2.text(5.0, 9.25, "Polysemantic Reasoning Backbone ($N_p$)", ha="center", fontsize=9.6, color="#0369a1", fontweight="bold")
ax2.text(5.0, 8.65, r"$\bullet$ Contextual Transformer Layers $\mathcal{L}_{1 \dots l}$ (Syntactic Schema Formation)", ha="center", fontsize=8.0, color="#0c4a6e")
ax2.text(5.0, 8.10, r"$\bullet$ Intermediate State $h_t^{(l)} \in \mathbb{R}^D$ ($l=3$ in Pythia-70M, $D=512$)", ha="center", fontsize=8.0, color="#0c4a6e")
ax2.text(5.0, 7.55, r"$\bullet$ High Relational Rank $\mathcal{R}(u) \gg 0$: Isomorphic Abstraction ($\mathtt{DNS} \leftrightarrow \mathtt{IP} \leftrightarrow \mathtt{Phone}$)", ha="center", fontsize=7.8, color="#0369a1")
ax2.text(5.0, 7.05, r"$\bullet$ Normalized Latent Direction: $q_l = h_t^{(l)} / \|h_t^{(l)}\|_2$", ha="center", fontsize=7.8, color="#0284c7", fontweight="bold")

# 2. Middle Layer Gate: Metric Operator Q at Layer l
box_q = patches.FancyBboxPatch((3.1, 4.35), 3.8, 1.60, boxstyle="round,pad=0.18",
                               facecolor="#ffffff", edgecolor="#1e293b", lw=1.5, zorder=5)
ax2.add_patch(box_q)
ax2.text(5.0, 5.55, "Intermediate Layer $l$ Operator", ha="center", fontsize=8.2, color="#0f172a", fontweight="bold", zorder=6)
ax2.text(5.0, 5.05, r"$Q(q_l, \mathcal{F}_{\mathrm{mono}}) \to d^* = \min_k \|q_l - \mu_k\|_2$", ha="center", fontsize=7.4, color="#1e293b", zorder=6)
ax2.text(5.0, 4.58, r"Evaluates $T_t$ vs. $R_{\mathrm{dom}}$ Boundary", ha="center", fontsize=7.2, color="#475569", zorder=6)

# Flow: Backbone down to Operator Q
ax2.annotate("", xy=(5.0, 5.95), xytext=(5.0, 6.75),
             arrowprops=dict(arrowstyle="->", color="#0369a1", lw=2.0))
ax2.text(5.0, 6.35, "Probe $q_l$ at Layer $l$", ha="center", va="center", fontsize=7.2, color="#0369a1",
         bbox=dict(boxstyle="square,pad=0.15", facecolor="#ffffff", edgecolor="#0284c7", lw=0.6))

# Branch 1 (CONFIRMED): Green Right Path -> EARLY EXIT
box_conf = patches.FancyBboxPatch((7.4, 4.30), 2.15, 1.65, boxstyle="round,pad=0.18",
                                  facecolor="#f0fdf4", edgecolor="#16a34a", lw=1.2, zorder=5)
ax2.add_patch(box_conf)
ax2.text(8.47, 5.55, r"$\mathbf{CONFIRMED}$", ha="center", fontsize=8.0, color="#15803d", fontweight="bold", zorder=6)
ax2.text(8.47, 5.08, r"$d^* \leq T_t \to \mathbf{v}_{k^*}$", ha="center", fontsize=7.4, color="#15803d", zorder=6)
ax2.text(8.47, 4.60, "EARLY-EXIT", ha="center", fontsize=7.4, color="#166534", fontweight="bold", zorder=6)
ax2.text(8.47, 4.25, "(Bypasses L-l Layers)", ha="center", fontsize=6.6, color="#166534", zorder=6)

# Arrow from Q right to CONFIRMED
ax2.annotate("", xy=(7.4, 5.15), xytext=(6.9, 5.15),
             arrowprops=dict(arrowstyle="->", color="#16a34a", lw=2.0))

# Branch 2 (SEMANTIC FAULT): Red Left Path -> HALT
box_fault = patches.FancyBboxPatch((0.45, 4.30), 2.20, 1.65, boxstyle="round,pad=0.18",
                                   facecolor="#fef2f2", edgecolor="#dc2626", lw=1.2, zorder=5)
ax2.add_patch(box_fault)
ax2.text(1.55, 5.55, r"$\mathbf{SEMANTIC\;FAULT}$", ha="center", fontsize=7.8, color="#b91c1c", fontweight="bold", zorder=6)
ax2.text(1.55, 5.08, r"$T_t < d^* \leq R_{\mathrm{dom}}$", ha="center", fontsize=7.4, color="#991b1b", zorder=6)
ax2.text(1.55, 4.60, r"$\bot_{\mathrm{Fault}}$ Halted Mid-Net", ha="center", fontsize=7.2, color="#b91c1c", fontweight="bold", zorder=6)
ax2.text(1.55, 4.25, "(Blocks Hallucination)", ha="center", fontsize=6.6, color="#991b1b", zorder=6)

# Arrow from Q left to SEMANTIC FAULT
ax2.annotate("", xy=(2.65, 5.15), xytext=(3.1, 5.15),
             arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2.0))

# Branch 3 (FLUID BYPASS): Loop path up to remaining layers L_l+1..L
ax2.annotate("", xy=(5.0, 6.75), xytext=(6.5, 5.75),
             arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.5, ls="--"))
ax2.text(6.65, 6.15, r"$d^* > R_{\mathrm{dom}}$" + "\n(Fluid Flow)", fontsize=6.8, color="#0369a1", ha="left")

# 3. Bottom Card: Monosemantic Discrete Registry (Nm / Filesystem Inodes)
box_m = patches.FancyBboxPatch((0.5, 0.40), 9.0, 3.10, boxstyle="round,pad=0.22",
                               facecolor="#fefce8", edgecolor="#ca8a04", lw=1.8)
ax2.add_patch(box_m)
ax2.text(5.0, 3.15, "Monosemantic Discrete Inode Registry ($N_m$ / Filesystem)", ha="center", fontsize=9.6, color="#a16207", fontweight="bold")
ax2.text(5.0, 2.55, r"$\bullet$ Zero associative degrees: $\mathcal{R}(N_m) = 0$  [Factual Ground Truth Memory]", ha="center", fontsize=8.0, color="#713f12")
ax2.text(5.0, 2.00, r"$\bullet$ Indexed Inode Centroids: $\mathcal{F}_{\mathrm{mono}} = \{(\mu_1, \mathbf{v}_1), \dots, (\mu_M, \mathbf{v}_M)\}$ ($d_{\min} = 0.212$)", ha="center", fontsize=8.0, color="#713f12")
ax2.text(5.0, 1.45, r"$\bullet$ Certified Atomic Literals: $\mathtt{8.8.8.8}$ (DNS), $\mathtt{24}$ (CIDR), $\mathtt{JOHN}$ (User), $\mathtt{22/80}$ (Ports)", ha="center", fontsize=7.8, color="#854d0e")
ax2.text(5.0, 0.75, r"$\star$ Early-Exit Architecture: Neither Serial Nor Parallel --- Layer-l Geometric Interception",
         ha="center", fontsize=7.5, color="#92400e", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef3c7", edgecolor="#d97706", lw=0.8))

# Arrow connecting Nm up to Q
ax2.annotate("", xy=(5.0, 4.35), xytext=(5.0, 3.50),
             arrowprops=dict(arrowstyle="<->", color="#ca8a04", lw=1.8))
ax2.text(5.0, 3.90, r"Query Inodes $\{\mu_k\}$", ha="center", va="center", fontsize=7.0, color="#a16207",
         bbox=dict(boxstyle="square,pad=0.15", facecolor="#ffffff", edgecolor="#ca8a04", lw=0.6))


# ==============================================================================
# PANEL (C): EMPIRICAL BOUNDARY PROBING (Pythia-70M Residual Stream D=512)
# ==============================================================================
ax3 = fig.add_subplot(gs[0, 2])

labels = [
    "JOHN\n(Admin)",
    "24\n(CIDR)",
    "22\n(SSH)",
    "8.8.8.8\n(DNS)",
    "80\n(HTTP)",
    "Speleo\n(Unindexed)",
    "Sky\n(Creative)"
]
distances = [0.0889, 0.1263, 0.3257, 0.3889, 0.4304, 0.4973, 1.0430]
bar_colors = ["#16a34a", "#16a34a", "#16a34a", "#16a34a", "#16a34a", "#dc2626", "#0284c7"]
edge_colors = ["#14532d", "#14532d", "#14532d", "#14532d", "#14532d", "#7f1d1d", "#0369a1"]

# Background zones: Green (Confirmed Basin), Red/Amber (Fault Band), Blue (Fluid)
ax3.fill_between([-0.6, 6.6], 0, 0.46, color="#ecfdf5", alpha=0.80, zorder=0)
ax3.fill_between([-0.6, 6.6], 0.46, 0.65, color="#fff7ed", alpha=0.80, zorder=0)
ax3.fill_between([-0.6, 6.6], 0.65, 1.25, color="#f0f9ff", alpha=0.80, zorder=0)

# Threshold lines T_t = 0.46 and R_dom = 0.65
ax3.axhline(0.46, color="#2563eb", ls="--", lw=1.8, zorder=3)
ax3.text(1.2, 0.472, r"$T_t = 0.46$ (Inode Boundary)",
         color="#1d4ed8", fontsize=7.6, fontweight="bold", ha="center")

ax3.axhline(0.65, color="#ea580c", ls="--", lw=1.8, zorder=3)
ax3.text(3.5, 0.665, r"$R_{\mathrm{dom}} = 0.65$ (Domain Boundary)",
         color="#c2410c", fontsize=7.6, fontweight="bold", ha="center")

bars = ax3.bar(range(len(labels)), distances, color=bar_colors, edgecolor=edge_colors, width=0.58, lw=1.2, zorder=4)

# Value annotations
for idx, (bar, dist) in enumerate(zip(bars, distances)):
    if idx == 4:  # HTTP 0.430
        ax3.text(bar.get_x() + bar.get_width() / 2.0, dist - 0.055, f"{dist:.3f}",
                 ha="center", va="center", fontsize=8.0, fontweight="bold",
                 color="white", zorder=6)
    else:
        val_y = dist + 0.022
        col = "#15803d" if dist <= 0.46 else ("#b91c1c" if dist <= 0.65 else "#0369a1")
        ax3.text(bar.get_x() + bar.get_width() / 2.0, val_y, f"{dist:.3f}",
                 ha="center", va="bottom", fontsize=8.0, fontweight="bold",
                 color=col, zorder=5)

ax3.set_xticks(range(len(labels)))
ax3.set_xticklabels(labels, fontsize=8.0)
ax3.set_ylabel(r"Metric Distance on $S^{511}$ ($\|q_l - \mu_k\|_2$)")
ax3.set_ylim(0, 1.25)
ax3.set_xlim(-0.6, 6.6)
ax3.grid(axis="y", ls=":", alpha=0.5, zorder=1)

legend_handles = [
    patches.Patch(facecolor="#ecfdf5", edgecolor="#15803d", label=r"Confirmed ($d^* \leq T_t$)"),
    patches.Patch(facecolor="#fff7ed", edgecolor="#ea580c", label=r"Semantic Fault ($T_t < d^* \leq R_{\mathrm{dom}}$)"),
    patches.Patch(facecolor="#f0f9ff", edgecolor="#0284c7", label=r"Fluid $N_p$ ($d^* > R_{\mathrm{dom}}$)")
]
ax3.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98), framealpha=0.92, fontsize=7.2)
ax3.set_title("(c) Metric Boundary Probing (Pythia-70M, Layer 3)")

plt.subplots_adjust(left=0.035, right=0.985, top=0.91, bottom=0.15)

pdf_path = "papers/figures/fig3_hypersphere_geometry.pdf"
png_path = "assets/fig3_hypersphere_geometry.png"
obsidian_png = "obsidian/assets/fig3_hypersphere_geometry.png"

plt.savefig(pdf_path)
plt.savefig(png_path, dpi=300)
if os.path.isdir("obsidian/assets"):
    shutil.copy(png_path, obsidian_png)

plt.close(fig)

print(f"[✓] Success: Refined concentric fig3_hypersphere_geometry saved to {pdf_path} and {png_path}.")
