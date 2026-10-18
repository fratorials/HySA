#!/usr/bin/env python3
"""
Generate Figure 3: Semantic Hypersphere Geometry, Unified Architecture, and Pythia-70M Boundary Probing.
High-precision typographical redesign matching NeurIPS / Nature / ICLR standards.
All labels formatted strictly on a single line (no multi-line line breaks) for maximum readability.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Publication-grade typographical settings
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "figure.titlesize": 12.0
})

fig = plt.figure(figsize=(17.0, 5.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.38, 1.15], wspace=0.25)

# ==============================================================================
# PANEL (A): SEMANTIC HYPERSPHERE GEOMETRY (Foglio 1 Blueprint)
# ==============================================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_aspect("equal")

center = np.array([0.0, 0.0])
T_t = 1.0

# 1. Subtle orthogonal latent coordinate axes
ax1.axhline(0, color="#cbd5e1", lw=0.9, ls="--", zorder=1)
ax1.axvline(0, color="#cbd5e1", lw=0.9, ls="--", zorder=1)
ax1.text(1.70, 0.04, r"$z_1$", fontsize=8.5, color="#64748b", fontweight="bold")
ax1.text(0.04, 1.70, r"$z_2$", fontsize=8.5, color="#64748b", fontweight="bold")

# 2. Concentric metric circles and basin shading
# Semantic Fault region background
bg_fault = patches.Rectangle((-1.95, -1.95), 3.90, 3.90, facecolor="#fef2f2", alpha=0.5, zorder=0)
ax1.add_patch(bg_fault)

# Inner Confirmed Basin: B(mu_k, T_t)
c_basin = patches.Circle(center, T_t, edgecolor="#2563eb", facecolor="#eff6ff", alpha=0.95, lw=2.2, zorder=2)
ax1.add_patch(c_basin)

# Concentric guide circles (0.5 T_t and 1.5 T_t)
for r_metric, ls, col in [(0.5, ":", "#93c5fd"), (1.5, ":", "#fca5a5")]:
    c = patches.Circle(center, r_metric, edgecolor=col, facecolor="none", lw=1.1, ls=ls, zorder=2)
    ax1.add_patch(c)

# Radius T_t dimension vector with callout
theta_dim = np.pi * 0.72
rad_pt = np.array([T_t * np.cos(theta_dim), T_t * np.sin(theta_dim)])
ax1.annotate("", xy=rad_pt, xytext=center, arrowprops=dict(arrowstyle="<->", color="#1d4ed8", lw=1.8))
ax1.text(-0.52, 0.44, r"Radius $T_t(T_p, T_m)$", fontsize=8.2, color="#1e40af", fontweight="bold", rotation=50)

# 3. Inode Centroid mu_k (Concept: CAT / GATTO)
ax1.scatter([0], [0], color="#1e3a8a", s=180, zorder=7, edgecolors="white", lw=2.0)
ax1.scatter([0], [0], color="none", s=280, zorder=6, edgecolors="#1e3a8a", lw=1.2, ls="--")
# Single-line centroid badge
ax1.text(0.18, 0.0, r"$\mathbf{Inode}\;\mu_{\mathrm{cat}}\;(\mathrm{Centroid})$",
         ha="left", va="center", fontsize=8.2, fontweight="bold", color="#1e3a8a",
         bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#93c5fd", alpha=0.92, lw=0.7))

# 4. Radial concept vectors A, B, C (Directly from Foglio 1) - ALL SINGLE LINE
ray_a = np.array([0.65, 0.38])    # GATTO (Synonym, inside)
ray_b = np.array([0.16, 0.88])    # KG (Attribute/Weight, near boundary)
ray_c = np.array([-0.90, 0.82])   # MACCHINA (Distant vehicle, far outside)

# Draw rays from center with single-line pill labels
for ray, label, col, badge_pos, halign, valign in [
    (ray_a, "Ray A: GATTO (Synonym, d=0.75)", "#059669", (0.75, 0.38), "left", "center"),
    (ray_b, "Ray B: KG (Attribute, d=0.89)", "#d97706", (0.22, 1.00), "left", "bottom"),
    (ray_c, "Ray C: MACCHINA (Distant, d=1.22)", "#dc2626", (-0.85, 1.02), "center", "bottom")
]:
    ax1.annotate("", xy=ray, xytext=center, arrowprops=dict(arrowstyle="->", color=col, lw=1.8))
    ax1.scatter([ray[0]], [ray[1]], color=col, s=75, zorder=6, edgecolors="white", lw=1.2)
    ax1.text(badge_pos[0], badge_pos[1], label, fontsize=7.6, color=col, fontweight="bold",
             ha=halign, va=valign,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor=col, alpha=0.95, lw=0.9))

# 5. Query point: Interrogated (inside confirmed basin) - SINGLE LINE
q_inter = np.array([0.42, -0.42])
d_inter = np.linalg.norm(q_inter)
ax1.plot([0, q_inter[0]], [0, q_inter[1]], color="#16a34a", lw=1.4, ls=":", zorder=5)
ax1.scatter([q_inter[0]], [q_inter[1]], color="#16a34a", marker="*", s=210, zorder=8, edgecolors="black", lw=0.8)
ax1.text(0.38, -0.28, r"$d_1 = 0.59$", fontsize=7.2, color="#15803d", fontweight="bold", rotation=-45)

ax1.annotate(r"$q_{\mathrm{valid}}$: $\mathbf{CONFIRMED}$ ($d = 0.59 \leq T_t$)",
             xy=q_inter, xytext=(0.08, -0.92),
             arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.5), fontsize=8.0, color="#15803d", fontweight="bold",
             bbox=dict(boxstyle="square,pad=0.22", facecolor="#f0fdf4", edgecolor="#16a34a", lw=0.9))

# 6. Query point: Drift / Semantic Fault (outside boundary) - SINGLE LINE
q_fault = np.array([-0.88, -0.88])
d_fault = np.linalg.norm(q_fault)
ax1.plot([0, q_fault[0]], [0, q_fault[1]], color="#b91c1c", lw=1.4, ls=":", zorder=5)
ax1.scatter([q_fault[0]], [q_fault[1]], color="#b91c1c", marker="X", s=150, zorder=8, edgecolors="black", lw=0.8)
ax1.text(-0.52, -0.46, r"$d_2 = 1.24$", fontsize=7.2, color="#b91c1c", fontweight="bold", rotation=45)

ax1.annotate(r"$q_{\mathrm{drift}}$: $\mathbf{SEMANTIC\;FAULT}$ ($d = 1.24 > T_t$)",
             xy=q_fault, xytext=(-0.70, -1.35), ha="center",
             arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.5), fontsize=8.0, color="#b91c1c", fontweight="bold",
             bbox=dict(boxstyle="square,pad=0.22", facecolor="#fef2f2", edgecolor="#b91c1c", lw=0.9))

# Soft card border
for spine in ax1.spines.values():
    spine.set_color("#94a3b8")
    spine.set_linewidth(1.0)

ax1.set_xlim(-1.95, 1.95)
ax1.set_ylim(-1.95, 1.95)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title(r"(a) Semantic Hypersphere Metric Ball $\mathcal{B}(\mu_k, T_t)$")


# ==============================================================================
# PANEL (B): UNIFIED MACRO-ARCHITECTURE (Neither Serial Nor Parallel)
# ==============================================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis("off")
ax2.set_title("(b) Unified Architecture: Forward & Verification Flow")

# 1. Top Card: Polysemantic Relational Engine (Np)
box_p = patches.FancyBboxPatch((0.5, 6.4), 9.0, 3.2, boxstyle="round,pad=0.25",
                               facecolor="#f0f9ff", edgecolor="#0284c7", lw=1.8)
ax2.add_patch(box_p)
ax2.text(5.0, 9.20, "Polysemantic Relational Engine ($N_p$)", ha="center", fontsize=9.8, color="#0369a1", fontweight="bold")
ax2.text(5.0, 8.55, r"$\bullet$ High relational connectivity: $\mathcal{R}(N_p) \gg 0$  [Associative Intelligence]", ha="center", fontsize=8.2, color="#0c4a6e")
ax2.text(5.0, 7.95, r"$\bullet$ Continuous Latent Trajectory $h_t \in \mathcal{S} \subset \mathbb{R}^D$ ($D = 512 \dots 4096$)", ha="center", fontsize=8.2, color="#0c4a6e")
ax2.text(5.0, 7.35, r"$\bullet$ Co-activates isomorphic schemas: $\mathtt{IP} \leftrightarrow \mathtt{DNS} \leftrightarrow \mathtt{White\;Pages} \leftrightarrow \mathtt{Phone}$", ha="center", fontsize=8.0, color="#0369a1")
ax2.text(5.0, 6.75, r"$\bullet$ Governed by Polysemantic Temperature $T_p$ (Exploratory Brownian Sampling)", ha="center", fontsize=8.0, color="#0284c7")

# 2. Middle Gateway: Internal Metric Inode Quantizer & Verification Operator Q
box_q = patches.FancyBboxPatch((3.1, 4.35), 3.8, 1.35, boxstyle="round,pad=0.2",
                               facecolor="#ffffff", edgecolor="#1e293b", lw=1.5, zorder=5)
ax2.add_patch(box_q)
ax2.text(5.0, 5.25, "Metric Verification Operator", ha="center", fontsize=8.2, color="#0f172a", fontweight="bold", zorder=6)
ax2.text(5.0, 4.65, r"$Q(h_t, \mathcal{F}_{\mathrm{mono}}, T_t) = [d^* = \min_k \|h_t - \mu_k\|_2]$", ha="center", fontsize=7.5, color="#1e293b", zorder=6)

# Flow 1: Np down to Operator Q (Candidate token)
ax2.annotate("", xy=(5.0, 5.70), xytext=(5.0, 6.40),
             arrowprops=dict(arrowstyle="->", color="#0369a1", lw=2.0))
ax2.text(5.0, 6.05, "Candidate $h_t$ (On-Demand Query)", ha="center", va="center", fontsize=7.2, color="#0369a1",
         bbox=dict(boxstyle="square,pad=0.15", facecolor="#ffffff", edgecolor="#0284c7", lw=0.6))

# Flow 2: Operator Q down to Monosemantic Registry (Distance probe)
ax2.annotate("", xy=(5.0, 3.45), xytext=(5.0, 4.35),
             arrowprops=dict(arrowstyle="->", color="#a16207", lw=1.8))
ax2.text(5.0, 3.90, r"Query Inodes $\{\mu_k\}$", ha="center", va="center", fontsize=7.2, color="#a16207",
         bbox=dict(boxstyle="square,pad=0.15", facecolor="#ffffff", edgecolor="#ca8a04", lw=0.6))

# Flow 3: Branch A (CONFIRMED) -> Green path from Nm returning literal v_k* back into Np
box_conf = patches.FancyBboxPatch((7.4, 4.25), 2.1, 1.55, boxstyle="round,pad=0.18",
                                  facecolor="#f0fdf4", edgecolor="#16a34a", lw=1.2, zorder=5)
ax2.add_patch(box_conf)
ax2.text(8.45, 5.35, r"$\mathbf{CONFIRMED}$", ha="center", fontsize=7.8, color="#15803d", fontweight="bold", zorder=6)
ax2.text(8.45, 4.88, r"$d^* \leq T_t \rightarrow \mathbf{v}_k^*$", ha="center", fontsize=7.4, color="#15803d", zorder=6)
ax2.text(8.45, 4.40, "Certified Literal Injected", ha="center", fontsize=7.0, color="#166534", zorder=6)

# Arrow from Nm up to CONFIRMED
ax2.annotate("", xy=(8.45, 4.25), xytext=(8.45, 3.45),
             arrowprops=dict(arrowstyle="->", color="#16a34a", lw=2.0))
# Arrow from CONFIRMED up into Np
ax2.annotate("", xy=(8.45, 6.40), xytext=(8.45, 5.80),
             arrowprops=dict(arrowstyle="->", color="#16a34a", lw=2.0))

# Flow 4: Branch B (SEMANTIC FAULT) -> Red exit path to the left (Halting / Exception)
box_fault = patches.FancyBboxPatch((0.5, 4.25), 2.1, 1.55, boxstyle="round,pad=0.18",
                                   facecolor="#fef2f2", edgecolor="#dc2626", lw=1.2, zorder=5)
ax2.add_patch(box_fault)
ax2.text(1.55, 5.35, r"$\mathbf{SEMANTIC\;FAULT}$", ha="center", fontsize=7.8, color="#b91c1c", fontweight="bold", zorder=6)
ax2.text(1.55, 4.88, r"$d^* > T_t \rightarrow \bot_{\mathrm{Fault}}$", ha="center", fontsize=7.4, color="#991b1b", zorder=6)
ax2.text(1.55, 4.40, "Hallucination Halted", ha="center", fontsize=7.0, color="#b91c1c", fontstyle="italic", zorder=6)

# Arrow from Q pointing LEFT to SEMANTIC FAULT
ax2.annotate("", xy=(2.6, 5.0), xytext=(3.1, 5.0),
             arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2.0))

# 3. Bottom Card: Monosemantic Discrete Registry (Nm / Filesystem Inodes)
box_m = patches.FancyBboxPatch((0.5, 0.35), 9.0, 3.1, boxstyle="round,pad=0.25",
                               facecolor="#fefce8", edgecolor="#ca8a04", lw=1.8)
ax2.add_patch(box_m)
ax2.text(5.0, 3.10, "Monosemantic Discrete Registry ($N_m$ / Filesystem Inodes)", ha="center", fontsize=9.8, color="#a16207", fontweight="bold")
ax2.text(5.0, 2.50, r"$\bullet$ Zero associative degrees: $\mathcal{R}(N_m) = 0$  [Deterministic Ground Truth Memory]", ha="center", fontsize=8.2, color="#713f12")
ax2.text(5.0, 1.95, r"$\bullet$ Metric Inode Manifold: $\mathcal{F}_{\mathrm{mono}} = \{\mu_1, \dots, \mu_M\}$, Cardinality $M$ ($d_{\min} = 0.212$)", ha="center", fontsize=8.2, color="#713f12")
ax2.text(5.0, 1.40, r"$\bullet$ Certified Atomic Literals: $\mathtt{8.8.8.8}$ (DNS), $\mathtt{24}$ (CIDR), $\mathtt{JOHN}$ (User), $\mathtt{22/80}$ (Ports)", ha="center", fontsize=8.0, color="#854d0e")
ax2.text(5.0, 0.70, r"$\star$ Hybrid Paradigm: Neither Serial Nor Parallel --- Event-Driven Interrogation",
         ha="center", fontsize=7.6, color="#92400e", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef3c7", edgecolor="#d97706", lw=0.8))


# ==============================================================================
# PANEL (C): EMPIRICAL BOUNDARY VERIFICATION (Pythia-70M Residual Stream D=512)
# ==============================================================================
ax3 = fig.add_subplot(gs[0, 2])

# ALL X-TICK LABELS STRICTLY ON A SINGLE LINE (NO '\n')
labels = [
    "JOHN (Admin)",
    "24 (CIDR)",
    "22 (SSH)",
    "8.8.8.8 (DNS)",
    "80 (HTTP)",
    "Telemetry (Drift)"
]
distances = [0.0889, 0.1263, 0.3257, 0.3889, 0.4304, 0.4973]
bar_colors = ["#16a34a", "#16a34a", "#16a34a", "#16a34a", "#16a34a", "#dc2626"]
edge_colors = ["#14532d", "#14532d", "#14532d", "#14532d", "#14532d", "#7f1d1d"]

# Background zones: Green (Confirmed Basin) and Red (Semantic Fault Zone)
ax3.fill_between([-0.6, 5.6], 0, 0.46, color="#ecfdf5", alpha=0.75, zorder=0)
ax3.fill_between([-0.6, 5.6], 0.46, 0.71, color="#fff1f2", alpha=0.75, zorder=0)

# Threshold line T_t = 0.46
ax3.axhline(0.46, color="#2563eb", ls="--", lw=2.2, zorder=3)
ax3.text(2.0, 0.472, r"$\mathbf{Topological\;Boundary}\;T_t = 0.46 \quad (\Delta_{\mathrm{margin}} = 0.067)$",
         color="#1d4ed8", fontsize=8.0, fontweight="bold", ha="center")

# Bars with rounded aesthetics & crisp edges
bars = ax3.bar(range(len(labels)), distances, color=bar_colors, edgecolor=edge_colors, width=0.55, lw=1.2, zorder=4)

# Value annotations on bars
for idx, (bar, dist) in enumerate(zip(bars, distances)):
    if idx == 4:  # HTTP 0.430: place inside bar in bold white to keep clean separation with T_t line
        ax3.text(bar.get_x() + bar.get_width() / 2.0, dist - 0.052, f"{dist:.3f}",
                 ha="center", va="center", fontsize=8.2, fontweight="bold",
                 color="white", zorder=6)
    else:
        val_y = dist + 0.016
        ax3.text(bar.get_x() + bar.get_width() / 2.0, val_y, f"{dist:.3f}",
                 ha="center", va="bottom", fontsize=8.2, fontweight="bold",
                 color="#15803d" if dist <= 0.46 else "#b91c1c", zorder=5)

# Grouping brackets at the top
# Bracket 1: Indexed Inodes (0 to 4) - Single Line
ax3.plot([-0.25, 4.25], [0.655, 0.655], color="#15803d", lw=1.2)
ax3.plot([-0.25, -0.25], [0.640, 0.655], color="#15803d", lw=1.2)
ax3.plot([4.25, 4.25], [0.640, 0.655], color="#15803d", lw=1.2)
ax3.text(2.0, 0.665, r"$\mathbf{Indexed\;Monosemantic\;Inodes}\;(\mathcal{F}_{\mathrm{mono}})\;\mathbf{[Verified]}$",
         ha="center", va="bottom", fontsize=7.4, color="#15803d", fontweight="bold")

# Bracket 2: Out of Distribution (5) - Single Line
ax3.plot([4.75, 5.25], [0.655, 0.655], color="#b91c1c", lw=1.2)
ax3.plot([4.75, 4.75], [0.640, 0.655], color="#b91c1c", lw=1.2)
ax3.plot([5.25, 5.25], [0.640, 0.655], color="#b91c1c", lw=1.2)
ax3.text(5.0, 0.665, r"$\mathbf{Drift}\;\mathbf{[\bot_{\mathrm{Fault}}]}$",
         ha="center", va="bottom", fontsize=7.4, color="#b91c1c", fontweight="bold")

# Clean single-line tick labels with sleek 18-degree angle for 100% legibility
ax3.set_xticks(range(len(labels)))
ax3.set_xticklabels(labels, fontsize=8.2, rotation=18, ha="right", rotation_mode="anchor")
ax3.set_ylabel(r"Euclidean Metric Distance on $S^{511}$ ($\|q - \mu_k\|_2$)")
ax3.set_ylim(0, 0.71)
ax3.set_xlim(-0.6, 5.6)
ax3.grid(axis="y", ls=":", alpha=0.5, zorder=1)

# Compact elegant legend placed in upper left
legend_handles = [
    patches.Patch(facecolor="#ecfdf5", edgecolor="#15803d", label=r"Confirmed Basin ($d \leq T_t$)"),
    patches.Patch(facecolor="#fff1f2", edgecolor="#b91c1c", label=r"Semantic Fault ($d > T_t$)")
]
ax3.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.90), framealpha=0.92, fontsize=7.4)
ax3.set_title("(c) Metric Boundary Probing (Pythia-70M, Layer 3)")

plt.subplots_adjust(left=0.035, right=0.985, top=0.91, bottom=0.15)
plt.savefig("fig3_hypersphere_geometry.pdf")
plt.savefig("fig3_hypersphere_geometry.png", dpi=300)
plt.close(fig)

print("[✓] Success: Refined fig3_hypersphere_geometry with single-line labels created.")
