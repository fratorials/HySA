#!/usr/bin/env python3
"""
Generate Figure 3: Semantic Hypersphere Geometry, Unified Architecture, and Pythia-70M Boundary Probing.
Faithfully renders the sketches from Handwritten Blueprint Sheet 1 and Sheet 2.
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
    "axes.titlesize": 10.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "figure.titlesize": 11.5
})

fig = plt.figure(figsize=(15, 4.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.1, 1.05], wspace=0.32)

# ==============================================================================
# PANEL (A): SEMANTIC HYPERSPHERE GEOMETRY (Foglio 1 Blueprint)
# ==============================================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_aspect("equal")

# Center and radius
center = np.array([0.0, 0.0])
T_t = 1.0

# Hypersphere boundary
circle_outer = patches.Circle(center, 1.55, edgecolor="#cbd5e1", facecolor="#f8fafc", ls=":", lw=1.2, zorder=1)
circle_tt = patches.Circle(center, T_t, edgecolor="#2563eb", facecolor="#eff6ff", alpha=0.65, lw=2.0, ls="-", zorder=2)
ax1.add_patch(circle_outer)
ax1.add_patch(circle_tt)

# Inode Centroid mu_k (CAT)
ax1.scatter([0], [0], color="#1e3a8a", s=130, zorder=6, edgecolors="white", lw=1.5)
ax1.text(0.04, -0.15, r"$\mu_k$ (Centroid / Inode)" + "\nCAT", fontsize=8.5, fontweight="bold", color="#1e3a8a")

# Radial vectors A, B, C (from Foglio 1)
ray_a = np.array([0.62, 0.48])    # GATTO (inside / close)
ray_b = np.array([0.15, 0.88])    # KG (near boundary)
ray_c = np.array([-1.05, 0.95])   # MACCHINA (far outside)
q_interrogated = np.array([0.48, 0.32]) # Interrogated query point (inside)
q_fault = np.array([-0.90, -0.85])       # Interrogated query point (outside)

# Draw rays from center
for ray, label, col in [(ray_a, "Ray A: GATTO", "#059669"),
                        (ray_b, "Ray B: KG", "#d97706"),
                        (ray_c, "Ray C: MACCHINA", "#dc2626")]:
    ax1.annotate("", xy=ray, xytext=center, arrowprops=dict(arrowstyle="->", color=col, lw=1.5, ls="--"))
    ax1.scatter([ray[0]], [ray[1]], color=col, s=70, zorder=5)
    offset = np.array([0.08, 0.04]) if ray[0] >= 0 else np.array([-0.72, 0.04])
    ax1.text(ray[0] + offset[0], ray[1] + offset[1], label, fontsize=8.0, color=col, fontweight="bold")

# Interrogated query points
ax1.scatter([q_interrogated[0]], [q_interrogated[1]], color="#16a34a", marker="*", s=160, zorder=7, edgecolors="black", lw=0.8)
ax1.annotate(r"$q_{\mathrm{valid}}$ (Interrogated)" + "\n" + r"$d \leq T_t \rightarrow \mathrm{CONFIRMED}$",
             xy=q_interrogated, xytext=(0.35, -0.75),
             arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.4), fontsize=8.0, color="#15803d", fontweight="bold")

ax1.scatter([q_fault[0]], [q_fault[1]], color="#b91c1c", marker="X", s=130, zorder=7, edgecolors="black", lw=0.8)
ax1.annotate(r"$q_{\mathrm{drift}}$" + "\n" + r"$d > T_t \rightarrow \mathrm{SEMANTIC\;FAULT}$",
             xy=q_fault, xytext=(-1.45, -0.35),
             arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.4), fontsize=8.0, color="#b91c1c", fontweight="bold")

# Radius marker T_t
ax1.plot([0, T_t * np.cos(np.pi/4)], [0, T_t * np.sin(np.pi/4)], color="#2563eb", lw=1.8)
ax1.text(0.38, 0.52, r"Radius $T_t(T_p, T_m)$", fontsize=8.5, color="#1d4ed8", rotation=45, fontweight="bold")

ax1.set_xlim(-1.6, 1.6)
ax1.set_ylim(-1.6, 1.6)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title(r"(a) Semantic Hypersphere $\mathcal{B}(\mu_k, T_t)$")

# ==============================================================================
# PANEL (B): UNIFIED MACRO-ARCHITECTURE (Neither Serial Nor Parallel)
# ==============================================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(10)
ax2.axis("off")
ax2.set_title(r"(b) Forward & Verification Architecture")

# Box 1: Polysemantic Network Np
box_p = patches.FancyBboxPatch((0.5, 5.8), 9.0, 3.4, boxstyle="round,pad=0.3",
                               facecolor="#f0f9ff", edgecolor="#0284c7", lw=1.8)
ax2.add_patch(box_p)
ax2.text(5.0, 8.6, "Polysemantic Relational Manifold ($N_p$)", ha="center", fontsize=9.2, color="#0369a1", fontweight="bold")
ax2.text(5.0, 7.9, r"$\cdot$ High relational connectivity (Generates Intelligence)", ha="center", fontsize=8.0, color="#0c4a6e")
ax2.text(5.0, 7.3, r"$\cdot$ Continuous Latent Trajectories $h_t \in \mathbb{R}^D$ ($D=512\dots4096$)", ha="center", fontsize=8.0, color="#0c4a6e")
ax2.text(5.0, 6.7, r"$\cdot$ Governed by Polysemantic Temperature $T_p$ (Exploration)", ha="center", fontsize=8.0, color="#0369a1")

# Arrow downwards: On-demand Metric Query API
ax2.annotate("", xy=(5.0, 4.3), xytext=(5.0, 5.6),
             arrowprops=dict(arrowstyle="->", color="#475569", lw=2.0))
ax2.text(5.15, 4.95, r"Metric Query Operator $Q(h_t, \mathcal{F}_{\mathrm{mono}}, T_t)$",
         ha="left", va="center", fontsize=8.2, color="#1e293b", fontweight="bold",
         bbox=dict(boxstyle="square,pad=0.15", facecolor="#ffffff", edgecolor="#cbd5e1", lw=0.8))

# Box 2: Monosemantic Registry Nm
box_m = patches.FancyBboxPatch((0.5, 0.7), 9.0, 3.3, boxstyle="round,pad=0.3",
                               facecolor="#fefce8", edgecolor="#ca8a04", lw=1.8)
ax2.add_patch(box_m)
ax2.text(5.0, 3.4, "Monosemantic Discrete Registry ($N_m$ / Filesystem)", ha="center", fontsize=9.2, color="#a16207", fontweight="bold")
ax2.text(5.0, 2.7, r"$\cdot$ Zero associative degrees ($\emptyset$ Intelligence, 100\% Rigor)", ha="center", fontsize=8.0, color="#713f12")
ax2.text(5.0, 2.1, r"$\cdot$ Indexed Inodes $\mathcal{F}_{\mathrm{mono}} = \{\mu_1, \dots, \mu_M\}$, Cardinality $M$", ha="center", fontsize=8.0, color="#713f12")
ax2.text(5.0, 1.5, r"$\cdot$ Immutable Atomic Literals: 8.8.8.8, 24, JOHN, Port 22/80", ha="center", fontsize=8.0, color="#854d0e")

# Return confirmation arrow
ax2.annotate("", xy=(8.6, 5.8), xytext=(8.6, 4.0),
             arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.8, ls="--"))
ax2.text(8.75, 4.85, "CONFIRMED\n(Certified Literal)", ha="left", va="center", fontsize=7.6, color="#15803d", fontweight="bold")

# Fault arrow
ax2.annotate("", xy=(1.4, 4.8), xytext=(1.4, 5.6),
             arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.8, ls="--"))
ax2.text(1.25, 4.85, "SEMANTIC\nFAULT\n($d > T_t$)", ha="right", va="center", fontsize=7.6, color="#b91c1c", fontweight="bold")


# ==============================================================================
# PANEL (C): EMPIRICAL BOUNDARY VERIFICATION (Pythia-70M Residual Stream D=512)
# ==============================================================================
ax3 = fig.add_subplot(gs[0, 2])

labels = [
    "JOHN\n(Admin)",
    "24\n(CIDR)",
    "22\n(SSH)",
    "8.8.8.8\n(DNS IP)",
    "80\n(HTTP)",
    "Telemetry\n(Unindexed)"
]
distances = [0.0889, 0.1263, 0.3257, 0.3889, 0.4304, 0.4973]
bar_colors = ["#16a34a", "#16a34a", "#16a34a", "#16a34a", "#16a34a", "#dc2626"]

bars = ax3.bar(range(len(labels)), distances, color=bar_colors, width=0.6, edgecolor="black", lw=0.8, alpha=0.85)

# Threshold line T_t = 0.46
ax3.axhline(0.46, color="#2563eb", ls="--", lw=1.8, label=r"Topological Threshold $T_t = 0.46$")
ax3.fill_between([-0.6, 5.6], 0, 0.46, color="#16a34a", alpha=0.06)
ax3.fill_between([-0.6, 5.6], 0.46, 0.60, color="#dc2626", alpha=0.06)

# Value annotations on bars
for idx, (bar, dist) in enumerate(zip(bars, distances)):
    ax3.text(bar.get_x() + bar.get_width() / 2.0, dist + 0.015, f"{dist:.3f}",
             ha="center", va="bottom", fontsize=7.5, fontweight="bold",
             color="#16a34a" if dist <= 0.46 else "#dc2626")

ax3.set_xticks(range(len(labels)))
ax3.set_xticklabels(labels, fontsize=7.5)
ax3.set_ylabel(r"Euclidean Distance on $S^{511}$ ($\|q - \mu_k\|_2$)")
ax3.set_ylim(0, 0.58)
ax3.set_xlim(-0.6, 5.6)
ax3.grid(axis="y", ls=":", alpha=0.4)
ax3.set_title(r"(c) Metric Boundary Probing (Pythia-70M, Layer 3)")

plt.savefig("fig3_hypersphere_geometry.pdf", bbox_inches="tight")
plt.savefig("fig3_hypersphere_geometry.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("[✓] Success: fig3_hypersphere_geometry.pdf and .png created successfully.")
