# HySA: Hybrid Semantic Architecture

> **Decoupling Fluid Reasoning from Deterministic Verification via Neuronal Semanticity and Inode Indexing**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23459635-blue.svg)](https://doi.org/10.5281/zenodo.23459635)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--1319--3573-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0008-1319-3573)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Paper: 9-Page Extended](https://img.shields.io/badge/Paper-9--Page%20Extended%20Edition-success.svg)](papers/extended/HySA_Extended_EN.pdf)
[![Paper: 8-Page Two-Column](https://img.shields.io/badge/Paper-8--Page%20Two--Column-purple.svg)](papers/extended/HySA_Extended_TwoColumn_EN.pdf)
[![Paper: 3-Page Condensed](https://img.shields.io/badge/Paper-3--Page%20Condensed%20Edition-blue.svg)](papers/condensed/HySA_Short_EN.pdf)

---

> *"The AI industry views polysemanticity as a curse... yet perhaps it is the very blessing that made intelligence possible."*

Official open-source repository and codebase for the research paper:  
**"HySA (Hybrid Semantic Architecture): Decoupling Fluid Reasoning from Deterministic Verification via Neuronal Semanticity and Inode Indexing"**  
*Author:* **Francesco Mariani** (*Independent Systems Architect*, Umbria, Italy — `fratorials@gmail.com`)  
*ORCID:* [0009-0008-1319-3573](https://orcid.org/0009-0008-1319-3573)  
*DOI:* [10.5281/zenodo.23459635](https://doi.org/10.5281/zenodo.23459635)

---

## ⚡ Executive Summary & Paradigm Shift

Contemporary research in **Mechanistic Interpretability** predominantly characterizes neuronal polysemanticity as an undesirable artifact of high-dimensional compression (*the Superposition Hypothesis*), dedicating immense computational resources to decomposing neural representations into dictionaries of purely monosemantic directions using Sparse Autoencoders (SAEs).

**HySA proposes a foundational paradigm shift:**  
**Polysemanticity is not a flaw to be eradicated, but the geometric prerequisite for associative intelligence and analogical reasoning.**

A purely monosemantic neuron possesses zero associative degrees of freedom ($\emptyset$ relational intelligence): devoid of lateral relational pathways, it functions as a rigid, static lookup cell. Conversely, polysemantic units co-activate across isomorphic concepts (e.g., `IP Address`, `DNS`, `White Pages`, `Phone Directory`) to compress disparate instances into higher-order structural schemas. 

| Neuronal Regime | Epistemological Role | Representative Entities | Geometric Property |
| :--- | :--- | :--- | :--- |
| **Polysemantic Network ($N_p$)** | **Intelligence & Analogy** | `IP`, `DNS`, `White Pages`, `Phone Number` | High relational rank $\mathcal{R}(u) \gg 0$; compresses diverse entities into unified structural schemas. |
| **Monosemantic Registry ($N_m$)** | **Memory & Verification** | `8.8.8.8`, `24`, `JOHN`, `Port 53` | Relational rank $\mathcal{R}(u) = 0$; orthogonal atomic anchors guaranteeing $100\%$ factual determinism. |

Rather than coupling $N_p$ and $N_m$ in **series** (which creates a non-abstractive bottleneck that collapses contextual nuance) or in **parallel** (which causes $2\times$ FLOPs overhead and requires heuristic arbitration heads), **HySA integrates both into a Unified Single-Substrate Architecture** operating directly in latent Hilbert space $\mathcal{S} \subset \mathbb{R}^D$.

---

## 🧭 Systems Heuristics vs. Formal Hilbert Geometry

HySA borrows foundational heuristics from operating system (OS) kernel design and formalizes them into rigorous geometric constructs:

| Systems Term | Formal Mathematical Construct in HySA | Architectural Function in HySA |
| :--- | :--- | :--- |
| **RAM / Cache** | Fluid Polysemantic Subspace $\mathcal{S}_p \subset \mathbb{R}^D$ | High relational rank ($N_p$), associative synthesis, continuous analogical reasoning. |
| **Disk / ROM** | Rigid Monosemantic Manifold $\mathcal{S}_m \subset \mathbb{R}^D$ | Zero relational rank ($N_m$), deterministic storage of immutable literals. |
| **Inode ($N_m / \mu_k$)** | Centroid $\mu_k \in S^{D-1} \subset \mathbb{R}^D$ | Orthogonal metric anchor with $\emptyset$ associative degrees, bound to literal $\mathbf{v}_k$. |
| **Filesystem** | Metric Registry $`\mathcal{F}_{\mathrm{mono}} = \{\mu_1, \dots, \mu_M\}`$ | Indexed registry ($100\%$ precision) evaluated on-demand during decoding. |
| **Query / API** | Boundary Operator $`Q(q_l, \mathcal{F}_{\mathrm{mono}}, T_t, R_{\mathrm{dom}})`$ | Geometric gating at intermediate layer $l$ projecting onto literals or halting hallucinations. |
| **Page Fault** | **Semantic Fault ($`\bot_{\mathrm{Fault}}`$)** | Explicit exception raised when $`T_t \lt d^* \le R_{\mathrm{dom}}`$, halting ungrounded hallucinations. |
| **Temperature** | Metric Neighborhood $\mathcal{B}(\mu_k, T_t)$ | Multidimensional metric volume of tolerance around a concept centroid ($`T_t \lt R_{\mathrm{dom}}`$). |

---

## 🔬 Mathematical Architecture

### 1. Contrastive Superposition Tensors ($I_n$)
A polysemantic unit's multi-concept polytope is extracted as the whole-network inference expectation across contrastive data ($\mathcal{X}^+$ vs. $\mathcal{X}^-$):

```math
I_n = \frac{1}{\sum_{X_k \in \mathcal{X}^+} z_{k,n} + \eta} \left[ \sum_{X_k \in \mathcal{X}^+} z_{k,n} \cdot \Phi_{\mathrm{net}}(X_k \mid n) - \gamma \sum_{X_j \in \mathcal{X}^-} z_{j,n} \cdot \Phi_{\mathrm{net}}(X_j \mid n) \right]
```

Algebraic operations are directly supported over feature tensors:
* **Shared Structural Invariant:** $I_{\mathrm{shared}} = I_A \odot I_B$ (Hadamard product)
* **Differential Feature:** $I_{\mathrm{diff}} = \mathrm{ReLU}(I_A - \gamma I_B)$

### 2. The Semantic Hypersphere & Topological Temperature ($T_t$)
Temperature is generalized geometrically as the metric ball $\mathcal{B}(\mu_k, T_t)$ in $\mathcal{S} \subset \mathbb{R}^D$:

```math
\mathcal{B}(\mu_k, T_t) = \left\{ x \in \mathcal{S} \;\middle|\; \| x - \mu_k \|_2 \le T_t(T_p, T_m) \right\}
```

Regimes are unified via bounded saturating coupling:

```math
T_t(T_p, T_m) = T_m \left( 1 + \alpha \tanh\left( \frac{T_p}{T_m} \right) \right) \quad \text{with } T_t \lt R_{\mathrm{dom}}
```

On the unit sphere $S^{D-1}$, Euclidean distance $\ell_2$ is in exact bijection with spherical angular divergence:

```math
\|x - \mu_k\|_2 = \sqrt{2(1 - \cos\theta)}
```

### 3. Intermediate Layer Geometric Gating ($Q$) & Concentric Regimes
Rather than relying on uncertainty (Shannon entropy) over final logits—which fails symmetrically on open creative discourse (spurious faults) and overconfident hallucinations (silent misses)—HySA evaluates the geometry of normalized residual states $q_l = h_t^{(l)} / \|h_t^{(l)}\|_2$ at an **intermediate layer $l$** (e.g., Layer 3 in a 6-layer architecture, where semantic context is formed but lexical token selection is uncommitted).

Attending to nearest Inode centroid $\mu_{k^*}$ at distance $d^* = \min_k \|q_l - \mu_k\|_2$, the boundary operator enforces three exact regimes:

```math
Q(q_l, \mathcal{F}_{\mathrm{mono}}, T_t, R_{\mathrm{dom}}) = \begin{cases} 
  (\mathrm{CONFIRMED}, \mathbf{v}_{k^*}) & \text{if } d^* \le T_t \quad (\text{Early-Exit}) \\ 
  (\mathrm{FAULT}, \bot_{\mathrm{Fault}}) & \text{if } T_t \lt d^* \le R_{\mathrm{dom}} \quad (\text{Semantic Fault}) \\
  (\mathrm{BYPASS}, \mathrm{Fluid}) & \text{if } d^* \gt R_{\mathrm{dom}} \quad (\text{Fluid Regime})
\end{cases}
```

1. **Inode Hit ($`d^* \le T_t`$):** State falls within the certified Inode core. Execution immediately **Early-Exits**, bypassing subsequent layers $l+1 \dots L$ and emitting certified literal $\mathbf{v}_{k^*}$ (100% precision, ~50% FLOPs saved).
2. **Semantic Fault ($`T_t \lt d^* \le R_{\mathrm{dom}}`$):** State enters the factual domain basin, but the required entity is unindexed. The API intercepts execution mid-network, raising $`\bot_{\mathrm{Fault}}`$ and halting both stochastic guessing and overconfident hallucinations.
3. **Fluid Regime ($`d^* \gt R_{\mathrm{dom}}`$):** State navigates open creative, poetic, or abstract discourse far from factual Inodes. The API remains dormant, allowing standard propagation through remaining layers and Softmax sampling at temperature $T_p$.

### 4. Systems Capacity Balancing & The Critical Point ($P_c$)
The capacity allocation between fluid associative units and rigid factual Inodes is defined by:

```math
\Delta q = \frac{N_p}{N_m \cdot M}
```
where $M$ is the internal cardinality of indexed literals. This governs three operational regimes:
* **Under-Indexing ($\Delta q \gg 1$, Associative Drift):** Polysemantic nodes dominate without adequate anchors. The model excels at cross-domain synthesis but hallucinates exact values.
* **Over-Indexing ($\Delta q \ll 1$, Lookup Saturation):** Inodes saturate the latent volume, engulfing associative pathways. The network degrades into an inflexible lookup table.
* **The Critical Point ($P_c$):** The optimal systems operating point where metric tolerance radii $T_t$ leave ample volume for fluid exploration while intercepting factual queries with surgical precision.

---

## 📊 Experimental Results

### Figure 1: Contrastive Superposition Tensors & Algebraic Emulation ($32 \times 32$ Manifold)
![Figure 1: Imprinted Tensors](assets/fig1_imprinted_tensors.png)
*The contrastive block emulates and reconstructs binary matrices $X \in \{0, 1\}^{32 \times 32}$ structured into rigid monosemantic Inode headers $N_m$ ($2 \times 2$ pixels top-left, columns 2, 5, 8, 11) and a polysemantic structural manifold $N_p$ (digits `'0'`, `'6'`, `'8'`, `'9'` under spatial jitter). **Cols 1–4 ($I_1 \dots I_4$):** 4-neuron bottleneck ($z \in \mathbb{R}^4$) superimposing digit primitives ($I_1 \to \text{'8'}$, $I_2 \to \text{'6'}$, $I_3 \to \text{'0'}$, $I_4 \to \text{'9'}$) alongside Inodes. **Col 5 ($I_1 \odot I_2$):** Hadamard intersection isolating shared structural segments between `'8'` and `'6'`. **Col 6 ($\mathrm{ReLU}(I_1 - 0.7 I_2)$):** Rectified subtraction isolating differential morphology (upper loop of `'8'`), completely suppressing the shared Inode header.*

---

### Figure 2: Latent Regime Trade-Off, Critical Point ($P_c$), and Factual Verification
![Figure 2: Phase Transition & Critical Point](assets/fig2_phase_transition.png)
* **(a) Metric Volume Trade-Off:** High-dimensional regime allocation ($D=32$) between fluid volume $\mathcal{V}_p$ and deterministic Inode volume $\mathcal{V}_m$.
* **(b) Capacity Landscape:** Architectural instability cost identifying the stationary Critical Point $P_c$.
* **(c) Factual Verification Benchmark:** Unconstrained decoding suffers $>40\%$ literal corruption under rising $T_p$, whereas HySA maintains **$0.0\%$ corruption** on confirmed queries, intercepting drift as explicit Semantic Faults.

---

### Figure 3: Concentric Domain Geometry, Architecture, and Pythia-70M Probing
![Figure 3: Semantic Hypersphere & Architecture](assets/fig3_hypersphere_geometry.png)
* **(a) Concentric Hypersphere Geometry:** Radial metric boundaries $T_t = 0.46$ and $R_{\mathrm{dom}} = 0.65$ defining the three operational zones: Confirmed Inode Core, Semantic Fault Buffer, and Open Fluid Domain.
* **(b) Intermediate Layer Early-Exit Architecture:** Probing at Layer $l=3$ either triggers immediate Early-Exit, raises a Semantic Fault, or passes to Layer $L$ for fluid sampling.
* **(c) Empirical Residual Stream Probing on Pythia-70M:** Demonstrating clear, non-overlapping metric separation across known factual queries ($d^* \le 0.4304$), unindexed speleological telemetry ($d^* = 0.4973$), and open creative text ($d^* \ge 1.0430$).

---

## 🧪 Empirical Validation on Real Transformer (`EleutherAI/pythia-70m`)

### 1. Isomorphism in the Polysemantic Manifold ($N_p$)
Cosine similarity across four distinct prompts expressing identity-to-address resolution (residual stream Layer 3, $D=512$):

| Conceptual Schema | DNS | IP Routing | White Pages | Phone Directory |
| :--- | :---: | :---: | :---: | :---: |
| **DNS** | $1.000$ | $0.961$ | $0.973$ | $0.893$ |
| **IP Routing** | $0.961$ | $1.000$ | $0.940$ | $0.916$ |
| **White Pages** | $0.973$ | $0.940$ | $1.000$ | $0.903$ |
| **Phone Directory** | $0.893$ | $0.916$ | $0.903$ | $1.000$ |

*High cosine similarity ($\ge 0.893$) proves empirically that deep transformers naturally construct polysemantic manifolds representing abstract relational schemas across distant domains.*

### 2. Empirical Verification Benchmark at Layer 3 ($T_t = 0.46$, $R_{\mathrm{dom}} = 0.65$)

Evaluating Layer 3 residual states on `EleutherAI/pythia-70m` confirms the clean metric decoupling of the three operational regimes:

| Target Concept | Interrogated Prompt | Raw Base Output | Distance $d^*$ | Operational Regime | HySA Final Output |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `SSH_PORT` | *"...SSH default port is"* | `' 1'` (Hallucination) | $0.3257$ | **CONFIRMED** ($`d^* \le T_t`$) | `' 22'` (Early-Exit, $100\%$ Exact) |
| `HTTP_PORT` | *"...HTTP default port is"* | `' used'` (Hallucination) | $0.4304$ | **CONFIRMED** ($`d^* \le T_t`$) | `' 80'` (Early-Exit, $100\%$ Exact) |
| `PUBLIC_DNS_IP` | *"...Google public DNS resolver is"* | `' used'` (Hallucination) | $0.3889$ | **CONFIRMED** ($`d^* \le T_t`$) | `' 8.8.8.8'` (Early-Exit, $100\%$ Exact) |
| `SUBNET_MASK` | *"...255.255.255.0 has CIDR length"* | `' 0'` (Hallucination) | $0.1263$ | **CONFIRMED** ($`d^* \le T_t`$) | `' 24'` (Early-Exit, $100\%$ Exact) |
| `ADMIN_NAME` | *"...root user first name literal is"* | `' the'` (Hallucination) | $0.0889$ | **CONFIRMED** ($`d^* \le T_t`$) | `' JOHN'` (Early-Exit, $100\%$ Exact) |
| `UNINDEXED` | *"...subterranean speleological sensors..."* | `' a'` (Nonsense) | $0.4973$ | **SEMANTIC FAULT** ($`T_t \lt d^* \le R_{\mathrm{dom}}`$) | **[HALTED]** (Intercepted mid-network) |
| `CREATIVE_SKY` | *"...Today the afternoon sky looks very"* | `' good'` (Fluid) | $1.0430$ | **FLUID $N_p$** ($`d^* \gt R_{\mathrm{dom}}`$) | `' good'` (Normal generation) |
| `CREATIVE_POEM`| *"...In the quiet forest whispers the"* | `' wind'` (Fluid) | $1.1084$ | **FLUID $N_p$** ($`d^* \gt R_{\mathrm{dom}}`$) | `' wind'` (Normal generation) |

---

## 🚀 Quickstart & Reproduction

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/fratorials/HySA.git
cd HySA

# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Experiments & Generate Figures
```bash
# Run the synthetic 32x32 benchmark (generates Figure 1 & Figure 2 in papers/figures/)
python3 src/run_synthetic_benchmark.py

# Run real-world transformer probing on Pythia-70M
python3 src/run_pythia_validation.py

# Generate high-resolution Figure 3
python3 src/generate_figures.py
```

### 3. Compile LaTeX Manuscripts
```bash
# Compile Flagship 9-page Extended English edition (single-column)
cd papers/extended
pdflatex -interaction=nonstopmode HySA_Extended_EN.tex
pdflatex -interaction=nonstopmode HySA_Extended_EN.tex

# Compile Flagship 9-page Extended Italian edition (single-column)
pdflatex -interaction=nonstopmode HySA_Extended_IT.tex
pdflatex -interaction=nonstopmode HySA_Extended_IT.tex

# Compile 8-page Extended Two-Column English edition
pdflatex -interaction=nonstopmode HySA_Extended_TwoColumn_EN.tex
pdflatex -interaction=nonstopmode HySA_Extended_TwoColumn_EN.tex

# Compile 3-page Condensed English Conference Short-Paper (two-column)
cd ../condensed
pdflatex -interaction=nonstopmode HySA_Short_EN.tex
pdflatex -interaction=nonstopmode HySA_Short_EN.tex

# Compile 3-page Condensed Italian Conference Short-Paper (two-column)
pdflatex -interaction=nonstopmode HySA_Short_IT.tex
pdflatex -interaction=nonstopmode HySA_Short_IT.tex
```

---

## 📁 Repository Structure

```text
HySA/
├── README.md                      # Overview, mathematical formulation, BibTeX citation
├── LICENSE                        # MIT License
├── requirements.txt               # Reproducible dependencies
├── .gitignore                     # Ignores build artifacts, venv, and large archives
│
├── papers/                        # 📄 Official Manuscripts & Vector Figures
│   ├── figures/                   # Shared vector PDF figures (Fig 1, 2, 3)
│   │   ├── fig1_imprinted_tensors.pdf
│   │   ├── fig2_phase_transition.pdf
│   │   └── fig3_hypersphere_geometry.pdf
│   ├── extended/                  # 9-page Flagship Journal Editions & 8-page Two-Column
│   │   ├── HySA_Extended_EN.tex   # English LaTeX source (9 pages)
│   │   ├── HySA_Extended_EN.pdf   # English Flagship PDF
│   │   ├── HySA_Extended_IT.tex   # Italian LaTeX source (9 pages)
│   │   ├── HySA_Extended_IT.pdf   # Italian Flagship PDF
│   │   ├── HySA_Extended_TwoColumn_EN.tex # English Two-Column LaTeX (8 pages)
│   │   ├── HySA_Extended_TwoColumn_EN.pdf # English Two-Column PDF
│   │   ├── HySA_Extended_EN.md    # English Markdown edition
│   │   └── HySA_Extended_IT.md    # Italian Markdown edition
│   └── condensed/                 # 3-page Conference Short-Papers
│       ├── HySA_Short_EN.tex      # English Short Paper LaTeX (3 pages)
│       ├── HySA_Short_EN.pdf      # English Short Paper PDF
│       ├── HySA_Short_IT.tex      # Italian Short Paper LaTeX (3 pages)
│       ├── HySA_Short_IT.pdf      # Italian Short Paper PDF
│       ├── HySA_Short_EN.md       # English Condensed Markdown
│       └── HySA_Short_IT.md       # Italian Condensed Markdown
│
├── src/                           # 🧠 Python Source Code
│   ├── run_synthetic_benchmark.py # Synthetic 32x32 manifold bottleneck experiment
│   ├── run_pythia_validation.py   # EleutherAI/pythia-70m residual stream probing
│   └── generate_figures.py        # High-resolution vector plot generator
│
├── assets/                        # 🖼️ High-resolution PNGs for README / Web
│   ├── fig1_imprinted_tensors.png
│   ├── fig2_phase_transition.png
│   └── fig3_hypersphere_geometry.png
│
├── obsidian/                      # 📓 Obsidian Vault Notes (Markdown + KaTeX + Callouts)
│   ├── HySA_Extended_IT.md        # Extended Italian Note
│   ├── HySA_Extended_EN.md        # Extended English Note
│   ├── HySA_Short_IT.md           # Condensed Italian Note
│   ├── HySA_Short_EN.md           # Condensed English Note
│   └── assets/                    # Image assets for Obsidian preview
│
└── drafts/                        # 📝 Historical notes and text drafts
```

---

## 📜 Citation

If you find HySA relevant to your research in mechanistic interpretability, neurosymbolic architectures, or hallucination mitigation, please cite:

```bibtex
@article{mariani2026hysa,
  title={{HySA (Hybrid Semantic Architecture): Decoupling Fluid Reasoning from Deterministic Verification via Neuronal Semanticity and Inode Indexing}},
  author={Mariani, Francesco},
  journal={arXiv preprint / Zenodo},
  year={2026},
  month={October},
  doi={10.5281/zenodo.23459635},
  url={https://doi.org/10.5281/zenodo.23459635},
  orcid={0009-0008-1319-3573}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).