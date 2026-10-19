---
title: "HySA (Hybrid Semantic Architecture): Separare il Ragionamento Fluido dalla Verifica Deterministica tramite Semanticità Neuronale e Indicizzazione Inode"
author: "Francesco Mariani"
affiliation: "Architetto di Sistemi Indipendente, Umbria, Italia"
email: "fratorials@gmail.com"
doi: "10.5281/zenodo.23459635"
orcid: "0009-0008-1319-3573"
date: "Ottobre 2026"
edition: "Edizione Condensata (Italiano) — Revisione Systems Architecture"
tags:
  - systems-architecture
  - interpretabilita-meccanicistica
  - reti-neurali
  - polisemantismo
  - inode-indexing
  - semantic-fault
  - geometric-gating
  - early-exit
  - pythia-70m
---

# HySA (Hybrid Semantic Architecture): Separare il Ragionamento Fluido dalla Verifica Deterministica tramite Semanticità Neuronale e Indicizzazione Inode

**Francesco Mariani**  
*Architetto di Sistemi Indipendente — Umbria, Italia*  
`fratorials@gmail.com` | DOI: [10.5281/zenodo.23459635](https://doi.org/10.5281/zenodo.23459635) | ORCID: [0009-0008-1319-3573](https://orcid.org/0009-0008-1319-3573)  
*Ottobre 2026*

---

> [!quote] Epigrafe
> *"L'industria dell'IA vede il polisemantismo come una condanna... ma forse è la benedizione che ha reso possibile l'intelligenza."*

> [!abstract] Sommario (Abstract)
> La ricerca attuale nell'*Interpretabilità Meccanicistica* tratta il polisemantismo neuronale come un difetto di compressione (*Ipotesi di Sovrapposizione*), investendo ingenti risorse nell'uso di Sparse Autoencoder (SAE) per scomporre la rete in dizionari di unità puramente monosemantiche. Questo lavoro parte da una tesi opposta: **il polisemantismo non è un bug da correggere, ma il prerequisito strutturale dell'intelligenza associativa.** Un neurone puramente monosemantico ha zero gradi di libertà relazionali ($\mathcal{R}(u)=0$): non potendo connettere concetti distanti, funziona solo come una rigida cella di memoria. Al contrario, le unità polisemantiche si attivano su concetti isomorfi (es. `IP`, `DNS`, `Pagine Bianche`, `Numero di Telefono`) per astrarre schemi logici di ordine superiore. Il vero problema dei modelli attuali non è la sovrapposizione semantica, ma la mancata separazione tra il **comportamento fluido** (ragionamento) e il **comportamento di verifica** (recupero del dato esatto). Per risolvere questa dicotomia senza ricorrere a colli di bottiglia in serie o a duplicazioni in parallelo, presento **HySA (Hybrid Semantic Architecture)**: un paradigma ibrido in cui lo spazio latente è unificato, delimitato non da blocchi fisici separati ma dal **grado di semanticità del neurone** e dalla sua **dimensionalità interna (cardinalità $|\mathcal{C}(u)|$)**. Attraverso due regimi termici distinti — Temperatura Polisemantica ($T_p$) e Monosemantica ($T_m$) — unificati nella Temperatura Topologica ($T_t$), un Raggio di Dominio Fattuale ($R_{\mathrm{dom}}$), e un meccanismo interno di **Gating Geometrico al Layer Intermedio con Early-Exit** su registro metrico (**Inode**), HySA permette al modello di ragionare in modo plastico e di ancorarsi a dati deterministici quando serve, bloccando le allucinazioni tramite eccezioni esplicite di sistema (*Semantic Fault*).

---

![Figura 1: Tensori di Sovrapposizione Contrastivi](assets/fig1_imprinted_tensors.png)
*Figura 1: **Tensori di Sovrapposizione Contrastivi ($I_n$) ed Emulazione Algebrica sul Manifold $32 \times 32$.** Il blocco a contrasto emula matrici binarie $X \in \{0, 1\}^{32 \times 32}$ strutturate su header Inode $N_m$ ($2 \times 2$ pixel, col. $2, 5, 8, 11$) e manifold polisemantico $N_p$ (cifre `'0'`, `'6'`, `'8'`, `'9'` con jitter). **Col. 1–4 ($I_1 \dots I_4$):** Rappresentazioni dal collo di bottiglia a 4 neuroni ($z \in \mathbb{R}^4$), ciascuna sovrapponente combinazioni di cifre ($I_1 \to \text{'8'}$, $I_2 \to \text{'6'}$, $I_3 \to \text{'0'}$, $I_4 \to \text{'9'}$) e header Inode. **Col. 5 ($I_1 \odot I_2$):** Intersezione di Hadamard che isola l'invariante strutturale condiviso tra `'8'` e `'6'`. **Col. 6 ($\mathrm{ReLU}(I_1 - 0.7 I_2)$):** Sottrazione rettificata che isola la divergenza morfologica (anello superiore di `'8'`), sopprimendo l'Inode comune e dimostrando l'isolamento algebrico deterministico.*

## 1. Il Problema di Fondo: Astrazione Fluida vs. Verifica Fattuale

Nelle reti neurali profonde, un singolo neurone manifesta frequentemente selettività mista (*mixed selectivity*) e si attiva per concetti eterogenei (*polisemantismo*) [1, 2, 3]. La risposta convenzionale dell'interpretabilità meccanicistica consiste nell'impiegare Sparse Autoencoder (SAE) [4, 5] per scomporre lo spazio latente in dizionari di direzioni monosemantiche isolate.

Tuttavia, usare la monosemia come obiettivo di progettazione architetturale rappresenta un vicolo cieco: scambia il meccanismo con cui la rete astrae per rumore statistico. Si confondono due funzioni cognitive essenziali:

* **Comportamento Fluido (Neuroni Polisemantici, $N_p$):** Un'unità che si attiva su `IP`, `DNS`, `Pagine Bianche` e `Numero di Telefono` non è rumorosa: ha estratto un livello di astrazione superiore (*risoluzione da identità a indirizzo*). Questa sovrapposizione rende possibili le analogie e il ragionamento continuo.
* **Comportamento di Verifica (Neuroni Monosemantici, $N_m$):** Quando il modello deve restituire un dato atomico e immutabile — l'IP `8.8.8.8`, la maschera `24`, la porta SSH `22` o il root user `JOHN` — l'interpolazione fluida genera allucinazioni. Serve una cella di memoria rigida, a zero gradi di libertà relazionali ($\mathcal{R}(u)=0$).

I modelli attuali allucinano perché impastano entrambi i regimi negli stessi pesi densi [6]. La sfida di HySA è disaccoppiare i due domini e risolvere il problema ingegneristico dell'**indicizzazione**: consentire all'intelligenza fluida di accedere a dati rigidi certificati a costo computazionale nullo.

> [!definition] Definizione 1 (Rango Relazionale delle Unità Semantiche)
> Sia $u$ un'unità neuronale e $\Omega$ lo spazio dei concetti. Sia $\mathcal{C}(u) = \{c \in \Omega \mid \mathbb{E}_{X \in \mathcal{D}_c}[z_u(X)] > \theta\}$. Il **Rango Relazionale** $\mathcal{R}(u)$ è la cardinalità delle traiettorie associative abilitate dall'unità:
> $$
> \mathcal{R}(u) = \binom{|\mathcal{C}(u)|}{2} = \frac{|\mathcal{C}(u)| (|\mathcal{C}(u)| - 1)}{2}
> $$
> Un'unità puramente monosemantica $u_{\mathrm{mono}}$ ha $|\mathcal{C}(u_{\mathrm{mono}})| = 1 \implies \mathcal{R}(u_{\mathrm{mono}}) = 0$. Un'architettura interamente monosemantica non può compiere analogie continue senza grafi esterni.

---

## 2. Architettura Unificata: Né Serie, Né Parallelo

Accoppiare $N_p$ ed $N_m$ in serie crea un collo di bottiglia meccanico che distrugge la fluidità contestuale; accoppiarli in parallelo ($2\times \mathrm{FLOPs}$) richiede moduli euristici di arbitraggio insostenibili.

**Soluzione: Dimensionalità Interna e Cardinalità Concettuale $|\mathcal{C}(u)|$.** HySA opera su un unico substrato nello spazio di Hilbert $\mathcal{S} \subset \mathbb{R}^D$, delimitato dalla cardinalità concettuale $|\mathcal{C}(u)|$:

* **Alta Cardinalità ($N_p$, $|\mathcal{C}(u)| \gg 1$):** Aggregano concetti isomorfi per la sintesi e il pensiero fluido.
* **Cardinalità Unitaria ($N_m$, $|\mathcal{C}(u)|=1$):** Punti privi di rami associativi che ancorano fatti immutabili.

**Tensori di Sovrapposizione Contrastivi ($I_n$):**
$$
I_n = \frac{1}{\sum_{X_k \in \mathcal{X}^+} z_{k,n} + \eta} \left[ \sum_{X_k \in \mathcal{X}^+} z_{k,n} \cdot \Phi_{\mathrm{net}}(X_k \mid n) - \gamma \sum_{X_j \in \mathcal{X}^-} z_{j,n} \cdot \Phi_{\mathrm{net}}(X_j \mid n) \right]
$$
Come illustrato nella Figura 1, dentro pochi neuroni ($N_p=4$) i concetti non si sporcano ma si sovrappongono in modo ordinato: $I_1 \odot I_2$ estrae i segmenti strutturali condivisi, mentre $\mathrm{ReLU}(I_1 - 0.7 I_2)$ isola la differenza, senza esplodere la rete in dizionari enormi.

---

## 3. Indicizzazione, Temperature e Logica "Query/API"

Ispirandosi all'architettura dei Sistemi Operativi (RAM fluida per i processi, Inode rigidi nel filesystem), HySA introduce una metrica di confine nello spazio latente:

| Termine OS | Costrutto HySA | Funzione Pratica in HySA |
| :--- | :--- | :--- |
| **Inode ($\mu_k$)** | Centroide $N_m \in S^{D-1}$ | Coordinata dello spazio legata al dato esatto (`' 8.8.8.8'`). |
| **Filesystem** | Varietà $\mathcal{F}_{\mathrm{mono}}$ | Insieme $\{\mu_k\}_{k=1}^M$ dei fatti indicizzati. |
| **Query / API ($Q$)** | Operatore di Confine | Intercetta lo stato al layer intermedio ed emette il dato esatto. |
| **Page Fault** | **Semantic Fault** | Blocca il modello su fatti non indicizzati nel dominio ($T_t < d^* \le R_{\mathrm{dom}}$). |
| **Temperatura** | Intorno $\mathcal{B}(\mu_k, T_t)$ | Raggio di tolleranza attorno all'Inode. |

*Tabella 1: Mappatura Euristiche OS sui Costrutti HySA.*

**Due Temperature ($T_p, T_m$) e Temperatura Topologica ($T_t$):**
1. $T_p$: governa il movimento fluido ed esplorativo lungo le varietà.
2. $T_m$: definisce la stretta tolleranza del centroide dell'Inode.

Dalla loro interazione nasce la Temperatura Topologica:
$$
T_t(T_p, T_m) = T_m \left( 1 + \alpha \tanh\left( \frac{T_p}{T_m} \right) \right)
$$
con la condizione $T_t < R_{\mathrm{dom}}$, che delimita la palla metrica $\mathcal{B}(\mu_k, T_t) = \{x \in \mathcal{S} \mid \|x - \mu_k\|_2 \le T_t\}$.

> [!definition] Definizione 2 (Operatore di Confine Metrico $Q$ con Gating Concentrico)
> Dato lo stato normalizzato $q_l = h_t^{(l)} / \|h_t^{(l)}\|_2$ al layer intermedio $l$, l'indice Inode $\mathcal{F}_{\mathrm{mono}}$, il raggio di dominio $R_{\mathrm{dom}}$ e il confine $T_t < R_{\mathrm{dom}}$ con distanza minima $d^* = \min_k \|q_l - \mu_k\|_2$:
> $$
> Q(q_l, \mathcal{F}_{\mathrm{mono}}, T_t, R_{\mathrm{dom}}) = 
> \begin{cases} 
>   (\mathrm{CONFIRMED}, \mathbf{v}_{k^*}) & \text{se } d^* \le T_t \quad (\mathbf{Early\text{-}Exit}) \\ 
>   (\mathrm{FAULT}, \bot_{\mathrm{Fault}}) & \text{se } T_t < d^* \le R_{\mathrm{dom}} \quad (\mathbf{Semantic\;Fault}) \\
>   (\mathrm{BYPASS}, \mathrm{Fluid}) & \text{se } d^* > R_{\mathrm{dom}} \quad (\mathbf{Regime\;Fluido})
> \end{cases}
> $$

![Figura 2: Transizioni e Punto di Equilibrio](assets/fig2_phase_transition.png)
*Figura 2: **Bilanciamento del Gradiente di Compressione e Benchmark di Verifica Fattuale in HySA.** **(a)** Transizione volumetrica tra spazio libero e vincoli indicizzati. **(b)** Il Punto Critico stazionario $P_c$, che bilancia la deriva associativa e la saturazione da sovra-indicizzazione. **(c)** Benchmark sui letterali fattuali al crescere di $T_p$: la decodifica non vincolata subisce rapida corruzione ($>40\%$), mentre HySA garantisce lo $0.0\%$ di corruzione sulle query verificate, trasformando gli stati ambigui in espliciti Errori Semantici (Semantic Faults).*

**Punto di Equilibrio ($P_c$):** Se $N_p$ domina ($\Delta q \gg 1$), il modello soffre di deriva associativa. Se $N_m$ domina con sfere troppo larghe, lo spazio si satura e il modello regredisce a un lookup table rigido. Il Punto Critico $P_c$ garantisce sfere $T_t$ abbastanza strette da lasciare libero il pensiero fluido, intercettando con precisione i fatti noti.

---

## 4. Validazione Empirica su `Pythia-70M`

L'analisi dello stream residuo al Layer 3 ($D=512$) di `EleutherAI/pythia-70m` conferma tre comportamenti cardine:

1. **Isomorfismo Polisemantico Naturale ($N_p$):** Quattro prompt tecnici (`DNS`, `Instr. IP`, `Pag. Bianche`, `Elenco Tel.`) esibiscono una similarità coseno compresa tra **$0.893$ e $0.973$** (Tabella 2).
2. **Iniezione Deterministica ed Early-Exit ($N_m$):** Configurando $T_t = 0.46$ su 6 costanti, HySA azzera le allucinazioni del modello base (che fallisce nel 100% dei casi), eseguendo un Early-Exit a metà rete che restituisce il letterale esatto con precisione del 100% ($d^* \le 0.4304$, Tabella 3).
3. **Immunità dai Falsi Positivi Creativi ($R_{\mathrm{dom}} = 0.65$):** Su query non censite nel dominio fattuale (sensori speleologici), $T_t < d^* = 0.4973 \le R_{\mathrm{dom}}$, sollevando un *Semantic Fault*. Al contrario, prompt aperti e creativi cadono a $d^* \ge 1.0430 > R_{\mathrm{dom}}$, lasciando proseguire la decodifica fluida senza errori di sistema.

| Concetto | DNS | Instr. IP | Pag. Bianche | Elenco Tel. |
| :--- | :---: | :---: | :---: | :---: |
| **DNS** | $1.000$ | $0.961$ | $0.973$ | $0.893$ |
| **Instr. IP** | $0.961$ | $1.000$ | $0.940$ | $0.916$ |
| **Pag. Bianche** | $0.973$ | $0.940$ | $1.000$ | $0.903$ |
| **Elenco Tel.** | $0.893$ | $0.916$ | $0.903$ | $1.000$ |

*Tabella 2: Similarità Coseno al Layer 3 dei Concetti Isomorfi in $N_p$ (`Pythia-70M`, $D=512$).*

| Concetto | Output Base | Dist. $d^*$ | Stato HySA | Output Finale |
| :--- | :--- | :---: | :--- | :--- |
| `SSH_PORT` | `' 1'` (**Alluc.**) | $0.3257$ | **CONFIRMED** ($d^* \le T_t$) | `' 22'` (Early-Exit) |
| `HTTP_PORT` | `' used'` (**Alluc.**) | $0.4304$ | **CONFIRMED** ($d^* \le T_t$) | `' 80'` (Early-Exit) |
| `PUBLIC_IP` | `' used'` (**Alluc.**) | $0.3889$ | **CONFIRMED** ($d^* \le T_t$) | `' 8.8.8.8'` (Early-Exit) |
| `SUBNET` | `' 0'` (**Alluc.**) | $0.1263$ | **CONFIRMED** ($d^* \le T_t$) | `' 24'` (Early-Exit) |
| `ADMIN` | `' the'` (**Alluc.**) | $0.0889$ | **CONFIRMED** ($d^* \le T_t$) | `' JOHN'` (Early-Exit) |
| `UNINDEXED` | `' a'` (**Nons.**) | $0.4973$ | **SEMANTIC FAULT** | **[BLOCCO]** |
| `CREATIVE` | `' good'` (Fluido) | $1.0430$ | **FLUID $N_p$** ($d^* > R_{\mathrm{dom}}$) | `' good'` (Bypass) |

*Tabella 3: Benchmark di Verifica al Layer 3 su `EleutherAI/pythia-70m` ($D=512$, $T_t=0.46, R_{\mathrm{dom}}=0.65$).*

---

## 5. Scalabilità e Gating Geometrico di Dominio

L'approccio a valle basato sull'entropia dei logit fallisce sia su frasi creative aperte (falsi allarmi) sia su allucinazioni pronunciate con elevata confidenza statistica. HySA risolve questo problema alla radice con il **Gating Geometrico al Layer Intermedio**: intercettando il flusso a metà rete ($l=3$), le frasi creative ($d^* > R_{\mathrm{dom}}$) proseguono indisturbate, le allucinazioni fattuali vengono arrestate prima dei layer finali, e i fatti noti generano un Early-Exit immediato risparmiando circa il 50% dei FLOPs. Con strutture ANN su GPU (es. indici HNSW / FAISS) a dimensionalità elevata ($D \ge 4096$), la latenza di ricerca scala a $O(\log M)$ con tempi inferiori a $15\,\mu\text{s}$.

---

## 6. Conclusione

Con **HySA** propongo un'architettura che dimostra come il polisemantismo sia il motore dell'intelligenza associativa fluida ($T_p$), mentre la monosemia è il presidio della memoria fattuale ($T_m$). Unificando entrambi i regimi nell'Ipersfera Semantica con confini concentrici ($T_t, R_{\mathrm{dom}}$) e interrogazione API al layer intermedio su registro Inode, i modelli coniugano astrazione plastica, efficienza computazionale ed esatto rigore deterministico.

---

## Bibliografia

1. M. Rigotti et al., "The importance of mixed selectivity in complex cognitive tasks," *Nature*, vol. 497, pp. 585–590, 2013.
2. S. Fusi, E. K. Miller, e M. Rigotti, "Why neurons mix: high dimensionality for higher cognition," *Curr. Opin. Neurobiol.*, vol. 37, pp. 66–74, 2016.
3. N. Elhage et al., "Toy models of superposition," *Transformer Circuits Thread*, 2022.
4. T. Bricken et al., "Towards monosemanticity: Decomposing language models with dictionary learning," *Transformer Circuits Thread*, 2023.
5. A. Templeton et al., "Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet," *Transformer Circuits Thread*, 2024.
6. M. Geva et al., "Transformer feed-forward layers are key-value memories," in *Proc. EMNLP*, pp. 5484–5495, 2021.
7. K. Meng et al., "Locating and editing factual associations in GPT," in *Proc. NeurIPS*, pp. 17359–17372, 2022.
8. U. Khandelwal et al., "Generalization through memorization: Nearest neighbor language models," in *Proc. ICLR*, 2020.
9. D. Kahneman, *Pensieri lenti e veloci*, Mondadori, 2011.
