---
title: "HySA (Hybrid Semantic Architecture): Separare il Ragionamento Fluido dalla Verifica Deterministica tramite Semanticità Neuronale e Indicizzazione Inode"
author: "Francesco Mariani"
affiliation: "Architetto di Sistemi Indipendente, Umbria, Italia"
email: "fratorials@gmail.com"
doi: "10.5281/zenodo.23459635"
orcid: "0009-0008-1319-3573"
date: "Ottobre 2026"
edition: "Edizione Estesa (Italiano) — Revisione Systems Architecture"
tags:
  - systems-architecture
  - interpretabilita-meccanicistica
  - reti-neurali
  - polisemantismo
  - inode-indexing
  - ipersfera-semantica
  - semantic-fault
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

> [!abstract] Abstract
> La ricerca attuale nell'*Interpretabilità Meccanicistica* tratta il polisemantismo neuronale come un difetto di compressione (*Ipotesi di Sovrapposizione*), investendo enormi risorse nell'uso di Sparse Autoencoder (SAE) per scomporre la rete in dizionari di unità puramente monosemantiche. Questo lavoro parte da una tesi opposta: **il polisemantismo non è un bug da correggere, ma il prerequisito strutturale dell'intelligenza associativa.** Un neurone puramente monosemantico ha zero gradi di libertà relazionali ($\mathcal{R}(u)=0$): non potendo connettere concetti distanti, funziona solo come una rigida cella di memoria. Al contrario, le unità polisemantiche si attivano su concetti isomorfi (es. `IP`, `DNS`, `Pagine Bianche`, `Numero di Telefono`) per astrarre schemi logici di ordine superiore.
> 
> Il vero problema dei modelli attuali non è la sovrapposizione semantica, ma la mancata separazione tra il **comportamento fluido** (ragionamento) e il **comportamento di verifica** (recupero del dato esatto). Per risolvere questa dicotomia senza ricorrere a colli di bottiglia in serie o a duplicazioni in parallelo, presento **HySA (Hybrid Semantic Architecture)**: un paradigma ibrido in cui lo spazio latente è un tutt'uno, delimitato non da blocchi fisici separati ma dal **grado di semanticità del neurone** e dalla sua **dimensionalità interna (cardinalità $|\mathcal{C}(u)|$)**. Attraverso l'introduzione di due regimi termici distinti — Temperatura Polisemantica ($T_p$) e Monosemantica ($T_m$) — unificati nella Temperatura Topologica ($T_t$), e di un meccanismo interno di **Query/API** su registro metrico (**Inode**), HySA permette al modello di ragionare in modo plastico e di ancorarsi a dati deterministici quando serve, bloccando le allucinazioni tramite eccezioni esplicite di sistema (*Semantic Fault*).

---

## 1. Il Problema di Fondo: Astrazione Fluida vs. Verifica Fattuale

Nelle reti neurali profonde, un singolo neurone manifesta frequentemente selettività mista (*mixed selectivity*) e si attiva per concetti eterogenei [1, 2, 3]. Chi lavora sull'interpretabilità meccanicistica vede questo comportamento (*polisemantismo*) come un limite dovuto al fatto che il modello deve comprimere più concetti ($M$) delle dimensioni fisiche che ha a disposizione ($D$) [3]. L'approccio standard oggi consiste nell'usare gli Sparse Autoencoder (SAE) [4, 5] per "spacchettare" questi neuroni in milioni di feature monosemantiche pure, dove a ogni unità corrisponde un solo significato.

Tuttavia, l'impiego degli SAE come strumento analitico esterno di ispezione non deve essere confuso con un obiettivo di progettazione architetturale. Forzare la monosemia come substrato operativo della rete rappresenta un vicolo cieco concettuale: scambia il meccanismo primario con cui la rete genera astrazione per mero rumore statistico.

Cercare la purezza monosemantica ovunque significa confondere due funzioni cognitive che nei sistemi informatici sono sempre state separate: **l'elaborazione logica** e **lo storage dei dati**.

1. **Il comportamento fluido (Neuroni Polisemantici, $N_p$):** Si consideri un neurone che si attiva quando il modello elabora `Indirizzo IP`, `DNS`, `Pagine Bianche` e `Numero di Telefono`. Non è un neurone confuso o rumoroso. Ha semplicemente estratto un livello di astrazione superiore: *il concetto di risoluzione da un'identità a un indirizzo*. Questa sovrapposizione è ciò che permette al modello di fare analogie e muoversi in modo fluido tra domini diversi.
2. **Il comportamento di verifica (Neuroni Monosemantici, $N_m$):** Quando però il modello deve restituire un dato secco, atomico e immutabile — l'IP `8.8.8.8`, la maschera CIDR `24`, la porta SSH `22` o l'utente `JOHN` — l'astrazione fluida diventa un problema. Qui non serve creatività o interpolazione: serve una cella di memoria rigida, a zero gradi di libertà associativi ($\mathcal{R}(u)=0$).

Gli LLM attuali allucinano perché impastano sintesi associativa e memoria fattuale dentro gli stessi pesi [6, 7]. Mentre approcci non parametrici [8] tentano di mitigare questo limite esternamente, la cognizione biologica separa efficacemente la sintesi associativa dalla verifica deliberata all'interno del medesimo substrato [9]. Durante l'addestramento, la rete tratta una costante di rete o una data storica esattamente come tratta una metafora poetica: come una distribuzione di probabilità plastica.

Il paradigma ibrido di HySA nasce per separare il comportamento fluido da quello di verifica. Ma una volta separati concettualmente, la vera sfida ingegneristica non è filosofica: **il difficile sta nell'indicizzare il tutto**, ovvero nel mettere l'intelligenza fluida in condizione di ripescare il dato rigido nel modo più efficiente, veloce e affidabile possibile, senza rallentare la generazione o snaturare la rete.

![Figura 1: Tensori di Sovrapposizione Contrastivi](assets/fig1_imprinted_tensors.png)
*Figura 1: **Tensori di Sovrapposizione Contrastivi ($I_n$) ed Emulazione Algebrica sul Manifold Sintetico $32 \times 32$.** Il blocco a contrasto emula e ricostruisce matrici binarie $X \in \{0, 1\}^{32 \times 32}$ strutturate su due regimi: un header monosemantico Inode $N_m$ ($2 \times 2$ pixel in alto a sinistra, colonne $2, 5, 8, 11$) e un manifold polisemantico $N_p$ che codifica le cifre geometriche `'0'`, `'6'`, `'8'` e `'9'` con jitter stocastico ($\Delta x, \Delta y \in \{-1, 0, 1\}$). **Colonne 1–4 ($I_1 \dots I_4$):** Rappresentazioni polisemantiche apprese dal collo di bottiglia a 4 neuroni ($z \in \mathbb{R}^4$), in cui ciascun tensore ricostruisce la sovrapposizione delle cifre ($I_1 \to \text{'8'}$, $I_2 \to \text{'6'}$, $I_3 \to \text{'0'}$, $I_4 \to \text{'9'}$) unitamente all'header di classe. **Colonna 5 ($I_1 \odot I_2$):** L'intersezione di Hadamard isola i segmenti strutturali condivisi tra le cifre `'8'` e `'6'` (anello inferiore comune). **Colonna 6 ($\mathrm{ReLU}(I_1 - 0.7 I_2)$):** La sottrazione differenziale rettificata estrae la divergenza morfologica (anello superiore di `'8'`), azzerando completamente l'header Inode comune e dimostrando l'isolamento algebrico deterministico delle feature senza distruggere la plasticità relazionale.*

> [!definition] Definizione 1 (Rango Relazionale delle Unità Semantiche)
> Sia $u$ un'unità neuronale e sia $\Omega$ lo spazio dei concetti semantici. Definisco l'insieme di feature associate $\mathcal{C}(u) = \{c \in \Omega \mid \mathbb{E}_{X \in \mathcal{D}_c}[z_u(X)] > \theta\}$, dove $\theta > 0$ è una soglia di attivazione statisticamente significativa. Il **Rango Relazionale** $\mathcal{R}(u)$ è la cardinalità delle traiettorie associative a coppie abilitate dall'unità:
> $$
> \mathcal{R}(u) = \binom{|\mathcal{C}(u)|}{2} = \frac{|\mathcal{C}(u)| (|\mathcal{C}(u)| - 1)}{2}
> $$
> Un'unità puramente monosemantica $u_{\mathrm{mono}}$ con $|\mathcal{C}(u_{\mathrm{mono}})| = 1$ ha rango relazionale nullo: $\mathcal{R}(u_{\mathrm{mono}}) = 0$. Di conseguenza, un'architettura limitata esclusivamente a neuroni monosemantici non può eseguire proiezioni analogiche continue senza introdurre un grafo relazionale esterno.

---

## 2. Architettura Unificata: Né in Serie, Né in Parallelo

Quando si progetta un sistema che deve gestire sia l'astrazione che il dato crudo, l'istinto ingegneristico classico porta a due strade: metterli in serie o metterli in parallelo. Entrambe falliscono:

* **Perché non in serie ($N_p \to N_m$ o viceversa):** Mettere uno strato monosemantico rigido in cascata dopo uno strato polisemantico crea un **bottleneck**. Attenzione: non è un bottleneck di astrazione (che nelle reti neurali è utile per comprimere le informazioni), ma una strozzatura meccanica che schiaccia le relazioni contestuali su categorie fisse troppo presto, uccidendo la fluidità del discorso.
* **Perché non in parallelo ($N_p \parallel N_m$):** Far girare due reti separate in parallelo è **estremamente inefficiente**. Raddoppia il costo computazionale ($2\times \mathrm{FLOPs}$) e introduce il problema irrisolto di dover costruire un modulo terzo di arbitraggio che decida, token dopo token, quale delle due reti abbia ragione.

### La soluzione: Dimensionalità interna e Cardinalità Concettuale $|\mathcal{C}(u)|$

In HySA, il sistema è tutt'uno. Non ci sono due reti fisiche distinte: c'è un unico substrato latente delimitato dalla logica operativa e, soprattutto, dal **tipo di "semanticità" del neurone**.

Per capire come convivono, introduco il concetto di **dimensionalità dentro i layer e dentro i neuroni**, definita dalla sua **Cardinalità Concettuale $|\mathcal{C}(u)|$**:

* Più che una misura geometrica spaziale, la cardinalità $|\mathcal{C}(u)|$ di un'unità neurale rappresenta **un livello di astrazione superiore**: indica quanti concetti o istanze distinte afferiscono a quello specifico nodo logico.
* I nodi ad **alta cardinalità ($N_p$, $|\mathcal{C}(u)| \gg 1$)** aggregano molteplici concetti isomorfi, creando il tessuto connettivo su cui scorre il pensiero fluido.
* I nodi a **cardinalità unitaria ($N_m$, dove $|\mathcal{C}(u)|=1$)** sono punti rigidi, privi di diramazioni associative laterali, che ancorano un singolo fatto immutabile.

I Tensori di Sovrapposizione Contrastivi $I_n$ sul benchmark sintetico $32 \times 32$ dimostrano che all'interno di un collo di bottiglia a 4 neuroni ($N_p=4$) addestrato su cifre con header di checksum Inode rigidi, i concetti non si degradano reciprocamente ma si sovrappongono in modo ordinato: mediante operazioni algebriche di Hadamard ($I_1 \odot I_2$) è possibile isolare i segmenti strutturali condivisi, mentre con sottrazioni rettificate ($\mathrm{ReLU}(I_1 - 0.7 I_2)$) si isola la divergenza specifica, senza dover frammentare lo spazio in milioni di direzioni isolate.

Formalizzo il Tensore di Sovrapposizione Contrastivo $I_n$ come:
$$
I_n = \frac{1}{\sum_{X_k \in \mathcal{X}^+} z_{k,n} + \eta} \left[ \sum_{X_k \in \mathcal{X}^+} z_{k,n} \cdot \Phi_{\mathrm{net}}(X_k \mid n) - \gamma \sum_{X_j \in \mathcal{X}^-} z_{j,n} \cdot \Phi_{\mathrm{net}}(X_j \mid n) \right]
$$

Per formalizzare la capacità di bilanciamento tra rappresentazione fluida e vincoli deterministici, la Figura 2 caratterizza le transizioni termodinamiche del sistema:
* **Transizione Volumetrica dell'Ipersfera ($T_t^D$, Fig. 2a):** All'aumentare del Vincolo di Determinismo ($\mathcal{D}$), la frazione di volume monosemantico vincolato $\mathcal{V}_m$ scala come $(T_t / R)^D$, subendo una rapida transizione di fase che restringe lo spazio libero $\mathcal{V}_p$.
* **Bilanciamento di Capacità e Punto Critico ($P_c$, Fig. 2b):** Il gradiente di compressione $\Delta q = \frac{N_p}{N_m \cdot M}$ definisce un paesaggio di instabilità in cui $P_c$ rappresenta l'equilibrio ottimale tra deriva associativa ($\Delta q \gg 1$) e saturazione rigida a dizionario ($\Delta q \ll 1$).
* **Verifica dei Letterali Fattuali ($T_p$ vs. $T_t$, Fig. 2c):** Al crescere della temperatura polisemantica $T_p$, la decodifica non vincolata subisce una rapida corruzione ($>40\%$), mentre HySA garantisce lo $0.0\%$ di errore sui dati verificati, convertendo le traiettorie incerte in espliciti *Semantic Fault*.

![Figura 2: Transizioni Termodinamiche e Punto Critico](assets/fig2_phase_transition.png)
*Figura 2: **Bilanciamento del Gradiente di Compressione e Benchmark di Verifica Fattuale in HySA.** **(a)** Transizione volumetrica ($T_t^D$) tra spazio libero $\mathcal{V}_p$ e vincoli indicizzati $\mathcal{V}_m$ sotto il Vincolo di Determinismo $\mathcal{D}$. **(b)** Il Punto Critico stazionario $P_c$, che bilancia la deriva associativa ($\Delta q \gg 1$) e la saturazione da sovra-indicizzazione ($\Delta q \ll 1$). **(c)** Benchmark sui letterali fattuali al crescere di $T_p$: la decodifica non vincolata subisce rapida corruzione ($>40\%$), mentre HySA garantisce lo $0.0\%$ di corruzione sulle query verificate, trasformando gli stati ambigui in espliciti Errori Semantici (Semantic Faults).*

---

## 3. Indicizzazione, Temperature ($T_p, T_m$) e la Logica "Query/API"

Se tutto vive nello stesso spazio vettoriale, come fa il modello a passare dalla generazione creativa alla verifica deterministica senza confondersi? Qui entra in gioco l'architettura ispirata ai Sistemi Operativi.

Nei sistemi operativi come Linux, la RAM gestisce l'esecuzione dinamica e fluida dei processi, mentre i dati che non devono corrompersi risiedono sul filesystem, agganciati a puntatori rigidi chiamati **Inode**. HySA porta esattamente questa logica dentro lo spazio latente del Transformer:

| Concetto di Sistema (OS) | Ruolo in HySA | Cosa fa in pratica |
| :--- | :--- | :--- |
| **Inode ($\mu_k$)** | Punto di ancoraggio Monosemantico ($N_m$) | È la coordinata esatta nello spazio del modello a cui è associato un dato immutabile (es. `' 8.8.8.8'`). |
| **Filesystem ($\mathcal{F}_{\mathrm{mono}}$)** | Registro dei fatti indicizzati | L'insieme dei punti Inode certificati, interrogabile ad alta velocità ($O(\log M)$). |
| **Query / API ($Q$)** | Operatore di verifica di confine | La chiamata interna che intercetta lo stato del modello e restituisce il dato deterministico. |
| **Page Fault ($\bot_{\mathrm{Fault}}$)** | **Semantic Fault** | L'errore di sistema che blocca il modello quando cerca un dato preciso non indicizzato, evitando che tiri a indovinare. |
| **Temperatura** | Intorno Metrico $\mathcal{B}(\mu_k, T_t)$ | Volume metrico multidimensionale di tolleranza attorno al centroide concettuale. |

*Tabella 1: Mappatura delle Euristiche dei Sistemi Operativi sui Costrutti Architetturali in HySA.*

### Le Due Temperature ($T_p$ e $T_m$) e la Temperatura Topologica ($T_t$)

Nei modelli standard esiste una sola temperatura, ed è un banale divisore scalare applicato alla fine di tutto (sui logit) per appiattire o appuntire le probabilità delle parole. È un parametro "cieco" rispetto a cosa sta dicendo il modello.

In HySA la temperatura diventa una proprietà geometrica dello spazio intorno al neurone (l'**Ipersfera Semantica**) e si divide in due regimi:

1. **Temperatura Polisemantica ($T_p$):** Governa il movimento fluido. Più è alta, più permette al modello di esplorare connessioni laterali, fare analogie e muoversi liberamente tra i concetti.
2. **Temperatura Monosemantica ($T_m$):** Governa la tolleranza dell'Inode. È stretta, rigida, e definisce quanto devi essere vicino a un concetto fattuale per poterlo attivare senza ambiguità.

Dall'interazione tra queste due nasce la **Temperatura Topologica ($T_t$)**, che definisce il raggio d'azione effettivo (il confine di tolleranza) attorno a ciascun Inode:
$$
\mathcal{B}(\mu_k, T_t) = \left\{ x \in \mathcal{S} \;\middle|\; \| x - \mu_k \|_2 \le T_t(T_p, T_m) \right\}
$$
formalizzata via accoppiamento saturante con bound $\alpha \in (0, 1)$:
$$
T_t(T_p, T_m) = T_m \left( 1 + \alpha \tanh\left( \frac{T_p}{T_m} \right) \right)
$$
Sulla sfera unitaria $S^{D-1}$, la distanza euclidea $\ell_2$ coincide esattamente con la deviazione angolare: $\|x - \mu_k\|_2 = \sqrt{2(1 - \cos\theta)}$.

![Figura 3: Geometria dell'Ipersfera e Architettura](assets/fig3_hypersphere_geometry.png)
*Figura 3: **Geometria Concentrica dei Domini Semantici, Architettura Early-Exit e Probing Empirico su `Pythia-70m`.** **(a)** Geometria concentrica attorno all'Inode $\mu_k$: nucleo deterministico $T_t = 0.46$ (conferma fattuale ed Early-Exit), fascia cuscinetto $R_{\mathrm{dom}} = 0.65$ (*Semantic Fault* su fatti non indicizzati nel dominio) e spazio fluido aperto $d^* > R_{\mathrm{dom}}$ (generazione libera $N_p$). **(b)** Macro-architettura con Early-Exit al layer intermedio ($l=3$): l'operatore $Q$ intercetta lo stato $q_l$; in caso di hit ($\le T_t$) esegue l'Early-Exit bypassando i layer $4 \dots 6$, mentre le traiettorie fuori dominio ($> R_{\mathrm{dom}}$) proseguono verso $\mathbf{W}_U$. **(c)** Distanze empiriche su `Pythia-70M` (Layer 3, $D=512$): separazione metrica netta tra fatti indicizzati ($d^* \le 0.4304$, Early-Exit), query non indicizzata ($d^* = 0.4973$, Semantic Fault) e prompt creativi (`Sky`, $d^* = 1.0430$, Bypass fluido).*

### Il meccanismo di "Query / API" interna e il Gating Geometrico di Dominio

Usando una terminologia propria dell'architettura dei sistemi, **l'operatore $Q$ di HySA funziona come una chiamata a un'API interna che impone consistenza deterministica al modello.**

Un errore comune nei sistemi ibridi è tentare di attivare la verifica fattuale misurando l'incertezza (entropia di Shannon) sui logit dell'ultimo layer. Questo approccio fallisce per due motivi opposti:
1. **Il Falso Positivo Creativo:** Davanti a una frase aperta (es. *"Oggi il cielo appare..."*), decine di aggettivi sono equiprobabili; l'entropia dei logit schizza in alto e un trigger basato sull'incertezza invocherebbe a sproposito il database dei fatti, mandando il sistema in *Semantic Fault* nel mezzo di un discorso creativo.
2. **L'Allucinazione Sicura (*Overconfident Hallucination*):** Gli LLM spesso generano dati errati con altissima confidenza statistica (entropia quasi nulla). Un controllo posto a valle sui logit non scatterebbe mai, lasciando passare l'errore.

In HySA, il Gating non dipende dall'incertezza sulle parole finali, ma dalla **posizione geometrica dello stato latente a metà rete** (Layer intermedio $l$, con $1 < l < L$, dove la struttura sintattica della richiesta è già formata ma la decodifica lessicale $\mathbf{W}_U$ non è ancora avvenuta).

Attorno a ciascun Inode (o cluster di Inode affini) esistono due confini concentrici:
* **Il Raggio di Dominio ($R_{\mathrm{dom}}$):** Delimita la "zona di giurisdizione" dei fatti rigidi (es. lo spazio semantico in cui cadono tutte le frasi che richiedono un indirizzo IP, una porta di rete, una costante fisica o un identificativo).
* **La Temperatura Topologica ($T_t$, con $T_t < R_{\mathrm{dom}}$):** Delimita il nucleo rigido di conferma dell'Inode specifico ($\mu_k$).

Durante il forward pass al layer intermedio $l$, il vettore normalizzato $q_l = h_t^{(l)} / \|h_t^{(l)}\|_2$ interroga il registro $\mathcal{F}_{\mathrm{mono}}$ calcolando la distanza dall'Inode più vicino $d^* = \min_k \|q_l - \mu_k\|_2$. Si aprono esattamente tre regimi operativi privi di ambiguità:

> [!definition] Definizione 2 (Operatore di Confine Metrico e Gating di Dominio al Layer $l$)
> Data la traiettoria latente normalizzata $q_l = h_t^{(l)} / \|h_t^{(l)}\|_2$ al layer intermedio $l$, l'insieme degli Inode indicizzati $\mathcal{F}_{\mathrm{mono}} = \{\mu_1, \dots, \mu_M\}$, la soglia $T_t$ e il raggio di dominio $R_{\mathrm{dom}}$:
> $$
> Q(q_l, \mathcal{F}_{\mathrm{mono}}, T_t, R_{\mathrm{dom}}) = 
> \begin{cases} 
>   (\mathrm{CONFIRMED}, \mathbf{v}_{k^*}) & \text{se } d^* \le T_t \quad (\mathbf{Hit\;Inode\;Deterministico}) \\ 
>   (\mathrm{FAULT}, \bot_{\mathrm{Fault}}) & \text{se } T_t < d^* \le R_{\mathrm{dom}} \quad (\mathbf{In\text{-}Domain\;Drift}) \\
>   (\mathrm{BYPASS}, \mathrm{Fluid}) & \text{se } d^* > R_{\mathrm{dom}} \quad (\mathbf{Fuori\;Dominio\;/\;Creativo})
> \end{cases}
> $$

1. **Fuori Dominio ($d^* > R_{\mathrm{dom}}$ — Regime Fluido $N_p$):** Lo stato sta navigando in territori discorsivi, astratti o creativi, lontano da qualsiasi struttura fattuale rigida. L'API resta inattiva, il vettore $h_t^{(l)}$ prosegue attraverso i restanti layer $l+1 \dots L$ e viene campionato normalmente con la temperatura polisemantica $T_p$.
2. **Entro il Raggio di Dominio ma fuori dall'Inode ($T_t < d^* \le R_{\mathrm{dom}}$ — Semantic Fault):** La struttura della frase è entrata inequivocabilmente nel bacino semantico di una richiesta fattuale (es. sta cercando il numero di porta o l'indirizzo di un protocollo specifico), ma il dato richiesto non corrisponde a nessun Inode certificato entro la tolleranza $T_t$. L'API intercetta la traiettoria prima che raggiunga gli ultimi layer e solleva un **Semantic Fault** ($\bot_{\mathrm{Fault}}$), bloccando sia il tirare a indovinare sia l'allucinazione sicura.
3. **Entro la Temperatura Topologica ($d^* \le T_t$ — Hit Deterministico $N_m$):** Lo stato cade esattamente nel nucleo dell'Inode $\mu_{k^*}$. L'API esegue un **Early-Exit** (bypassando il calcolo dei layer successivi $l+1 \dots L$ e la proiezione sul vocabolario $\mathbf{W}_U$) ed emette direttamente il letterale certificato $\mathbf{v}_{k^*}$, risparmiando fino al 50% dei FLOPs.

### Il Punto di Equilibrio (Compression Gradient)

Quanto spazio occorre assegnare all'astrazione fluida ($N_p$) e quanto agli Inode rigidi ($N_m$)? Tutto dipende dal bilanciamento parametrizzato dal gradiente di compressione $\Delta q = \frac{N_p}{N_m \cdot M}$:

* **Sotto-indicizzazione ($\Delta q \gg 1$, Troppo fluido):** Se $N_p$ domina senza ancore $N_m$, il modello è un grande affabulatore ma inventa i dati tecnici (deriva associativa non vincolata).
* **Sovra-indicizzazione ($\Delta q \ll 1$, Troppo rigido):** Se si riempie lo spazio di Inode con raggi $T_t$ troppo larghi, le sfere di controllo finiscono per occupare tutto lo spazio di manovra. Il modello smette di ragionare, perde la capacità di fare sintesi e si blocca in una gigantesca tabella di lookup statica.
* **Il Punto Critico ($P_c$):** È il punto di lavoro ottimale dell'architettura, dove i raggi $T_t$ sono abbastanza stretti da lasciare libero lo spazio per il ragionamento polisemantico, ma abbastanza presenti da catturare con precisione chirurgica le traiettorie quando il discorso converge su un fatto noto.

> [!example] Algoritmo 1: Forward Pass Unificato con Gating Geometrico al Layer Intermedio ed Early-Exit
> **Input:** Token di contesto $\{x_{\le t}\}$, Layer del Transformer $\mathcal{L}_{1 \dots L}$, Layer di probing $l$ (es. $l=3$), Registro Monosemantico $\mathcal{F}_{\mathrm{mono}} = \{(\mu_k, \mathbf{v}_k)\}_{k=1}^M$, Temperature $T_p, T_m$, Raggio di Dominio Fattuale $R_{\mathrm{dom}}$ (con $T_t < R_{\mathrm{dom}}$).  
> **Output:** Token emesso $x_{t+1}$ oppure eccezione esplicita $\bot_{\mathrm{Fault}}$.
> 
> 1. **Propagazione fino al layer intermedio $l$:**
>    $$h_t^{(l)} \leftarrow \mathcal{L}_{1 \dots l}(x_{\le t}), \quad q_l \leftarrow \frac{h_t^{(l)}}{\|h_t^{(l)}\|_2}$$
> 2. **Calcolo della Temperatura Topologica:**
>    $$T_t \leftarrow T_m \left( 1 + \alpha \tanh\left(\frac{T_p}{T_m}\right) \right)$$
> 3. **Query sull'indice degli Inode ($\mathcal{F}_{\mathrm{mono}}$):**
>    $$k^* = \arg\min_{k \in \{1, \dots, M\}} \|q_l - \mu_k\|_2, \quad d^* = \|q_l - \mu_{k^*}\|_2$$
> 4. **se** $d^* \le T_t$ **allora** *(Hit sull'Inode — Verifica Deterministica)*
>    * **Early-Exit:** interrompi i layer $l+1 \dots L$ e restituisci il dato certificato: $x_{t+1} \leftarrow \mathbf{v}_{k^*}$
> 5. **altrimenti se** $T_t < d^* \le R_{\mathrm{dom}}$ **allora** *(Nel Dominio Fattuale ma Dato Ignoto)*
>    * Solleva **Semantic Fault**: ritorna $\bot_{\mathrm{Fault}}$ *(Allucinazione intercettata a metà rete)*
> 6. **altrimenti** *(Fuori dal Dominio Fattuale: $d^* > R_{\mathrm{dom}}$ — Regime Fluido)*
>    * Completa la propagazione fino all'ultimo layer $L$: $h_t^{(L)} \leftarrow \mathcal{L}_{l+1 \dots L}(h_t^{(l)})$
>    * Campiona normalmente dai logit con Temperatura Polisemantica $T_p$:
>      $$x_{t+1} \sim \mathrm{Softmax}\left(\frac{\mathbf{W}_U h_t^{(L)}}{T_p}\right)$$

---

## 4. Validazione Empirica su Transformer Reale (`Pythia-70m`): Early-Exit e Limiti del Bias Sintattico

Per verificare la fattibilità dell'intercettazione geometrica e misurare l'impatto computazionale di un vero Early-Exit, ho condotto test sperimentali sullo stream residuo al Layer 3 ($D=512$) del modello causale open-source `EleutherAI/pythia-70m`. L'indagine adotta un approccio rigoroso e comparativo, evidenziando sia il guadagno di latenza hardware ottenuto arrestando il grafo, sia i limiti fisici intrinseci della rappresentazione latente autoregressiva (*Lexical Recency Bias*).

I riscontri sperimentali documentano tre proprietà fondamentali:

1. **Vero Early-Exit e Abbattimento della Latenza Hardware:**  
   L'interruzione della computazione al Layer 3 non è simulata a posteriori, ma implementata arrestando fisicamente l'esecuzione del modello tramite forward hook PyTorch. In questo modo si bypassano interamente i layer rimanenti ($l=3 \dots 5$) e la costosa proiezione non vincolata sul vocabolario $\mathbf{W}_U \in \mathbb{R}^{512 \times 50304}$. Il benchmark empirico su GPU (CUDA) registra una riduzione della latenza da **$15.08\,\text{ms}$** (forward pass completo) a **$3.70\,\text{ms}$** (Early-Exit al Layer 3), pari a uno **speedup effettivo di $4.07\times$** (**risparmio di latenza del $75.5\%$**).

2. **Gating Concentrico sotto Template Controllati (Regime A):**  
   Quando le query condividono una struttura sintattica coerente con gli Inode censiti, l'operatore $Q$ separa nettamente i fatti noti ($d^* \in [0.0889, 0.4304] \le T_t$) innescando l'Early-Exit deterministico. Contestualmente, richieste relative a protocolli non indicizzati (*"telemetry protocol for subterranean speleological sensors..."*) cadono nella fascia di cuscinetto $T_t < d^* = 0.4973 \le R_{\mathrm{dom}}$, sollevando un **Semantic Fault** che intercetta la potenziale allucinazione a metà rete (Tabella 3).

3. **Emersione del Lexical Recency Bias sulle Query Naturali (Regime B):**  
   Nei modelli causali decoder-only, lo stato latente dell'ultimo token $h_t^{[-1]}$ a layer intermedi è fortemente condizionato dai token immediatamente antecedenti (sintassi locale $n$-gram). Riformulando la medesima richiesta in linguaggio naturale aperto (*"Which port does SSH listen on by default?"*), la distanza sale a $d^* = 0.9399 \dots 1.0127$, finendo oltre $R_{\mathrm{dom}}$ e rendendosi indistinguibile dal discorso fluido (Regime C, $d^* = 1.0430$).

| Regime | Concetto Target | Prompt di Query | Distanza $d^*$ | Stato HySA ($T_t=0.46, R_{\mathrm{dom}}=0.65$) | Output Operativo |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **A: Template** | `SSH_PORT` | *"...remote server management... SSH default port is"* | $0.3257$ | **CONFIRMED** ($d^* \le T_t$) | `' 22'` (Vero Early-Exit) |
| **A: Template** | `HTTP_PORT` | *"...hyper-text HTTP default plaintext port is"* | $0.4304$ | **CONFIRMED** ($d^* \le T_t$) | `' 80'` (Vero Early-Exit) |
| **A: Template** | `PUBLIC_DNS_IP` | *"...Google public DNS IPv4 literal address is"* | $0.3889$ | **CONFIRMED** ($d^* \le T_t$) | `' 8.8.8.8'` (Vero Early-Exit) |
| **A: Template** | `SUBNET_MASK` | *"...Class C 255.255.255.0 has CIDR bit length of"* | $0.1263$ | **CONFIRMED** ($d^* \le T_t$) | `' 24'` (Vero Early-Exit) |
| **A: Template Drift** | `UNINDEXED` | *"...speleological cave sensors default port number is"* | $0.4973$ | **SEMANTIC FAULT** ($T_t < d^* \le R_{\mathrm{dom}}$) | **[BLOCCATO]** (Mid-Net Fault) |
| **B: Naturale** | `SSH_PORT` | *"Which port does SSH listen on by default?"* | $1.0127$ | **FLUID $N_p$** ($d^* > R_{\mathrm{dom}}$) | *Bypass* (Recency Bias) |
| **B: Naturale** | `SSH_PORT` | *"What is the standard port for the SSH service?"* | $0.9399$ | **FLUID $N_p$** ($d^* > R_{\mathrm{dom}}$) | *Bypass* (Recency Bias) |
| **B: Naturale** | `SPELEO` | *"What is the port number for cave telemetry sensors?"* | $0.9473$ | **FLUID $N_p$** ($d^* > R_{\mathrm{dom}}$) | *Bypass* (Recency Bias) |
| **C: Fluido** | `CREATIVE_SKY` | *"Today the clear morning sky feels particularly..."* | $1.0430$ | **FLUID $N_p$** ($d^* > R_{\mathrm{dom}}$) | Generazione libera |

*Tabella 3: Benchmark di Probing Metrico al Layer 3 su `EleutherAI/pythia-70m` ($D=512$, $T_t=0.46, R_{\mathrm{dom}}=0.65$): Template Controllati vs. Linguaggio Naturale Aperto.*

> **Implicazione Architetturale:**  
> Questo riscontro prova che l'estrazione grezza dello stato all'ultimo token rappresenta un utile banco di prova preliminare sotto vincoli di template, ma non garantisce l'invarianza concettuale alle parafrasi aperte. Per il deployment di HySA in scenari open-vocabulary, l'operatore di verifica $Q$ richiede l'accoppiamento con un **Linear Probe supervisionato** ($W_{\mathrm{probe}}: \mathbb{R}^D \to \mathbb{R}^{D_{\mathrm{mono}}}$) o un meccanismo di **Attention Pooling** sui token rilevanti della sequenza.

---

## 5. Le Sfide Aperte: Come scalare davvero l'indicizzazione

Un prototipo su una manciata di Inode dimostra il principio meccanico, ma portare HySA in produzione su milioni di fatti richiede di risolvere due colli di bottiglia architetturali:

1. **Dal Flag manuale al Gating Geometrico al Layer Intermedio:**  
   Affidarsi all'entropia a valle dei logit fallisce sia sui discorsi creativi (falsi positivi per via delle molte parole plausibili) sia sulle allucinazioni pronunciate con sicurezza (falsi negativi a entropia quasi nulla). Posizionando il Gating a metà rete ($l=3$), HySA separa geometricamente lo spazio: le frasi creative ($d^* > R_{\mathrm{dom}}$) proseguono indisturbate, le allucinazioni fattuali vengono arrestate prima dei layer finali, e i fatti noti generano un Early-Exit immediato.
2. **Evitare il sovraffollamento dello spazio latente (Indexing ad alta dimensione):**  
   Indicizzare milioni di fatti dentro i 512 canali di un modello piccolo come Pythia-70m farebbe toccare e sovrapporre le sfere $T_t$. Tuttavia, passando a modelli reali ($D = 4096$ o $8192$) e strutturando il registro $\mathcal{F}_{\mathrm{mono}}$ non come lista piatta ma come un **grafo gerarchico indicizzato** (usando strutture HNSW / FAISS con quantizzazione IVF-PQ), lo spazio a disposizione cresce esponenzialmente grazie alla quasi-ortogonalità dei vettori casuali in alta dimensione, e il tempo di ricerca resta sub-lineare nell'ordine dei microsecondi ($O(\log M)$, $<15\,\mu\text{s}$).

---

## 6. Conclusione

Cercare di eliminare la polisemia dalle reti neurali per renderle interpretabili, equivale a voler smantellare la capacità associativa della cognizione perché non assimilabile a un archivio relazionale strutturato.

Con **HySA** propongo una prospettiva fondata sull’ingegneria dei sistemi reali: demandare ai neuroni polisemantici la sintesi associativa e creativa dell’astrazione analogica (governati dalla temperatura $T_p$), affiancando loro, nel medesimo spazio vettoriale unificato, un piano di verifica deterministica basato su Inode monosemantici ($T_m$). Non due modelli disgiunti in serie o in parallelo, ma una singola architettura capace di interrogare se stessa tramite chiamate API interne non appena la fluidità del discorso deve cedere il passo alla precisione del dato.

---

## Bibliografia

1. M. Rigotti, O. Barak, M. R. Warden, X.-J. Wang, N. D. Daw, E. K. Miller, e S. Fusi, "The importance of mixed selectivity in complex cognitive tasks," *Nature*, vol. 497, no. 7451, pp. 585–590, 2013.
2. S. Fusi, E. K. Miller, e M. Rigotti, "Why neurons mix: high dimensionality for higher cognition," *Current Opinion in Neurobiology*, vol. 37, pp. 66–74, 2016.
3. N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S. Kravec, Z. Hatfield-Dodds, R. Lasenby, D. Drain, C. Chen, et al., "Toy models of superposition," *Transformer Circuits Thread*, 2022.
4. T. Bricken, A. Templeton, J. Marks, S. Marcus, E. Mindermann, L. Petrini, S. Kundu, N. Pangh, e C. Olah, "Towards monosemanticity: Decomposing language models with dictionary learning," *Transformer Circuits Thread*, 2023.
5. A. Templeton, T. Conerly, N. Marcus, T. Bricken, K. Trendell, L. Petrini, T. Henighan, e C. Olah, "Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet," *Transformer Circuits Thread*, 2024.
6. M. Geva, R. Schuster, J. Berant, e O. Levy, "Transformer feed-forward layers are key-value memories," in *Proc. EMNLP*, pp. 5484–5495, 2021.
7. K. Meng, D. Bau, A. Andonian, e Y. Belinkov, "Locating and editing factual associations in GPT," in *Proc. NeurIPS*, vol. 35, pp. 17359–17372, 2022.
8. U. Khandelwal, O. Levy, D. Jurafsky, L. Zettlemoyer, e M. Lewis, "Generalization through memorization: Nearest neighbor language models," in *Proc. ICLR*, 2020.
9. D. Kahneman, *Pensieri lenti e veloci*, Mondadori, 2011.
