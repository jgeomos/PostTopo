# Technical note — what the PostTopo notebooks compute

J. Giraud · 11 November 2026

## Scope

The two notebooks of the [PostTopo repository](https://github.com/jgeomos/PostTopo) describe each model of a posterior ensemble by a feature vector, embed the vectors in two dimensions with Uniform Manifold Approximation and Projection (UMAP; McInnes et al. 2018), cluster the embedding with Density-Based Spatial Clustering of Applications with Noise (DBSCAN; Ester et al. 1996), and read the clusters with geometrical and topological motifs. These steps follow the posterior analysis workflow of Giraud et al. (2026b), except the motif reading, which extends it. Both notebooks run the same steps (Fig. 1) on two ensembles:

| Notebook | Ensemble |
| --- | --- |
| `cluster_workflow_synthetic.ipynb` | Synthetic ensemble of 3 × 1000 models whose geometry is known model by model |
| `cluster_workflow_field.ipynb` | Boulia gravity ensemble: 2000 post-burn-in models from five pseudo-trans-dimensional inversion chains (Giraud et al. 2026b) |

Section numbers below follow those of the notebooks; the code is in `modules/`.

![Figure 1: steps of the two notebooks](fig1_workflow.png)

**Figure 1.** Steps of the two notebooks, labelled with the notebook section numbers. The feature route (left) groups the models without reference to geology; the motif route (right) describes each model in geological terms. The enrichment reads the clusters of the first route with the motifs of the second. Hypothesis probabilities are computed over the whole ensemble or one cluster, per chain or per cluster.

## 1. Data

Both notebooks work on an array `core` of shape (n\_z, n\_y, n\_x, N): N models of n\_z × n\_y × n\_x cells, depth index 0 at the top, lateral padding of the inversion grid removed.

**Models.** The pseudo-trans-dimensional (trans-D) sampler builds each model as a grid of densities from n\_r signed-distance fields φ\_i, positive inside rock unit i, zero on its contacts and negative outside, and from the unit densities ρ\_i (Osher & Sethian 1988; Giraud et al. 2021, 2026a):

```math
m(\mathbf{x}) = \sum_{i=1}^{n_r} \rho_i\, H\big(\phi_i(\mathbf{x})\big) \prod_{j \ne i} \Big(1 - H\big(\phi_j(\mathbf{x})\big)\Big),
```

where m(**x**) is the density at cell **x** and H the Heaviside step function, so that each cell takes the density of the unit whose signed distance is positive there. A model therefore doubles as a grid of unit labels. The number n\_r of signed-distance fields is fixed to a maximum number of units, and births and deaths activate and deactivate units, so that the number of live units (units with φ\_i > 0 somewhere) varies across the ensemble (Giraud et al. 2026a). Densities are sampled as well, so that the same unit carries a slightly different value in each model. The synthetic generator below draws models of this kind directly, without signed distances.

**Synthetic ensemble** (`synthetic_ensemble.py`). No data are inverted: each model is drawn independently by a generator that mimics the trans-D sampler, on a grid of 36 × 54 × 64 cells with 6 padding cells on each lateral side (36 × 42 × 52 once removed). The prior model holds a cover (above depth index 6), sediments (down to 16) and a basement, of nominal densities 2300, 2550 and 2800 kg m⁻³, sampled with a standard deviation of 10 kg m⁻³. Interfaces are roughened by smoothed Gaussian noise; a model may hold a basement high (amplitude 6–18 cells) and born lenses (2950 kg m⁻³, in the sediments, tilt 0–25°), dykes (3000 kg m⁻³, dip 45–90°) and pipes (2900 kg m⁻³, plunge 30–90°, from the basement into the cover); a few isolated cells per model, carrying the density of another unit, mimic sampler noise. The dip and plunge ranges straddle the motif thresholds of Section 5, so that the ensemble holds bodies on both sides of them. Three parameter sets mimic three chains:

| Chain | p\_lens | p\_dyke | p\_pipe | p\_high | Roughness (cells) |
| --- | --- | --- | --- | --- | --- |
| chain 1 (`A_baseline`) | 0.60 | 0.10 | 0.30 | 0.50 | 2.0 |
| chain 2 (`B_birthy`) | 0.80 | 0.40 | 0.50 | 0.50 | 2.0 |
| chain 3 (`C_smooth`) | 0.30 | 0.05 | 0.15 | 0.20 | 1.0 |

where p is the probability that a model holds the body. The table `truth` records what was drawn in each model; it is used to read the clusters (Section 3) and to check the motifs (Section 6).

**Boulia ensemble** (`data/boulia_5chains_2000models.npz`). The models come from the pseudo-trans-D inversion of 2001 gravity data points in the Boulia region (Queensland, Australia) described by Giraud et al. (2026b). The prior model holds six units (Rashidifard et al. 2024), between 6 and 20 units may be live, and births are not allowed shallower than 700 m. Five chains, run with slightly different weighting parameters (their Table 1), were thinned to every 100th model, and their first 250,000 iterations were discarded as burn-in, leaving 17,500 models. The sampler forces the acceptance of a small fraction of birth, death and geometry-simplification proposals, so that the chains form an exploration ensemble guided by the data rather than a sample of the exact posterior (Giraud et al. 2026b). The file holds 400 of these models from each chain, evenly spaced after burn-in, lateral padding removed; on loading, the depth axis is flipped and the two lateral axes are swapped.

## 2. Features

Each model m\_i (i = 1, …, N) is described by a vector **f**\_i of 523 values: 512 from a fixed-kernel encoder, 6 topological scalars and 5 density moments (`ensemble_features.py`). None of these requires units to be matched across models. They follow Giraud et al. (2026b, Sect. 3), except that the repository computes five density moments (no median) and labels connected components with face connectivity rather than the 26-cell neighbourhood.

**Encoder (512 values).** The encoder is the untrained convolutional network of Giraud et al. (2026b, Sect. 3.3). The densities are first scaled to [0, 1] with the global minimum and maximum of the ensemble. The first block applies 351 fixed 3 × 3 × 3 difference kernels, one per unordered pair (a, b) of the 27 cells of the neighbourhood, with +1/√2 at offset **o**\_a and −1/√2 at offset **o**\_b. Kernel (a, b) applied at cell **x** therefore returns

```math
r_{ab}(\mathbf{x}) = \frac{1}{\sqrt{2}}\,\big|\,m(\mathbf{x} + \mathbf{o}_a) - m(\mathbf{x} + \mathbf{o}_b)\,\big|,
```

where m(**x**) is the scaled density at **x**, followed by 2 × 2 × 2 max pooling. Blocks 2 and 3 apply the same kernels depthwise (kernel c on channel c), again with absolute value and max pooling. The output of each block is average-pooled (2 × 2 × 2), flattened and projected on a fixed random matrix with orthonormal columns (QR decomposition of a standard normal matrix, seeds 1, 2, 3), to 128, 128 and 256 dimensions (Johnson & Lindenstrauss 1984). Each branch is L2-normalised and the three are concatenated. Since r\_ab vanishes inside a unit, the 512 values describe where the unit contacts lie and their density contrast, at three scales.

**Topological scalars (6 values, `topo_features.py`).** Units are the distinct density values of the model; a connected component is a maximal set of cells of one unit linked through shared faces (6-connectivity; Silversmith 2021).

- k: number of units, set by the birth–death process.
- total\_cc = Σ\_u n\_u, where n\_u is the number of connected components of unit u; total\_cc ≥ k.
- Mean Herfindahl–Hirschman index (HHI; Herfindahl 1950): for unit u, HHI\_u = Σ\_i f\_i², where f\_i = V\_i / V\_u is the volume fraction of its component i; HHI\_u = 1 for one body and 1/n for n equal pieces. The mean is taken over the k units.
- Weighted Fiedler value (Fiedler 1973; Chung 1996): the second-smallest eigenvalue of the Laplacian of the unit contact graph,

```math
F_w = \lambda_2(\mathbf{L}), \qquad \mathbf{L} = \mathbf{D} - \mathbf{A}_w, \qquad D_{ii} = \sum_j (\mathbf{A}_w)_{ij},
```

where (A\_w)\_ij is the number of cell faces shared by units i and j and D is the diagonal degree matrix. F\_w is small when the units split into two groups sharing little contact.

- Hyperedge entropy (Berge 1989; Shannon 1948): at each grid vertex, the units of the 8 cells sharing it are collected; a vertex where three or more units meet is a junction, and its type is the set of those units, i.e. which units meet, not how many cells each occupies. Each type is a hyperedge of the contact hypergraph, linking all the units that meet there at once. With n\_i the number of vertices of type i among t types,

```math
H_\varepsilon = -\sum_{i=1}^{t} p_i \log_2 p_i, \qquad p_i = \frac{n_i}{\sum_{j=1}^{t} n_j},
```

in bits; H\_ε = 0 when the model holds no junction (a layered model) or a single junction type (lenses of one unit lying on the same interface).

- Mean ellipticity: for each unit, E = 1 − λ\_min / λ\_max, where λ\_min and λ\_max are the smallest and largest eigenvalues of the covariance matrix of the cell indices of its largest component; E = 0 for an equant body and tends to 1 for an elongated or flat one. The mean is taken over the units.

**Density moments (5 values).** Mean, standard deviation, skewness, excess kurtosis and range of the densities of all cells of the model.

**Standardisation and cache.** The 523 columns are standardised (zero mean, unit variance over the ensemble), so that features of different units and magnitudes (encoder values below 1, counts, densities in kg m⁻³) become comparable. The three groups are saved in `data/` with a fingerprint of the ensemble and of the version of the feature definitions (`FEATURES_VERSION` in `ensemble_features.py`), and are reloaded only when the notebook runs again on the same models with the same definitions.

## 3. Embedding and clustering

The standardised feature vectors are embedded in two dimensions, and the embedding, not the 523-dimensional space, is clustered.

**UMAP (cell 3.1).** UMAP builds a graph of the `UMAP_N_NEIGHBORS` nearest neighbours of each model in feature space, here with the cosine distance

```math
d_{\cos}(\mathbf{f}_i, \mathbf{f}_j) = 1 - \frac{\mathbf{f}_i \cdot \mathbf{f}_j}{\lVert \mathbf{f}_i \rVert\, \lVert \mathbf{f}_j \rVert},
```

which depends on the angle between two vectors and not on their lengths, and finds 2-D coordinates whose neighbourhood graph matches it (McInnes et al. 2018). `UMAP_MIN_DIST` (0.05) sets how tightly neighbouring points may pack. With few neighbours (3 for the synthetic ensemble, 5 for Boulia), the embedding keeps local structure: models differing by small changes, typically consecutive models of a chain, form thin strings rather than blobs.

**DBSCAN (cell 3.2).** Two points of the embedding are neighbours if they are closer than `EPS` (embedding units). A point with at least `MIN_SAMPLES` points within that distance (5, itself included) is a core point; clusters are the sets of points reachable through chains of core points, and points reachable from no core point are noise, labelled −1 (Ester et al. 1996). `EPS` depends on N and on the UMAP settings, and is chosen near the knee of the k-distance curve, the sorted distance of each point to its `MIN_SAMPLES`-th neighbour (0.225 for the synthetic ensemble, 0.45 for Boulia). With `CLUSTERER = 'hdbscan'`, hierarchical DBSCAN (HDBSCAN; Campello et al. 2013) replaces `EPS` by a minimum cluster size; it adapts to clusters of varying density but tends to merge the strings. On the 17,500 Boulia models, Giraud et al. (2026b) used 30 neighbours, a minimum distance of 0, EPS = 0.25 and MIN\_SAMPLES = 5.

**Cluster table (cell 3.3, commented out by default).** For each cluster c, sorted by size: the number of models |c|, the share of each chain, the mean over cells of the standard-deviation model of Section 4 (spread of the cluster, in density units), and the representative model, i.e. the member closest to the centroid of the cluster in the embedding. For the synthetic ensemble, the same cell gives the enrichment of each drawn body (basement high, lens, dyke, pipe),

```math
E_{\text{body}, c} = \frac{P(\text{body} \mid c)}{P(\text{body})},
```

which compares the fraction of the members of c holding that body with the fraction over the whole ensemble (noise included): E ≫ 1 marks what the cluster is made of, E ≈ 0 what it excludes.

## 4. Cluster statistics and viewer

Each cluster is summarised, cell by cell, by the median and standard-deviation models of its members:

```math
\tilde m_c(\mathbf{x}) = \operatorname{median}_{i \in c}\, m_i(\mathbf{x}), \qquad s_c(\mathbf{x}) = \Big(\frac{1}{|c|}\sum_{i \in c} \big(m_i(\mathbf{x}) - \bar m_c(\mathbf{x})\big)^2\Big)^{1/2},
```

where m\_i(**x**) is the density of model i at cell **x**, |c| the number of members of cluster c and m̄\_c(**x**) their mean. The median keeps sharp interfaces where most members agree; s\_c is large where they disagree. Sections are named by the axis held fixed: the section at y = `IY` is an x–z panel, at x = `IX` a y–z panel, and the map at z = `IZ` an x–y panel.

**Tiles (cell 4.1).** Median and standard-deviation models of several clusters, one tile per cluster. The synthetic notebook shows the 12 clusters whose median models differ most (`select='diverse'`: chosen greedily from the largest, by the mean absolute difference over all cells), each on the y-section where its median model has most lateral structure (`'best'`: most cells departing from the median of their depth slice by more than 10 % of the colour range of the densities). The field notebook shows the 30 largest clusters on the section y = `IY`.

**Largest clusters (cell 4.2).** Median, standard-deviation and representative models of the `N_SHOW` largest clusters, on the section y = `IY` and the map z = `IZ`.

**Viewer (cell 4.3).** A self-contained HTML page (Bokeh), also saved in `outputs/`: hovering over a point of the embedding shows the section of its model, its cluster, its chain and, for the synthetic ensemble, the bodies drawn in it; tapping a point, or choosing in the menu, shows the median, standard-deviation and representative models of its cluster; the page also holds the cluster table. Clusters smaller than `MIN_SAMPLES`, and those beyond the `MAX_CLUSTERS` largest, are grey.

## 5. Motifs and topology of the clusters

Independently of the features, each model is reduced to a graph of its bodies and to a row of motif counts (`motif_analysis.py`); comparing motif frequencies inside and outside a cluster names the cluster in geological terms (right-hand route of Fig. 1).

**Unit identity across models (cell 5.2).** Motifs need to know which unit is the cover, which the basement and which units were born, but sampled densities cannot be compared across models (label switching; Stephens 2000; Jasra et al. 2005). Each model is matched to a reference model r: the prior model of the generator (synthetic) or of the sampler (`data/boulia_prior_model.npy`, oriented on the grid of `core` automatically; a modal stand-in is built without it). Reference units are numbered k = 0, …, K − 1 by increasing mean depth, so that 0 is the cover and K − 1 the basement; units occupying less than `MIN_UNIT_SHARE` (1 %) of the grid are absorbed by their nearest unit. For a model with distinct density values v,

```math
T_{vk} = \#\{\mathbf{x} : m(\mathbf{x}) = v,\ r(\mathbf{x}) = k\}
```

is the number of cells carrying v in m and lying in reference unit k. Value v takes code k when each is the best match of the other (k = argmax\_k′ T\_vk′ and v = argmax\_v′ T\_v′k); a reference unit left unmatched goes to the largest value with at least half of its cells in it (a thinned or displaced unit). All other values are born units, coded K, K + 1, ….

**Body graph.** A body b is a maximal set of cells of one unit connected through shared faces (6-connectivity; Rosenfeld & Pfaltz 1966; Silversmith 2021), and |b| its number of cells. Bodies with |b| < V\_min (`VMIN_VOXELS`: 8 for the synthetic ensemble, 52 for Boulia) are fragments, i.e. discretisation noise of the level sets, and are dropped. The body graph has one node per remaining body and one edge per pair of bodies sharing at least one face (Thiele et al. 2016a). For body b and unit u, n\_↑(b, u), n\_↓(b, u) and n\_↔(b, u) are the numbers of faces across which u lies above, below and beside b; they give superposition without surface normals.

**Shape.** The shape of b is summarised by its second-moment tensor

```math
\mathbf{M}_b = \frac{1}{|b|}\sum_{j \in b} (\mathbf{x}_j - \bar{\mathbf{x}}_b)(\mathbf{x}_j - \bar{\mathbf{x}}_b)^{\mathsf T} + \frac{1}{12}\,\mathrm{diag}(\Delta z^2, \Delta y^2, \Delta x^2),
```

where **x**\_j is the centre of cell j, x̄\_b the centroid of b, (Δz, Δy, Δx) the cell sizes (`CELL_SIZES`) and the second term the second moment of one cell. Its eigenvalues λ\_1 ≥ λ\_2 ≥ λ\_3, with unit eigenvectors **e**\_1 (long axis) and **e**\_3 (short axis), give the shape and strength parameters of Woodcock (1977),

```math
K = \frac{\ln(\lambda_1/\lambda_2)}{\ln(\lambda_2/\lambda_3)}, \qquad C = \ln\frac{\lambda_1}{\lambda_3},
```

with K > 1 for rod-like and K < 1 for sheet-like bodies, and C = 0 for an equant body, whose K is undefined. A body receives a shape class only if |b| ≥ 50, C ≥ 1 (axis ratio above about 1.65) and at most 25 % of its surface lies on the lateral faces of the model (a truncated body has the shape of the cut). It is a rod if K > 2, a sheet if K < 0.5, and steep if it is a rod with |**e**\_1 · **ẑ**| > 0.7 (plunge above about 44°) or a sheet with |**e**\_3 · **ẑ**| < 0.3 (dip above about 73°), **ẑ** being the vertical unit vector. Lateral continuity is measured by the footprint fraction F\_b = n\_c(b) / (n\_y n\_x) and the aspect ratio A\_b = h̄\_b / √(n\_c(b) Δy Δx), where n\_c(b) is the number of map-view columns holding a cell of b and h̄\_b = |b| Δz / n\_c(b) its mean thickness.

**Motifs.** In network science a motif is a recurring subgraph pattern (Milo et al. 2002); here it is a body together with predicates on its attributes and contacts. c\_k(m) is the number of bodies of model m realising motif k. Defaults are those of `ma.DEFAULTS`, overridden by `THR` in cell 5.1.

| Motif | Predicate (defaults) | Geological reading |
| --- | --- | --- |
| `enclosed_body` | touches no face of the model; ≥ 75 % of its surface against one unit | lens, enclave |
| `layer` | F\_b ≥ 0.25, A\_b ≤ 0.3, bottom contact on ≤ 50 % of its footprint, different dominant units above and below (or the model top above) | laterally continuous sheet |
| `steep_body` | shaped and steep | dyke or pipe |
| `cross_cutting` | steep, lateral contacts with ≥ 2 distinct units | intrusion cutting several units |
| `dyke` | `cross_cutting` and a sheet | dyke cutting the layering |
| `chimney` | a rod with ≥ `amin_link` (1) faces against the basement and as many against the cover, any plunge | pipe linking basement and cover |
| `window` | basement body with ≥ V\_min faces directly below the cover or on the model top | basement breaching the cover |
| `bridge` | articulation point (Hopcroft & Tarjan 1973), not a layer, whose removal leaves ≥ 2 sub-graphs reaching the boundary, each at least as voluminous as the body | separator |
| `order_violation` | contact whose majority superposition contradicts the stratigraphic order, over ≥ `amin_violation` inverted faces (1; 100 for Boulia) | inverted contact |
| `fragment` | body below V\_min | sampler noise (diagnostic only) |

**Topology key.** Merging all bodies of a unit gives the unit-level contact graph (Thiele et al. 2016a, b). The key T(m) of a model is the set of pairs of unit classes in contact, the classes being the K reference units plus one class grouping all born units, together with, for each class, its number of bodies capped at `MAX_BODIES` (5). Grouping born units and capping the counts keeps sampling noise from making every model a class of its own. Topology classes are numbered by decreasing frequency in the ensemble (#1 most frequent).

**Enrichment (cells 5.3 and 5.4).** For a cluster c of |c| models and a motif k,

```math
P(k \mid c) = \frac{1}{|c|}\sum_{i \in c} \mathbf{1}\{c_k(m_i) > 0\}, \qquad P(k) = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\{c_k(m_i) > 0\}, \qquad E_{kc} = \frac{P(k \mid c)}{P(k)},
```

where 1{·} is 1 when the condition holds and 0 otherwise, P(k | c) and P(k) are the fractions of models holding motif k in the cluster and in the ensemble (noise included), and E\_kc is the enrichment, i.e. over-representation against the ensemble taken as null model (Milo et al. 2002). E\_kc ≫ 1 marks the motifs that characterise the cluster and E\_kc ≈ 0 those it excludes. The clusters come from the features, not from the motifs, so that the enrichment is an independent reading of them. Cell 5.3 lists P(k | c) and the most frequent topology class of each cluster; E\_kc is shown in the motif viewer of cell 5.4.

**Motif viewer (cell 5.4).** The viewer of Section 4 plus the motifs, numbers of units and born units and topology class of the model under the cursor, colouring of the embedding by motif, and a heatmap of E\_kc (clusters × motifs, logarithmic colour scale; tapping a row selects the cluster).

## 6. Hypotheses and examples

A geological statement becomes a posterior probability by turning it into an indicator evaluated on the motif table and averaging it over the selected models.

**Estimator.** A hypothesis H (e.g. "the basement breaches the cover") defines the indicator 1\_H(m), equal to 1 when model m satisfies H and 0 otherwise. Its posterior probability is the expectation of the indicator, estimated by the sample proportion (Robert & Casella 2004)

```math
P(H \mid d) \approx \frac{1}{N_s}\sum_{i=1}^{N_s} \mathbf{1}_H(m_i),
```

where d denotes the data, the sum runs over the selected models and N\_s is their number: the whole ensemble (`CLUSTER = None`) or the members of one cluster, read in the viewer; within a cluster, the estimate is P(H | d, c). Cell 6.4 computes it separately for each chain (`GROUP_BY = 'chain'`) or for each of the `N_GROUPS` largest clusters (`GROUP_BY = 'cluster'`), N\_s being then the number of models of that group, and shows the binomial standard error √(P (1 − P) / N\_s), where P = P(H | d), as error bars. This error assumes independent samples; consecutive models of a chain are autocorrelated, so that N\_s overstates the effective number of samples (Geyer 1992). The chains were run with different hyperparameters, so that a hypothesis whose probability changes from chain to chain depends on them; per cluster, the hypotheses that separate the clusters stand out. For Boulia, the chains form an exploration ensemble rather than a sample of the exact posterior (Section 1), so that P(H | d) reads as the frequency of H in that ensemble.

| Hypothesis | Indicator |
| --- | --- |
| at least one unit born | `n_born` > 0, the number of units not matched to a reference unit |
| basement is one body | the deepest reference unit has a single body |
| lens / enclosed body present | `enclosed_body` > 0 |
| basement breaches the cover | `window` > 0 |
| pipe from basement into cover | `chimney` > 0 (synthetic only, see below) |
| steep dyke cutting several units | `dyke` > 0 (not `cross_cutting`, which also counts pipes) |
| basement within `DEPTH_CELLS` cells of the surface | `z_deep_top` < `DEPTH_CELLS`, the depth index of the shallowest basement cell |

Further hypotheses are added in `MY_HYPOTHESES` as functions of the motif table; they combine with Boolean operators, and the probability of a conjunction can be compared with the product of the separate probabilities.

**Clearest examples (cells 6.2 and 6.3).** For each motif, up to 200 models carrying it are checked, and the body shown is the one realising it most clearly. Each criterion j of the motif compares an attribute x\_j(b) of body b with its threshold t\_j, through the normalised margin

```math
\mu_j(b) = \min\!\Big(1, \max\!\Big(0, \frac{x_j(b) - t_j}{r_j - t_j}\Big)\Big), \qquad s(b) = \min_j \mu_j(b),
```

where r\_j is a value regarded as fully typical, so that μ\_j = 0 at the threshold and 1 when fully typical (on log x\_j, over one decade past the threshold for K and two for contact areas and volume ratios); the clarity s(b) is their minimum, the conjunction of fuzzy logic (Zadeh 1965). Two criteria of the geological reading that the definitions do not test are added: the plunge of a `chimney` (0 at about 45°, 1 when vertical), since the definition accepts flat rods lying between basement and cover, and a localised `window` (0 when half of the basement footprint lies under the cover, 1 at a tenth or less). Bodies with s ≥ 0.5 are ranked first, then those carrying no other motif, then by s and by the number of other flagged bodies in their model. Clarity only selects what is shown; it does not change counts or probabilities.

**Motifs shown.** `SHOW_MOTIFS` (cell 6.1) selects the motifs ranked in cell 6.2 and summarised in cell 6.4: in the synthetic notebook, the four motifs that each correspond to one kind of drawn body (`cm.VALIDATED_MOTIFS`: `chimney` and pipes, `dyke` and dykes dipping more than about 73°, `window` and basement highs, `enclosed_body` and lenses); in the field notebook, seven motifs. Cell 6.3 redefines the list for its catalogue, and cell 6.4 uses the redefined list when run after it. In the synthetic notebook, cell 6.4 also gives, next to P(motif present), the fraction of the selected models holding the corresponding drawn body (`P drawn`, from `truth`), which shows where a motif falls short of the object it is meant to capture: a basement high, for instance, gives a `window` only where it reaches the cover.

**Boulia.** Born bodies are less regular than the synthetic ones, and their shape classes depend on `CELL_SIZES` and `THR`. The hypothesis on pipes is left out (`DROP_HYPOTHESES`): `chimney` counts any rod touching both basement and cover, whatever its plunge, and does not separate pipes from rods lying along the basement–cover contact.

## Practical notes

- **Comparability.** Two ensembles are comparable in feature space only if encoded with the same density range (`vmin`, `vmax` of `DiffNetEncoder.encode_ensemble`). `EPS` is chosen anew for each ensemble, and UMAP embeddings change between UMAP versions.
- **Small clusters.** P(k | c) is a proportion over |c| models: in a cluster of 10 models, an enrichment of 2 for a motif present in 20 % of the ensemble rests on 4 models.
- **Thresholds.** Motif probabilities are conditional on `THR`, `VMIN_VOXELS` and `CELL_SIZES`, which are reported with them. A probability that follows a threshold is a property of the definition, not of the models.
- **Cell sizes.** With `CELL_SIZES = (1, 1, 1)`, a placeholder in the field notebook, shape classes, steepness and layer aspect are computed in cells, not in metres; the Boulia mesh is set before these are reported.
- **Outputs.** Features are cached in `data/`; the embedding, cluster labels (`outputs/<FILE_PREFIX>_cluster_labels.npy`) and viewers are written to `outputs/`, which is not tracked by git. Both notebooks run in Google Colab from cell 0.1 without a GPU.

## References

- Berge, C., 1989. *Hypergraphs: Combinatorics of Finite Sets*, North-Holland, Amsterdam.
- Campello, R.J.G.B., Moulavi, D. & Sander, J., 2013. Density-based clustering based on hierarchical density estimates, *PAKDD*, LNCS 7819, 160–172.
- Chung, F., 1996. *Spectral Graph Theory*, CBMS Regional Conference Series in Mathematics, 92, American Mathematical Society.
- Ester, M., Kriegel, H.-P., Sander, J. & Xu, X., 1996. A density-based algorithm for discovering clusters in large spatial databases with noise, *Proc. KDD-96*, 226–231.
- Fiedler, M., 1973. Algebraic connectivity of graphs, *Czech. Math. J.*, 23(2), 298–305.
- Geyer, C.J., 1992. Practical Markov chain Monte Carlo, *Stat. Sci.*, 7(4), 473–483.
- Giraud, J., Lindsay, M. & Jessell, M., 2021. Generalization of level-set inversion to an arbitrary number of geologic units in a regularized least-squares framework, *Geophysics*, 86(4), R623–R637.
- Giraud, J., Ogarko, V., Caumon, G., Pirot, G., Portes, L., Cupillard, P. & Herrero, J., 2026a. Pseudo trans-dimensional 3-D geometrical inversion: a proof of concept using gravity data, *Geophys. J. Int.*, 244, doi:10.1093/gji/ggaf501.
- Giraud, J., Lindsay, M., Ogarko, V., Caumon, G., Portes, L., Pirot, G., Cupillard, P., Herrero, J., Rashidifard, M. & Jessell, M.W., 2026b. 3D pseudo-trans-dimensional geometrical inversion and posterior ensemble analysis for undercover imaging in the Boulia region (Queensland, Australia), preprint, SSRN, [abstract 7673742](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7673742).
- Herfindahl, O.C., 1950. *Concentration in the U.S. Steel Industry*, PhD thesis, Columbia University.
- Hopcroft, J. & Tarjan, R., 1973. Algorithm 447: efficient algorithms for graph manipulation, *Commun. ACM*, 16(6), 372–378.
- Jasra, A., Holmes, C.C. & Stephens, D.A., 2005. Markov chain Monte Carlo methods and the label switching problem in Bayesian mixture modeling, *Stat. Sci.*, 20(1), 50–67.
- Johnson, W.B. & Lindenstrauss, J., 1984. Extensions of Lipschitz mappings into a Hilbert space, *Contemp. Math.*, 26, 189–206.
- McInnes, L., Healy, J. & Melville, J., 2018. UMAP: Uniform Manifold Approximation and Projection for dimension reduction, arXiv:1802.03426.
- Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N., Chklovskii, D. & Alon, U., 2002. Network motifs: simple building blocks of complex networks, *Science*, 298(5594), 824–827.
- Osher, S. & Sethian, J.A., 1988. Fronts propagating with curvature-dependent speed: algorithms based on Hamilton–Jacobi formulations, *J. Comput. Phys.*, 79(1), 12–49.
- Rashidifard, M., Giraud, J., Lindsay, M. & Jessell, M., 2024. Cooperative geophysical inversion integrated with 3-D geological modelling in the Boulia region, QLD, *Geophys. J. Int.*, 238, 860–880.
- Robert, C.P. & Casella, G., 2004. *Monte Carlo Statistical Methods*, 2nd edn, Springer.
- Rosenfeld, A. & Pfaltz, J.L., 1966. Sequential operations in digital picture processing, *J. ACM*, 13(4), 471–494.
- Shannon, C.E., 1948. A mathematical theory of communication, *Bell Syst. Tech. J.*, 27(3), 379–423.
- Silversmith, W., 2021. cc3d: connected components on multilabel 3D images, software, https://github.com/seung-lab/connected-components-3d.
- Stephens, M., 2000. Dealing with label switching in mixture models, *J. R. Stat. Soc. B*, 62(4), 795–809.
- Thiele, S.T., Jessell, M.W., Lindsay, M., Ogarko, V., Wellmann, J.F. & Pakyuz-Charrier, E., 2016a. The topology of geology 1: Topological analysis, *J. Struct. Geol.*, 91, 27–38.
- Thiele, S.T., Jessell, M.W., Lindsay, M., Wellmann, J.F. & Pakyuz-Charrier, E., 2016b. The topology of geology 2: Topological uncertainty, *J. Struct. Geol.*, 91, 74–87.
- Woodcock, N.H., 1977. Specification of fabric shapes using an eigenvalue method, *Geol. Soc. Am. Bull.*, 88(9), 1231–1236.
- Zadeh, L.A., 1965. Fuzzy sets, *Information and Control*, 8(3), 338–353.
