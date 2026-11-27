# PostTopo

Posterior ensemble analysis for structural and geometrical models models.

The notebooks cluster an ensemble of posterior models and interpret the clusters with geometrical and topological motifs. Each model is described by a feature vector (encoder values, topological scalars and density moments), embedded with UMAP (Uniform Manifold Approximation and Projection; McInnes et al., 2018) and clustered with DBSCAN (Density-Based Spatial Clustering of Applications with Noise; Ester et al., 1996).

## Running in the cloud
| Notebook | Ensemble | Open in Colab |
|---|---|---|
| `cluster_workflow_synthetic.ipynb` | Synthetic ensemble, geometry known model by model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jgeomos/PostTopo/blob/main/cluster_workflow_synthetic.ipynb) |
| `cluster_workflow_field.ipynb` | Boulia field ensemble (different (pseudo)trans-dimensional inversion chains) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jgeomos/PostTopo/blob/main/cluster_workflow_field.ipynb) |

# Running the examples

## Fully run results
Repository with results: 
https://zenodo.org/records/23997718

## Browser-based examples

You can run the notebooks in your browser without local installation using Google Colab:

- [Synthetic ensemble](https://colab.research.google.com/github/jgeomos/PostTopo/blob/main/cluster_workflow_synthetic.ipynb)
- [Boulia field ensemble](https://colab.research.google.com/github/jgeomos/PostTopo/blob/main/cluster_workflow_field.ipynb)

Run the first code cell (*0.1 Setup*) once: it clones this repository into `/content/PostTopo` (modules and data included), installs the packages that Colab lacks and imports the modules. The remaining cells can then be run in order (*Runtime > Run all* also works). A Google account is required; no GPU is needed.

Files written by the notebooks (cached features in `data/`; embedding, cluster labels and interactive cluster viewers in `outputs/`) are stored in the Colab session and are lost when the runtime is disconnected. They can be downloaded from the *Files* panel on the left-hand side.

## Local execution

Requirements: Python ≥ 3.9 and Jupyter.

```bash
git clone https://github.com/jgeomos/PostTopo.git
cd PostTopo
pip install -r requirements.txt
jupyter notebook cluster_workflow_synthetic.ipynb
```

The notebooks must be launched from the repository root so that `modules/` and `data/` are found.

## Repository content

- `cluster_workflow_synthetic.ipynb`, `cluster_workflow_field.ipynb`: course notebooks.
- `modules/`: Python modules used by the notebooks (synthetic ensemble generator, feature extraction, clustering, interactive viewer with unit entropy and born units after loopUI, motif analysis and plotting).
- `data/`: Boulia ensemble (2000 post-burn-in models from five chains, `boulia_5chains_2000models.npz`) and prior model (`boulia_prior_model.npy`); features of the Boulia and synthetic ensembles saved by the notebooks (`*_features_*.npy`, `*_topo.npy`, with a fingerprint file), reloaded instead of being computed again.
- `technical_note.md`: what the notebooks compute, with equations and references.
- `requirements.txt`: Python packages needed by the notebooks.
- `LICENSE`: GNU General Public License v3.0.

A run of the notebooks writes the embedding, the cluster labels and the interactive cluster viewers (self-contained HTML pages, to be opened in a web browser) in `outputs/`, which is not tracked by git.

## Relevant references 

Equations and the full list of method references are given in `technical_note.md`. The main references are listed below.

### Methods used in the notebooks

- Campello, R.J.G.B., Moulavi, D. & Sander, J., 2013. Density-based clustering based on hierarchical density estimates, *PAKDD*, LNCS 7819, 160–172.
- Ester, M., Kriegel, H.-P., Sander, J. & Xu, X., 1996. A density-based algorithm for discovering clusters in large spatial databases with noise, *Proc. KDD-96*, 226–231.
- Haralick, R.M., Shanmugam, K. & Dinstein, I., 1973. Textural features for image classification, *IEEE Trans. Syst. Man Cybern.*, SMC-3(6), 610–621, doi:10.1109/TSMC.1973.4309314.
- Johnson, W.B. & Lindenstrauss, J., 1984. Extensions of Lipschitz mappings into a Hilbert space, *Contemp. Math.*, 26, 189–206, doi:10.1090/conm/026/737400.
- Lindsay, M.D., Jessell, M.W., Ailleres, L., Perrouty, S., de Kemp, E. & Betts, P.G., 2013. Geodiversity: exploration of 3D geological model space, *Tectonophysics*, 594, 27–37.
- McInnes, L., Healy, J. & Melville, J., 2018. UMAP: Uniform Manifold Approximation and Projection for dimension reduction, arXiv:1802.03426.
- Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N., Chklovskii, D. & Alon, U., 2002. Network motifs: simple building blocks of complex networks, *Science*, 298(5594), 824–827.
- Pakyuz-Charrier, E., Jessell, M., Giraud, J., Lindsay, M. & Ogarko, V., 2019. Topological analysis in Monte Carlo simulation for uncertainty propagation, *Solid Earth*, 10(5), 1663–1684, doi:10.5194/se-10-1663-2019.
- Portes, L., Pirot, G., Nzikou, M.M., Giraud, J., Lindsay, M., Jessell, M. & Cripps, E., 2025. Feature fusion-enhanced t-SNE image atlas for geophysical features discovery, *Sci. Rep.*, 15, 17152, doi:10.1038/s41598-025-01333-3.
- Schubert, E., Sander, J., Ester, M., Kriegel, H.-P. & Xu, X., 2017. DBSCAN revisited, revisited: why and how you should (still) use DBSCAN, *ACM Trans. Database Syst.*, 42(3), 19, doi:10.1145/3068335.
- Stephens, M., 2000. Dealing with label switching in mixture models, *J. R. Stat. Soc. B*, 62(4), 795–809.
- Thiele, S.T., Jessell, M.W., Lindsay, M., Ogarko, V., Wellmann, J.F. & Pakyuz-Charrier, E., 2016a. The topology of geology 1: Topological analysis, *J. Struct. Geol.*, 91, 27–38, doi:10.1016/j.jsg.2016.08.009.
- Thiele, S.T., Jessell, M.W., Lindsay, M., Wellmann, J.F. & Pakyuz-Charrier, E., 2016b. The topology of geology 2: Topological uncertainty, *J. Struct. Geol.*, 91, 74–87, doi:10.1016/j.jsg.2016.08.010.
- Woodcock, N.H., 1977. Specification of fabric shapes using an eigenvalue method, *Geol. Soc. Am. Bull.*, 88(9), 1231–1236.



