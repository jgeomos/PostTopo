# PostTopo

Posterior ensemble analysis: notebooks and code for the short course *Probabilistic geometrical inversion: from model sampling to (geological) assessment* (ASEG Discover 2026).

The notebooks cluster an ensemble of posterior models and interpret the clusters with geometrical and topological motifs. Each model is described by a feature vector (encoder values, topological scalars and density moments), embedded with UMAP (Uniform Manifold Approximation and Projection; McInnes et al., 2018) and clustered with DBSCAN (Density-Based Spatial Clustering of Applications with Noise; Ester et al., 1996).

| Notebook | Ensemble | Open in Colab |
|---|---|---|
| `cluster_workflow_synthetic.ipynb` | Synthetic ensemble, geometry known model by model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jgeomos/PostTopo/blob/main/cluster_workflow_synthetic.ipynb) |
| `cluster_workflow_field.ipynb` | Boulia field ensemble (different (pseudo)trans-dimensional inversion chains) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jgeomos/PostTopo/blob/main/cluster_workflow_field.ipynb) |

# Running the examples

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
- `modules/`: Python modules used by the notebooks (synthetic ensemble generator, feature extraction, clustering, interactive viewer, motif analysis and plotting).
- `data/`: Boulia ensemble (2000 post-burn-in models from five chains, `boulia_5chains_2000models.npz`) and prior model (`boulia_prior_model.npy`); features of the Boulia and synthetic ensembles saved by the notebooks (`*_features_*.npy`, `*_topo.npy`, with a fingerprint file), reloaded instead of being computed again.
- `technical_note.md`: what the notebooks compute, with equations and references.
- `requirements.txt`: Python packages needed by the notebooks.
- `LICENSE`: GNU General Public License v3.0.

A run of the notebooks writes the embedding, the cluster labels and the interactive cluster viewers (self-contained HTML pages, to be opened in a web browser) in `outputs/`, which is not tracked by git.
