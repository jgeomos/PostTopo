"""
Embedding, clustering and cluster statistics of a model ensemble, following
test_notebook3.ipynb: UMAP on the feature vectors, DBSCAN on the 2D embedding,
median and standard-deviation models per cluster, tile views.

Conventions
-----------
- `core` is the ensemble as (nz, ny, nx, N), axis 0 = depth with index 0 at the
  top, model axis last (the analysis region, padding removed).
- `coords` is the 2D embedding (N, 2); `labels` the cluster of each model, -1 for
  DBSCAN noise.
- Sections: 'y' is the vertical section at a fixed y index (an x-z panel,
  core[:, iy, :]), 'x' the vertical section at a fixed x index (a y-z panel,
  core[:, :, ix]), 'z' the map at a fixed depth index (core[iz]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Embedding and clustering
# ---------------------------------------------------------------------------
def umap_embed(features, n_neighbors=5, min_dist=0.01, metric='cosine', random_state=42, **kw):
    """UMAP embedding (N, 2) of the standardised features, with the parameters of the notebook. Returns (coords, reducer)."""
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                        random_state=random_state, **kw)
    coords = reducer.fit_transform(features)
    return np.asarray(coords, dtype=np.float64), reducer


def cluster_dbscan(coords, eps=0.25, min_samples=5):
    """DBSCAN on the embedding; -1 = noise. eps is in embedding units (see kdist_curve)."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)


def cluster_hdbscan(coords, min_cluster_size=10, min_samples=None):
    """HDBSCAN on the embedding (no eps to choose); -1 = noise."""
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(coords)


def kdist_curve(coords, k=5):
    """Sorted distance of each point to its k-th neighbour: the 'knee' is a usual choice of eps for DBSCAN with min_samples = k."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    d, _ = nn.kneighbors(coords)
    return np.sort(d[:, -1])


def cluster_summary(labels):
    """Number of clusters and of noise points."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters, int((labels == -1).sum())


def cluster_ids_by_size(labels, min_size=1):
    """Cluster ids (noise excluded) sorted by decreasing size, keeping clusters with at least min_size models."""
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = np.argsort(-counts, kind='stable')
    return [int(i) for i, c in zip(ids[order], counts[order]) if c >= min_size]


# ---------------------------------------------------------------------------
# Cluster statistics
# ---------------------------------------------------------------------------
def cluster_stats(core, labels, cl, median=True):
    """
    Median model, standard-deviation model and member indices of cluster cl. Returns (median, std, idx);
    median is None with median=False. The members are gathered one depth slice at a time, so that a large
    cluster is never copied as a whole (the extra memory is that of one slice of all its members).
    """
    idx = np.flatnonzero(labels == cl)
    nz = core.shape[0]
    med = np.empty(core.shape[:3], dtype=np.float32) if median else None
    std = np.empty(core.shape[:3], dtype=np.float32)
    for z in range(nz):
        members = core[z][..., idx].astype(np.float32)          # (ny, nx, n)
        std[z] = members.std(axis=-1)
        if median:
            med[z] = np.median(members, axis=-1)
    return med, std, idx


def representative_index(coords, labels, cl):
    """Member of cluster cl closest to the cluster centroid in the embedding (utils.get_cluster_representative)."""
    idx = np.flatnonzero(labels == cl)
    pts = coords[idx]
    return int(idx[np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1))])


def section(model, orient, iy=None, ix=None, iz=None):
    """2D section of a (nz, ny, nx) model: 'y' -> (nz, nx) at y = iy, 'x' -> (nz, ny) at x = ix, 'z' -> (ny, nx) at z = iz."""
    nz, ny, nx = model.shape
    if orient == 'y':
        return model[:, ny // 2 if iy is None else iy, :]
    if orient == 'x':
        return model[:, :, nx // 2 if ix is None else ix]
    if orient == 'z':
        return model[nz // 2 if iz is None else iz]
    raise ValueError(orient)


def cluster_table(core, coords, labels, chain_id=None, truth=None, truth_cols=('high', 'lens', 'dyke', 'pipe'),
                  min_size=1):
    """
    One row per cluster (noise excluded), sorted by size: number of models, share of each chain,
    mean voxel standard deviation (spread of the cluster in model space, in density units), index of the
    representative model, and, when a truth table is given, the fraction of members carrying each body.
    """
    rows = []
    chains = np.unique(chain_id) if chain_id is not None else []
    for cl in cluster_ids_by_size(labels, min_size):
        idx = np.flatnonzero(labels == cl)
        row = {'cluster': cl, 'n': len(idx)}
        for c in chains:
            row[f'frac_{c}'] = float(np.mean(chain_id[idx] == c))
        row['mean_voxel_std'] = float(cluster_stats(core, labels, cl, median=False)[1].mean())
        row['representative'] = representative_index(coords, labels, cl)
        if truth is not None:
            for col in truth_cols:
                if col in truth:
                    row[f'P({col})'] = float(np.mean(truth[col].values[idx].astype(bool)))
        rows.append(row)
    return pd.DataFrame(rows).set_index('cluster')


def truth_enrichment(truth, labels, cols=('high', 'lens', 'dyke', 'pipe'), min_size=1):
    """
    For synthetic ensembles: P(body | cluster) / P(body) for each cluster and body type, the same
    enrichment as used to name clusters with motifs. Noise points are excluded from the clusters but
    kept in the ensemble-wide P(body).
    """
    cols = [c for c in cols if c in truth]
    P = truth[cols].astype(float).mean()
    rows = {}
    for cl in cluster_ids_by_size(labels, min_size):
        idx = labels == cl
        rows[cl] = truth.loc[idx, cols].astype(float).mean() / P.replace(0, np.nan)
    enr = pd.DataFrame(rows).T
    enr.index.name = 'cluster'
    return enr


def topology_classes(keys):
    """
    Class number of each model's topology key (1 = most frequent class of the ensemble, 2 = next, ...),
    numbered as in motif_analysis.topology_census (decreasing frequency, ties in order of first occurrence).
    Returns (class number per model (N,), number of models per class, indexed by class number - 1).
    """
    arr = np.empty(len(keys), dtype=object)
    for i, k in enumerate(keys):                       # element-wise: keys are nested tuples
        arr[i] = k
    codes, _ = pd.factorize(arr)
    n = np.bincount(codes)
    order = np.argsort(-n, kind='stable')
    number = np.empty(len(n), dtype=int)
    number[order] = np.arange(1, len(n) + 1)
    return number[codes], n[order]


def cluster_topology(classes, labels, min_size=1):
    """
    Per cluster (sorted by size): most frequent topology class, its share in the cluster and in the ensemble,
    and the number of distinct classes in the cluster. `classes` from topology_classes.
    """
    classes = np.asarray(classes)
    N = len(classes)
    p_all = np.bincount(classes) / N
    rows = []
    for cl in cluster_ids_by_size(labels, min_size):
        c = classes[labels == cl]
        cnt = np.bincount(c)
        mode = int(cnt.argmax())
        rows.append({'cluster': cl, 'n': len(c), 'topology_class': mode, 'share_in_cluster': cnt[mode] / len(c),
                     'share_in_ensemble': p_all[mode], 'n_classes': int((cnt > 0).sum())})
    return pd.DataFrame(rows).set_index('cluster')


# ---------------------------------------------------------------------------
# Figures (matplotlib)
# ---------------------------------------------------------------------------
def cluster_palette(n):
    """n distinct colours (colorcet glasbey when available, tab20 otherwise)."""
    try:
        import colorcet as cc
        pal = list(cc.glasbey)
    except ImportError:
        pal = [plt.cm.tab20(i % 20) for i in range(20)]
    return [pal[i % len(pal)] for i in range(max(n, 1))]


def plot_embedding(coords, labels, ax=None, label_clusters=True, min_size=1, s=12, title=None):
    """Scatter of the embedding coloured by cluster, noise in light grey, cluster ids at the centroids."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    ids = cluster_ids_by_size(labels, min_size)
    pal = cluster_palette(len(ids))
    noise = labels == -1
    ax.scatter(coords[noise, 0], coords[noise, 1], s=s / 3, c='lightgray', alpha=0.5, linewidths=0, label='noise')
    for col, cl in zip(pal, ids):
        m = labels == cl
        ax.scatter(coords[m, 0], coords[m, 1], s=s, color=col, alpha=0.9, edgecolors='black', linewidths=0.1)
        if label_clusters:
            cx, cy = coords[m].mean(axis=0)
            ax.text(cx, cy, str(cl), fontsize=6, fontweight='bold', ha='center', va='center',
                    bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
    ax.set_aspect('equal')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    n_cl, n_noise = cluster_summary(labels)
    ax.set_title(title or f'{n_cl} clusters, {n_noise} noise points, N = {len(labels)}')
    return ax


def plot_kdist(coords, k=5, eps=None, ax=None):
    """k-distance curve used to choose eps for DBSCAN (min_samples = k)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    d = kdist_curve(coords, k)
    ax.plot(d, lw=1)
    if eps is not None:
        ax.axhline(eps, color='r', ls='--', lw=1, label=f'eps = {eps}')
        ax.legend()
    ax.set_xlabel('points, sorted'); ax.set_ylabel(f'distance to {k}-th neighbour'); ax.set_yscale('log')
    return ax


def _edges(centres):
    """Cell edges from cell-centre coordinates (half a cell beyond the first and last centres)."""
    c = np.asarray(centres, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])


def section_extent(orient, shape, axis_coords=None):
    """
    imshow extent (left, right, bottom, top) of a section of a (nz, ny, nx) grid, and the names of its horizontal
    and vertical axes. axis_coords: None (cell indices) or dict with the cell-centre coordinates along the axes of
    the grid, keys 'z' (depth, index 0 = top), 'y', 'x'; missing keys fall back to cell indices.
    """
    nz, ny, nx = shape
    n = {'z': nz, 'y': ny, 'x': nx}
    hk, vk = {'y': ('x', 'z'), 'x': ('y', 'z'), 'z': ('x', 'y')}[orient]
    e = {k: _edges(axis_coords[k]) if axis_coords is not None and k in axis_coords else np.arange(n[k] + 1) - 0.5
         for k in (hk, vk)}
    for k in (hk, vk):
        assert len(e[k]) == n[k] + 1, f"axis_coords['{k}'] has {len(e[k]) - 1} values, the grid {n[k]} cells along {k}"
    if orient == 'z':                                  # map view, drawn with origin='lower'
        extent = (e[hk][0], e[hk][-1], e[vk][0], e[vk][-1])
    else:                                              # vertical section, origin='upper': depth index 0 at the top
        extent = (e[hk][0], e[hk][-1], e[vk][-1], e[vk][0])
    return extent, hk, ('depth' if vk == 'z' else vk)


def plot_cluster_tiles(core, labels, orient='y', iy=None, ix=None, iz=None, stat='median', min_size=5,
                       max_tiles=36, vmin=None, vmax=None, cmap='viridis', tile_w=2.2, axis_coords=None,
                       units='cell'):
    """
    Tile view: one section per cluster (median or std model), clusters sorted by size, as in test_notebook3.
    Each tile shows its coordinates (tick values); the axis names are given on the left column and the bottom row.
    axis_coords: None for cell indices, or dict of cell-centre coordinates along the axes of core ('z', 'y', 'x'),
    e.g. in metres, with `units` their unit; the tiles keep the proportions of the grid in cells.
    Returns the figure.
    """
    ids = cluster_ids_by_size(labels, min_size)[:max_tiles]
    panels, ns = [], []
    for cl in ids:
        med, std, idx = cluster_stats(core, labels, cl, median=(stat == 'median'))
        panels.append(section(med if stat == 'median' else std, orient, iy, ix, iz))
        ns.append(len(idx))
    if not panels:
        raise ValueError('no cluster with at least min_size models')
    h, w = panels[0].shape
    n = len(panels)
    n_cols = int(np.ceil(np.sqrt(n * w / h))) if n > 1 else 1
    n_rows = int(np.ceil(n / n_cols))
    if vmin is None:
        vmin = min(p.min() for p in panels)
    if vmax is None:
        vmax = max(p.max() for p in panels)
    extent, hname, vname = section_extent(orient, core.shape[:3], axis_coords)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(tile_w * n_cols, tile_w * h / w * n_rows + 0.45 * n_rows),
                             squeeze=False, constrained_layout=True)
    for ax in axes.flat:
        ax.axis('off')
    for k, (ax, p, cl, m) in enumerate(zip(axes.flat, panels, ids, ns)):
        ax.axis('on')
        ax.imshow(p, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper' if orient != 'z' else 'lower',
                  extent=extent, aspect='equal' if axis_coords is None else 'auto', interpolation='nearest')
        ax.set_title(f'cluster {cl} (n={m})', fontsize=8)
        ax.tick_params(labelsize=6, length=2, pad=1)
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
        r, c = divmod(k, n_cols)
        if c == 0:
            ax.set_ylabel(f'{vname} ({units})', fontsize=7)
        if k + n_cols >= n:                              # lowest tile of its column
            ax.set_xlabel(f'{hname} ({units})', fontsize=7)
    where = {'y': f'section y = {iy}', 'x': f'section x = {ix}', 'z': f'map z = {iz}'}[orient]
    fig.suptitle(f'{stat} model per cluster, {where}', fontsize=11)
    return fig


def plot_cluster_sections(core, coords, labels, cl, orient='y', iy=None, ix=None, iz=None, vmin=None, vmax=None,
                          vmax_std=None, cmap='viridis', cmap_std='magma'):
    """Median, standard deviation and representative model of one cluster on one section. Returns the figure."""
    med, std, idx = cluster_stats(core, labels, cl)
    rep = core[..., representative_index(coords, labels, cl)]
    panels = [(section(med, orient, iy, ix, iz), f'median (n={len(idx)})', cmap, vmin, vmax),
              (section(std, orient, iy, ix, iz), 'standard deviation', cmap_std, 0, vmax_std),
              (section(rep, orient, iy, ix, iz), 'representative model', cmap, vmin, vmax)]
    h, w = panels[0][0].shape
    fig, axes = plt.subplots(1, 3, figsize=(3 * 3.6, 3.6 * h / w + 0.8), constrained_layout=True)
    for ax, (p, t, cm, lo, hi) in zip(axes, panels):
        im = ax.imshow(p, cmap=cm, vmin=lo, vmax=hi, origin='upper' if orient != 'z' else 'lower',
                       aspect='equal', interpolation='nearest')
        ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle(f'cluster {cl}', fontsize=11)
    return fig
