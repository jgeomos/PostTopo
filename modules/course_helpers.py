"""
Helpers of the course notebooks cluster_workflow_synthetic.ipynb and cluster_workflow_field.ipynb.

They hold the code that the notebooks need but that is not discussed in the course (loading and reorienting the
ensembles, caching the features, timing, figure layout, saving and displaying the viewer pages, assembling tables),
so that the notebook cells show the parameters and the main calls only. Every function is a thin wrapper around the
course modules (synthetic_ensemble, ensemble_features, cluster_tools, cluster_viewer, motif_analysis, motif_plots,
cluster_motifs) and adds no method of its own.

Conventions as in cluster_tools: `core` is the ensemble (nz, ny, nx, N), axis 0 = depth with index 0 at the top,
model axis last; `coords` the 2D embedding (N, 2); `labels` the cluster of each model (-1 = noise); `chain_id` the
chain of each model.
"""

from __future__ import annotations

import hashlib
import os
import sys
import warnings
from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import synthetic_ensemble as se
import ensemble_features as ef
import cluster_tools as ct
import cluster_viewer as cv
import motif_analysis as ma
import motif_plots as mp
import cluster_motifs as cm

try:
    from IPython.display import display, IFrame
except ImportError:                                   # outside IPython: tables are printed, pages not embedded
    display, IFrame = print, None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _spread(n, max_models):
    """Indices of max_models models evenly spread over n, or None to keep all."""
    if max_models is None or max_models >= n:
        return None
    return np.unique(np.linspace(0, n - 1, max_models).round().astype(int))


def rename_chains(chain_id):
    """
    Chain names replaced by 'chain 1', 'chain 2', ..., numbered in order of first appearance in the ensemble.
    Returns (new names (N,), {original name: new name}).
    """
    names, first, inv = np.unique(chain_id, return_index=True, return_inverse=True)
    new = np.array([f'chain {r + 1}' for r in np.argsort(np.argsort(first))])
    return new[inv.ravel()], {str(n): str(m) for n, m in zip(names, new)}


def _print_ensemble(core, chain_id, mapping):
    counts = dict(zip(*np.unique(chain_id, return_counts=True)))
    original = {v: k for k, v in mapping.items()}
    print(f"core {core.shape} (depth, y, x, model), {core.nbytes / 1e9:.2f} GB")
    print("chains: " + ", ".join(f"{c} ({original[c]}): {n} models" if original[c] != c else f"{c}: {n} models"
                                 for c, n in sorted(counts.items(), key=lambda t: int(t[0].split()[-1]))))


def load_synthetic(n_per_chain=1000, seed=0, padd=6, max_models=None):
    """
    Synthetic ensemble of synthetic_ensemble.make_ensemble (three chains), lateral padding removed as each model is
    generated, chains renamed 'chain 1' to 'chain 3'. Returns (core, chain_id, truth), truth being the table of what
    was drawn in each model.
    """
    core, chain_id, _, truth = se.make_ensemble(n_per_chain=n_per_chain, padd=padd, seed=seed, return_truth=True,
                                                crop_padding=True)
    keep = _spread(core.shape[3], max_models)
    if keep is not None:
        core, chain_id = np.ascontiguousarray(core[..., keep]), chain_id[keep]
        truth = truth.iloc[keep].reset_index(drop=True)
    chain_id, mapping = rename_chains(chain_id)
    _print_ensemble(core, chain_id, mapping)
    print("bodies drawn (fraction of models):", truth[['high', 'lens', 'dyke', 'pipe']].mean().round(2).to_dict())
    return core, chain_id, truth


def load_field(path, max_models=None, flip_depth=True, swap_yx=True):
    """
    Post-burn-in models saved by export_boulia_subset_cell.py (.npz with models (depth, y, x, model), chain_id,
    chains, burn_in, padd; lateral padding already removed). The depth axis is flipped and the two lateral axes are
    swapped (flip_depth, swap_yx), as in the data cell of the original notebook. Returns (core, chain_id, padd).
    """
    with np.load(path) as d:
        core, chain_id, padd = d['models'].astype(np.float32, copy=False), d['chain_id'], int(d['padd'])
        n_chains, burn_in = len(d['chains']), int(d['burn_in'])
    if flip_depth:
        core = core[::-1]
    if swap_yx:
        core = np.swapaxes(core, 1, 2)
    keep = _spread(core.shape[3], max_models)
    if keep is not None:
        core, chain_id = np.ascontiguousarray(core[..., keep]), chain_id[keep]
    chain_id, mapping = rename_chains(chain_id)
    print(f"{os.path.basename(path)}: {n_chains} chains, models stored from {burn_in} (burn-in) on, "
          f"lateral padding of {padd} cells removed")
    _print_ensemble(core, chain_id, mapping)
    return core, chain_id, padd


def density_range(core, q=(0.5, 99.5), n_sample=200):
    """Colour range of the densities: percentiles q of up to n_sample models spread over the ensemble."""
    N = core.shape[-1]
    lo, hi = np.percentile(core[..., ::max(N // n_sample, 1)], q)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _fingerprint(core, n=20):
    """Shape of the ensemble and hash of n models spread over it."""
    N = core.shape[-1]
    idx = np.unique(np.linspace(0, N - 1, min(N, n)).round().astype(int))
    return f"{core.shape} " + hashlib.sha1(np.ascontiguousarray(core[..., idx]).tobytes()).hexdigest()


def _load_cached(files, fp_file, fingerprint, N):
    """The three saved feature groups if they belong to this ensemble, else (None, reason)."""
    if not all(os.path.exists(f) for f in files.values()):
        return None, "no saved features"
    parts = [np.load(f) for f in files.values()]
    if not all(len(p) == N for p in parts):
        return None, f"saved features hold {len(parts[0])} models, the ensemble {N}"
    if os.path.exists(fp_file):
        with open(fp_file) as f:
            if f.read().strip() != fingerprint:
                return None, "saved features belong to another ensemble (fingerprint differs)"
    else:                                   # saved before fingerprints existed: same N, assumed same models
        print("no fingerprint saved with the features: assumed to match this ensemble (same number of models)")
        with open(fp_file, 'w') as f:
            f.write(fingerprint)
    return parts, None


def ensemble_features(core, cache=None, recompute=False, n_jobs=-1):
    """
    Feature vectors of the ensemble, as in ensemble_features: 512 values of the fixed-kernel encoder (densities
    scaled to 0-1 with the extrema of the ensemble), 6 topological scalars and 5 density moments, concatenated and
    standardised column by column.

    cache : path prefix of the saved features (e.g. 'data/boulia_5chains'), or None to compute without saving.
        With a prefix, the three groups are loaded when they were computed before for the same models (checked with
        a fingerprint of the ensemble), and computed and saved otherwise; the standardised features are saved as
        <cache>_features.npy. recompute=True forces a new computation.

    Returns (features (N, 523), topo (N, 6)); shows summary statistics of the 11 topological and density features.
    """
    N = core.shape[-1]
    parts, reason = None, "no cache"
    if cache is not None:
        files = {k: f"{cache}_{k}.npy" for k in ('features_latent_codes', 'topo', 'features_density')}
        fp_file, fingerprint = f"{cache}_features_fingerprint.txt", _fingerprint(core)
        if recompute:
            reason = "recompute = True"
        else:
            parts, reason = _load_cached(files, fp_file, fingerprint, N)
    if parts is not None:
        latent, topo, dens = parts
        print(f"features loaded from {os.path.dirname(os.path.abspath(cache))}")
    else:
        print(f"computing the features ({reason})")
        t0 = time()
        enc = ef.DiffNetEncoder(core.shape[:3])               # fixed random projections for this grid (seeds 1, 2, 3)
        print(f"  encoder: pooled dimensions {enc.in_dims} -> {enc.dims} ({time() - t0:.0f} s for the projections)")
        t0 = time()
        latent = enc.encode_ensemble(core, n_jobs=n_jobs)
        print(f"  encoder: {latent.shape} in {time() - t0:.0f} s")
        t0 = time()
        topo, dens = ef.topo_and_density_features(core, n_jobs=n_jobs)
        print(f"  topological and density features: {topo.shape}, {dens.shape} in {time() - t0:.0f} s")
        if cache is not None:
            for f, a in zip(files.values(), (latent, topo, dens)):
                np.save(f, a)
            with open(fp_file, 'w') as f:
                f.write(fingerprint)
            print(f"  saved in {os.path.dirname(os.path.abspath(cache))}")
    features, _ = ef.combine_features(latent, topo, dens)
    if cache is not None:
        np.save(f"{cache}_features.npy", features)
    print(f"features: {features.shape} = 512 encoder + 6 topological + 5 density values, standardised")
    display(pd.DataFrame(np.hstack([topo, dens]), columns=ef.TOPO_FEATURE_NAMES + ef.DENSITY_FEATURE_NAMES)
            .describe().round(3))
    return features, topo


# ---------------------------------------------------------------------------
# Embedding and clustering
# ---------------------------------------------------------------------------
def embed(features, n_neighbors=5, min_dist=0.05, metric='cosine', chain_id=None, prefix=None, random_state=42):
    """
    UMAP embedding (N, 2) of the features (cluster_tools.umap_embed), drawn coloured by chain. With a prefix, the
    embedding is saved as a CSV (columns comp_1, comp_2, then the features), as the 2b notebook writes it.
    """
    t0 = time()
    with warnings.catch_warnings():               # UMAP notes that a fixed seed runs on one thread; tqdm, no widgets
        warnings.filterwarnings('ignore', message='n_jobs value')
        warnings.filterwarnings('ignore', message='IProgress not found')
        coords, _ = ct.umap_embed(features, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                                  random_state=random_state)
    print(f"UMAP: {coords.shape} in {time() - t0:.0f} s")
    if prefix is not None:
        name = (f"{prefix}_umap_{n_neighbors}Features_cnn_topo_dens_DiffNN_{min_dist}_metrics_{metric}__.csv")
        pd.concat([pd.DataFrame(coords, columns=['comp_1', 'comp_2']), pd.DataFrame(features)], axis=1) \
            .to_csv(name, index=False)
        print("saved:", name)
    fig, ax = plt.subplots(figsize=(7, 6))
    groups = np.unique(chain_id) if chain_id is not None else [None]
    for c in groups:
        m = np.ones(len(coords), bool) if c is None else chain_id == c
        ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.7, label=c)
    ax.set_aspect('equal'); ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    if chain_id is not None:
        ax.legend(markerscale=3, fontsize=8)
    ax.set_title('UMAP embedding' + (', coloured by chain' if chain_id is not None else ''))
    plt.show()
    return coords


def cluster(coords, method='dbscan', eps=0.25, min_samples=5, min_cluster_size=10,
            eps_scan=(0.2, 0.3, 0.35, 0.5, 0.75), prefix=None):
    """
    Clusters of the embedding (-1 = noise) with DBSCAN or HDBSCAN (cluster_tools). For DBSCAN, the numbers of
    clusters and noise points are printed for the values in eps_scan, and the k-distance curve is drawn with eps.
    With a prefix, the labels are saved as <prefix>_cluster_labels.npy. Returns the labels.
    """
    if method == 'dbscan':
        for e in eps_scan:
            n_cl, n_noise = ct.cluster_summary(ct.cluster_dbscan(coords, eps=e, min_samples=min_samples))
            print(f"eps = {e:<5} -> {n_cl:3d} clusters, {n_noise:4d} noise points")
        ct.plot_kdist(coords, k=min_samples, eps=eps)
        plt.show()
        labels = ct.cluster_dbscan(coords, eps=eps, min_samples=min_samples)
    elif method == 'hdbscan':
        labels = ct.cluster_hdbscan(coords, min_cluster_size=min_cluster_size, min_samples=min_samples)
    else:
        raise ValueError(f"method must be 'dbscan' or 'hdbscan', not {method!r}")
    n_cl, n_noise = ct.cluster_summary(labels)
    print(f"{method}: {n_cl} clusters, {n_noise} noise points, N = {len(labels)}")
    if prefix is not None:
        np.save(f"{prefix}_cluster_labels.npy", labels)
    ct.plot_embedding(coords, labels, min_size=min_samples)
    plt.show()
    return labels


# ---------------------------------------------------------------------------
# Cluster statistics and viewer
# ---------------------------------------------------------------------------
def plot_tiles(core, labels, orient='y', index=None, min_size=5, vmin=None, vmax=None, max_tiles=30,
               axis_coords=None, units='cell'):
    """Median model, then standard-deviation model, of every cluster on one section (cluster_tools.plot_cluster_tiles)."""
    at = {'y': 'iy', 'x': 'ix', 'z': 'iz'}[orient]
    for stat, lo, hi, cmap in (('median', vmin, vmax, 'viridis'), ('std', 0, None, 'magma')):
        ct.plot_cluster_tiles(core, labels, orient=orient, stat=stat, min_size=min_size, max_tiles=max_tiles,
                              vmin=lo, vmax=hi, cmap=cmap, axis_coords=axis_coords, units=units, **{at: index})
        plt.show()


def plot_largest_clusters(core, coords, labels, n_show=4, iy=None, ix=None, iz=None, min_size=5, vmin=None,
                          vmax=None, axis_coords=None, units='cell'):
    """Median, standard deviation and representative model of the n_show largest clusters, section y = iy and map z = iz."""
    for cl in ct.cluster_ids_by_size(labels, min_size)[:n_show]:
        ct.plot_cluster_sections(core, coords, labels, cl, orient=('y', 'z'), iy=iy, ix=ix, iz=iz, vmin=vmin,
                                 vmax=vmax, axis_coords=axis_coords, units=units)
        plt.show()


def show_viewer(layout, filename, width=1250, height=1150):
    """
    Save a viewer layout (cluster_viewer.build_viewer) as a self-contained HTML page and show it below the cell:
    the saved page in a frame, or, in Colab (which cannot serve a local file), the layout itself.
    """
    cv.save_viewer(layout, filename, title=os.path.splitext(os.path.basename(filename))[0].replace('_', ' '))
    print(f"saved: {filename} ({os.path.getsize(filename) / 1e6:.1f} MB); open it in a browser for a full-size view")
    if 'google.colab' in sys.modules:
        from bokeh.io import output_notebook, show
        output_notebook(hide_banner=True)
        show(layout)
    elif IFrame is not None:
        display(IFrame(filename, width=width, height=height))


def viewer_extra(df_motifs, topo):
    """Per-model scalars shown as cluster means in the motif viewer: numbers of units and born units, topological features."""
    return pd.concat([df_motifs[['n_units', 'n_born']].reset_index(drop=True),
                      pd.DataFrame(topo, columns=ef.TOPO_FEATURE_NAMES)], axis=1)


# ---------------------------------------------------------------------------
# Motifs
# ---------------------------------------------------------------------------
def reference(core, prior, padd, min_share=0.01, row=0, check=None):
    """
    Reference codes for the unit matching (cluster_motifs.reference_model): from the prior model (array, or path of
    a .npy file, also looked for in the working directory) oriented on the grid of core, or a modal stand-in when
    there is none. check=(iy, iz): shows the orientation table and the prior next to the median of the models.
    Returns the codes (0 = top unit, K-1 = deepest).
    """
    if isinstance(prior, (str, os.PathLike)):
        path = prior if os.path.exists(prior) else os.path.basename(prior)
        if os.path.exists(path):
            print(f"prior model: {path}")
            prior = np.load(path)
        else:
            print(f"{prior} not found")
            prior = None
    ref, prior_core, table = cm.reference_model(core, prior, padd, min_share=min_share, row=row)
    if table is not None and check is not None:
        display(table.head(4).round(3))
        cm.plot_prior_check(prior_core, core, *check)
        plt.show()
    return ref


def motifs_per_cluster(df_motifs, labels, motifs, min_size=5, truth=None,
                       truth_motifs=('enclosed_body', 'dyke', 'chimney', 'window')):
    """
    Per cluster (at least min_size models, sorted by size): P(motif | cluster), enrichment E, most frequent topology
    class; with a truth table, the enrichment of the drawn bodies next to that of the recovered motifs.
    Returns the topology class of each model (1 = most frequent in the ensemble).
    """
    ids = ct.cluster_ids_by_size(labels, min_size)
    p_cl, enr = ma.cluster_enrichment(df_motifs, labels)
    print("P(motif | cluster):")
    display(p_cl.loc[ids, ['n_models'] + list(motifs)].round(2))
    print("enrichment E = P(motif | cluster) / P(motif)   (empty: motif absent from the ensemble)")
    display(enr.loc[ids, list(motifs)].round(2).replace([np.inf, -np.inf], np.nan))
    topo_class, topo_n = ct.topology_classes(df_motifs['topology'].to_numpy())
    N = len(df_motifs)
    print(f"{len(topo_n)} topology classes; the most frequent hold "
          + ", ".join(f"#{k + 1}: {n / N:.0%}" for k, n in enumerate(topo_n[:5])) + " of the models")
    display(ct.cluster_topology(topo_class, labels, min_size).round(2))
    if truth is not None:
        print("enrichment of the drawn bodies (truth) and of the recovered motifs:")
        display(pd.concat({'drawn': ct.truth_enrichment(truth, labels, min_size=min_size),
                           'recovered': enr.loc[ids, list(truth_motifs)]}, axis=1).round(2))
    return topo_class


def selection(df_motifs, core, kw, labels, cluster=None, motifs=()):
    """
    Models used in section 6: all models, or the members of one cluster. Stops if a motif name is unknown or if the
    motif table does not belong to core and kw (motif_analysis.check_table). Returns (indices, name of the selection).
    """
    unknown = sorted(set(motifs) - set(ma.MOTIFS))
    if unknown:
        raise ValueError(f"unknown motifs {unknown}; valid: {ma.MOTIFS}")
    ma.check_table(df_motifs, core, **kw)
    members, where = cm.select_models(labels, cluster)
    print(f"models used below: {len(members)} ({where})")
    return members, where


# ---------------------------------------------------------------------------
# Clearest examples of a motif
# ---------------------------------------------------------------------------
# Motifs carried by the same body without changing its reading (a steep dyke is also a steep, cross-cutting body).
COMPATIBLE = {'dyke': {'steep_body', 'cross_cutting'}, 'chimney': {'steep_body', 'cross_cutting'},
              'cross_cutting': {'steep_body', 'dyke', 'chimney'}, 'steep_body': {'cross_cutting', 'dyke', 'chimney'},
              'order_violation': {'layer'}}       # the units of an inverted contact are usually layers


def _margin(x, t, ref):
    """Normalised margin of x past threshold t: 0 at t, 1 at ref, linear in between (as in motif_analysis)."""
    return float(np.clip((x - t) / (ref - t), 0.0, 1.0)) if ref != t else float(x >= t)


def reading_terms(G, n, motif):
    """
    Criteria of the geological reading of `motif` that its definition does not test, as margins in [0, 1]:
    chimney: plunge of the long axis (0 at the steep_cos threshold, ~45 deg, 1 when vertical), since the definition
        accepts rods of any plunge, e.g. flat bodies lying between basement and cover;
    window: localised breach (0 when half of the basement footprint lies under the cover, 1 at a tenth or less),
        since the definition also accepts a basement surface lying under the cover everywhere.
    """
    a, p = G.nodes[n], G.graph['thr']
    if motif == 'chimney':
        return {'plunge': _margin(a['axis_vz'], p['steep_cos'], 1.0)}
    if motif == 'window':
        return {'localised breach': _margin(a['window_faces'] / max(a['footprint'], 1), 0.5, 0.1)}
    return {}


def _rank_model(model, kw, motifs, i):
    """For each motif of `motifs` present in model i, its clearest body: (key, i, body, clarity, criteria, others)."""
    _, G, _, flags = ma.motif_counts(model, return_graph=True, **kw)
    n_flagged = sum(1 for f in flags.values() if f - {'layer'})
    out = {}
    for motif in motifs:
        for n, f in flags.items():
            if motif not in f:
                continue
            crit = {**ma.motif_score(G, n, motif)[1], **reading_terms(G, n, motif)}
            clarity = min(crit.values()) if crit else 0.0
            others = sorted(f - {motif} - COMPATIBLE.get(motif, set()))
            vol = G.nodes[n]['volume']
            key = (not others, round(clarity, 1), -n_flagged, round(clarity, 2),
                   round(float(np.mean(list(crit.values()))), 2) if crit else 0.0,
                   -vol if motif == 'order_violation' else vol)   # a violation flags both bodies: show the smaller,
            #                                                         usually the misplaced one
            if motif not in out or key > out[motif][0]:
                out[motif] = (key, int(i), int(n), clarity, crit, others)
    return out


def rank_examples(core, df_motifs, kw, motifs, members, n_check=200, n_jobs=-1):
    """
    Clearest realisations of each motif among the selected models (members): for each motif, up to n_check of the
    models carrying it (spread over the selection) are checked, and their bodies ranked by
      1. carrying no other motif than those compatible with the reading (COMPATIBLE),
      2. clarity, to 0.1: smallest margin of the criteria of the typicality score (motif_analysis.motif_score) and of
         the reading (reading_terms), each 0 at the threshold and 1 when fully typical,
      3. fewer other flagged bodies in the model (less clutter in the figures),
      4. clarity to 0.01, then mean margin, then volume (largest body first; for order_violation, which flags both
         bodies of an inverted contact, the smallest, usually the misplaced one).
    Returns {motif: [(key, model, body, clarity, criteria, other motifs), ...] best first}. Clarity only selects
    what is shown; it does not change the motif counts or probabilities.
    """
    members = np.asarray(members)
    motifs = list(dict.fromkeys(motifs))
    pick = set()
    for m in motifs:
        cand = members[df_motifs[m].to_numpy()[members] > 0]
        if len(cand):
            pick.update(cand[np.unique(np.linspace(0, len(cand) - 1, min(n_check, len(cand))).round().astype(int))]
                        .tolist())
    pick = sorted(pick)
    t0 = time()
    res = Parallel(n_jobs=n_jobs)(delayed(_rank_model)(core[..., i], kw, motifs, i) for i in pick)
    out = {m: sorted((r[m] for r in res if m in r), key=lambda t: t[0], reverse=True) for m in motifs}
    print(f"{len(pick)} models checked in {time() - t0:.0f} s; clarity of the clearest example: "
          + ", ".join(f"{m} {out[m][0][3]:.2f}" if out[m] else f"{m} absent" for m in motifs))
    return out


def clarity_summary(examples, clear=0.5):
    """
    Per motif, from rank_examples: models checked, clarity of the clearest example, median clarity, fraction of the
    checked models whose clearest body has a clarity >= `clear` and carries no other motif, and the criterion that
    most often limits the clarity. Sorted by that fraction, clearest motifs first.
    """
    rows = {}
    for motif, ex in examples.items():
        if not ex:
            rows[motif] = {'models checked': 0}
            continue
        c = np.array([e[3] for e in ex])
        weakest = pd.Series([min(e[4], key=e[4].get) for e in ex if e[4]])
        rows[motif] = {'models checked': len(ex), 'best clarity': c.max(), 'median clarity': np.median(c),
                       f'P(clarity >= {clear:g}, no other motif)': np.mean([(e[3] >= clear) and not e[5] for e in ex]),
                       'most limiting criterion': weakest.mode().iloc[0] if len(weakest) else ''}
    tab = pd.DataFrame(rows).T
    return tab.sort_values(f'P(clarity >= {clear:g}, no other motif)', ascending=False, na_position='last')


def _example_title(vw, G, i, n, clarity, crit, others):
    weak = min(crit, key=crit.get) if crit else ''
    inv = G.nodes[n].get('viol_faces', 0)
    return (f"model #{i}, body {n}: {vw.uname(G.nodes[n]['unit'])}, {G.nodes[n]['volume']} cells"
            + (f", {inv} faces in inverted contact" if inv else '') + "\n"
            f"clarity {clarity:.2f}" + (f" (weakest: {weak} {crit[weak]:.2f})" if crit else '')
            + (f", also {', '.join(others)}" if others else ''))


def show_example(vw, kw, examples, motif, labels, chain_id):
    """
    The clearest example of `motif` (from rank_examples): criteria, table of the bodies of the model, and sections
    through the bodies carrying the motif.
    """
    ex = examples.get(motif)
    if not ex:
        print(f"{motif} does not occur in the selected models")
        return
    _, i0, n0, clarity, crit, others = ex[0]
    c, G, body_labels, flags = ma.motif_counts(vw.core[..., i0], return_graph=True, **kw)
    print(f"model #{i0} (cluster {labels[i0]}, {chain_id[i0]}): {c['n_units']} units ({c['n_born']} born), "
          f"{c['n_bodies']} bodies; body {n0} realises {motif} with clarity {clarity:.2f}")
    print("criteria (0 at the threshold, 1 fully typical): " + ", ".join(f"{k} {v:.2f}" for k, v in crit.items()))
    if others:
        print(f"no clearer example: the body also carries {', '.join(others)}")
    if clarity < 0.5:
        print(f"no clear example of {motif} among the models checked (clarity < 0.5)")
    display(mp.bodies_table(vw, i0, G, body_labels, flags))
    mp.plot_flagged_bodies(vw, i0, G, body_labels, {b: (f if motif in f else set()) for b, f in flags.items()})


def plot_catalogue(vw, df_motifs, kw, examples, motifs, n_ex, members, where):
    """
    The n_ex clearest examples of each motif (from rank_examples), the body realising the motif outlined on the
    sections holding most of it (window: the basement cells under the cover). P in the titles: fraction of the
    selected models carrying the motif.
    """
    members = np.asarray(members)
    for motif in motifs:
        p = float((df_motifs[motif].to_numpy()[members] > 0).mean())
        ex = examples.get(motif, [])[:n_ex]
        if not ex:
            print(f"{motif}: P = {p:.3f} in {where}, never occurs")
            continue
        fig, axes = plt.subplots(len(ex), 2, figsize=(10, 2.3 * len(ex)), squeeze=False,
                                 gridspec_kw=dict(width_ratios=vw.pair_ratios()), constrained_layout=True)
        for (_, i, n, clarity, crit, others), (axy, axx) in zip(ex, axes):
            _, G, body_labels, flags = ma.motif_counts(vw.core[..., i], return_graph=True, **kw)
            mask = ma.motif_mask(G, body_labels, {n: flags[n]}, motif)
            vy, vx = mp.best_view(mask, 1), mp.best_view(mask, 2)
            im = vw.draw(axy, i, *vy, mask)
            vw.draw(axx, i, *vx, mask)
            axy.set_title(_example_title(vw, G, i, n, clarity, crit, others) + f" — section {mp.view_name(*vy)}",
                          fontsize=8)
            axx.set_title(f"section {mp.view_name(*vx)}", fontsize=8)
            vw.colorbar(fig, im, [axy, axx])
        fig.suptitle(f"{motif}: P = {p:.3f} in {where} ({mp.READING.get(motif, '')}) — "
                     f"{len(ex)} clearest of {len(examples[motif])} models checked", fontsize=11)
        plt.show()


def motif_presence(df_motifs, motifs, members, where, truth=None, thr=None):
    """
    P(motif present) in the ensemble and in the selection (cluster_motifs.motif_probability_table); with a truth
    table, the fraction of the selected models holding the object each motif was checked against
    (cluster_motifs.drawn_objects).
    """
    prob = cm.motif_probability_table(df_motifs, motifs, members, where)
    if truth is not None:
        drawn = cm.drawn_objects(truth, (thr or {}).get('flat_cos'))
        prob[f'P drawn, {where}'] = drawn.iloc[members].mean().reindex(list(motifs))   # NaN: no drawn counterpart
    display(prob.round(3))


def hypothesis_probabilities(df_motifs, members, where, labels, chain_id, basement, group_by='chain', min_size=1,
                             n_groups=6, depth_cells=3, extra=None):
    """
    P(H | d) of the hypotheses of cluster_motifs.hypotheses, plus `extra` ({name: function of the motif table}),
    per chain or per cluster, as a table and as bars with binomial standard errors.
    """
    rows, groups = cm.hypothesis_groups(members, labels, chain_id, group_by, min_size, n_groups)
    d = df_motifs.iloc[rows].reset_index(drop=True)
    H = {**cm.hypotheses(d, basement=basement, depth_cells=depth_cells),
         **{k: f(d) for k, f in (extra or {}).items()}}
    tab = ma.hypothesis_table(H, groups=groups)
    display(tab['P'].round(3))
    mp.plot_hypotheses(tab, title=f'P(H | d) in {where}, per {group_by}', legend_title=group_by)
