"""
Motif analysis of the clusters of cluster_workflow.ipynb: reference model for the unit matching, the motifs and
hypotheses validated on the synthetic ensemble, and the selection of models by cluster or chain.

A thin layer over motif_analysis (motif table, unit matching) and motif_plots (figures), which are shared with the
motif workflow (motif_workflow_boulia_v4.ipynb) and are not modified here.

Conventions as in cluster_tools: `core` is the ensemble (nz, ny, nx, N), axis 0 = depth with index 0 at the top,
model axis last; `labels` the cluster of each model (-1 = noise); `chain_id` the chain of each model.

Validation (motif workflow, section *Validation against the synthetic ground truth*, seed 0, 1,000 models): the
motifs recovered from the sampled densities agree model by model with the motif definitions applied to the true
geometry, except three models lying at a threshold. VALIDATED_MOTIFS are those that also correspond to one kind of
drawn body: chimney (299 of the 304 pipes drawn), dyke (dykes dipping more than about 73 deg), window (never without a
basement high) and enclosed_body (56 % of the lenses, a lower bound on lenses). steep_body and cross_cutting are
carried by dykes and pipes alike, layer also by dipping dykes; bridge and order_violation never occur in the
synthetic ensemble.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import motif_analysis as ma
from cluster_tools import cluster_ids_by_size, section

VALIDATED_MOTIFS = ['chimney', 'dyke', 'window', 'enclosed_body']


# ---------------------------------------------------------------------------
# Reference model for the unit matching
# ---------------------------------------------------------------------------
def _test_models(core, n_test):
    """n_test models spread over the ensemble, and in each the cells carrying lateral structure (density different
    from the most common one at the same depth), which separate mirror images of a laterally structured prior."""
    N = core.shape[-1]
    models = [core[..., i] for i in np.unique(np.linspace(0, N - 1, n_test).round().astype(int))]
    lateral = []
    for m in models:
        mode_z = np.array([v[c.argmax()] for v, c in (np.unique(mz, return_counts=True) for mz in m)])
        lateral.append(m != mode_z[:, None, None])
    return models, lateral


def prior_agreement(prior, models, lateral):
    """
    Fraction of the cells whose unit, matched to the prior (ma.canonical_labels), is the prior unit: over the cells
    carrying lateral structure in the test models, and over all cells. Returns (lateral, all).
    """
    r = ma.reference_codes(prior)
    hits = [ma.canonical_labels(m, r) == r for m in models]
    with_lat = [(h, l) for h, l in zip(hits, lateral) if l.any()]
    return (np.mean([h[l].mean() for h, l in with_lat]) if with_lat else np.nan,
            np.mean([h.mean() for h in hits]))


def orient_prior(prior, core, padd, n_test=5, tol=0.05):
    """
    Candidate orientations of a prior model on the grid of `core`: the prior with its lateral padding (`padd` cells
    on each side) removed or as saved, its two lateral axes swapped or not, any of its axes flipped; only the
    candidates with the shape of core.shape[:3] are kept. They are ranked as in ensemble_io.find_prior_orientation
    of the motif workflow: first the candidates within `tol` of the best agreement over all cells (which removes
    depth flips and wrong axis orders), then by agreement over the cells with lateral structure (which separates
    mirror images). The transformations of the data cell therefore need not be repeated on the prior.

    Returns (candidates, table): the oriented arrays in ranked order, and a DataFrame with one row per candidate.
    """
    models, lateral = _test_models(core, n_test)
    crop = prior[:, padd:prior.shape[1] - padd, padd:prior.shape[2] - padd]
    rows = []
    for name, p in [('padding removed', crop)] + ([('as saved', prior)] if padd > 0 else []):
        for swap in (False, True):
            q = np.swapaxes(p, 1, 2) if swap else p
            if q.shape != core.shape[:3]:
                continue
            for fl in product((False, True), repeat=3):
                flips = tuple(k for k in range(3) if fl[k])
                cand = np.ascontiguousarray(np.flip(q, axis=flips) if flips else q)
                rows.append(prior_agreement(cand, models, lateral) + (name, swap, flips, cand))
    if not rows:
        raise ValueError(f"prior model {prior.shape} cannot be brought to the grid of core {core.shape[:3]} "
                         f"(lateral padding {padd}, lateral axes swapped or not)")
    best = max(r[1] for r in rows)
    rows.sort(key=lambda r: (r[1] < best - tol, -np.nan_to_num(r[0]), -r[1]))
    table = pd.DataFrame([r[:5] for r in rows], columns=['agreement, lateral structure', 'agreement, all cells',
                                                        'prior', 'y and x swapped', 'flipped axes (0 = depth)'])
    return [r[5] for r in rows], table


def absorb_small_units(ref, min_share):
    """
    Reference codes after absorbing the units occupying less than `min_share` of the grid (slivers of prior units
    lying mostly in the padding) into their nearest unit (Euclidean distance between cells), as in the motif
    workflow, so that the deepest code is a unit present in the analysis region. Returns (codes, absorbed codes).
    """
    share = np.bincount(ref.ravel()) / ref.size
    small = np.flatnonzero(share < min_share)
    if not small.size:
        return ref, []
    from scipy import ndimage
    near = ndimage.distance_transform_edt(share[ref] < min_share, return_distances=False, return_indices=True)
    return ma.reference_codes(ref[tuple(near)]), small.tolist()


def reference_model(core, prior=None, padd=0, min_share=0.01, row=0, n_modal=200):
    """
    Reference codes for matching the units of the models (label switching): from the prior model when given,
    oriented by orient_prior (row: rank of the candidate kept, 0 = best), otherwise a modal stand-in built from
    n_modal models (ma.modal_reference); units below min_share of the grid are then absorbed.

    Returns (ref, prior_core, table): ref the codes (0 = top unit, K-1 = deepest), prior_core the oriented prior
    (None for the modal stand-in) and the table of orient_prior (None for the modal stand-in).
    """
    if prior is not None:
        cands, table = orient_prior(prior, core, padd)
        prior_core = cands[row]
        r = table.iloc[row]
        print(f"prior model {prior.shape} -> {prior_core.shape}: {r['prior']}, lateral axes swapped: "
              f"{r['y and x swapped']}, flipped axes: {r['flipped axes (0 = depth)']} (row {row} of the table)")
        ref = ma.reference_codes(prior_core)
    else:
        prior_core, table = None, None
        ref = ma.modal_reference(core, n_sample=n_modal)
        print(f"no prior model: modal stand-in from {min(n_modal, core.shape[-1])} models")
    ref, absorbed = absorb_small_units(ref, min_share)
    if absorbed:
        print(f"reference units absorbed (< {min_share:.0%} of the grid): {absorbed}")
    K = int(ref.max()) + 1
    print(f"reference: {K} units (0 = top, {K - 1} = deepest), shares of the grid:",
          np.round(np.bincount(ref.ravel()) / ref.size, 3).tolist())
    return ref, prior_core, table


def plot_prior_check(prior_core, core, iy, iz, n_median=100):
    """Prior model and median of up to n_median models on the vertical section y = iy and the depth map z = iz:
    interfaces and highs should coincide; a mirror image means another row of the orientation table."""
    N = core.shape[-1]
    med = np.median(core[..., ::max(N // n_median, 1)], axis=3)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for j, (m, t) in enumerate(((prior_core, 'prior model'), (med, 'median of the models'))):
        axes[0, j].imshow(section(m, 'y', iy=iy), aspect='equal', interpolation='nearest')
        axes[0, j].set_title(f'{t}, section y = {iy}', fontsize=9)
        axes[1, j].imshow(section(m, 'z', iz=iz), origin='lower', aspect='equal', interpolation='nearest')
        axes[1, j].set_title(f'{t}, depth map z = {iz}', fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# Selection of models, probabilities and hypotheses
# ---------------------------------------------------------------------------
def select_models(labels, cluster=None):
    """Indices of the models of `cluster` (all models if None) and the name of the selection."""
    if cluster is None:
        return np.arange(len(labels)), 'the ensemble'
    members = np.flatnonzero(labels == cluster)
    if not members.size:
        raise ValueError(f"no model in cluster {cluster}; clusters by size: {cluster_ids_by_size(labels)}")
    return members, f'cluster {cluster}'


def motif_probability_table(df, motifs, members=None, where=None):
    """P(motif present) in the ensemble, and in the selected models with their enrichment when a selection is given."""
    present = df[motifs] > 0
    t = pd.DataFrame({'P, ensemble': present.mean()})
    if members is not None and len(members) < len(df):
        t[f'P, {where}'] = present.iloc[members].mean()
        t['enrichment'] = t[f'P, {where}'] / t['P, ensemble'].replace(0, np.nan)
    return t


def hypotheses(d, basement, depth_cells=3):
    """
    Indicators (one boolean per row of the motif table d) of the hypotheses checked against the synthetic ground
    truth; `basement` is the code of the deepest reference unit. The dyke hypothesis uses `dyke`, not
    `cross_cutting`, which also counts pipes.
    """
    return {
        'at least one unit born':                           d['n_born'] > 0,
        'basement is one body':                             d['topology'].map(
            lambda k: ma.bodies_per_unit(k).get(float(basement), 0) == 1),
        'lens / enclosed body present':                     d['enclosed_body'] > 0,
        'basement breaches the cover':                      d['window'] > 0,
        'pipe from basement into cover':                    d['chimney'] > 0,
        'steep dyke cutting several units':                 d['dyke'] > 0,
        f'basement within {depth_cells} cells of surface':  d['z_deep_top'] < depth_cells,
    }


def hypothesis_groups(members, labels, chain_id, group_by='chain', min_size=1, n_groups=6):
    """
    Models and group names for per-group probabilities: by chain (all the selected models), or by cluster (the
    selected models lying in the n_groups largest clusters with at least min_size models). Returns (rows, groups).
    """
    members = np.asarray(members)
    if group_by == 'cluster':
        shown = cluster_ids_by_size(labels[members], min_size)[:n_groups]
        rows = members[np.isin(labels[members], shown)]
        return rows, np.array([f'cluster {l}' for l in labels[rows]])
    if group_by == 'chain':
        return members, np.asarray(chain_id)[members]
    raise ValueError(f"group_by must be 'chain' or 'cluster', not {group_by!r}")


# ---------------------------------------------------------------------------
# Synthetic ensemble: drawn objects
# ---------------------------------------------------------------------------
def drawn_objects(truth, flat_cos=None):
    """
    Synthetic ensemble only: for each validated motif, whether each model holds the drawn object that the motif was
    checked against (P_drawn of the validation), from the truth table of synthetic_ensemble.make_ensemble:
    chimney -> pipe; dyke -> dyke dipping more than arccos(flat_cos) (the steepness threshold of the sheets, default
    ma.DEFAULTS['flat_cos'], i.e. about 73 deg); window -> basement high; enclosed_body -> lens. The drawn object
    does not always realise the motif: a window also needs the high to reach the cover, an enclosed body a lens that
    is neither truncated by the cover or the basement nor touching a side of the region.
    """
    fc = ma.DEFAULTS['flat_cos'] if flat_cos is None else flat_cos
    dip_min = np.degrees(np.arccos(fc))
    return pd.DataFrame({'chimney': truth['pipe'].astype(bool),
                         'dyke': truth['dyke'].astype(bool) & (truth['dyke_dip'].fillna(0) > dip_min),
                         'window': truth['high'].astype(bool),
                         'enclosed_body': truth['lens'].astype(bool)})
