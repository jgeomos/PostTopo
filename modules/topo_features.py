"""
Topological feature extraction for posterior model ensembles.

Computes 6 scalar features per 3D labelled model:
  0. k             — number of discrete geological units
  1. total_cc      — total connected components across all units
  2. fiedler_w     — weighted Fiedler value (algebraic connectivity)
  3. he_entropy    — hyperedge entropy of the junction types (sets of >= 3 units meeting at a vertex)
  4. mean_herf     — mean Herfindahl index (fragmentation concentration)
  5. mean_ellip    — mean ellipticity of largest component per unit

Dependencies: numpy, cc3d
"""

import numpy as np
import cc3d
from collections import Counter


def _unit_level_contacts(model):
    """
    Count shared faces between each pair of distinct unit labels.

    Returns dict: (min_label, max_label) -> face_count
    """
    contacts = Counter()
    for axis in range(3):
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[axis] = slice(None, -1)
        sl_b[axis] = slice(1, None)
        a = model[tuple(sl_a)].ravel()
        b = model[tuple(sl_b)].ravel()
        diff = a != b
        if not diff.any():
            continue
        lo = np.minimum(a[diff], b[diff])
        hi = np.maximum(a[diff], b[diff])
        pairs, counts = np.unique(
            np.stack([lo, hi], axis=1), axis=0, return_counts=True
        )
        for (va, vb), cnt in zip(pairs, counts):
            contacts[(va, vb)] += int(cnt)
    return contacts


def _weighted_fiedler(units, contacts):
    """Fiedler value of the unit-level graph weighted by contact area."""
    k = len(units)
    if k <= 1:
        return 0.0
    idx = {u: i for i, u in enumerate(units)}
    A = np.zeros((k, k))
    for (va, vb), w in contacts.items():
        i, j = idx[va], idx[vb]
        A[i, j] = w
        A[j, i] = w
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals = np.sort(np.linalg.eigvalsh(L))
    # Second eigenvalue; guard against numerical noise
    return float(max(eigvals[1], 0.0)) if len(eigvals) > 1 else 0.0


def _hyperedge_entropy(model):
    """
    Shannon entropy (bits) of the junction types of a model. At each grid vertex, the units of the 8 cells
    sharing it are collected; a vertex where three or more units meet is a junction, and its type is the set
    of those units (which units meet, not how many cells each occupies), i.e. one column of the incidence
    matrix of the contact hypergraph.
    """
    corners = np.stack([
        model[:-1, :-1, :-1], model[:-1, :-1, 1:],
        model[:-1, 1:, :-1],  model[:-1, 1:, 1:],
        model[1:, :-1, :-1],  model[1:, :-1, 1:],
        model[1:, 1:, :-1],   model[1:, 1:, 1:],
    ], axis=-1).reshape(-1, 8)

    s = np.sort(corners, axis=1)
    dup = np.zeros(s.shape, dtype=bool)
    dup[:, 1:] = s[:, 1:] == s[:, :-1]            # label repeated within the vertex
    junction = (~dup).sum(axis=1) >= 3
    if not junction.any():
        return 0.0

    # Junction type = set of units: repeated labels are replaced by a filler and moved to the end of the row,
    # so that vertices joining the same units give the same row whatever the number of cells of each unit.
    s, dup = s[junction], dup[junction]
    fill = np.inf if np.issubdtype(s.dtype, np.floating) else np.iinfo(s.dtype).max
    s = np.sort(np.where(dup, fill, s), axis=1)
    keys = np.ascontiguousarray(s).view(np.dtype((np.void, s.dtype.itemsize * 8))).ravel()
    _, counts = np.unique(keys, return_counts=True)

    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)) + 0.0)    # + 0.0: a single type gives 0.0, not -0.0


def _ellipticity(coords):
    """Ellipticity from covariance eigenvalues: 1 - min/max."""
    if len(coords) < 4:
        return 0.0
    cov = np.cov(coords.T)
    ev = np.linalg.eigvalsh(cov)
    return float(1.0 - ev[0] / ev[-1]) if ev[-1] > 0 else 0.0


def compute_topo_features(model):
    """
    Compute 6 topological scalars for a single 3D labelled model.

    Parameters
    ----------
    model : ndarray, shape (nz, ny, nx)
        Discrete unit labels (density contrast values).

    Returns
    -------
    ndarray, shape (6,)
        [k, total_cc, fiedler_w, he_entropy, mean_herf, mean_ellip]
    """
    units = np.unique(model)
    k = len(units)

    total_cc = 0
    herfindahl_vals = []
    ellipticity_vals = []

    for unit in units:
        binary = (model == unit).astype(np.uint8)
        labels_cc = cc3d.connected_components(binary, connectivity=6)
        cc_ids = np.unique(labels_cc)
        cc_ids = cc_ids[cc_ids != 0]
        n_cc = len(cc_ids)
        total_cc += n_cc

        # Component volumes
        volumes = np.array([np.sum(labels_cc == cid) for cid in cc_ids])

        # Herfindahl index
        if n_cc > 0:
            fracs = volumes / volumes.sum()
            herfindahl_vals.append(float(np.sum(fracs ** 2)))

        # Ellipticity of largest component
        if n_cc > 0:
            largest = cc_ids[np.argmax(volumes)]
            coords = np.argwhere(labels_cc == largest).astype(float)
            ellipticity_vals.append(_ellipticity(coords))

    # Unit-level weighted adjacency
    contacts = _unit_level_contacts(model)
    fiedler_w = _weighted_fiedler(units, contacts)

    # Hyperedge entropy
    he_entropy = _hyperedge_entropy(model)

    mean_herf = np.mean(herfindahl_vals) if herfindahl_vals else 1.0
    mean_ellip = np.mean(ellipticity_vals) if ellipticity_vals else 0.0

    return np.array([k, total_cc, fiedler_w, he_entropy, mean_herf, mean_ellip],
                    dtype=np.float32)


# Feature names for reference
FEATURE_NAMES = [
    'k', 'total_cc', 'fiedler_w', 'he_entropy', 'mean_herfindahl', 'mean_ellipticity'
]
