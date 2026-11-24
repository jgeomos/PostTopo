"""
Topological feature extraction for posterior model ensembles.

Computes 6 scalar features per 3D labelled model:
  0. k             — number of discrete geological units
  1. total_cc      — total connected components across all units
  2. fiedler_w     — weighted Fiedler value (algebraic connectivity)
  3. he_entropy    — hyperedge entropy of the junction types (sets of >= 3 units meeting at a vertex)
  4. mean_herf     — mean Herfindahl index (fragmentation concentration)
  5. mean_ellip    — mean ellipticity of largest component per unit

The units are replaced by integer codes 0..k-1 (in increasing order of value) once, and the connected components of
all units are labelled by a single cc3d call (6-connectivity; cells of different units are never joined), so that
the cost per model does not grow with the number of units or of components.

Dependencies: numpy, cc3d
"""

import numpy as np
import cc3d


def _unit_contact_matrix(codes, k):
    """
    Number of shared faces between each pair of units, as a symmetric (k, k) float array with a zero diagonal.
    codes: integer unit codes 0..k-1 of the cells.
    """
    A = np.zeros(k * k, dtype=np.int64)
    for axis in range(3):
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[axis] = slice(None, -1)
        sl_b[axis] = slice(1, None)
        a = codes[tuple(sl_a)].ravel()
        b = codes[tuple(sl_b)].ravel()
        diff = a != b
        if diff.any():
            a, b = a[diff].astype(np.int64), b[diff].astype(np.int64)
            A += np.bincount(np.minimum(a, b) * k + np.maximum(a, b), minlength=k * k)
    A = A.reshape(k, k)
    return (A + A.T).astype(np.float64)


def _weighted_fiedler(A):
    """Fiedler value of the unit-level graph weighted by contact area (A: weighted adjacency matrix)."""
    if len(A) <= 1:
        return 0.0
    L = np.diag(A.sum(axis=1)) - A
    eigvals = np.sort(np.linalg.eigvalsh(L))
    # Second eigenvalue; guard against numerical noise
    return float(max(eigvals[1], 0.0))


def _hyperedge_entropy(codes):
    """
    Shannon entropy (bits) of the junction types of a model (codes: integer unit codes of the cells). At each grid
    vertex, the units of the 8 cells sharing it are collected; a vertex where three or more units meet is a
    junction, and its type is the set of those units (which units meet, not how many cells each occupies), i.e. one
    column of the incidence matrix of the contact hypergraph.
    """
    corners = np.stack([
        codes[:-1, :-1, :-1], codes[:-1, :-1, 1:],
        codes[:-1, 1:, :-1],  codes[:-1, 1:, 1:],
        codes[1:, :-1, :-1],  codes[1:, :-1, 1:],
        codes[1:, 1:, :-1],   codes[1:, 1:, 1:],
    ], axis=-1).reshape(-1, 8)

    s = np.sort(corners, axis=1)
    dup = np.zeros(s.shape, dtype=bool)
    dup[:, 1:] = s[:, 1:] == s[:, :-1]            # unit repeated within the vertex
    junction = (~dup).sum(axis=1) >= 3
    if not junction.any():
        return 0.0

    # Junction type = set of units: repeated codes are replaced by a filler (larger than any code) and moved to the
    # end of the row, so that vertices joining the same units give the same row whatever the number of cells of each
    # unit. Each row is then read as one key: a 64-bit integer for one-byte codes (fewer than 255 units), a byte
    # string otherwise.
    s, dup = s[junction], dup[junction]
    s = np.sort(np.where(dup, np.iinfo(s.dtype).max, s), axis=1)
    keys = np.ascontiguousarray(s).view(np.dtype((np.void, s.dtype.itemsize * 8))).ravel() if s.itemsize > 1 \
        else np.ascontiguousarray(s).view(np.uint64).ravel()
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
    model = np.asarray(model)
    units, inv = np.unique(model, return_inverse=True)
    k = len(units)
    codes = inv.reshape(model.shape).astype(np.uint8 if k < 255 else np.uint16)   # 0..k-1; the top value is a filler

    # Connected components of all units at once (cc3d does not join cells of different labels); components are
    # numbered in the order of their first cell in the grid, as when each unit is labelled on its own.
    labels_cc, total_cc = cc3d.connected_components(codes.astype(np.uint32) + 1, connectivity=6, return_N=True)
    flat = labels_cc.reshape(-1).astype(np.intp)
    volumes = np.bincount(flat, minlength=total_cc + 1)[1:]                    # component volumes (cells)
    comp_unit = np.empty(total_cc, dtype=np.intp)
    comp_unit[flat - 1] = codes.reshape(-1)                                     # unit code of each component
    by_unit = np.argsort(comp_unit, kind='stable')                              # grouped by unit, in label order
    n_comp = np.bincount(comp_unit, minlength=k)                                # components of each unit
    ends = np.cumsum(n_comp)

    herfindahl_vals = []
    ellipticity_vals = []
    for u in range(k):
        ids = by_unit[ends[u] - n_comp[u]:ends[u]]                              # components of unit u
        vols = volumes[ids]
        # Herfindahl index
        fracs = vols / vols.sum()
        herfindahl_vals.append(float(np.sum(fracs ** 2)))
        # Ellipticity of largest component
        coords = np.argwhere(labels_cc == ids[np.argmax(vols)] + 1).astype(float)
        ellipticity_vals.append(_ellipticity(coords))

    # Unit-level weighted adjacency
    fiedler_w = _weighted_fiedler(_unit_contact_matrix(codes, k))

    # Hyperedge entropy
    he_entropy = _hyperedge_entropy(codes)

    mean_herf = np.mean(herfindahl_vals) if herfindahl_vals else 1.0
    mean_ellip = np.mean(ellipticity_vals) if ellipticity_vals else 0.0

    return np.array([k, total_cc, fiedler_w, he_entropy, mean_herf, mean_ellip],
                    dtype=np.float32)


# Feature names for reference
FEATURE_NAMES = [
    'k', 'total_cc', 'fiedler_w', 'he_entropy', 'mean_herfindahl', 'mean_ellipticity'
]
