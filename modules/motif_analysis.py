"""
Geological motif analysis of 3D labelled models and posterior ensembles.

Conventions (same as topology_tools.py / topo_features.py)
-----------------------------------------------------------
- model : ndarray (nz, ny, nx) of unit labels (density or density-contrast
  values, one value per unit). Axis 0 is depth, index 0 is the top.
- ensembles : ndarray (nz, ny, nx, n_models), model i = data_models[..., i].
- 6-connectivity (face adjacency) everywhere, as in geological_topology().
- cell_sizes = (dz, dy, dx) in physical units. Shape and orientation tests
  (steep, prolate, oblate, layer aspect) are only meaningful in physical
  units: with anisotropic cells, pass the true cell sizes.

What it does
------------
0. Unit identity across models (optional, `reference=`). Unit densities are
   sampled, so one unit carries a different value in each model (the label
   switching problem of MCMC, Stephens, 2000). Each model is relabelled
   against a reference model (prior, or modal stand-in): reference units get
   codes 0..K-1 ordered from top to bottom, born units get codes >= K.
1. One model -> a body-level contact graph (nodes = connected components,
   not units) with shape, orientation, footprint and above/below/lateral
   contact attributes. A single cc3d call labels all bodies of all units at
   once; adjacency and orientation come from axis-shifted comparisons of the
   body-label volume.
2. Motif predicates on that graph -> a count vector per model.
3. Ensemble tables: P(motif | d), co-occurrence, cluster enrichment,
   prior/posterior enrichment, topology census, distance to prior,
   per-motif "where" maps.

Shape classes (Woodcock, 1977)
------------------------------
Eigenvalues lam1 >= lam2 >= lam3 of the second-moment tensor of the body
(voxel boxes, physical units), K = ln(lam1/lam2) / ln(lam2/lam3) (shape) and
C = ln(lam1/lam3) (strength). A body gets a shape class only if it is
`shaped`: volume >= vmin_shape voxels, C >= c_min (not equant: K is
undefined, 0/0, for equant bodies) and at most f_side_max of its surface on
the lateral faces of the model (a body cut by the model boundary has the
shape of the cut, not its own). prolate: K > k_prolate; oblate: K < k_oblate.

Motifs (geological reading in brackets)
---------------------------------------
enclosed_body   touches no model face, >= enclosed_host of its
                surface against one unit                   [lens, enclave, intrusion]
layer           footprint >= layer_footprint of the model columns, mean
                thickness / sqrt(footprint area) <= layer_aspect, not the
                half-space (bottom contact on <= bottom_max of the footprint),
                different dominant units above and below (or model top
                above)                                     [laterally continuous sheet]
steep_body      shaped and (prolate with |e1.z| > steep_cos, or oblate
                with |e3.z| < flat_cos)                    [dyke, pipe]
cross_cutting   steep_body with lateral contacts against
                >= cross_units distinct units              [intrusion cutting several units: dyke or pipe]
dyke            cross_cutting body that is oblate (a steep
                sheet)                                     [dyke cutting the layering]
vertical_trap   touches neither model top nor bottom, same dominant unit
                above and below, covering >= trap_frac of both the top and
                bottom faces of the body                   [lens inside a layer]
                (not reported by default: see EXCLUDED)
bridge          articulation point (not a layer) whose removal leaves
                >= bridge_parts face-reaching sub-graphs, each of volume
                >= bridge_ratio x the body volume          [separator: splits the model into compartments]
chimney         shaped and prolate (a rod), in contact with a `deep_units`
                unit and with a `top_units` unit over at least amin_link
                faces each, whatever its plunge            [pipe linking the basement to the cover]
window          body of a `deep_units` unit whose faces directly below a
                `top_units` unit plus its faces on the model top number
                >= amin_window                             [basement breach of the cover]
order_violation edge where the majority above/below sense contradicts the
                stratigraphic order `order`
fragment        body dropped by the volume floor           [sampler noise, not a motif]

chimney and dyke ask only what their reading states: contacts and the shape
class. chimney does not test the plunge nor which units lie above and below
the body: for an inclined rod, the horizontal faces lie along its whole
length, so that the dominant units above and below describe what the rod
crosses, not where its ends are. For the same reason, vertical_trap (same
dominant unit above and below) also flags inclined rods crossing a layer; it
is computed but not reported (EXCLUDED), and can be reported again by
removing it from EXCLUDED (a module constant, so that parallel workers see it).

All thresholds are in DEFAULTS and can be changed per call with thr={...}
(passed through motif_table, so that parallel workers see them).

Usage
-----
    import motif_analysis as ma
    ref = ma.reference_codes(prior_model)           # or ma.modal_reference(models)
    kw = dict(vmin_voxels=8, reference=ref, cell_sizes=(dz, dy, dx))
    df = ma.motif_table(models, n_jobs=-1, **kw)    # deep/top units and order default
    ma.motif_probabilities(df)                      # to basement / cover / prior order

Dependencies: numpy, cc3d, networkx, pandas, joblib.

Background and references
-------------------------
Contact graphs of geological models and topological uncertainty:
    Thiele et al. (2016a, b); Pakyuz-Charrier et al. (2019).
Connected-component labelling: Rosenfeld & Pfaltz (1966); cc3d (Silversmith).
Shape fabric from eigenvalues (K and C parameters): Woodcock (1977).
Articulation points: Hopcroft & Tarjan (1973).
Label switching across MCMC samples: Stephens (2000); Jasra et al. (2005).
Network motifs and null-model enrichment: Milo et al. (2002).
Posterior probabilities as Monte Carlo averages of indicators:
    Robert & Casella (2004); effective sample size: Geyer (1992).
Trans-D sampling and model selection: Green (1995); Sambridge et al. (2006);
    Bodin & Sambridge (2009).
Generalised (Gibbs) posterior: Bissiri et al. (2016).
Bayes factors: Kass & Raftery (1995).
Lift (association rules): Brin et al. (1997).
Jaccard distance: Jaccard (1912); Levandowsky & Winter (1971).
Voxel probability maps of model ensembles: Wellmann & Regenauer-Lieb (2012).

Beaumont, M.A., Zhang, W. & Balding, D.J., 2002. Approximate Bayesian
    computation in population genetics, Genetics, 162(4), 2025-2035.
Bissiri, P.G., Holmes, C.C. & Walker, S.G., 2016. A general framework for
    updating belief distributions, J. R. Stat. Soc. B, 78(5), 1103-1130.
Bodin, T. & Sambridge, M., 2009. Seismic tomography with the reversible jump
    algorithm, Geophys. J. Int., 178(3), 1411-1436.
Brin, S., Motwani, R., Ullman, J.D. & Tsur, S., 1997. Dynamic itemset counting
    and implication rules for market basket data, Proc. ACM SIGMOD, 255-264.
Geyer, C.J., 1992. Practical Markov chain Monte Carlo, Stat. Sci., 7(4),
    473-483.
Green, P.J., 1995. Reversible jump Markov chain Monte Carlo computation and
    Bayesian model determination, Biometrika, 82(4), 711-732.
Hopcroft, J. & Tarjan, R., 1973. Algorithm 447: efficient algorithms for graph
    manipulation, Commun. ACM, 16(6), 372-378.
Jaccard, P., 1912. The distribution of the flora in the alpine zone,
    New Phytol., 11(2), 37-50.
Jasra, A., Holmes, C.C. & Stephens, D.A., 2005. Markov chain Monte Carlo
    methods and the label switching problem in Bayesian mixture modeling,
    Stat. Sci., 20(1), 50-67.
Kass, R.E. & Raftery, A.E., 1995. Bayes factors, J. Am. Stat. Assoc.,
    90(430), 773-795.
Levandowsky, M. & Winter, D., 1971. Distance between sets, Nature, 234, 34-35.
Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N., Chklovskii, D. & Alon, U.,
    2002. Network motifs: simple building blocks of complex networks,
    Science, 298(5594), 824-827.
Pakyuz-Charrier, E., Jessell, M., Giraud, J., Lindsay, M. & Ogarko, V., 2019.
    Topological analysis in Monte Carlo simulation for uncertainty
    propagation, Solid Earth, 10(5), 1663-1684.
Robert, C.P. & Casella, G., 2004. Monte Carlo Statistical Methods, 2nd edn,
    Springer.
Rosenfeld, A. & Pfaltz, J.L., 1966. Sequential operations in digital picture
    processing, J. ACM, 13(4), 471-494.
Sambridge, M., Gallagher, K., Jackson, A. & Rickwood, P., 2006.
    Trans-dimensional inverse problems, model comparison and the evidence,
    Geophys. J. Int., 167(2), 528-542.
Silversmith, W. cc3d: connected components on multilabel 3D images, software,
    https://github.com/seung-lab/connected-components-3d.
Stephens, M., 2000. Dealing with label switching in mixture models,
    J. R. Stat. Soc. B, 62(4), 795-809.
Thiele, S.T., Jessell, M.W., Lindsay, M., Ogarko, V., Wellmann, J.F. &
    Pakyuz-Charrier, E., 2016a. The topology of geology 1: Topological
    analysis, J. Struct. Geol., 91, 27-38.
Thiele, S.T., Jessell, M.W., Lindsay, M., Wellmann, J.F. & Pakyuz-Charrier,
    E., 2016b. The topology of geology 2: Topological uncertainty,
    J. Struct. Geol., 91, 74-87.
Wellmann, J.F. & Regenauer-Lieb, K., 2012. Uncertainties have a meaning:
    information entropy as a quality measure for 3-D geological models,
    Tectonophysics, 526-529, 207-216.
Woodcock, N.H., 1977. Specification of fabric shapes using an eigenvalue
    method, Geol. Soc. Am. Bull., 88(9), 1231-1236.
"""

from __future__ import annotations

import numpy as np
import cc3d
import networkx as nx
import pandas as pd
from joblib import Parallel, delayed

ALL_MOTIFS = ['enclosed_body', 'layer', 'steep_body', 'cross_cutting', 'dyke',
              'vertical_trap', 'bridge', 'chimney', 'window',
              'order_violation', 'fragment']
# Motifs computed but neither flagged nor counted (edit here: parallel workers import this module).
EXCLUDED = ('vertical_trap',)
MOTIFS = [m for m in ALL_MOTIFS if m not in EXCLUDED]

# Thresholds of the motif predicates; override per call with thr={...}.
DEFAULTS = dict(
    k_prolate=2.0,         # Woodcock K above which a shaped body is prolate (rod-like)
    k_oblate=0.5,          # K below which a shaped body is oblate (sheet-like)
    c_min=1.0,             # minimum strength C = ln(lam1/lam3), i.e. axis ratio >= e^(1/2) ~ 1.65
    vmin_shape=50,         # minimum number of voxels to give a body a shape class
    f_side_max=0.25,       # maximum fraction of the body surface on the lateral model faces
    steep_cos=0.7,         # prolate and |e1.z| > 0.7: long axis plunges more than ~45 deg
    flat_cos=0.3,          # oblate and |e3.z| < 0.3: sheet dips more than ~73 deg
    layer_footprint=0.25,  # minimum fraction of the model columns covered by a layer
    layer_aspect=0.3,      # maximum mean thickness / sqrt(footprint area) of a layer
    bottom_max=0.5,        # maximum fraction of the footprint on the model bottom (half-space)
    trap_frac=0.5,         # host fraction of both the top and bottom faces of a trapped body (vertical_trap)
    amin_window=None,      # minimum basement-under-cover faces for a window (None: vmin_voxels)
    amin_link=1,           # minimum faces of a chimney against each of the deep and top units (1: any contact)
    cross_units=2,         # minimum number of distinct units in lateral contact with a cross-cutting body or dyke
    bridge_parts=2,        # minimum number of face-reaching compartments left when a bridge is removed
    bridge_ratio=1.0,      # each compartment has at least this times the volume of the bridge
    enclosed_host=0.75,    # minimum fraction of the body surface against its main host unit
)


def _params(thr):
    p = dict(DEFAULTS)
    if thr:
        unknown = set(thr) - set(DEFAULTS)
        if unknown:
            raise KeyError(f"unknown thresholds {sorted(unknown)}; valid: {sorted(DEFAULTS)}")
        p.update(thr)
    return p


# ---------------------------------------------------------------------------
# 0. Unit identity across models
# ---------------------------------------------------------------------------
def reference_codes(reference):
    """
    Integer codes 0..K-1 of the units of a reference model, ordered by
    increasing mean depth (0 = shallowest, e.g. cover; K-1 = deepest, e.g.
    basement). Idempotent on its own output.
    """
    ref = np.asarray(reference)
    vals, inv = np.unique(ref, return_inverse=True)
    inv = inv.reshape(-1)
    z = np.broadcast_to(np.arange(ref.shape[0])[:, None, None], ref.shape).reshape(-1)
    zmean = np.bincount(inv, weights=z) / np.bincount(inv)
    rank = np.empty(len(vals), np.int16)
    rank[np.argsort(zmean, kind='stable')] = np.arange(len(vals))
    return rank[inv].reshape(ref.shape)


def canonical_labels(model, ref_codes, return_map=False, min_frac=0.5):
    """
    Relabel the units of `model` against reference codes (from
    reference_codes()), so that units can be compared across models.

    Let T[v, k] be the number of voxels of model value v inside reference
    unit k. Value v takes code k if (i) v and k are mutual best overlaps,
    v = argmax_v' T[v', k] and k = argmax_k' T[v, k']; or (ii) k was left
    unmatched by (i), k = argmax_k' T[v, k'], at least `min_frac` of the
    voxels of v lie in k, and v is the largest such value (a thinned or
    displaced reference unit). All other values are born units and get codes
    K, K+1, ... in increasing order of value.

    Returns int16 labels of the shape of `model` (and {value: code} if
    return_map).

    References
    ----------
    Relabelling of sampled models against a reference: the label-switching
    problem of MCMC (Stephens, 2000; Jasra et al., 2005).
    """
    model = np.asarray(model)
    ref = np.asarray(ref_codes).reshape(-1).astype(np.int64)
    K = int(ref.max()) + 1
    vals, inv = np.unique(model, return_inverse=True)
    inv = inv.reshape(-1)
    nv = len(vals)
    T = np.bincount(inv * K + ref, minlength=nv * K).reshape(nv, K)
    best_k, best_v = T.argmax(1), T.argmax(0)
    code = np.full(nv, -1, np.int64)
    mutual = best_v[best_k] == np.arange(nv)
    code[mutual] = best_k[mutual]
    free_k = np.setdiff1d(np.arange(K), code[mutual])
    if free_k.size:
        frac = T[np.arange(nv), best_k] / np.maximum(T.sum(1), 1)
        for k in free_k:
            cand = np.flatnonzero((code < 0) & (best_k == k) & (frac >= min_frac))
            if cand.size:
                code[cand[T[cand, k].argmax()]] = k
    born = np.flatnonzero(code < 0)
    code[born] = K + np.arange(born.size)
    lab = code.astype(np.int16)[inv].reshape(model.shape)
    if return_map:
        return lab, dict(zip(vals.tolist(), code.tolist()))
    return lab


def modal_reference(models, n_sample=200, n_iter=2):
    """
    Modal stand-in for a missing prior model, robust to sampled densities.

    Starts from a sampled model with the modal number of units, relabels
    `n_sample` evenly spaced models against it with canonical_labels(),
    takes the voxel-wise most frequent reference code (born units ignored)
    and repeats `n_iter` times. Returns reference codes (0..K-1, top to
    bottom). Units present in few models vanish from the mode.
    """
    N = models.shape[-1]
    idx = np.unique(np.linspace(0, N - 1, min(n_sample, N)).round().astype(int))
    nu = np.array([len(np.unique(models[..., i])) for i in idx])
    v, c = np.unique(nu, return_counts=True)
    ref = reference_codes(models[..., idx[np.flatnonzero(nu == v[c.argmax()])[0]]])
    shape, nvox = ref.shape, ref.size
    vox = np.arange(nvox, dtype=np.int64)
    for _ in range(n_iter):
        K = int(ref.max()) + 1
        cnt = np.zeros(K * nvox, np.int64)
        for i in idx:
            lab = canonical_labels(models[..., i], ref).reshape(-1).astype(np.int64)
            ok = lab < K
            cnt += np.bincount(lab[ok] * nvox + vox[ok], minlength=K * nvox)
        ref = reference_codes(cnt.reshape(K, nvox).argmax(0).reshape(shape))
    return ref


# ---------------------------------------------------------------------------
# 1. Body graph
# ---------------------------------------------------------------------------
def body_graph(model, vmin_voxels=1, cell_sizes=(1.0, 1.0, 1.0), thr=None):
    """
    Body-level contact graph of one model.

    Parameters
    ----------
    model : ndarray (nz, ny, nx), unit labels.
    vmin_voxels : int
        Bodies smaller than this are dropped from the graph (volume floor).
        Their count is returned in G.graph['n_fragments'].
    cell_sizes : (dz, dy, dx), physical cell sizes (shape, areas, aspect).
    thr : dict of thresholds overriding DEFAULTS (shape-class part).

    Returns
    -------
    G : nx.Graph
        node = body id (int >= 1). Node attrs:
            unit, volume (voxels), centroid (iz, iy, ix) in cell indices,
            zmin, zmax, touches_top, touches_bottom, touches_side,
            f_side    fraction of the body surface on the lateral model faces
            f_bottom  bottom-face voxels / footprint
            footprint number of (y, x) columns occupied; f_footprint = same
                      divided by ny * nx
            aspect    mean thickness / sqrt(footprint area), physical units
            K, C      Woodcock (1977) shape and strength
            shaped, prolate, oblate, steep (bools)
            axis_vz   |z-component| of the long axis e1
            normal_vz |z-component| of the short axis e3 (sheet normal)
        edge attrs:
            area   contact area (physical)
            above  {b: number of horizontal faces where body b lies directly
                   above the other body}, stored for both ends
            lat    number of vertical (lateral) contact faces
        G.graph: n_fragments, units (values), unit_of (unit per body id,
        nan for 0), shape, cell_sizes.
    labels : ndarray (nz, ny, nx) of body ids (0 = dropped fragment)

    References
    ----------
    Bodies are 6-connected components (Rosenfeld & Pfaltz, 1966; cc3d);
    the graph follows the contact-graph representation of geological models
    of Thiele et al. (2016a), at the level of bodies rather than units.
    Shape classes from the eigenvalues of the second-moment tensor
    (Woodcock, 1977).
    """
    p = _params(thr)
    model = np.asarray(model)
    n_z, n_y, n_x = model.shape
    dz, dy, dx = map(float, cell_sizes)

    # --- one cc3d call for all units (labels are not re-used across units)
    units_all, inv = np.unique(model, return_inverse=True)
    shifted = (inv.reshape(model.shape) + 1).astype(np.uint32)
    labels, n_body = cc3d.connected_components(shifted, connectivity=6, return_N=True)
    labels = labels.astype(np.int64, copy=False)
    nb1 = n_body + 1

    lab = labels.reshape(-1)
    vol = np.bincount(lab, minlength=nb1)
    keep = vol >= vmin_voxels
    keep[0] = False
    n_frag = int((~keep[1:] & (vol[1:] > 0)).sum())
    # Drop fragments: relabel them to 0 (they still separate bodies, which is
    # the conservative choice: a fragment-mediated contact is not a contact).
    if n_frag:
        labels = np.where(keep[labels], labels, 0)
        lab = labels.reshape(-1)

    m = lab > 0
    unit_of = np.full(nb1, np.nan)
    unit_of[lab[m]] = model.reshape(-1)[m]          # same value for every voxel of a body
    G = nx.Graph()
    G.graph.update(n_fragments=n_frag, units=units_all, unit_of=unit_of,
                   shape=model.shape, cell_sizes=(dz, dy, dx))
    ids = np.flatnonzero(keep)
    if ids.size == 0:
        return G, labels

    # --- per-body moments via bincount (cell indices)
    iz, iy, ix = (a.reshape(-1) for a in np.indices(model.shape))
    lab_m, izm, iym, ixm = lab[m], iz[m], iy[m], ix[m]
    w = np.bincount(lab_m, minlength=nb1).astype(float)
    w[w == 0] = 1.0

    def mean_of(a):
        return np.bincount(lab_m, weights=a, minlength=nb1) / w

    cz, cy, cx = mean_of(izm), mean_of(iym), mean_of(ixm)
    szz, syy, sxx = mean_of(izm * izm) - cz ** 2, mean_of(iym * iym) - cy ** 2, mean_of(ixm * ixm) - cx ** 2
    szy, szx, syx = mean_of(izm * iym) - cz * cy, mean_of(izm * ixm) - cz * cx, mean_of(iym * ixm) - cy * cx
    zmin = np.full(nb1, n_z); np.minimum.at(zmin, lab_m, izm)
    zmax = np.full(nb1, -1); np.maximum.at(zmax, lab_m, izm)

    # --- footprint (columns occupied) and boundary faces
    ncol = n_y * n_x
    colkey = np.unique(lab_m * ncol + (iym * n_x + ixm))
    footprint = np.bincount(colkey // ncol, minlength=nb1)

    def count(a):
        return np.bincount(a.reshape(-1), minlength=nb1)

    a_z, a_y, a_x = dy * dx, dz * dx, dz * dy          # face areas normal to z, y, x
    n_top, n_bot = count(labels[0]), count(labels[-1])
    s_side = (count(labels[:, 0, :]) + count(labels[:, -1, :])) * a_y \
        + (count(labels[:, :, 0]) + count(labels[:, :, -1])) * a_x
    s_tot = s_side + (n_top + n_bot) * a_z
    n_side = count(labels[:, 0, :]) + count(labels[:, -1, :]) + count(labels[:, :, 0]) + count(labels[:, :, -1])

    # --- contacts from axis shifts (all faces, incl. fragments, for the surface)
    areas = (a_z, a_y, a_x)
    contacts = []
    for axis in range(3):
        lo = [slice(None)] * 3; hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        a, b = labels[tuple(lo)].reshape(-1), labels[tuple(hi)].reshape(-1)   # a shallower along z
        d = a != b
        s_tot = s_tot + (np.bincount(a[d], minlength=nb1) + np.bincount(b[d], minlength=nb1)) * areas[axis]
        dd = d & (a > 0) & (b > 0)
        if dd.any():
            key, cnt = np.unique(a[dd] * nb1 + b[dd], return_counts=True)
            contacts.append((axis, key // nb1, key % nb1, cnt))
    f_side = s_side / np.maximum(s_tot, 1e-12)

    # --- nodes
    S = np.array([dz, dy, dx])
    box = np.diag(S ** 2) / 12.0                     # second moment of one voxel box
    for b in ids:
        Ci = np.array([[szz[b], szy[b], szx[b]],
                       [szy[b], syy[b], syx[b]],
                       [szx[b], syx[b], sxx[b]]])
        ev, vec = np.linalg.eigh(S[:, None] * Ci * S[None, :] + box)   # ascending; columns = vectors
        l3, l2, l1 = np.clip(ev, 1e-12, None)
        C = np.log(l1 / l3)
        K = np.log(l1 / l2) / max(np.log(l2 / l3), 1e-12)
        axis_vz, normal_vz = abs(vec[0, 2]), abs(vec[0, 0])
        shaped = vol[b] >= p['vmin_shape'] and C >= p['c_min'] and f_side[b] <= p['f_side_max']
        prolate = shaped and K > p['k_prolate']
        oblate = shaped and K < p['k_oblate']
        steep = (prolate and axis_vz > p['steep_cos']) or (oblate and normal_vz < p['flat_cos'])
        fp = max(int(footprint[b]), 1)
        aspect = (vol[b] * dz / fp) / np.sqrt(fp * dy * dx)
        G.add_node(int(b), unit=float(unit_of[b]), volume=int(vol[b]),
                   centroid=(float(cz[b]), float(cy[b]), float(cx[b])),
                   zmin=int(zmin[b]), zmax=int(zmax[b]),
                   touches_top=bool(n_top[b]), touches_bottom=bool(n_bot[b]), n_top=int(n_top[b]),
                   touches_side=bool(n_side[b]), f_side=float(f_side[b]),
                   f_bottom=float(n_bot[b] / fp), footprint=fp, f_footprint=fp / ncol,
                   aspect=float(aspect), K=float(min(K, 1e3)), C=float(C),
                   shaped=bool(shaped), prolate=bool(prolate), oblate=bool(oblate),
                   steep=bool(steep), axis_vz=float(axis_vz), normal_vz=float(normal_vz))

    # --- edges
    for axis, us, vs, cnt in contacts:
        for u, v, c in zip(us.tolist(), vs.tolist(), cnt.tolist()):
            if not G.has_edge(u, v):
                G.add_edge(u, v, area=0.0, above={u: 0, v: 0}, lat=0)
            e = G.edges[u, v]
            e['area'] += c * areas[axis]
            if axis == 0:               # u (index k) is directly above v (index k+1)
                e['above'][u] += c
            else:
                e['lat'] += c
    return G, labels


# ---------------------------------------------------------------------------
# 2. Motifs on one graph
# ---------------------------------------------------------------------------
def _contacts(G, n):
    """Units directly above, below and beside body n, as {unit: faces}."""
    above, below, lateral = {}, {}, {}
    for nb in G.neighbors(n):
        e = G.edges[n, nb]
        u = G.nodes[nb]['unit']
        if e['above'][nb]:
            above[u] = above.get(u, 0) + e['above'][nb]    # nb over n
        if e['above'][n]:
            below[u] = below.get(u, 0) + e['above'][n]     # n over nb
        if e['lat']:
            lateral[u] = lateral.get(u, 0) + e['lat']
    return above, below, lateral


def motif_flags(G, deep_units=(), top_units=(), order=None, thr=None, amin=1):
    """
    Per-body motif flags for a graph from body_graph().

    Parameters
    ----------
    deep_units, top_units : labels (e.g. basement, cover) for the `window`
        and `chimney` motifs.
    order : list of unit labels from top (youngest) to bottom (oldest), or None.
        Used for `order_violation`: an edge (u over v) with u deeper than v in
        `order` by face majority is a violation. Units absent from `order`
        (e.g. born units) are ignored.
    thr : dict of thresholds overriding DEFAULTS.
    amin : default minimum number of faces for a window (used when
        thr['amin_window'] is None; motif_counts passes vmin_voxels).

    Returns
    -------
    flags : dict {body: set(motif names)}
    n_violations : int

    References
    ----------
    Motifs in the sense of attributed subgraph patterns (Milo et al., 2002);
    articulation points after Hopcroft & Tarjan (1973), as implemented in
    networkx; shape classes from Woodcock (1977).
    """
    p = _params(thr)
    amin_w = p['amin_window'] if p['amin_window'] is not None else max(int(amin), 1)
    flags = {n: set() for n in G.nodes}
    if G.number_of_nodes() == 0:
        return flags, 0

    # bridge: articulation point whose removal leaves >= bridge_parts
    # face-reaching sub-graphs, each at least bridge_ratio times as voluminous
    # as the body (a separator is smaller than what it separates; the
    # basement holding a pipe is not a separator).
    vol = nx.get_node_attributes(G, 'volume')
    reach = {n for n, a in G.nodes(data=True)
             if a['touches_top'] or a['touches_bottom'] or a['touches_side']}
    necks = set()
    for n in G.nodes:
        G.nodes[n]['bridge_minratio'] = 0.0
    if G.number_of_edges():
        for n in nx.articulation_points(G):
            H = G.subgraph([v for v in G if v != n])
            ratios = sorted((sum(vol[v] for v in cc) / vol[n] for cc in nx.connected_components(H) if cc & reach),
                            reverse=True)
            if len(ratios) >= p['bridge_parts']:
                # volume ratio of the smallest of the bridge_parts largest compartments
                G.nodes[n]['bridge_minratio'] = ratios[p['bridge_parts'] - 1]
                if ratios[p['bridge_parts'] - 1] >= p['bridge_ratio']:
                    necks.add(n)

    deep, topu = set(map(float, deep_units)), set(map(float, top_units))
    rank = {float(u): i for i, u in enumerate(order)} if order is not None else None
    n_viol = 0

    for n, a in G.nodes(data=True):
        above, below, lateral = _contacts(G, n)
        dom_up = max(above, key=above.get) if above else None
        dom_dn = max(below, key=below.get) if below else None
        f_up = above[dom_up] / sum(above.values()) if above else 0.0
        f_dn = below[dom_dn] / sum(below.values()) if below else 0.0
        touches = a['touches_top'] or a['touches_bottom'] or a['touches_side']
        tot = {}
        for d in (above, below, lateral):
            for u, c in d.items():
                tot[u] = tot.get(u, 0) + c
        w_faces = sum(above.get(u, 0) for u in topu) + (a['n_top'] if a['unit'] in deep else 0)
        deep_faces = sum(c for u, c in tot.items() if u in deep)     # faces against deep units, any direction
        top_faces = sum(c for u, c in tot.items() if u in topu)      # faces against top units, any direction
        a.update(dom_up=dom_up, dom_dn=dom_dn, f_up=f_up, f_dn=f_dn, n_lat_units=len(lateral),
                 host_frac=max(tot.values()) / sum(tot.values()) if tot else 0.0,
                 window_faces=w_faces, deep_faces=deep_faces, top_faces=top_faces, viol_faces=0)
        f = flags[n]

        if not touches and a['host_frac'] >= p['enclosed_host']:
            f.add('enclosed_body')
        # layer: laterally extensive, thin, not the half-space, different
        # units above and below (or the model top above).
        if (touches and a['f_footprint'] >= p['layer_footprint']
                and a['aspect'] <= p['layer_aspect'] and a['f_bottom'] <= p['bottom_max']
                and dom_dn is not None
                and ((dom_up is not None and dom_up != dom_dn) or a['touches_top'])):
            f.add('layer')
        if a['steep']:
            f.add('steep_body')
            if len(lateral) >= p['cross_units']:
                f.add('cross_cutting')
                if a['oblate']:
                    f.add('dyke')                         # steep sheet cutting several units
        if (not a['touches_top'] and not a['touches_bottom'] and dom_up is not None
                and dom_up == dom_dn and f_up >= p['trap_frac'] and f_dn >= p['trap_frac']):
            f.add('vertical_trap')
        if n in necks and 'layer' not in f:
            f.add('bridge')
        if (a['prolate'] and deep_faces >= max(p['amin_link'], 1)
                and top_faces >= max(p['amin_link'], 1)):          # rod linking a deep unit to a top unit
            f.add('chimney')
        if a['unit'] in deep and w_faces >= amin_w:       # faces under the cover + faces on the model top
            f.add('window')

    if rank is not None:
        for u, v, e in G.edges(data=True):
            ru, rv = rank.get(G.nodes[u]['unit']), rank.get(G.nodes[v]['unit'])
            if ru is None or rv is None or ru == rv:
                continue
            over_u, over_v = e['above'][u], e['above'][v]     # faces u-over-v, v-over-u
            if over_u == over_v:
                continue
            upper, lower = (u, v) if over_u > over_v else (v, u)
            if rank[G.nodes[upper]['unit']] > rank[G.nodes[lower]['unit']]:
                n_viol += 1
                flags[upper].add('order_violation'); flags[lower].add('order_violation')
                for b in (upper, lower):
                    G.nodes[b]['viol_faces'] += max(over_u, over_v)
    for f in flags.values():                              # motifs computed but not reported
        f.difference_update(EXCLUDED)
    G.graph.update(thr=p, amin_window=amin_w, vmin_voxels=max(int(amin), 1))
    return flags, n_viol


# ---------------------------------------------------------------------------
# Typicality of a motif
# ---------------------------------------------------------------------------
def _lin(x, t, ref):
    # Normalised margin of x past threshold t: 0 at t, 1 at ref (ref < t for criteria of the form x <= t).
    return float(np.clip((x - t) / (ref - t), 0.0, 1.0)) if ref != t else float(x >= t)


def _logm(x, t, decades):
    # Normalised margin on a log scale: 0 at t, 1 at t * 10**decades.
    return float(np.clip(np.log10(max(x, 1e-12) / t) / decades, 0.0, 1.0)) if t > 0 else 1.0


def _shape_terms(a, p, orientation=True):
    c = {'strength C': _lin(a['C'], p['c_min'], 2.0 * p['c_min']),
         'inside model': _lin(a['f_side'], p['f_side_max'], 0.0)}
    if a['prolate']:
        c['rod shape K'] = _logm(a['K'], p['k_prolate'], 1.0)
        if orientation:
            c['steepness'] = _lin(a['axis_vz'], p['steep_cos'], 1.0)
    else:
        c['sheet shape K'] = float(np.clip(np.log10(p['k_oblate'] / max(a['K'], 1e-12)), 0.0, 1.0))
        if orientation:
            c['steepness'] = _lin(a['normal_vz'], p['flat_cos'], 0.0)
    return c


def motif_score(G, n, motif):
    """
    Typicality of body n of graph G (from motif_counts(..., return_graph=True))
    as an example of `motif`: how clearly the body satisfies the definition.

    Each criterion j of the motif gives a normalised margin mu_j in [0, 1]: 0 at
    the threshold of the definition, 1 at a value regarded as fully typical,
    linear in between (on a log scale for K and for areas). The score is the
    smallest margin, s = min_j mu_j, so that a body is typical only if it meets
    every criterion clearly (the minimum is the conjunction of fuzzy logic,
    Zadeh, 1965). Fully typical values: C = 2 c_min; K = 10 k_prolate (rods) or
    k_oblate / 10 (sheets); |e1.z| = 1 or |e3.z| = 0; f_side = 0; F_b = 1;
    A_b = layer_aspect / 3; f_bottom = 0; f_up = f_down = 1; 2 more lateral units than
    cross_units; compartments 100 times the bridge volume; window and violating
    contacts of 100 times the minimum area; contacts of a chimney with the deep
    and top units of 100 times amin_link. Criteria without a threshold (contact
    fractions f_up, f_down, host fraction) enter as their value.

    Returns (score, {criterion: margin}); score is nan if n does not realise the motif.

    References
    ----------
    Zadeh, L.A., 1965. Fuzzy sets, Inform. Control, 8(3), 338-353.
    """
    a, p = G.nodes[n], G.graph['thr']
    if motif in ('steep_body', 'cross_cutting', 'dyke'):
        if not a['steep'] or (motif == 'dyke' and not a['oblate']):
            return np.nan, {}
        c = _shape_terms(a, p)
        if motif in ('cross_cutting', 'dyke'):
            c['units cut'] = _lin(a['n_lat_units'], p['cross_units'] - 1, p['cross_units'] + 1)
    elif motif == 'chimney':
        if not a['prolate']:
            return np.nan, {}
        c = _shape_terms(a, p, orientation=False)
        c['contact with deep unit'] = _logm(a['deep_faces'], max(p['amin_link'], 1), 2.0)
        c['contact with top unit'] = _logm(a['top_faces'], max(p['amin_link'], 1), 2.0)
    elif motif == 'layer':
        c = {'extent F': _lin(a['f_footprint'], p['layer_footprint'], 1.0),
             'thinness A': _lin(a['aspect'], p['layer_aspect'], p['layer_aspect'] / 3.0),
             'not half-space': _lin(a['f_bottom'], p['bottom_max'], 0.0),
             'contact above': 1.0 if a['touches_top'] and not a['f_up'] else a['f_up'],
             'contact below': a['f_dn']}
    elif motif == 'vertical_trap':
        c = {'host above': _lin(a['f_up'], p['trap_frac'], 1.0),
             'host below': _lin(a['f_dn'], p['trap_frac'], 1.0),
             'inside model': 1.0 - a['f_side']}
    elif motif == 'enclosed_body':
        c = {'single host': _lin(a['host_frac'], p['enclosed_host'], 1.0)}
    elif motif == 'bridge':
        c = {'compartments': _logm(a['bridge_minratio'], max(p['bridge_ratio'], 1e-12), 2.0)}
    elif motif == 'window':
        c = {'window area': _logm(a['window_faces'], G.graph['amin_window'], 2.0)}
    elif motif == 'order_violation':
        c = {'violating contact': _logm(a['viol_faces'], G.graph['vmin_voxels'], 2.0)}
    else:
        raise KeyError(motif)
    return (min(c.values()) if c else np.nan), c


def topology_key(G, decimals=1, born_from=None, max_bodies=2):
    """
    Canonical key of the unit-level topology of a model: sorted tuple of
    (unit_lo, unit_hi) contact pairs plus bodies per unit.

    born_from : if given, every unit label >= born_from is merged into the
        single class `born_from` ("born unit"), so that models differing only
        by the density or number of their born units share a key.
    max_bodies : bodies per unit are capped at this value (default 2: one
        body vs fragmented). None keeps exact counts, which makes almost
        every model of a large ensemble a class of its own.

    References
    ----------
    Unit-level topology as in Thiele et al. (2016a); used for the topology
    census of Thiele et al. (2016b) and Pakyuz-Charrier et al. (2019).
    """
    def cls(u):
        return born_from if (born_from is not None and u >= born_from) else round(u, decimals)

    edges = set()
    for u, v in G.edges:
        a, b = cls(G.nodes[u]['unit']), cls(G.nodes[v]['unit'])
        if a != b:
            edges.add((min(a, b), max(a, b)))
    bodies = {}
    for _, a in G.nodes(data=True):
        c = cls(a['unit'])
        bodies[c] = bodies.get(c, 0) + 1
    if max_bodies is not None:
        bodies = {u: min(nb, max_bodies) for u, nb in bodies.items()}
    return (tuple(sorted(edges)), tuple(sorted(bodies.items())))


def motif_counts(model, vmin_voxels=1, deep_units=None, top_units=None, order=None,
                 cell_sizes=(1.0, 1.0, 1.0), reference=None, thr=None,
                 max_bodies=2, return_graph=False):
    """
    Motif count vector of one model: dict with MOTIFS keys + n_units, n_bodies,
    n_born (with a reference), z_deep_top (shallowest cell index reached by a
    `deep_units` body; nz if none), the topology key, and score_<motif>, the
    typicality (motif_score) of the body realising each motif most clearly
    (nan if the motif is absent).

    reference : reference codes (reference_codes()) or a reference model of
        the same shape. When given, the model is relabelled with
        canonical_labels(); `deep_units`, `top_units` and `order` then refer
        to codes and default to [K-1], [0] and [0, ..., K-1].

    With return_graph=True, also returns G (with G.graph['deep_units'],
    ['top_units'], ['K']), the body labels and the flags.
    """
    K = None
    if reference is not None:
        ref = reference_codes(reference)
        K = int(ref.max()) + 1
        model = canonical_labels(model, ref)
        deep_units = [K - 1] if deep_units is None else deep_units
        top_units = [0] if top_units is None else top_units
        order = list(range(K)) if order is None else order
    deep_units = () if deep_units is None else deep_units
    top_units = () if top_units is None else top_units

    G, labels = body_graph(model, vmin_voxels, cell_sizes, thr)
    G.graph.update(deep_units=tuple(map(float, deep_units)),
                   top_units=tuple(map(float, top_units)), K=K)
    flags, n_viol = motif_flags(G, deep_units, top_units, order, thr, amin=vmin_voxels)
    out = {m: 0 for m in MOTIFS}
    for f in flags.values():
        for mm in f:
            if mm != 'order_violation':
                out[mm] += 1
    out['order_violation'] = n_viol
    out['fragment'] = G.graph['n_fragments']
    out['n_units'] = len(G.graph['units'])
    out['n_bodies'] = G.number_of_nodes()
    if K is not None:
        out['n_born'] = int((G.graph['units'] >= K).sum())
    deep = set(map(float, deep_units))
    zs = [a['zmin'] for _, a in G.nodes(data=True) if a['unit'] in deep]
    out['z_deep_top'] = min(zs) if zs else G.graph['shape'][0]
    out['topology'] = topology_key(G, born_from=K, max_bodies=max_bodies)
    for mm in MOTIFS[:-1]:                                  # best typicality of each motif (nan if absent)
        sc = [motif_score(G, n, mm)[0] for n, f in flags.items() if mm in f]
        out['score_' + mm] = max(sc) if sc else np.nan
    if return_graph:
        return out, G, labels, flags
    return out


def motif_mask(G, labels, flags, motif):
    """
    Voxels realising `motif` in one model (from motif_counts(..., return_graph=True)).
    Body motifs: the voxels of the tagged bodies. `window`: only the voxels of
    the tagged basement bodies lying directly below a top unit or at the model
    top, i.e. where the cover sits on the basement.
    """
    tagged = [n for n, f in flags.items() if motif in f]
    if not tagged:
        return np.zeros(labels.shape, bool)
    mask = np.isin(labels, tagged)
    if motif == 'window':
        uv = G.graph['unit_of'][labels]
        lid = np.zeros(labels.shape, bool)
        lid[0] = True
        lid[1:] = np.isin(uv[:-1], list(G.graph.get('top_units', ())))
        mask &= lid
    return mask


# ---------------------------------------------------------------------------
# 3. Ensemble tables
# ---------------------------------------------------------------------------
def _iter_models(data_models):
    if isinstance(data_models, np.ndarray) and data_models.ndim == 4:
        return (data_models[..., i] for i in range(data_models.shape[-1]))
    return iter(data_models)


def motif_table(data_models, n_jobs=-1, **kw):
    """
    Motif counts for every model of an ensemble (nz, ny, nx, n_models) or of a
    list/iterable of 3D models. kw are passed to motif_counts().

    Returns a DataFrame indexed by model with MOTIFS columns, n_units,
    n_bodies, n_born, z_deep_top and the topology key.
    """
    if kw.get('reference') is not None:
        kw['reference'] = reference_codes(kw['reference'])     # small int array per task
    rows = Parallel(n_jobs=n_jobs)(delayed(motif_counts)(m, **kw) for m in _iter_models(data_models))
    return pd.DataFrame(rows)


def check_table(df, data_models, n_check=3, **kw):
    """
    Raise if `df` does not correspond to `data_models` and `kw` (e.g. a table
    left over from an earlier selection of models): recomputes n_check
    evenly spaced rows and compares them with df.
    """
    N = data_models.shape[-1]
    if len(df) != N:
        raise ValueError(f"df has {len(df)} rows but the ensemble has {N} models: re-run motif_table().")
    cols = MOTIFS + ['n_bodies']
    for i in np.unique(np.linspace(0, N - 1, n_check).round().astype(int)):
        c = motif_counts(data_models[..., i], **kw)
        if any(c[k] != df.iloc[i][k] for k in cols) or c['topology'] != df.iloc[i]['topology']:
            raise ValueError(f"df row {i} does not match model {i} with the current settings: "
                             "re-run motif_table() after changing the data or kw.")


def motif_probabilities(df, groups=None):
    """
    P(motif present | d) = fraction of models with count > 0, and mean count.
    `groups` (array-like, len n_models, e.g. chain id or hyperparameter set)
    gives one row per group instead of one global row.

    References
    ----------
    Monte Carlo estimate of the posterior probability (Robert & Casella, 2004).
    """
    present = (df[MOTIFS] > 0).astype(float)
    if groups is None:
        return pd.DataFrame({'P_present': present.mean(), 'mean_count': df[MOTIFS].mean()})
    g = pd.Series(np.asarray(groups), index=df.index, name='group')
    return pd.concat({'P_present': present.groupby(g).mean(),
                      'mean_count': df[MOTIFS].groupby(g).mean(),
                      'n_models': present.groupby(g).size()}, axis=1)


def hypothesis_table(indicators, groups=None):
    """
    Posterior probability of geological hypotheses from an ensemble:

        P(H | d) ~= (1/N) * sum_i 1_H(m_i)

    where 1_H(m_i) is 1 if sampled model m_i satisfies hypothesis H and 0
    otherwise, N is the number of retained samples. The standard error is the
    binomial one, sqrt(P (1 - P) / N), which assumes the retained samples are
    approximately independent (thin the chains first).

    Parameters
    ----------
    indicators : dict {hypothesis name: boolean array-like of length N}
    groups : array-like of length N (chain id, hyperparameter set...), or None.

    Returns
    -------
    DataFrame with P, SE and N, per group if given.

    References
    ----------
    Monte Carlo estimate (Robert & Casella, 2004); for autocorrelated chains,
    replace N by the effective sample size (Geyer, 1992). Hypotheses on the
    number of units are trans-D model selection (Sambridge et al., 2006).
    """
    X = pd.DataFrame({k: np.asarray(v, bool) for k, v in indicators.items()})
    if groups is None:
        P = X.mean()
        return pd.DataFrame({'P': P, 'SE': np.sqrt(P * (1 - P) / len(X)), 'N': len(X)})
    g = pd.Series(np.asarray(groups), index=X.index, name='group')
    P = X.groupby(g).mean()
    N = X.groupby(g).size()
    SE = np.sqrt(P * (1 - P)).div(np.sqrt(N), axis=0)
    return pd.concat({'P': P, 'SE': SE}, axis=1).assign(N=N)


def bodies_per_unit(topology_key):
    """{unit: number of bodies (capped as in topology_key())} from a topology key."""
    return dict(topology_key[1])


def motif_cooccurrence(df):
    """
    Joint presence P(i & j) and lift P(i & j) / (P(i) P(j)) between motifs.
    lift > 1: motifs appear together more than independence predicts;
    lift < 1: they tend to exclude each other.

    References
    ----------
    Lift as defined for association rules (Brin et al., 1997).
    """
    X = (df[MOTIFS] > 0).astype(float).values
    n = X.shape[0]
    joint = X.T @ X / n
    p = X.mean(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        lift = joint / np.outer(p, p)
    return (pd.DataFrame(joint, MOTIFS, MOTIFS),
            pd.DataFrame(lift, MOTIFS, MOTIFS))


def cluster_enrichment(df, labels, noise_label=-1):
    """
    Per-cluster motif presence and enrichment = P(motif | cluster) / P(motif).
    `labels` from UMAP/HDBSCAN etc. Enrichment >> 1 names what a cluster is.

    References
    ----------
    Clusters typically from UMAP + HDBSCAN on other features; enrichment
    follows the over-representation logic of Milo et al. (2002).
    """
    present = (df[MOTIFS] > 0).astype(float)
    lab = pd.Series(np.asarray(labels), index=df.index)
    p_all = present.mean()
    p_cl = present[lab != noise_label].groupby(lab[lab != noise_label]).mean()
    enr = p_cl / p_all.replace(0, np.nan)
    p_cl['n_models'] = lab[lab != noise_label].value_counts().sort_index()
    return p_cl, enr


def prior_enrichment(df_post, df_prior):
    """
    Milo-style enrichment: P(motif | posterior) / P(motif | prior samples),
    with a z-score against the prior frequency. Values >> 1 mean the data
    create the configuration; ~1 means it is inherited from the prior or the
    perturbation mechanics.

    References
    ----------
    Null model = prior ensemble (sampler run without the data, w = 0 in the
    Gibbs posterior of Bissiri et al., 2016), in the spirit of Milo et al.
    (2002). ln of posterior odds / prior odds is the (generalised) Bayes
    factor (Kass & Raftery, 1995).
    """
    pp = (df_post[MOTIFS] > 0).mean()
    pq = (df_prior[MOTIFS] > 0).mean()
    n = len(df_post)
    se = np.sqrt(pq * (1 - pq) / n).replace(0, np.nan)
    return pd.DataFrame({'P_posterior': pp, 'P_prior': pq,
                         'enrichment': pp / pq.replace(0, np.nan),
                         'z': (pp - pq) / se})


def topology_census(df, top=10):
    """
    Posterior over unit-level topologies: frequency of each distinct topology
    key, sorted by decreasing frequency, and the first model of each class as
    representative. The index holds the keys (an Index of tuples).

    References
    ----------
    Posterior pmf over topologies (Thiele et al., 2016b; Pakyuz-Charrier et
    al., 2019).
    """
    codes, uniq = pd.factorize(df['topology'].to_numpy())
    n = np.bincount(codes)
    _, first = np.unique(codes, return_index=True)
    order = np.argsort(-n, kind='stable')
    idx = pd.Index([uniq[k] for k in order], tupleize_cols=False, name='topology')
    out = pd.DataFrame({'n_models': n[order], 'P': n[order] / len(df),
                        'representative': df.index.to_numpy()[first[order]]}, index=idx)
    out['n_units'] = [len(k[1]) for k in out.index]
    out['n_contacts'] = [len(k[0]) for k in out.index]
    return out.head(top)


def topological_distance(df, key_ref):
    """
    Jaccard distance between each model's unit-contact edge set and a
    reference key (e.g. the prior model's topology_key).
    0 = same contacts as the reference, 1 = no contact in common.

    References
    ----------
    Jaccard (1912) distance, a metric on sets (Levandowsky & Winter, 1971).
    """
    ref = set(key_ref[0])
    d = []
    for k in df['topology']:
        e = set(k[0])
        u = len(e | ref)
        d.append(1.0 - len(e & ref) / u if u else 0.0)
    return pd.Series(d, index=df.index, name='jaccard_to_ref')


def _where_one(model, motif, kw):
    _, G, labels, flags = motif_counts(model, return_graph=True, **kw)
    return motif_mask(G, labels, flags, motif)


def motif_where_map(data_models, motif, n_jobs=-1, **kw):
    """
    Voxel probability of lying in the region realising `motif` (see
    motif_mask(); for `window`, the basement voxels under the cover)
    (nz, ny, nx). kw as for motif_counts().

    References
    ----------
    Extends unit-probability maps of model ensembles (Wellmann &
    Regenauer-Lieb, 2012) from units to configurations.
    """
    if kw.get('reference') is not None:
        kw['reference'] = reference_codes(kw['reference'])
    acc, n = None, 0
    for msk in Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(_where_one)(m, motif, kw) for m in _iter_models(data_models)):
        acc = msk.astype(np.int32) if acc is None else acc + msk
        n += 1
    return acc / max(n, 1)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _demo_model(shape=(30, 40, 50)):
    """Three layers, an enclosed lens, a steep dyke through all layers, a chimney."""
    nz = shape[0]
    m = np.zeros(shape, float)
    m[:nz // 3] = -50.; m[nz // 3:2 * nz // 3] = 0.; m[2 * nz // 3:] = 100.
    m[12:15, 10:22, 8:26] = 200.                     # flat lens inside the middle layer
    m[:, 5:35, 30:32] = 300.                         # dyke through everything
    m[5:2 * nz // 3 + 2, 33:36, 42:45] = 150.        # pipe from basement up into cover
    return m


if __name__ == '__main__':
    model = _demo_model()
    ref = reference_codes(np.where(model >= 150, 0., model))  # prior: the three layers
    c, G, labels, flags = motif_counts(model, vmin_voxels=5, reference=ref, return_graph=True)
    print({k: v for k, v in c.items() if k != 'topology'})
    print('topology:', c['topology'])
    for n, f in flags.items():
        a = G.nodes[n]
        print(f"body {n:2d} unit {a['unit']:3.0f} vol {a['volume']:6d} K={a['K']:7.2f} C={a['C']:4.2f} "
              f"shaped={a['shaped']!s:5} steep={a['steep']!s:5} f_side={a['f_side']:.2f} "
              f"fp={a['f_footprint']:.2f} asp={a['aspect']:.2f} -> {sorted(f)}")
