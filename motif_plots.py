"""
Figures of the motif workflow: sections of models with bodies outlined, motif and
topology catalogues, census sections, hypothesis and enrichment charts, maps.

Sections are drawn from the analysis-region ensemble `core` (nz, ny, nx, N) by a
Viewer, which matches the units of each model to the reference
(motif_analysis.canonical_labels), so that a given unit has the same colour in
every model. Pixels are square in cells (1:1) by default, or in metres with
physical_aspect=True and the cell sizes.

Section axes: axis 0 = map view at a depth index, 1 = vertical section at a y
index (x horizontal), 2 = vertical section at an x index (y horizontal).

pick_example and plot_motif_catalogue accept models= (indices of the models among
which examples are taken, e.g. the members of a cluster); plot_hypotheses accepts
legend_title=. Both additions leave the default behaviour unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

import motif_analysis as ma

try:
    from IPython.display import display
except ImportError:
    display = print

# Geological reading of each motif, used in figure titles only.
READING = {
    'enclosed_body':   'lens, enclave, intrusion',
    'layer':           'laterally continuous sheet',
    'steep_body':      'dyke, pipe',
    'cross_cutting':   'steep body (dyke or pipe) cutting several units',
    'dyke':            'steep sheet cutting several units',
    'vertical_trap':   'lens inside a layer',
    'bridge':          'separator: splits the model into compartments',
    'chimney':         'pipe linking the basement to the cover',
    'window':          'basement breaching the cover',
    'order_violation': 'geologically inconsistent contact',
}


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def sec(vol, axis, index):
    """Section of a 3D grid along `axis` (0 = map view, 1 = at y, 2 = at x)."""
    return vol[index] if axis == 0 else (vol[:, index, :] if axis == 1 else vol[:, :, index])


def best_index(mask, axis):
    """Index along `axis` of the section holding the most voxels of `mask`."""
    return int(mask.sum(axis=tuple(a for a in range(3) if a != axis)).argmax())


def best_view(mask, axis, sat=0.6):
    """
    (axis, index) of the section along `axis` holding the most voxels of `mask`
    without being filled by it (a section along the strike of a dyke lies inside
    the dyke); otherwise a map view at the best depth.
    """
    for ax_ in (axis, 0):
        n = mask.sum(axis=tuple(a for a in range(3) if a != ax_))
        ok = (n > 0) & (n <= sat * mask.size / mask.shape[ax_])
        if ok.any():
            return ax_, int(np.where(ok, n, -1).argmax())
    return axis, best_index(mask, axis)


def view_name(axis, index):
    return f"{'depth' if axis == 0 else 'y' if axis == 1 else 'x'}={index}" + (" (map view)" if axis == 0 else "")


def pixel_aspect(axis, cell_sizes=(1.0, 1.0, 1.0), physical=False):
    """imshow aspect (height / width of a pixel) of a section: 1 in cells, or the cell-size ratio in metres."""
    if not physical:
        return 1.0
    dz, dy, dx = cell_sizes
    return {0: dy / dx, 1: dz / dx, 2: dz / dy}[axis]


def _outline(ax, mask2d):
    if mask2d.any():
        s = mask2d.astype(float)
        ax.contour(s, levels=[0.5], colors='k', linewidths=2.2)
        ax.contour(s, levels=[0.5], colors='w', linewidths=1.0)


def _labels(ax, axis):
    ax.set_xlabel(('y' if axis == 2 else 'x') + ' (cell)', fontsize=8)
    ax.set_ylabel(('y' if axis == 0 else 'depth') + ' (cell)', fontsize=8)
    ax.tick_params(labelsize=7)


class Viewer:
    """
    Section plots of the models of `core` (nz, ny, nx, N).

    ref : reference codes (motif_analysis.reference_codes): 0 = top unit ... K-1 = basement.
    unit_names : {code: name}; codes 0 and K-1 default to 'cover' and 'basement'.
    show : 'units' (matched units, same colour in every model) or 'density'.
    physical_aspect : False = square pixels in cells; True = proportions in metres (cell_sizes).
    """

    def __init__(self, core, ref, unit_names=None, cell_sizes=(1.0, 1.0, 1.0), show='units',
                 physical_aspect=False, cmap_density='viridis', n_born_col=4):
        self.core, self.ref = core, ref
        self.K = int(ref.max()) + 1
        self.names = {0: 'cover', self.K - 1: 'basement', **(unit_names or {})}
        self.cell_sizes, self.show, self.physical_aspect = cell_sizes, show, physical_aspect
        self.cmap_density, self.n_born_col = cmap_density, n_born_col
        born = ['#d62728', '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2'][:n_born_col]
        self.cmap = ListedColormap(list(plt.cm.Pastel1(np.arange(self.K) % 9)) + born)
        self.norm = BoundaryNorm(np.arange(-0.5, self.K + n_born_col), self.cmap.N)
        self._canon = {}

    # --- units
    def uname(self, u):
        u = int(u)
        return self.names.get(u, f"U{u}") if u < self.K else f"born{u - self.K + 1}"

    def canon(self, i):
        """Matched unit codes of model i (cached)."""
        if i not in self._canon:
            self._canon[i] = ma.canonical_labels(self.core[..., i], self.ref)
        return self._canon[i]

    def density_of(self, i, labels, n):
        """Sampled density of body n of model i."""
        return float(self.core[..., i].reshape(-1)[np.argmax(labels.reshape(-1) == n)])

    def describe_key(self, k, max_bodies=None):
        """Readable topology key: unit classes with their number of bodies, and number of contacts."""
        cap = lambda n: f"{n}+" if max_bodies is not None and n >= max_bodies else f"{n}"
        b = ', '.join(f"{self.uname(u) if u < self.K else 'born'}×{cap(n)}" for u, n in k[1])
        return f"[{b}], {len(k[0])} contacts"

    # --- drawing
    def aspect(self, axis):
        return pixel_aspect(axis, self.cell_sizes, self.physical_aspect)

    def pair_ratios(self):
        """Width ratios of a (section at y, section at x) pair giving panels of equal height."""
        _, ny, nx, _ = self.core.shape
        if self.physical_aspect:
            _, dy, dx = self.cell_sizes
            return [nx * dx, ny * dy]
        return [nx, ny]

    def draw(self, ax, i, axis, index, mask=None):
        """Section of model i along `axis` at `index`, with `mask` (3D) outlined."""
        if self.show == 'units':
            img = np.minimum(sec(self.canon(i), axis, index), self.K + self.n_born_col - 1)
            im = ax.imshow(img, cmap=self.cmap, norm=self.norm, aspect=self.aspect(axis), interpolation='nearest')
        else:
            m = self.core[..., i]
            im = ax.imshow(sec(m, axis, index), cmap=self.cmap_density, vmin=m.min(), vmax=m.max(),
                           aspect=self.aspect(axis), interpolation='nearest')
        if mask is not None:
            _outline(ax, sec(mask, axis, index))
        _labels(ax, axis)
        return im

    def colorbar(self, fig, im, axes):
        if self.show == 'units':
            nb = self.n_born_col
            cb = fig.colorbar(im, ax=axes, fraction=0.03, ticks=np.arange(self.K + nb))
            cb.ax.set_yticklabels([self.uname(u) for u in range(self.K + nb - 1)] + [f"born{nb}+"], fontsize=7)
        else:
            fig.colorbar(im, ax=axes, fraction=0.03, label='density')


# ---------------------------------------------------------------------------
# Prior model check
# ---------------------------------------------------------------------------
def plot_prior_check(prior_model, models, padd, padd_bottom, iy=None, ix=None, iz=None, n_models=3,
                     cell_sizes=(1.0, 1.0, 1.0), physical_aspect=False, cmap='viridis'):
    """
    Prior model and n_models sampled models on the same sections of the full grid,
    same colour scale (1st-99th percentiles of the prior, which ignores isolated
    extreme values). Dashed box: analysis region. Interfaces should coincide up
    to the sampled deformation; a mirror image means a wrong flip or axis order.
    """
    nz, ny, nx = models.shape[:3]
    iy = ny // 2 if iy is None else iy
    ix = nx // 2 if ix is None else ix
    iz = nz // 4 if iz is None else iz
    vmin, vmax = np.percentile(prior_model, [1, 99])
    idx = np.unique(np.linspace(0, models.shape[3] - 1, n_models).round().astype(int))
    rows = [('prior', prior_model)] + [(f"model #{i}", models[..., i]) for i in idx]
    views = [(1, iy, (padd, nx - padd), (0, nz - padd_bottom)),
             (2, ix, (padd, ny - padd), (0, nz - padd_bottom)),
             (0, iz, (padd, nx - padd), (padd, ny - padd))]
    fig, axes = plt.subplots(len(rows), 3, figsize=(14, 2.6 * len(rows)), squeeze=False, constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[nx, ny, nx]))
    for (name, m), ax in zip(rows, axes):
        for a, (axis, index, (x0, x1), (y0, y1)) in zip(ax, views):
            im = a.imshow(sec(m, axis, index), cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect=pixel_aspect(axis, cell_sizes, physical_aspect), interpolation='nearest')
            a.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec='w', ls='--', lw=1.0))
            a.set_title(f"{name}: section {view_name(axis, index)}", fontsize=8)
            _labels(a, axis)
    fig.colorbar(im, ax=axes, fraction=0.02, label='density')
    plt.show()


# ---------------------------------------------------------------------------
# One model
# ---------------------------------------------------------------------------
def bodies_table(vw, i, G, labels, flags):
    """One row per body of model i: unit, density, volume, shape and footprint attributes, motifs."""
    return pd.DataFrame([
        dict(body=n, unit=vw.uname(a['unit']), density=vw.density_of(i, labels, n), volume=a['volume'],
             degree=G.degree(n), K=round(a['K'], 2), C=round(a['C'], 2), shaped=a['shaped'], steep=a['steep'],
             f_side=round(a['f_side'], 2), F=round(a['f_footprint'], 2), A=round(a['aspect'], 2),
             top=a['touches_top'], motifs=', '.join(sorted(flags[n])))
        for n, a in G.nodes(data=True)]).sort_values('volume', ascending=False)


def plot_flagged_bodies(vw, i, G, labels, flags, skip=('layer',), row_h=2.2, width=8):
    """
    Each body of model i flagged with a motif other than those in `skip`, one row
    per body: section along x (at the y index holding most of the body) and along
    y (at the x index holding most of it).
    """
    flagged = [n for n in G.nodes if flags[n] - set(skip)]
    if not flagged:
        print("no flagged body"); return
    fig, axes = plt.subplots(len(flagged), 2, figsize=(width, row_h * len(flagged)), squeeze=False,
                             gridspec_kw=dict(width_ratios=vw.pair_ratios()), constrained_layout=True)
    for (axy, axx), n in zip(axes, flagged):
        mask = labels == n
        vy, vx = best_view(mask, 1), best_view(mask, 2)
        im = vw.draw(axy, i, *vy, mask)
        vw.draw(axx, i, *vx, mask)
        axy.set_title(f"body {n} ({vw.uname(G.nodes[n]['unit'])}): {', '.join(sorted(flags[n]))} — "
                      f"section {view_name(*vy)}", fontsize=8)
        axx.set_title(f"section {view_name(*vx)}", fontsize=8)
    vw.colorbar(fig, im, axes)
    plt.show()


def plot_all_bodies(vw, i, G, labels, flags, nodes=None, ncol=4):
    """Every body of model i outlined on one common section at mid y, then at mid x."""
    nz, ny, nx, _ = vw.core.shape
    nodes = sorted(G.nodes, key=lambda n: -G.nodes[n]['volume']) if nodes is None else list(nodes)
    for axis, index in [(1, ny // 2), (2, nx // 2)]:
        nc = min(len(nodes), ncol)
        nr = int(np.ceil(len(nodes) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(4.5 * nc, 2.4 * nr), squeeze=False,
                                 sharex=True, sharey=True, constrained_layout=True)
        for ax, n in zip(axes.flat, nodes):
            mask = labels == n
            im = vw.draw(ax, i, axis, index, mask)
            m = ', '.join(sorted(flags[n])) or 'no motif'
            title = f"b{n} ({vw.uname(G.nodes[n]['unit'])}, {G.nodes[n]['volume']} vox): {m}"
            ax.set_title(title + ("" if sec(mask, axis, index).any() else "\n(not in this section)"), fontsize=7)
        for ax in axes.flat[len(nodes):]:
            ax.axis('off')
        vw.colorbar(fig, im, axes)
        fig.suptitle(f"model #{i}, section {view_name(axis, index)}: all {G.number_of_nodes()} bodies "
                     f"({G.graph['n_fragments']} fragments < V_min not shown)", fontsize=10)
        plt.show()


def pick_example(core, kw, motif, n_sample=200, tol=0.05, n_check=20, models=None):
    """
    Model of `core` realising `motif` most clearly: among n_sample models spread
    over the ensemble (or over `models`, indices of a subset such as the members
    of a cluster), those within `tol` of the best typicality score
    (motif_analysis.motif_score; at most n_check of them, spread over the
    ensemble) are compared, and the one whose most typical body is largest is
    returned, so that ties (frequent at score 1) do not fall on the first model.
    Returns (model index in core, score), or (first model of the pool, nan) if the motif never occurs.
    """
    N = core.shape[-1]
    pool = np.arange(N) if models is None else np.asarray(models, dtype=int)
    smp = pool[np.unique(np.linspace(0, len(pool) - 1, min(len(pool), n_sample)).round().astype(int))]
    s = ma.motif_table(core[..., smp], n_jobs=-1, **kw)[f'score_{motif}'].to_numpy()
    if np.isnan(s).all():
        return int(pool[0]), np.nan
    near = smp[s >= np.nanmax(s) - tol]
    near = near[np.unique(np.linspace(0, len(near) - 1, min(n_check, len(near))).round().astype(int))]
    best = None
    for i in near:
        _, G, _, fl = ma.motif_counts(core[..., i], return_graph=True, **kw)
        for n in G.nodes:
            if motif in fl[n]:
                key = (ma.motif_score(G, n, motif)[0], G.nodes[n]['volume'])
                if best is None or key > best[0]:
                    best = (key, int(i))
    return best[1], best[0][0]


# ---------------------------------------------------------------------------
# Motif catalogue
# ---------------------------------------------------------------------------
def plot_motif_catalogue(vw, df, kw, motifs=None, n_ex=4, select='best', tol=0.05, n_cand=40, reading=None,
                         row_h=2.2, width=10, models=None, label=None):
    """
    For each motif, n_ex example models, with the body realising the motif most
    clearly (highest motif_analysis.motif_score) outlined on the sections holding
    most of it; for `window`, the basement cells directly below the cover.

    select = 'best': models with the highest typicality score (column
        score_<motif> of df), among which n_ex are spread over the ensemble
        when more than n_ex lie within `tol` of the best score (near-duplicate
        consecutive samples are thus avoided);
      'largest' / 'spread': n_cand candidate models evenly spread over those
        carrying the motif, examples chosen by body volume (largest, or spread
        over the range of volumes).
    kw: motif settings, as used to compute df. The panel titles give the score
    and its weakest criterion.
    models: indices (rows of df, models of vw.core) among which the examples are taken, e.g. the members of
        a cluster; P in the titles is then the fraction of these models carrying the motif. label: name of
        that subset in the titles.
    """
    reading = READING if reading is None else reading
    motifs = [m for m in ma.MOTIFS if m != 'fragment'] if motifs is None else motifs
    allowed = np.ones(len(df), bool)
    if models is not None:
        allowed[:] = False
        allowed[np.asarray(models, dtype=int)] = True
    graphs = {}

    def graph_of(i):
        if i not in graphs:
            _, Gi, labi, fli = ma.motif_counts(vw.core[..., i], return_graph=True, **kw)
            graphs[i] = (Gi, labi, fli)
        return graphs[i]

    def best_body(i, motif):
        # Body of model i realising the motif most clearly (ties: largest), with its score and criteria.
        Gi, _, fli = graph_of(i)
        sc = [(ma.motif_score(Gi, n, motif), Gi.nodes[n]['volume'], n) for n in Gi.nodes if motif in fli[n]]
        (s, comp), vol, n = max(sc, key=lambda t: (np.nan_to_num(t[0][0], nan=-1.0), t[1]))
        return n, vol, s, comp

    where = f" in {label}" if label else (" in the selected models" if models is not None else "")
    for motif in motifs:
        has = df[motif].to_numpy() > 0
        p = has[allowed].mean()
        cand = np.flatnonzero(has & allowed)
        if not len(cand):
            print(f"{motif}: P = {p:.3f}{where}, never occurs"); continue

        if select == 'best' and f'score_{motif}' in df:
            s_all = df[f'score_{motif}'].to_numpy()[cand]
            order = cand[np.argsort(-np.nan_to_num(s_all, nan=-1.0), kind='stable')]
            near = cand[np.nan_to_num(s_all, nan=-1.0) >= np.nanmax(s_all) - tol]
            pool = np.sort(near) if len(near) >= n_ex else np.sort(order[:n_ex])
            chosen = pool[np.unique(np.linspace(0, len(pool) - 1, min(n_ex, len(pool))).round().astype(int))]
            ex = [(int(i),) + best_body(int(i), motif) for i in chosen]
            ex.sort(key=lambda t: -np.nan_to_num(t[3], nan=-1.0))
            info = f"most typical, {len(near)} models within {tol:g} of the best score"
        else:
            cand = cand[np.unique(np.linspace(0, len(cand) - 1, min(n_cand, len(cand))).astype(int))]
            found = sorted(((int(i),) + best_body(int(i), motif) for i in cand), key=lambda t: -t[2])
            ex = found[:n_ex] if select == 'largest' or len(found) <= n_ex else \
                [found[k] for k in np.unique(np.linspace(0, len(found) - 1, n_ex).round().astype(int))]
            info = f"{len(ex)} of {len(found)} screened examples ({select})"

        fig, axes = plt.subplots(len(ex), 2, figsize=(width, row_h * len(ex)), squeeze=False,
                                 gridspec_kw=dict(width_ratios=vw.pair_ratios()), constrained_layout=True)
        for (i, n, vol, s, comp), (axy, axx) in zip(ex, axes):
            Gi, labi, fli = graph_of(i)
            mask = ma.motif_mask(Gi, labi, {n: fli[n]}, motif)
            vy, vx = best_view(mask, 1), best_view(mask, 2)
            im = vw.draw(axy, i, *vy, mask)
            vw.draw(axx, i, *vx, mask)
            weak = min(comp, key=comp.get) if comp else ''
            axy.set_title(f"model #{i}, body {n}: {vw.uname(Gi.nodes[n]['unit'])} ({vw.density_of(i, labi, n):.4g}), "
                          f"{vol} vox\nscore {s:.2f} (weakest: {weak} {comp.get(weak, np.nan):.2f}) — "
                          f"section {view_name(*vy)}", fontsize=8)
            axx.set_title(f"section {view_name(*vx)}", fontsize=8)
            vw.colorbar(fig, im, [axy, axx])
        fig.suptitle(f"{motif}: P = {p:.3f}{where} ({reading.get(motif, '')}) — {info}", fontsize=11)
        plt.show()


# ---------------------------------------------------------------------------
# Topology census
# ---------------------------------------------------------------------------
def census_table(vw, census, max_bodies=None):
    """Census with readable keys as index."""
    out = census.copy()
    out.index = [vw.describe_key(k, max_bodies) for k in out.index]
    return out[['n_models', 'P', 'representative', 'n_units', 'n_contacts']]


def census_entropy(df):
    """Number of distinct topologies, Shannon entropy of the census (nats) and its maximum ln(n)."""
    p = df['topology'].value_counts(normalize=True).values
    return len(p), float(-(p * np.log(p)).sum()), float(np.log(len(p)))


def plot_census_sections(vw, census, n=4, max_bodies=None):
    """Representative models of the n most probable topologies, through the cells holding most born units."""
    nz, ny, nx, _ = vw.core.shape
    reps = [(k, row, int(row['representative'])) for k, row in census.head(n).iterrows()]
    fig, axes = plt.subplots(2, len(reps), figsize=(4.5 * len(reps), 5.5), squeeze=False, constrained_layout=True)
    for j, (k, row, i) in enumerate(reps):
        born = vw.canon(i) >= vw.K
        iy = best_index(born, 1) if born.any() else ny // 2
        ix = best_index(born, 2) if born.any() else nx // 2
        im = vw.draw(axes[0, j], i, 1, iy)
        axes[0, j].set_title(f"P = {row['P']:.3f}   (model #{i})\n{vw.describe_key(k, max_bodies)}\nsection y={iy}",
                             fontsize=8)
        vw.draw(axes[1, j], i, 2, ix)
        axes[1, j].set_title(f"section x={ix}", fontsize=8)
    vw.colorbar(fig, im, axes)
    plt.show()


def plot_topology_catalogue(vw, census, n_top=6, max_bodies=None):
    """
    Unit contact graph of the n_top most probable topology classes: node = unit
    class (born units grouped) at its mean (x, depth) (top row) or (y, depth)
    (bottom row) in the representative model, ×n = number of bodies, edge = contact.
    """
    top = census.head(n_top)
    K = vw.K
    fig, axes = plt.subplots(2, len(top), figsize=(3.6 * len(top), 7.4), squeeze=False, constrained_layout=True)
    for j, (k, row) in enumerate(top.iterrows()):
        i = int(row['representative'])
        lab = vw.canon(i)
        Gu = networkx.Graph()
        for u, nb in k[1]:
            z, y, x = np.nonzero(lab >= K if u >= K else lab == int(u))
            if len(z):
                Gu.add_node(u, nb=nb, pos_x=(x.mean(), -z.mean()), pos_y=(y.mean(), -z.mean()))
        Gu.add_edges_from((u, v) for u, v in k[0] if u in Gu and v in Gu)
        colours = [vw.cmap.colors[min(int(u), K)] for u in Gu]
        cap = lambda nb: f"{nb}+" if max_bodies is not None and nb >= max_bodies else f"{nb}"
        for ax, key, xl in [(axes[0, j], 'pos_x', 'mean x (cell)'), (axes[1, j], 'pos_y', 'mean y (cell)')]:
            pos = networkx.get_node_attributes(Gu, key)
            networkx.draw_networkx_edges(Gu, pos, ax=ax, edge_color='0.5')
            networkx.draw_networkx_nodes(Gu, pos, ax=ax, node_color=colours, edgecolors='k',
                                         node_size=[300 + 60 * Gu.nodes[u]['nb'] for u in Gu])
            networkx.draw_networkx_labels(Gu, pos, ax=ax, font_size=7,
                                          labels={u: f"{vw.uname(u) if u < K else 'born'}\n×{cap(Gu.nodes[u]['nb'])}"
                                                  for u in Gu})
            ax.set_xlabel(xl, fontsize=8); ax.set_ylabel('− mean depth (cell)', fontsize=8)
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=7)
            ax.margins(0.25)
        axes[0, j].set_title(f"P = {row['P']:.3f} ({int(row['n_models'])} models), model #{i}\n"
                             f"{len(Gu)} unit classes, {Gu.number_of_edges()} contacts", fontsize=8)
    fig.suptitle('Topology catalogue: unit contact graphs (node = unit class, ×n = bodies, edge = contact)', fontsize=10)
    plt.show()


# ---------------------------------------------------------------------------
# Probabilities, co-occurrence, distance, maps, enrichment
# ---------------------------------------------------------------------------
def plot_hypotheses(tab, title='Posterior probability of geological hypotheses, per chain', legend_title='chain'):
    """Grouped bars of P(H | d) per group (chain) with binomial standard errors, from hypothesis_table(groups=...)."""
    P, SE = tab['P'], tab['SE']
    groups = list(P.index)
    fig, ax = plt.subplots(figsize=(12, 4))
    w = 0.8 / len(groups); x = np.arange(P.shape[1])
    for j, g in enumerate(groups):
        ax.bar(x + j * w, P.loc[g], w, yerr=SE.loc[g], capsize=2, label=g)
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(list(P.columns), rotation=25, ha='right', fontsize=8)
    ax.set_ylabel(r'$P(H \mid d)$'); ax.set_ylim(0, 1); ax.legend(fontsize=8, title=legend_title)
    ax.set_title(title); plt.tight_layout(); plt.show()


def plot_lift(df, min_p=0.02):
    """Lift matrix of the motifs present in more than min_p of the models."""
    _, lift = ma.motif_cooccurrence(df)
    present = [m for m in ma.MOTIFS if (df[m] > 0).mean() > min_p and m != 'fragment']
    L = lift.loc[present, present]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(L.values, cmap='RdBu_r', vmin=0, vmax=2)
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(present))); ax.set_yticklabels(present, fontsize=8)
    for i in range(len(present)):
        for j in range(len(present)):
            ax.text(j, i, f"{L.values[i, j]:.2f}", ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax, label='lift'); ax.set_title('Motif co-occurrence (lift)'); plt.tight_layout(); plt.show()


def plot_distance(dist, groups):
    """Histograms of the Jaccard distance to the prior topology, per group (chain)."""
    groups = np.asarray(groups)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for g in np.unique(groups):
        ax.hist(dist[groups == g], bins=np.linspace(0, 1, 26), alpha=0.5, label=g)
    ax.set_xlabel('Jaccard distance of unit contacts to the prior model'); ax.set_ylabel('models')
    ax.legend(fontsize=8); plt.tight_layout(); plt.show()


def plot_where(W, motif, zs=None, cell_sizes=(1.0, 1.0, 1.0), physical_aspect=False):
    """Map view of the maximum of the voxel probability W over the top zs cells, and section through its maximum."""
    nz = W.shape[0]
    zs = max(nz // 2, 1) if zs is None else zs
    iz, iy, ix = np.unravel_index(W.argmax(), W.shape)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    im = axes[0].imshow(W[:zs].max(axis=0), cmap='magma', vmin=0, vmax=W.max(),
                        aspect=pixel_aspect(0, cell_sizes, physical_aspect))
    axes[0].set_title(f"'{motif}': max over the top {zs} cells (map view)", fontsize=9); _labels(axes[0], 0)
    plt.colorbar(im, ax=axes[0], label='P')
    im = axes[1].imshow(W[:, iy, :], cmap='magma', vmin=0, vmax=W.max(),
                        aspect=pixel_aspect(1, cell_sizes, physical_aspect))
    axes[1].set_title(f"section y={iy} through the maximum (P={W.max():.2f})", fontsize=9); _labels(axes[1], 1)
    plt.colorbar(im, ax=axes[1], label='P')
    plt.show()


def plot_enrichment(enrich, z_thr=3.0):
    """Posterior/prior enrichment per motif, red where |z| > z_thr (from motif_analysis.prior_enrichment)."""
    e = enrich.dropna().drop(index='fragment', errors='ignore')
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(range(len(e)), e['enrichment'], color=np.where(e['z'].abs() > z_thr, '#B2182B', '#AAAAAA'))
    ax.axhline(1, color='k', lw=0.8)
    ax.set_xticks(range(len(e))); ax.set_xticklabels(e.index, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('posterior / prior'); ax.set_title(f'Motif enrichment by the data (red: |z| > {z_thr:g})')
    plt.tight_layout(); plt.show()


def plot_link_contacts(lc, sens=None, groups='chain', amin_link=None):
    """
    Contacts of the chimney candidates (motif_diagnostics.link_contacts): (a)
    distribution of the smaller of the two contacts of each rod with the deep
    and top units (faces, logarithmic bins), per group; (b) if `sens` is given
    (rows: amin_link values, columns: groups), P(chimney) against amin_link.
    A plateau in (b) means that the chimneys do not rest on grazing contacts.
    """
    fig, axes = plt.subplots(1, 2 if sens is not None else 1, figsize=(11 if sens is not None else 5.5, 3.6),
                             squeeze=False, constrained_layout=True)
    ax = axes[0, 0]
    if len(lc):
        top = max(int(lc['min_faces'].max()), 2)
        bins = np.unique(np.round(np.geomspace(1, top + 1, 20))).astype(float) - 0.5
        keys = list(dict.fromkeys(lc[groups])) if groups in lc else [None]
        for g in keys:
            v = lc['min_faces'] if g is None else lc.loc[lc[groups] == g, 'min_faces']
            ax.hist(v, bins=bins, histtype='step', lw=1.8, label=g if g is not None else None)
        if amin_link is not None:
            ax.axvline(amin_link - 0.5, color='k', ls='--', lw=1)
        ax.set_xscale('log')
        if keys != [None]:
            ax.legend(fontsize=8, title=groups)
    ax.set_xlabel('smaller contact with the deep or top unit (faces)')
    ax.set_ylabel('rods')
    ax.set_title('a) How firmly the rods link the deep and top units', fontsize=9, loc='left')
    if sens is not None:
        ax = axes[0, 1]
        for g in sens.columns:
            ax.plot(sens.index, sens[g], 'o-', label=g)
        ax.set_xscale('log', base=2)
        ax.set_xticks(list(sens.index)); ax.set_xticklabels([str(v) for v in sens.index])
        ax.set_ylim(0, max(1e-3, float(np.nanmax(sens.values))) * 1.15)
        ax.set_xlabel('amin_link (minimum faces with each unit)'); ax.set_ylabel('P(chimney)')
        ax.set_title('b) P(chimney) against the minimum contact', fontsize=9, loc='left')
        ax.legend(fontsize=8)
    plt.show()
