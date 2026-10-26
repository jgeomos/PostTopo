"""
Synthetic stand-in for a trans-D posterior ensemble (undercover setting:
cover / sediments / basement, plus born units), used to run the motif
workflow end to end when the real `all_models` memmap is not at hand.

Shapes follow the SW WA notebook: models_4d is (nz, ny, nx, n_models),
axis 0 = depth with index 0 at the top, `padd` padding cells on the sides.

Three "chains" mimic three hyperparameter families: different birth rates
and interface roughness. As in the trans-D sampler, unit densities are
sampled: each model draws its own density for every unit around the nominal
value (standard deviation `density_sd`), so that unit identity across models
cannot be read from density values alone (this exercises
motif_analysis.canonical_labels).

The born bodies have random geometry, drawn per model from the ranges in
GEOMETRY: lenses of random size, azimuth and tilt; dykes of random dip, strike,
thickness and length, some of them blind (stopping below the cover); pipes of
random plunge, trend and radius; elongated basement highs of random azimuth.
The ranges straddle the default motif thresholds (steep dykes dip more than
~73 deg, steep pipes plunge more than ~44 deg), so that the ensemble contains
bodies on both sides of each threshold.

Ground truth
------------
With return_truth=True, the drawn geometry of every model is returned (one row
per model): presence of the basement high and of each born body, lens tilt,
dyke dip and strike, blind dykes, pipe plunge and trend, and the isolated cells
added as sampler noise, as a list of (iz, iy, ix, code before, code after).
With return_labels=True, the exact unit of every cell before the isolated cells
are added is returned as int8 codes (CODES: 0 cover, 1 sediments, 2 basement,
3 lens, 4 dyke, 5 pipe), with the shape of the models. Truth and labels are
what synthetic_validation.py compares the motif analysis with. Neither option
changes the models drawn for a given seed.

Everything here is made up; it only has to exercise the same code paths as
the real ensemble. Distances are in cells.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

# Nominal unit densities (kg/m3). Born units get values from BORN_DENSITIES.
COVER, SED, BASEMENT = 2300., 2550., 2800.
BORN_DENSITIES = {'lens': 2950., 'dyke': 3000., 'pipe': 2900.}
CODES = ('cover', 'sediments', 'basement', 'lens', 'dyke', 'pipe')   # unit codes of the labels

CHAINS = {
    # name: (p_lens, p_dyke, p_pipe, p_high, roughness, n_fragments_max)
    'A_baseline':  (0.60, 0.10, 0.30, 0.50, 2.0, 3),
    'B_birthy':    (0.80, 0.40, 0.50, 0.50, 2.0, 6),
    'C_smooth':    (0.30, 0.05, 0.15, 0.20, 1.0, 1),
}
PRIOR = (0.30, 0.30, 0.30, 0.30, 2.0, 3)      # "no data" perturbation rates

# Ranges (min, max) of the uniform distributions of the born-body geometry.
# Angles in degrees; azimuths measured in map view from +y towards +x;
# dips and plunges measured from the horizontal. Set min = max to fix a value
# (e.g. dyke_dip=(90, 90) for vertical dykes).
GEOMETRY = dict(
    lens_axes=((1.5, 3.0), (4.0, 8.0), (5.0, 10.0)),  # semi-axes: across, along strike, down dip
    lens_azimuth=(0.0, 180.0),     # strike of the lens
    lens_tilt=(0.0, 25.0),         # dip of the lens plane
    dyke_dip=(45.0, 90.0),         # steep_body needs > ~73 deg with flat_cos = 0.3
    dyke_strike=(0.0, 360.0),      # strike; the dip direction is strike + 90
    dyke_thickness=(1.8, 3.5),     # >= sqrt(3) keeps the voxelised sheet face-connected
    dyke_length=(0.6, 1.0),        # along-strike length, as a fraction of the model interior
    p_blind=0.3,                   # probability that the dyke top lies below the cover
    pipe_plunge=(30.0, 90.0),      # steep_body needs > ~44 deg with steep_cos = 0.7
    pipe_trend=(0.0, 360.0),       # azimuth towards which the pipe plunges
    pipe_radius=(1.0, 2.0),        # >= sqrt(3)/2 keeps the voxelised rod face-connected
    high_height=(6.0, 18.0),       # basement high amplitude
    high_radius=(6.0, 12.0),       # basement high radius along its short axis
    high_elongation=(1.0, 2.0),    # long / short radius of the basement high
)


def _smooth_field(rng, ny, nx, amp, sigma=6.0):
    f = gaussian_filter(rng.standard_normal((ny, nx)), sigma)
    return amp * f / (f.std() + 1e-9)


def _unit(azimuth, plunge):
    """Unit vector (z, y, x), z down, plunging `plunge` deg below the horizontal towards `azimuth` deg."""
    a, p = np.deg2rad(azimuth), np.deg2rad(plunge)
    return np.array([np.sin(p), np.cos(p) * np.cos(a), np.cos(p) * np.sin(a)])


def _plane_axes(strike, dip):
    """Strike vector s, down-dip vector d and normal n of a plane (orthonormal, z down)."""
    s, d = _unit(strike, 0.0), _unit(strike + 90.0, dip)
    return s, d, np.cross(s, d)


def prior_model(shape=(36, 54, 64), z_cover=6, z_base=16):
    nz, ny, nx = shape
    Z = np.arange(nz)[:, None, None]
    m = np.where(Z < z_cover, COVER, np.where(Z < z_base, SED, BASEMENT)).astype(np.float32)
    return np.ascontiguousarray(np.broadcast_to(m, shape))


def sample_model(rng, params, shape=(36, 54, 64), padd=6, z_cover=6, z_base=16, density_sd=0.0,
                 geometry=None, truth=None, return_labels=False):
    """
    One synthetic model.

    geometry : dict overriding entries of GEOMETRY.
    truth : dict or None. If a dict, it is filled with the drawn geometry
        (nan for absent bodies) and the isolated cells.
    return_labels : if True, also returns the unit code (CODES) of every cell
        before the isolated cells are added, as int8.
    """
    g = dict(GEOMETRY, **(geometry or {}))
    U = lambda key: rng.uniform(*g[key])
    p_lens, p_dyke, p_pipe, p_high, rough, nfrag = params
    nz, ny, nx = shape
    rho = lambda nominal: np.float32(nominal + density_sd * rng.standard_normal())   # sampled density
    cover, sed, base = rho(COVER), rho(SED), rho(BASEMENT)
    code_of = {cover: 0, sed: 1, base: 2}
    info = dict(high=False, lens=False, lens_tilt=np.nan, dyke=False, dyke_dip=np.nan, dyke_strike=np.nan,
                dyke_blind=False, pipe=False, pipe_plunge=np.nan, pipe_trend=np.nan)

    # Interfaces.
    zc = z_cover + _smooth_field(rng, ny, nx, 0.5 * rough)
    zb = z_base + _smooth_field(rng, ny, nx, rough)
    yy, xx = np.mgrid[:ny, :nx]
    if rng.random() < p_high:
        # basement high, elongated, located where the (pretend) data want it, with posterior scatter
        y0 = int(np.clip(rng.normal(0.45 * ny, 3), padd + 8, ny - padd - 8))
        x0 = int(np.clip(rng.normal(0.35 * nx, 3), padd + 8, nx - padd - 8))
        r, e, a = U('high_radius'), U('high_elongation'), np.deg2rad(rng.uniform(0, 180))
        u = (yy - y0) * np.cos(a) + (xx - x0) * np.sin(a)          # along the long axis
        v = -(yy - y0) * np.sin(a) + (xx - x0) * np.cos(a)
        zb = zb - U('high_height') * np.exp(-((u / e) ** 2 + v ** 2) / (2 * r ** 2))
        info['high'] = True
    zb = np.maximum(zb, 0)
    zc = np.clip(zc, 1, None)
    Z = np.arange(nz)[:, None, None]
    lab = np.where(Z < np.round(zc), 0, np.where(Z < np.round(zb), 1, 2)).astype(np.int8)
    m = np.array([cover, sed, base], np.float32)[lab]

    # Born units: coordinates relative to a centre, open grids broadcast to 3D.
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    interior = np.zeros((1, ny, nx), bool)
    interior[:, padd:ny - padd, padd:nx - padd] = True
    dot = lambda v, c: v[0] * (zz - c[0]) + v[1] * (yy - c[1]) + v[2] * (xx - c[2])
    iy, ix = ny - 2 * padd, nx - 2 * padd                         # interior size

    if rng.random() < p_lens:                       # lens in the sediments, random size, azimuth and tilt
        c = (rng.integers(z_cover + 2, z_base - 2),
             rng.integers(padd + 6, ny - padd - 6), rng.integers(padd + 6, nx - padd - 6))
        tilt = U('lens_tilt')
        s, d, n = _plane_axes(U('lens_azimuth'), tilt)
        a, b, cc = (rng.uniform(*r) for r in g['lens_axes'])
        inside = (dot(n, c) / a) ** 2 + (dot(s, c) / b) ** 2 + (dot(d, c) / cc) ** 2 <= 1
        rho_lens = rho(BORN_DENSITIES['lens'])
        sel = inside & (lab == 1)
        m[sel], lab[sel] = rho_lens, 3
        code_of[rho_lens] = 3
        info.update(lens=True, lens_tilt=tilt)

    if rng.random() < p_dyke:                       # sheet of random dip, strike, thickness, length
        blind = rng.random() < g['p_blind']
        ztop = int(rng.integers(z_cover + 1, z_base)) if blind else 0
        c = ((ztop + nz) / 2,
             padd + rng.uniform(0.3, 0.7) * iy, padd + rng.uniform(0.3, 0.7) * ix)
        dip = U('dyke_dip')
        strike = U('dyke_strike')
        s, d, n = _plane_axes(strike, dip)
        half_len = 0.5 * U('dyke_length') * np.hypot(iy * s[1], ix * s[2])   # interior length along strike
        inside = ((np.abs(dot(n, c)) <= U('dyke_thickness') / 2) & (np.abs(dot(s, c)) <= half_len)
                  & (zz >= ztop) & interior)
        rho_dyke = rho(BORN_DENSITIES['dyke'])
        m[inside], lab[inside] = rho_dyke, 4
        code_of[rho_dyke] = 4
        info.update(dyke=True, dyke_dip=dip, dyke_strike=strike % 360.0, dyke_blind=blind)

    if rng.random() < p_pipe:                       # rod from the basement up into the cover, random plunge
        ztop, zbot = int(rng.integers(1, z_cover)), z_base + 4
        c = ((ztop + zbot) / 2,
             np.clip(rng.normal(0.6 * ny, 2.5), padd + 4, ny - padd - 4),
             np.clip(rng.normal(0.7 * nx, 2.5), padd + 4, nx - padd - 4))
        plunge = U('pipe_plunge')
        trend = U('pipe_trend')
        u = _unit(trend, plunge)
        t = dot(u, c)
        r2 = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 - t ** 2   # squared distance to the axis
        inside = (r2 <= U('pipe_radius') ** 2) & (zz >= ztop) & (zz < zbot) & interior
        rho_pipe = rho(BORN_DENSITIES['pipe'])
        m[inside], lab[inside] = rho_pipe, 5
        code_of[rho_pipe] = 5
        info.update(pipe=True, pipe_plunge=plunge, pipe_trend=trend % 360.0)

    # Sampler noise: isolated cells carrying the value of one of the model's units.
    vals = np.unique(m)
    isolated = []
    for _ in range(rng.integers(0, nfrag + 1)):
        z, y, x = rng.integers(0, nz), rng.integers(0, ny), rng.integers(0, nx)
        v = rng.choice(vals)
        isolated.append((int(z), int(y), int(x), int(code_of[m[z, y, x]]), int(code_of[v])))
        m[z, y, x] = v
    info['isolated'] = isolated
    if truth is not None:
        truth.update(info)
    return (m, lab) if return_labels else m


def _allocate(shape, padd, n, crop_padding, return_labels):
    """Output arrays filled model by model (no list of models, no np.stack: the peak memory is the output itself)."""
    nz, ny, nx = shape
    p = padd if crop_padding else 0
    out_shape = (nz, ny - 2 * p, nx - 2 * p, n)
    models = np.empty(out_shape, dtype=np.float32)
    labs = np.empty(out_shape, dtype=np.int8) if return_labels else None
    sl = (slice(None), slice(p, ny - p), slice(p, nx - p))
    return models, labs, sl


def make_ensemble(n_per_chain=400, shape=(36, 54, 64), padd=6, seed=0, chains=CHAINS, density_sd=10.0,
                  geometry=None, return_truth=False, return_labels=False, crop_padding=False):
    """
    Returns
    -------
    models_4d : (nz, ny, nx, N) float32; with crop_padding=True, the lateral padding (padd cells on each
                side) is dropped as each model is generated: (nz, ny - 2 padd, nx - 2 padd, N), the `core`
                of the notebooks, at 60 % of the memory for the default shape
    chain_id  : (N,) str
    misfit    : (N,) float, a fake data misfit (lower = better)
    truth     : pandas DataFrame of the drawn geometry, one row per model (only if return_truth)
    labels    : (nz, ny, nx, N) int8 unit codes (CODES) before the isolated cells (only if return_labels)

    Memory: N x nz x ny x nx x 4 bytes, e.g. 14.9 GB for 30 000 models of 36 x 54 x 64 cells (9.4 GB cropped).
    The models are drawn in the same order from the same random stream whatever crop_padding, so the
    ensemble is identical to the uncropped one restricted to the core.
    """
    rng = np.random.default_rng(seed)
    n = n_per_chain * len(chains)
    models, labs, sl = _allocate(shape, padd, n, crop_padding, return_labels)
    chain_id, misfit, rows = [], [], []
    k = 0
    for name, params in chains.items():
        for _ in range(n_per_chain):
            info = {}
            m, lab = sample_model(rng, params, shape, padd, density_sd=density_sd, geometry=geometry,
                                  truth=info, return_labels=True)
            models[..., k] = m[sl]
            if return_labels:
                labs[..., k] = lab[sl]
            chain_id.append(name); rows.append(info)
            # fake misfit: dykes and pipes "fit" slightly worse; purely illustrative
            misfit.append(abs(rng.normal(1.0, 0.3)) + 0.4 * info['dyke'] + 0.2 * info['pipe'])
            k += 1
    out = (models, np.array(chain_id), np.array(misfit))
    if return_truth:
        import pandas as pd
        out = out + (pd.DataFrame(rows),)
    if return_labels:
        out = out + (labs,)
    return out


def make_prior_ensemble(n=400, shape=(36, 54, 64), padd=6, seed=1, density_sd=10.0, geometry=None,
                        return_truth=False, return_labels=False, crop_padding=False):
    """Prior ensemble (PRIOR rates). Returns the models, then the truth table and labels if asked, as make_ensemble."""
    rng = np.random.default_rng(seed)
    models, labs, sl = _allocate(shape, padd, n, crop_padding, return_labels)
    rows = []
    for k in range(n):
        info = {}
        m, lab = sample_model(rng, PRIOR, shape, padd, density_sd=density_sd, geometry=geometry,
                              truth=info, return_labels=True)
        models[..., k] = m[sl]
        if return_labels:
            labs[..., k] = lab[sl]
        rows.append(info)
    if not (return_truth or return_labels):
        return models
    out = (models,)
    if return_truth:
        import pandas as pd
        out = out + (pd.DataFrame(rows),)
    if return_labels:
        out = out + (labs,)
    return out