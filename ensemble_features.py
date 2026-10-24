"""
Feature vectors for an ensemble of 3D labelled models, as in
2b_Encoding_with_3D_Autoencoder_topology_red.ipynb, without TensorFlow.

Each model (nz, ny, nx) is described by 523 numbers:

  [0:512]    fixed-kernel encoder ("DiffNN"): 512 values
  [512:518]  topological scalars (topo_features.compute_topo_features): 6 values
  [518:523]  density moments (mean, std, skewness, kurtosis, range): 5 values

The encoder is the untrained network of the notebook, re-implemented with NumPy:

  block 1   351 fixed 3x3x3 difference kernels (+1/sqrt2 at one voxel of a
            3x3x3 neighbourhood, -1/sqrt2 at another, one kernel per unordered
            pair of voxels), zero padding ('same'), absolute value, 2x2x2 max
            pooling;
  blocks 2-3  the same kernels applied depthwise (kernel c on channel c), abs,
            2x2x2 max pooling;
  pooling   2x2x2 average pooling of the output of each block, flattened in
            the Keras order (z, y, x, channel);
  projection  fixed column-orthonormal random projections (QR of a standard
            normal matrix, seeds 1, 2, 3) to 128, 128 and 256 dimensions;
  output    each branch L2-normalised, concatenated: 512 values.

A kernel with +s at offset i and -s at offset j, applied by cross-correlation,
is the difference of two shifted copies of the input; the encoder is therefore
computed by shifting, not by convolution. The flatten order and the random
projections follow the notebook (numpy.random.default_rng with the same seeds,
QR in float32), so that the latent codes match those of the TensorFlow model up
to floating-point rounding.

Because all weights are fixed, no training, no GPU and no TensorFlow are
needed; the only ensemble-dependent step is the global 0-1 scaling of the
densities before the encoder, as in the notebook.

Usage
-----
    enc = DiffNetEncoder(core.shape[:3])         # projections built once per grid shape
    latent = enc.encode_ensemble(core)           # (N, 512)
    topo, dens = topo_and_density_features(core) # (N, 6), (N, 5)
    features, scaler = combine_features(latent, topo, dens)   # standardised (N, 523)

Models are passed as (nz, ny, nx, N) arrays, model axis last, as in the motif
workflow (ensemble_io, synthetic_ensemble).
"""

from __future__ import annotations

import os
import numpy as np
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed

from topo_features import compute_topo_features, FEATURE_NAMES as TOPO_FEATURE_NAMES

DENSITY_FEATURE_NAMES = ['density_mean', 'density_std', 'density_skew', 'density_kurtosis', 'density_range']

# Unordered pairs of the 27 voxels of a 3x3x3 neighbourhood, in the order of
# build_difference_kernels() in the notebook: (0,1), (0,2), ..., (25,26).
PAIRS = np.array([(i, j) for i in range(27) for j in range(i + 1, 27)], dtype=np.intp)   # (351, 2)
N_KERNELS = len(PAIRS)                                                                   # 351
_SCALE = np.float32(1.0 / np.sqrt(2.0))
# Offsets (a, b, c) in the padded array of kernel element k = 9a + 3b + c (k.reshape(3, 3, 3) order).
_OFF = np.array([(k // 9, (k // 3) % 3, k % 3) for k in range(27)], dtype=np.intp)        # (27, 3)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
def orthogonal_projection(input_dim, output_dim, seed, chunk_rows=65536):
    """
    Column-orthonormal projection (input_dim, output_dim): QR of a standard normal matrix, as in the notebook.
    The normal matrix is drawn by blocks of rows directly into float32 (the same numbers as one float64 draw
    cast to float32, without the float64 copy), which halves the transient memory for large input_dim.
    """
    assert output_dim <= input_dim, (input_dim, output_dim)
    rng = np.random.default_rng(seed)
    W = np.empty((input_dim, output_dim), dtype=np.float32)
    for i0 in range(0, input_dim, chunk_rows):
        i1 = min(i0 + chunk_rows, input_dim)
        W[i0:i1] = rng.standard_normal((i1 - i0, output_dim))
    Q, _ = np.linalg.qr(W)
    del W
    return np.ascontiguousarray(Q, dtype=np.float32)


def _pool(x, op):
    """
    2x2x2 pooling (op = np.maximum or np.add) on the last three axes of x, 'valid' as in Keras (odd trailing
    slices dropped), done axis by axis with pairwise binary ops, which is several times faster than a
    multi-axis reduction over a reshaped array. For np.add the result is divided by 8 (average pooling).
    """
    c, nz, ny, nx = x.shape
    z, y, w = nz // 2, ny // 2, nx // 2
    v = x[:, :2 * z, :2 * y, :2 * w]
    v = op(v[:, 0::2], v[:, 1::2])
    v = op(v[:, :, 0::2], v[:, :, 1::2])
    v = op(v[:, :, :, 0::2], v[:, :, :, 1::2])
    return v * np.float32(0.125) if op is np.add else v


def _difference_block(x, chunk=27):
    """
    One block of the encoder. x: (C, nz, ny, nx) float32, C = 1 (block 1) or N_KERNELS (depthwise blocks).
    Returns (N_KERNELS, nz//2, ny//2, nx//2): |shift_i(x_c) - shift_j(x_c)| / sqrt(2), max-pooled 2x2x2.
    The kernels are processed by groups of `chunk` and each group is pooled at once, so that the full-resolution
    response of all 351 kernels (110 MB for a 36 x 42 x 52 grid) is never held in memory.
    """
    c, nz, ny, nx = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (1, 1)))                  # zero padding = Keras 'same'
    out = np.empty((N_KERNELS, nz // 2, ny // 2, nx // 2), dtype=np.float32)
    tmp = np.empty((chunk, nz, ny, nx), dtype=np.float32)
    for k0 in range(0, N_KERNELS, chunk):
        k1 = min(k0 + chunk, N_KERNELS)
        for j, k in enumerate(range(k0, k1)):
            (pz, py, px), (qz, qy, qx) = _OFF[PAIRS[k, 0]], _OFF[PAIRS[k, 1]]
            ch = 0 if c == 1 else k                                    # depthwise: kernel k on channel k
            np.subtract(xp[ch, pz:pz + nz, py:py + ny, px:px + nx], xp[ch, qz:qz + nz, qy:qy + ny, qx:qx + nx],
                        out=tmp[j])
        t = tmp[:k1 - k0]
        np.abs(t, out=t)
        t *= _SCALE
        out[k0:k1] = _pool(t, np.maximum)
    return out


def pooled_descriptors(model_01):
    """
    The three average-pooled, flattened block outputs of one model (densities scaled to 0-1).
    Returns (p1, p2, p3), float32 vectors, flattened in the Keras order (z, y, x, channel).
    """
    x = np.asarray(model_01, dtype=np.float32)[None]                   # (1, nz, ny, nx)
    out = []
    for _ in range(3):
        x = _difference_block(x)                                        # (351, nz/2, ny/2, nx/2)
        p = _pool(x, np.add)                                               # (351, nz/4, ny/4, nx/4)
        out.append(np.ascontiguousarray(np.moveaxis(p, 0, -1)).ravel())  # channels last, as Keras Flatten
    return tuple(out)


def pooled_dims(shape):
    """Dimensions of p1, p2, p3 for a grid (nz, ny, nx), without computing anything."""
    dims = []
    s = np.array(shape)
    for _ in range(3):
        s = s // 2                        # max pooling of the block
        dims.append(int(np.prod(s // 2)) * N_KERNELS)   # average pooling, times the channels
    return tuple(dims)


def _batch_of(models_4d, i0, i1):
    """Models i0..i1-1 of an (nz, ny, nx, N) ensemble as a contiguous (n, nz, ny, nx) float32 array."""
    return np.ascontiguousarray(np.moveaxis(models_4d[..., i0:i1], -1, 0), dtype=np.float32)


def _pooled_batch(batch, vmin, scale):
    """Pooled descriptors of a batch of models (n, nz, ny, nx), stacked by branch."""
    P = [pooled_descriptors((m - vmin) * scale) for m in batch]
    return tuple(np.stack([p[k] for p in P]) for k in range(3))


class DiffNetEncoder:
    """
    The fixed-kernel encoder of the notebook for a given grid shape.

    Parameters
    ----------
    shape : (nz, ny, nx) of the models.
    dims : output dimensions of the three branches (128, 128, 256 in the notebook).
    seeds : seeds of the three random projections (1, 2, 3 in the notebook).

    The projections are built at construction (a few seconds and ~1.5 GB for a
    54 x 94 x 30 grid, dominated by the first branch) and reused for every model.
    """

    def __init__(self, shape, dims=(128, 128, 256), seeds=(1, 2, 3)):
        self.shape = tuple(int(s) for s in shape)
        self.dims = tuple(dims)
        self.in_dims = pooled_dims(self.shape)
        self.Q = [orthogonal_projection(n_in, n_out, seed)
                  for n_in, n_out, seed in zip(self.in_dims, self.dims, seeds)]

    @property
    def latent_dim(self):
        return sum(self.dims)

    def project(self, P):
        """P: tuple of three (n, in_dim) batches of pooled descriptors -> (n, 512) latent codes, each branch L2-normalised."""
        parts = []
        for p, Q in zip(P, self.Q):
            f = np.asarray(p, dtype=np.float32) @ Q
            f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)   # tf.math.l2_normalize
            parts.append(f)
        return np.hstack(parts)

    def encode(self, model_01):
        """Latent code (512,) of one model whose densities are scaled to 0-1."""
        return self.project(tuple(p[None] for p in pooled_descriptors(model_01)))[0]

    def encode_ensemble(self, models_4d, vmin=None, vmax=None, n_jobs=-1, batch=25, verbose=0):
        """
        Latent codes (N, 512) of an ensemble (nz, ny, nx, N).

        The densities are scaled to 0-1 with the global minimum and maximum of
        the ensemble (vmin, vmax; computed if None), as in the notebook. The
        pooled descriptors are computed in parallel by batches of `batch`
        models and projected in the main process.
        """
        models_4d = np.asarray(models_4d)
        assert models_4d.shape[:3] == self.shape, (models_4d.shape, self.shape)
        n = models_4d.shape[3]
        vmin = float(models_4d.min()) if vmin is None else float(vmin)
        vmax = float(models_4d.max()) if vmax is None else float(vmax)
        scale = 1.0 / (vmax - vmin) if vmax > vmin else 1.0

        bounds = [(i, min(i + batch, n)) for i in range(0, n, batch)]
        out = np.empty((n, self.latent_dim), dtype=np.float32)
        # Each task receives only its own batch of models (model axis first, contiguous). The pooled
        # descriptors (about 1.8 MB per model for a 36 x 42 x 52 grid) are projected group by group as the
        # workers return them, so that memory does not grow with the number of models.
        with Parallel(n_jobs=n_jobs, verbose=verbose, max_nbytes=None) as parallel:   # no memmapping of the batches
            group = 4 * max(parallel.n_jobs if parallel.n_jobs > 0 else os.cpu_count() or 1, 1)
            for g in range(0, len(bounds), group):
                part = bounds[g:g + group]
                res = parallel(delayed(_pooled_batch)(_batch_of(models_4d, i0, i1), vmin, scale) for i0, i1 in part)
                for (i0, i1), P in zip(part, res):
                    out[i0:i1] = self.project(P)
                del res
        return out


# ---------------------------------------------------------------------------
# Topological and density features
# ---------------------------------------------------------------------------
def compute_density_features(model):
    """Moments of the per-voxel density distribution of one model: [mean, std, skewness, kurtosis (Fisher), range]."""
    v = np.asarray(model).ravel().astype(np.float64)
    return np.array([v.mean(), v.std(), skew(v), kurtosis(v), v.max() - v.min()], dtype=np.float32)


def _topo_dens_batch(batch):
    """Topological and density features of a batch of models (n, nz, ny, nx)."""
    return [(compute_topo_features(m), compute_density_features(m)) for m in batch]


def topo_and_density_features(models_4d, n_jobs=-1, verbose=0, batch=50):
    """Topological (N, 6) and density (N, 5) features of an ensemble (nz, ny, nx, N), on the raw label field."""
    n = models_4d.shape[3]
    bounds = [(i, min(i + batch, n)) for i in range(0, n, batch)]
    res = Parallel(n_jobs=n_jobs, verbose=verbose, max_nbytes=None)(       # batches pickled, not memmapped
        delayed(_topo_dens_batch)(_batch_of(models_4d, i0, i1)) for i0, i1 in bounds)
    res = [r for part in res for r in part]
    topo = np.stack([r[0] for r in res])
    dens = np.stack([r[1] for r in res])
    return topo, dens


def combine_features(latent, topo, dens):
    """Concatenate [latent | topo | dens] and standardise each column (StandardScaler). Returns (features, scaler)."""
    from sklearn.preprocessing import StandardScaler
    raw = np.hstack([latent, topo, dens]).astype(np.float64)
    scaler = StandardScaler().fit(raw)
    return scaler.transform(raw).astype(np.float32), scaler


FEATURE_GROUPS = {'CNN': (0, 512), 'Topological': (512, 518), 'Petrophysical': (518, 523)}


def feature_names(latent_dim=512):
    return [f'cnn_{i}' for i in range(latent_dim)] + list(TOPO_FEATURE_NAMES) + DENSITY_FEATURE_NAMES


def classify_feature(idx, latent_dim=512):
    """Group of feature index idx: 'CNN', 'Topological' or 'Petrophysical' (as pca_analysis.classify_feature)."""
    if idx < latent_dim:
        return 'CNN'
    if idx < latent_dim + len(TOPO_FEATURE_NAMES):
        return 'Topological'
    return 'Petrophysical'


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def check_against_convolution(shape=(24, 28, 36), seed=0):
    """
    Compare pooled_descriptors with a literal implementation of the Keras definition (scipy cross-correlation
    with the explicit 3x3x3 kernels, zero padding, abs, max pooling). Returns the largest absolute difference
    of the three branches; it is of the order of 1e-7 (float32 rounding).
    """
    from scipy.ndimage import correlate
    rng = np.random.default_rng(seed)
    x = rng.random(shape).astype(np.float32)
    s = np.float32(1 / np.sqrt(2))
    K = np.zeros((N_KERNELS, 27), np.float32)
    K[np.arange(N_KERNELS), PAIRS[:, 0]], K[np.arange(N_KERNELS), PAIRS[:, 1]] = s, -s
    K = K.reshape(N_KERNELS, 3, 3, 3)

    def pool(v, op):
        c, nz, ny, nx = v.shape
        z, y, w = nz // 2, ny // 2, nx // 2
        return op(v[:, :2 * z, :2 * y, :2 * w].reshape(c, z, 2, y, 2, w, 2), axis=(2, 4, 6))

    b = pool(np.abs(np.stack([correlate(x, K[c], mode='constant', cval=0.0) for c in range(N_KERNELS)])), np.max)
    ref = [np.moveaxis(pool(b, np.mean), 0, -1).ravel()]
    for _ in range(2):
        b = pool(np.abs(np.stack([correlate(b[c], K[c], mode='constant', cval=0.0) for c in range(N_KERNELS)])), np.max)
        ref.append(np.moveaxis(pool(b, np.mean), 0, -1).ravel())
    p = pooled_descriptors(x)
    return max(float(np.abs(a - r).max()) for a, r in zip(p, ref))


if __name__ == '__main__':
    print('largest difference with the brute-force convolution:', check_against_convolution())
