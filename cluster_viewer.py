"""
Interactive cluster viewer (standalone HTML, Bokeh), in the spirit of the
linked-brushing t-SNE mosaic of test_notebook2_original_file.ipynb.

Left: the 2D embedding coloured by cluster, with a hover tooltip showing a
section of the model under the cursor. Right: the median model, the
standard-deviation model and the representative model of the cluster chosen
in the drop-down list (or tapped in the embedding), on a vertical section at
a fixed y (x-z panel), at a fixed x (y-z panel) or on a depth map (x-y panel).
A table lists the clusters with their size, chain composition and spread.

The page is self-contained (BokehJS inlined) and needs no server.

    from cluster_viewer import build_viewer, save_viewer
    layout = build_viewer(core, coords, labels, chain_id, iy=21, ix=26, iz=12, truth=truth)
    save_viewer(layout, 'cluster_viewer.html')
"""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib

from bokeh.embed import file_html
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource, HoverTool, TapTool, CustomJS, Select, RadioButtonGroup, Div,
                          LinearColorMapper, ColorBar, DataTable, TableColumn, NumberFormatter,
                          CustomJSTickFormatter, Range1d)
from bokeh.palettes import Viridis256, Magma256
from bokeh.plotting import figure
from bokeh.resources import INLINE

from cluster_tools import cluster_ids_by_size, cluster_stats, representative_index, section, cluster_palette

_ORIENTS = [('y', 'section at fixed y (x-z)'), ('x', 'section at fixed x (y-z)'), ('z', 'depth map (x-y)')]


def _png_b64(arr2d, vmin, vmax, cmap='viridis', scale=3, flip=True):
    """Colour-mapped PNG (base64 data URI) of a 2D array, enlarged `scale` times; row 0 at the top when flip is False."""
    cm = matplotlib.colormaps[cmap]
    a = np.clip((np.asarray(arr2d, dtype=np.float32) - vmin) / (vmax - vmin + 1e-12), 0, 1)
    rgb = (cm(a)[..., :3] * 255).astype(np.uint8)
    im = Image.fromarray(rgb)
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = BytesIO()
    im.save(buf, format='png')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def _hex(c):
    return matplotlib.colors.to_hex(c)


def _image_figure(title, w_cells, h_cells, mapper, orient, width=320):
    """Figure holding one image glyph; returns (figure, source). Vertical sections are drawn with depth increasing downwards."""
    height = int(width * h_cells / w_cells) + 40
    p = figure(width=width, height=height, title=title, x_range=Range1d(0, w_cells), y_range=Range1d(0, h_cells),
               tools='', toolbar_location=None, match_aspect=True)
    src = ColumnDataSource(data=dict(image=[np.zeros((h_cells, w_cells), np.float32)]))
    p.image(image='image', x=0, y=0, dw=w_cells, dh=h_cells, source=src, color_mapper=mapper)
    if orient != 'z':
        p.yaxis.formatter = CustomJSTickFormatter(code=f'return ({h_cells} - tick).toFixed(0);')
        p.yaxis.axis_label = 'depth (cells)'
    else:
        p.yaxis.axis_label = 'y (cells)'
    p.xaxis.axis_label = 'x (cells)' if orient != 'x' else 'y (cells)'
    p.add_layout(ColorBar(color_mapper=mapper, width=8), 'right')
    return p, src


def build_viewer(core, coords, labels, chain_id=None, iy=None, ix=None, iz=None, truth=None,
                 min_size=5, max_clusters=60, vmin=None, vmax=None, vmax_std=None, thumb_orient='y',
                 title='Cluster viewer', with_table=True):
    """
    Build the Bokeh layout of the viewer.

    core : (nz, ny, nx, N) ensemble (analysis region), coords : (N, 2) embedding, labels : (N,) clusters (-1 noise).
    chain_id : (N,) chain of each model (optional). iy, ix, iz : section indices (centre of the grid if None).
    truth : synthetic truth table (optional): body flags are shown in the tooltip and in the cluster table.
    min_size, max_clusters : clusters shown (at least min_size models, the max_clusters largest).
    vmin, vmax : density colour range (percentiles of the ensemble if None); vmax_std : colour range of the std panels.
    with_table : include the cluster table below the embedding.
    """
    nz, ny, nx, N = core.shape
    iy = ny // 2 if iy is None else int(iy)
    ix = nx // 2 if ix is None else int(ix)
    iz = nz // 2 if iz is None else int(iz)
    if vmin is None or vmax is None:
        lo, hi = np.percentile(core[..., ::max(N // 200, 1)], [0.5, 99.5])
        vmin = float(lo) if vmin is None else vmin
        vmax = float(hi) if vmax is None else vmax
    ids = cluster_ids_by_size(labels, min_size)[:max_clusters]
    pal = cluster_palette(len(ids))
    col_of = {cl: _hex(c) for cl, c in zip(ids, pal)}

    # --- per-cluster statistics: sections of the median, std and representative models ---
    stats = {k: [] for k in ('cluster', 'n', 'members', 'info')}
    for o, _ in _ORIENTS:
        for s in ('med', 'std', 'rep'):
            stats[f'{s}_{o}'] = []
    chains = np.unique(chain_id) if chain_id is not None else []
    for cl in ids:
        med, std, idx = cluster_stats(core, labels, cl)
        rep = core[..., representative_index(coords, labels, cl)]
        stats['cluster'].append(cl)
        stats['n'].append(int(len(idx)))
        stats['members'].append([int(i) for i in idx])
        for o, _ in _ORIENTS:
            for s, m in (('med', med), ('std', std), ('rep', rep)):
                sec = section(m, o, iy, ix, iz).astype(np.float32)
                stats[f'{s}_{o}'].append(sec[::-1] if o != 'z' else sec)   # top of the model drawn at the top
        info = [f'<b>Cluster {cl}</b>: {len(idx)} models']
        if chain_id is not None:
            info.append('chains: ' + ', '.join(f'{c} {np.mean(chain_id[idx] == c):.0%}' for c in chains))
        info.append(f'mean voxel std: {std.mean():.1f}')
        if truth is not None:
            flags = [c for c in ('high', 'lens', 'dyke', 'pipe') if c in truth]
            info.append('bodies: ' + ', '.join(f'{c} {np.mean(truth[c].values[idx].astype(bool)):.0%}' for c in flags))
        stats['info'].append('<br>'.join(info))
    if not ids:
        raise ValueError('no cluster with at least min_size models')
    stats_src = ColumnDataSource(data=stats)
    if vmax_std is None:
        vmax_std = float(max(np.percentile(a, 99) for a in stats['std_y'] + stats['std_x'])) or 1.0

    # --- embedding scatter with thumbnails ---
    thumbs = [_png_b64(section(core[..., i], thumb_orient, iy, ix, iz), vmin, vmax,
                       flip=(thumb_orient != 'z')) for i in range(N)]
    pos = {cl: k for k, cl in enumerate(ids)}
    sc = dict(x=coords[:, 0], y=coords[:, 1], index=np.arange(N), cluster=labels.astype(int),
              pos=[pos.get(int(l), -1) for l in labels],
              color=[col_of.get(int(l), '#c8c8c8') for l in labels],
              size=[8 if int(l) in pos else 4 for l in labels],
              alpha=[0.9 if int(l) in pos else 0.4 for l in labels],
              thumb=thumbs,
              chain=list(chain_id) if chain_id is not None else [''] * N)
    tooltip_rows = ['<b>model @index</b> &middot; cluster @cluster &middot; @chain']
    if truth is not None:
        flags = [c for c in ('high', 'lens', 'dyke', 'pipe') if c in truth]
        for c in flags:
            sc[c] = ['yes' if v else 'no' for v in truth[c].values.astype(bool)]
        tooltip_rows.append(' '.join(f'{c}: @{c}' for c in flags))
    sc_src = ColumnDataSource(data=sc)
    p_emb = figure(width=560, height=520, title='UMAP embedding (hover: model section; tap: select the cluster)',
                   tools='pan,wheel_zoom,box_zoom,reset,tap', match_aspect=True, active_scroll='wheel_zoom')
    r = p_emb.scatter('x', 'y', source=sc_src, size='size', fill_color='color', fill_alpha='alpha',
                      line_color='black', line_width=0.2, nonselection_fill_alpha=0.12, nonselection_line_alpha=0.1,
                      selection_fill_alpha=1.0, selection_line_color='black', selection_line_width=0.6)
    p_emb.add_tools(HoverTool(renderers=[r], tooltips=f"""
        <div style="width:220px">
          <div style="font-size:11px">{'<br>'.join(tooltip_rows)}</div>
          <img src="@thumb" style="width:210px; image-rendering:pixelated; margin-top:3px"/>
        </div>"""))
    p_emb.xaxis.axis_label = 'UMAP 1'; p_emb.yaxis.axis_label = 'UMAP 2'

    # --- image panels ---
    mapper = LinearColorMapper(palette=Viridis256, low=vmin, high=vmax)
    mapper_std = LinearColorMapper(palette=Magma256, low=0, high=vmax_std)
    dims = {'y': (nx, nz), 'x': (ny, nz), 'z': (nx, ny)}
    panels, img_srcs = {}, {}
    for o, _ in _ORIENTS:
        w, h = dims[o]
        panels[o], img_srcs[o] = {}, {}
        for s, t, m in (('med', 'median model', mapper), ('std', 'standard deviation', mapper_std),
                        ('rep', 'representative model', mapper)):
            panels[o][s], img_srcs[o][s] = _image_figure(t, w, h, m, o)
            panels[o][s].visible = (o == 'y')
    info_div = Div(text=stats['info'][0], width=560)
    select = Select(title='Cluster (sorted by size)', value='0',
                    options=[(str(k), f'cluster {cl}  (n = {n})') for k, (cl, n) in enumerate(zip(ids, stats['n']))],
                    width=260)
    orient = RadioButtonGroup(labels=[lab for _, lab in _ORIENTS], active=0, width=400)

    flat = {f'src_{s}_{o}': img_srcs[o][s] for o in img_srcs for s in img_srcs[o]}
    flat.update({f'panel_{s}_{o}': panels[o][s] for o in panels for s in panels[o]})
    cb = CustomJS(args=dict(stats=stats_src, select=select, orient=orient, sc=sc_src, info=info_div,
                            orients=[o for o, _ in _ORIENTS], **flat),
                  code="""
        const srcs = {y: {med: src_med_y, std: src_std_y, rep: src_rep_y},
                      x: {med: src_med_x, std: src_std_x, rep: src_rep_x},
                      z: {med: src_med_z, std: src_std_z, rep: src_rep_z}};
        const panels = {y: {med: panel_med_y, std: panel_std_y, rep: panel_rep_y},
                        x: {med: panel_med_x, std: panel_std_x, rep: panel_rep_x},
                        z: {med: panel_med_z, std: panel_std_z, rep: panel_rep_z}};
        const k = parseInt(select.value);
        const o = orients[orient.active];
        for (const oo of orients) {
            for (const s of ['med', 'std', 'rep']) {
                panels[oo][s].visible = (oo === o);
            }
        }
        for (const s of ['med', 'std', 'rep']) {
            srcs[o][s].data = {image: [stats.data[s + '_' + o][k]]};
        }
        info.text = stats.data['info'][k];
        sc.selected.indices = Array.from(stats.data['members'][k]);
    """)
    select.js_on_change('value', cb)
    orient.js_on_change('active', cb)
    # tapping one point selects its cluster (a single selected index that belongs to a shown cluster)
    sc_src.selected.js_on_change('indices', CustomJS(args=dict(sc=sc_src, select=select), code="""
        const ind = sc.selected.indices;
        if (ind.length !== 1) return;
        const k = sc.data['pos'][ind[0]];
        if (k < 0) return;
        if (select.value !== String(k)) { select.value = String(k); }
    """))

    # --- cluster table ---
    tab = {'cluster': ids, 'n': stats['n'],
           'mean_voxel_std': [float(np.mean(a)) for a in [cluster_stats(core, labels, cl)[1] for cl in ids]]}
    cols = [TableColumn(field='cluster', title='cluster', width=60), TableColumn(field='n', title='n', width=50),
            TableColumn(field='mean_voxel_std', title='mean voxel std', formatter=NumberFormatter(format='0.0'))]
    for c in chains:
        tab[f'frac_{c}'] = [float(np.mean(chain_id[labels == cl] == c)) for cl in ids]
        cols.append(TableColumn(field=f'frac_{c}', title=f'{c}', formatter=NumberFormatter(format='0.00')))
    if truth is not None:
        for c in [c for c in ('high', 'lens', 'dyke', 'pipe') if c in truth]:
            tab[f'P_{c}'] = [float(np.mean(truth[c].values[labels == cl].astype(bool))) for cl in ids]
            cols.append(TableColumn(field=f'P_{c}', title=f'P({c})', formatter=NumberFormatter(format='0.00')))
    table = DataTable(source=ColumnDataSource(data=tab), columns=cols, width=560, height=260, index_position=None)

    header = Div(text=f'<h3 style="margin:4px 0">{title}</h3>'
                      f'<span style="font-size:12px">N = {N} models, {len(ids)} clusters shown '
                      f'(at least {min_size} models each); sections at y = {iy}, x = {ix}, z = {iz}.</span>', width=900)
    right = column(row(select, orient), info_div,
                   *[row(panels[o]['med'], panels[o]['std'], panels[o]['rep']) for o, _ in _ORIENTS])
    layout = column(header, row(column(p_emb, table) if with_table else p_emb, right))

    # initial state: first cluster, y section
    k0 = 0
    for s in ('med', 'std', 'rep'):
        img_srcs['y'][s].data = dict(image=[stats[f'{s}_y'][k0]])
    sc_src.selected.indices = stats['members'][k0]
    return layout


def save_viewer(layout, filename, title='Cluster viewer'):
    """Write the layout to a self-contained HTML file (BokehJS inlined)."""
    html = file_html(layout, INLINE, title)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    if layout.document is not None:          # leave the layout free for a later show() in a notebook
        layout.document.remove_root(layout)
    return filename
