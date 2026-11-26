import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D


RADII = (60, 70, 90, 100)


def _grid(rows=1, cols=1, width=5, height=3.5, **kwargs):
    return plt.subplots(rows, cols, figsize=(width, height), squeeze=False,
                        layout="constrained", **kwargs)


def _palette(keys, preset=None):
    keys = list(dict.fromkeys(keys))
    default = plt.get_cmap("tab20", max(20, len(keys)))
    return {key: (preset or {}).get(key, default(i)) for i, key in enumerate(keys)}


def _legend(ax, palette, ncol=1):
    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=4, label=k)
               for k, c in palette.items()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.02),
              ncol=ncol, frameon=False, fontsize=7, columnspacing=0.8, handletextpad=0.3)


def _umap(ax, cells, column, palette=None, label=True, legend=False, alpha=0.65):
    palette = _palette(cells[column].dropna().unique(), palette)
    for key, part in cells.groupby(column, sort=False, observed=True):
        ax.scatter(part.umap1, part.umap2, s=0.5, c=[palette[key]], alpha=alpha,
                   linewidths=0, rasterized=True)
        if label and len(part):
            ax.text(part.umap1.median(), part.umap2.median(), str(key), fontsize=7,
                    ha="center", va="center", zorder=4,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    if legend:
        _legend(ax, palette, ncol=2)


def _heat(ax, matrix, label, limits=None, cmap="coolwarm", fmt=None, bar=True):
    values = matrix.to_numpy(dtype=float)
    lo, hi = limits if limits is not None else (np.nanmin(values), np.nanmax(values))
    im = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=lo, vmax=hi)
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=7)
    ax.tick_params(length=0, pad=2)
    if fmt:
        for (r, c), val in np.ndenumerate(values):
            if np.isfinite(val):
                rgb = im.cmap(im.norm(val))[:3]
                ink = "white" if np.dot(rgb, [0.299, 0.587, 0.114]) < 0.5 else "#222222"
                ax.text(c, r, format(val, fmt), color=ink, fontsize=6, ha="center", va="center")
    if bar:
        ax.figure.colorbar(im, ax=ax, label=label, shrink=0.7, pad=0.012)
    return im


#Fig. S8a
def fig_s8a(embeddings_by_radius):
    fig,axes = _grid(1,4,14,4.1)
    for ax,radius in zip(axes.flat,RADII):
        cells = embeddings_by_radius[radius]
        order = list(pd.unique(cells.ecosystem))
        palette = {s:{"SE1":"#377eb8","SE2":"#ff8b2d","SE3":"#4daf4a","SE4":"#df3c34"}[s[:3]] for s in order}
        _umap(ax,cells,"ecosystem",palette,alpha=0.4)
        ax.set_title(f"{radius} μm")
    return fig


#Fig. S8b
def fig_s8b(pearson_r):
    labels = [f"{r} μm · SE{s}" for r in (60,70,80,90,100) for s in range(1,5)]
    matrix = pearson_r.loc[labels,labels]
    fig,axes = _grid(width=10.5,height=9)
    ax = axes[0,0]
    _heat(ax,matrix,"Pearson r",(0.7,1),fmt=".3f")
    for k in [3.5,7.5,11.5,15.5]:
        ax.axhline(k,c="white",lw=1.5)
        ax.axvline(k,c="white",lw=1.5)
    for j,r in enumerate((60,70,80,90,100)):
        ax.text(4*j+1.5,1.013,f"{r} μm",ha="center",transform=ax.get_xaxis_transform(),fontsize=8)
    return fig
