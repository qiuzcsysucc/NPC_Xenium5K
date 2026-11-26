import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import ConnectionPatch, Rectangle, Patch


MAJOR = dict(zip(
    ["B cells", "CD4 T cells", "CD8 T cells", "Dendritic cells", "Endothelial cells",
     "Fibroblasts", "Macrophages", "Mast cells", "Neutrophils", "Normal epithelial cells",
     "Pericytes", "Plasma cells", "Schwann cells", "Tumor cells"],
    ["#1f77b4", "#aec7e8", "#ffbb78", "#2ca02c", "#d62728", "#ff9896", "#c5b0d5",
     "#8c564b", "#e377c2", "#f7b6d2", "#c7c7c7", "#bcbd22", "#17becf", "#9edae5"]
))


CN = dict(zip(
    ["CN1_Normal_epithelium", "CN2_Tumor_boundary", "CN3_Tumor_core", "CN4_Neutrophil_enriched",
     "CN5_B_enriched", "CN6_T-APC_enriched", "CN7_Plasma_enriched", "CN8_Stroma_enriched"],
    ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#f0e442", "#a65628", "#f781bf"]
))


def _palette(keys, preset=None):
    keys = list(dict.fromkeys(keys))
    default = plt.get_cmap("tab20", max(20, len(keys)))
    return {key: (preset or {}).get(key, default(i)) for i, key in enumerate(keys)}


def _spatial_points(ax, data, column, preset, size=1):
    categories = pd.Categorical(data[column])
    palette = _palette(categories.categories, preset)
    cmap = colors.ListedColormap(list(palette.values()) or ["#dddddd"])
    codes = categories.codes
    keep = codes >= 0
    ax.scatter(data.loc[keep,"x"], data.loc[keep,"y"], c=codes[keep], cmap=cmap,
               vmin=-0.5, vmax=max(len(palette)-0.5,0.5), s=size, lw=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()


def _zoom(cells, sample, roi, layout, boundaries=None):
    xmin, xmax, ymin, ymax = roi
    selected = cells.loc[cells.batch.astype(str).str.replace("-", "", regex=False).eq(sample)]
    region = selected.loc[selected.x.between(xmin, xmax) & selected.y.between(ymin, ymax)]
    rows, cols = len(layout), len(layout[0])
    fig = plt.figure(figsize=(cols*3.1,rows*3.3+1.1),layout="constrained")
    gs = fig.add_gridspec(rows, cols)
    axes = {}
    for row in range(rows):
        for col in range(cols):
            entry = layout[row][col]
            if entry in axes:
                continue
            places = [(r,c) for r in range(rows) for c in range(cols) if layout[r][c] == entry]
            ax = fig.add_subplot(gs[min(r for r,c in places):max(r for r,c in places)+1,
                                    min(c for r,c in places):max(c for r,c in places)+1])
            axes[entry] = ax
            column, detail = entry
            palette, title = (CN,"Cellular niche") if column=="cn" else (MAJOR,"Major cell types")
            _spatial_points(ax, region if detail else selected, column, palette, 3 if detail else 0.35)
            ax.set_title(f"{title} ({'zoom' if detail else sample})", fontsize=9)
            ax.set(xlabel="spatial1", ylabel="spatial2")
            if detail:
                ax.set(xlim=(xmin,xmax), ylim=(ymax,ymin))
            else:
                ax.add_patch(Rectangle((xmin,ymin), xmax-xmin,ymax-ymin,fill=False,lw=1))
            if boundaries is not None:
                for line in boundaries:
                    ax.plot(*np.asarray(line).T, color="0.15", ls="--", lw=1, scalex=False, scaley=False)
    for column in ("cn","cell_type"):
        if (column,False) not in axes or (column,True) not in axes:
            continue
        whole, zoom = axes[column,False], axes[column,True]
        left = whole.get_subplotspec().colspan.start < zoom.get_subplotspec().colspan.start
        for y in (ymin,ymax):
            fig.add_artist(ConnectionPatch((xmax if left else xmin,y),(xmin if left else xmax,y),
                           coordsA="data",coordsB="data",axesA=whole,axesB=zoom,color="0.3",ls="--",lw=0.7))
    fig.legend(handles=[Patch(color=v,label=k) for k,v in {**CN,**MAJOR}.items()],
               loc="outside lower center", ncol=5, fontsize=6, frameon=False)
    return fig


#Fig. S3a
def fig_s3a(cells, roi):
    layout = [[("cn",False),("cn",True)],[("cell_type",False),("cell_type",True)]]
    return _zoom(cells,"PTR_01",roi,layout)


#Fig. S3b
def fig_s3b(cells, roi):
    layout = [[("cn",False),("cn",True)],[("cell_type",False),("cell_type",True)]]
    return _zoom(cells,"PTR_10",roi,layout)


#Fig. S3c
def fig_s3c(cells, roi):
    layout = [[("cell_type",False),("cell_type",True),("cn",False),("cn",True)]]
    return _zoom(cells,"RT_09",roi,layout)
