import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import ConnectionPatch, Rectangle, Patch
from scipy.cluster.hierarchy import dendrogram


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


GROUP = {"PT-NR": "#719d60", "PT-R": "#537da1", "RT": "#c7533e"}


def _grid(rows=1, cols=1, width=5, height=3.5, **kwargs):
    return plt.subplots(rows, cols, figsize=(width, height), squeeze=False,
                        layout="constrained", **kwargs)


def _palette(keys, preset=None):
    keys = list(dict.fromkeys(keys))
    default = plt.get_cmap("tab20", max(20, len(keys)))
    return {key: (preset or {}).get(key, default(i)) for i, key in enumerate(keys)}


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


def _ptext(p):
    return "<0.001" if float(p) < 0.001 else f"{float(p):.3f}"


def _boxes(ax, specimens, features, ylabel="Proportion", p_values=None, group_col="group",
           batch_col="batch", pair_col=None, scale=1, labels=None):
    groups = specimens[group_col].replace({"PT_NR":"PT-NR", "PT_R":"PT-R", "PTNR":"PT-NR", "PTR":"PT-R"})
    pairs = (specimens[pair_col] if pair_col is not None else
             specimens[batch_col].astype(str).str.extract(r"[_-](\d+)$", expand=False))
    offset = 0.24
    rng = np.random.default_rng(13)
    total = specimens.loc[:, features].to_numpy(dtype=float)*scale
    span = max(float(np.nanmax(total) - np.nanmin(total)), 1e-8)
    for i, feature in enumerate(features):
        positions = {}
        for j, (group, color) in enumerate(GROUP.items()):
            mask = groups.eq(group) & specimens[feature].notna()
            values = specimens.loc[mask, feature].to_numpy(float)*scale
            if not len(values):
                continue
            center = i + (j-1)*offset
            jitter = center + rng.uniform(-0.025, 0.025, len(values))
            ax.boxplot([values], positions=[center], widths=offset*0.83, patch_artist=True,
                       showfliers=False, manage_ticks=False, boxprops={"facecolor":color,"edgecolor":"0.4"},
                       medianprops={"color":"0.15"}, whiskerprops={"color":"0.4"}, capprops={"color":"0.4"})
            ax.scatter(jitter, values, s=8, c=color, edgecolors="0.25", lw=0.25, zorder=4)
            if group in ("PT-R", "RT"):
                coords = pd.DataFrame({"id":pairs[mask].to_numpy(), "x":jitter,"y":values}).dropna(subset=["id"])
                if coords.id.duplicated().any():
                    raise ValueError("Each matched specimen needs one unique pair ID within its group.")
                positions[group] = coords.set_index("id")
        if "PT-R" in positions and "RT" in positions:
            a, b = positions["PT-R"].align(positions["RT"], join="inner", axis=0)
            ax.plot([a.x.to_numpy(), b.x.to_numpy()], [a.y.to_numpy(), b.y.to_numpy()],
                    color="0.6", alpha=0.55, lw=0.6, zorder=3)
        if p_values is not None:
            height = float(np.nanmax(total[:, i])) + span*0.045
            for j, comparison in enumerate(("PT-NR_vs_PT-R", "PT-R_vs_RT")):
                p = p_values.loc[feature, comparison]
                if pd.isna(p):
                    continue
                x0 = i + (j-1)*offset
                h = height+j*span*0.075
                ax.plot([x0,x0,x0+offset,x0+offset], [h-span*0.014,h,h,h-span*0.014], lw=0.7, color="0.2")
                ax.text(x0+offset/2,h,_ptext(p),va="bottom",ha="center",fontsize=7,
                        fontweight="bold" if p < 0.05 else "normal")
            ax.update_datalim([(i, height+span*0.16)])
    ax.autoscale_view()
    ax.set_xticks(range(len(features)), labels if labels is not None else features,
                  rotation=38 if len(features)>1 else 0, ha="right" if len(features)>1 else "center", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.legend(handles=[Patch(color=c,label=k) for k,c in GROUP.items()], ncol=3, fontsize=7,
              frameon=False, loc="lower right", bbox_to_anchor=(1,1))


def _single_boxes(data, features, ylabel, p_values=None):
    fig, axes = _grid(width=max(5, len(features)*1.1), height=4.2)
    _boxes(axes[0,0], data, features, ylabel, p_values)
    return fig


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


def _atlas(cells, column, preset):
    fig, axes = _grid(3, 12, 17, 5.6)
    samples = cells["batch"].astype(str).str.replace("-", "", regex=False)
    for r, prefix in enumerate(("PTNR","PTR","RT")):
        for c, ax in enumerate(axes[r]):
            name = f"{prefix}_{c+1:02d}"
            part = cells.loc[samples.eq(name)]
            _spatial_points(ax, part, column, preset, 0.2)
            ax.set_title(name, fontsize=8)
    fig.legend(handles=[Patch(color=v, label=k) for k,v in preset.items()], loc="outside lower center",
               ncol=min(7,len(preset)), fontsize=8, frameon=False)
    return fig


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


#Fig. 2a
def fig_2a(log2_enrichment, column_linkage):
    matrix = log2_enrichment.loc[list(CN)]
    fig,axes = _grid(2,1,7,5,gridspec_kw={"height_ratios":[1,6]})
    tree = dendrogram(column_linkage,ax=axes[0,0],no_labels=True,color_threshold=0,above_threshold_color="0.35")
    axes[0,0].set_axis_off()
    heat = _heat(axes[1,0],matrix.iloc[:,tree["leaves"]],"log₂ fold enrichment",(-2,2),bar=False)
    fig.colorbar(heat,ax=axes[:,0].tolist(),shrink=0.55,label="log₂ fold enrichment")
    return fig


#Fig. 2b
def fig_2b(specimens, p_values=None):
    return _single_boxes(specimens, list(CN), "Proportion", p_values)


#Fig. 2c
def fig_2c(cells):
    return _atlas(cells, "cn", CN)


#Fig. 2d
def fig_2d(cells, roi):
    return _zoom(cells,"PTNR_03",roi,[[("cn",False),("cn",True),("cell_type",True)]])


#Fig. 2e
def fig_2e(cells, roi):
    return _zoom(cells,"PTNR_05",roi,[[("cn",False),("cn",True),("cell_type",True)]])


#Fig. 2f
def fig_2f(cells, roi, boundaries=None):
    layout = [[("cell_type",False),("cell_type",True),("cn",False)],
              [("cell_type",False),("cn",True),("cn",False)]]
    return _zoom(cells,"PTR_03",roi,layout,boundaries)
