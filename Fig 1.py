import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, leaves_list


MAJOR = dict(zip(
    ["B cells", "CD4 T cells", "CD8 T cells", "Dendritic cells", "Endothelial cells",
     "Fibroblasts", "Macrophages", "Mast cells", "Neutrophils", "Normal epithelial cells",
     "Pericytes", "Plasma cells", "Schwann cells", "Tumor cells"],
    ["#1f77b4", "#aec7e8", "#ffbb78", "#2ca02c", "#d62728", "#ff9896", "#c5b0d5",
     "#8c564b", "#e377c2", "#f7b6d2", "#c7c7c7", "#bcbd22", "#17becf", "#9edae5"]
))


GROUP = {"PT-NR": "#719d60", "PT-R": "#537da1", "RT": "#c7533e"}


MAJOR_ORDER = ["Tumor cells", "Plasma cells", "Macrophages", "CD4 T cells", "CD8 T cells",
               "Neutrophils", "Fibroblasts", "B cells", "Endothelial cells",
               "Normal epithelial cells", "Dendritic cells", "Pericytes", "Mast cells", "Schwann cells"]


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


def _bands(ax, groups, top=1.02):
    labels = list(groups)
    start = 0
    while start < len(labels):
        stop = start + 1
        while stop < len(labels) and labels[stop] == labels[start]:
            stop += 1
        trans = ax.get_xaxis_transform()
        ax.plot([start-0.45, stop-0.55], [top, top], transform=trans, clip_on=False, color="0.2", lw=2)
        ax.text((start+stop-1)/2, top+0.02, str(labels[start]), transform=trans, rotation=90,
                fontsize=7, ha="center", va="bottom")
        start = stop


def _dot(ax, means, fractions, gene_groups=None, counts=None, tree=None, orientation="normal",
         color_title="Scaled mean expression", size_ticks=(20, 40, 60, 80, 100)):
    fractions = fractions.loc[means.index, means.columns]
    if tree is not None:
        order = leaves_list(tree)
        means, fractions = means.iloc[order], fractions.iloc[order]
    if orientation == "transposed":
        display, sizes = means.T, fractions.T
    else:
        display, sizes = means, fractions
    y, x = np.indices(display.shape)
    dots = ax.scatter(x, y, s=95*sizes.to_numpy(dtype=float)/100, c=display.to_numpy(dtype=float),
                      cmap="coolwarm", vmin=0, vmax=1, edgecolors="#666666", linewidths=0.25)
    ax.set(xlim=(-0.5, display.shape[1]-0.5), ylim=(display.shape[0]-0.5, -0.5))
    ax.set_xticks(range(display.shape[1]), display.columns, rotation=90, fontsize=7)
    if orientation == "transposed":
        ax.xaxis.tick_top()
    ax.set_yticks(range(display.shape[0]), display.index, fontsize=7)
    ax.tick_params(length=0)
    if gene_groups is not None:
        groups = [gene_groups[g] for g in means.columns]
        edges = np.flatnonzero(np.asarray(groups[1:]) != np.asarray(groups[:-1])) + 0.5
        for edge in edges:
            if orientation == "transposed":
                ax.axhline(edge, color="0.25", lw=1)
            else:
                ax.axvline(edge, color="0.25", lw=1)
        if orientation != "transposed":
            _bands(ax, groups)
    fig = ax.figure
    handles = [ax.scatter([], [], s=95*t/100, color="0.65", label=f"{t}%") for t in size_ticks]
    ax.legend(handles=handles, title="Fraction of cells", fontsize=7, title_fontsize=7,
              frameon=False, ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5,-0.42))
    fig.colorbar(dots, ax=ax, label=color_title, shrink=0.6, pad=(0.32 if counts is not None and orientation != "transposed" else 0.03))
    if counts is not None:
        values = counts.loc[means.index]
        if orientation == "transposed":
            side = ax.inset_axes([0, -0.23, 1, 0.12])
            side.bar(range(len(values)), values, color="#e9b468")
            side.set_xlim(ax.get_xlim())
            side.invert_yaxis()
            side.set_xticks([])
            side.set_ylabel("Count", fontsize=6)
        else:
            side = ax.inset_axes([1.015, 0, 0.14, 1])
            side.barh(range(len(values)), values, color="#e9b468")
            side.set_ylim(ax.get_ylim())
            side.set_yticks([])
            side.set_xlabel("Count", fontsize=6)
        side.tick_params(labelsize=6)
    if tree is not None:
        if orientation == "transposed":
            branch = ax.inset_axes([0, -0.32, 1, 0.08])
            dendrogram(tree, ax=branch, orientation="bottom", no_labels=True, above_threshold_color="0.25",
                       color_threshold=0)
        else:
            branch = ax.inset_axes([1.17, 0, 0.12, 1])
            dendrogram(tree, ax=branch, orientation="right", no_labels=True,
                       above_threshold_color="0.25", color_threshold=0)
            branch.invert_yaxis()
        branch.set_axis_off()


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


#Fig. 1b
def fig_1b(cells, scaled_mean, fraction_pct):
    genes = ["MS4A1","CD79A","CD2","CD3E","CD4","CD8A","CD8B","IRF8","CSF2RA","PLVAP",
             "COL4A1","COL5A1","POSTN","CTSL","CD68","KIT","GATA2","HCAR2","HCAR3","FOXJ1",
             "MUC5B","PDGFRB","RGS5","MZB1","TENT5C","SOX10","MPZ","EPCAM","CLDN1"]
    rows = ["Plasma cells","B cells","Endothelial cells","Mast cells","CD8 T cells","CD4 T cells",
            "Dendritic cells","Macrophages","Tumor cells","Neutrophils","Normal epithelial cells",
            "Schwann cells","Fibroblasts","Pericytes"]
    modules = ["B","CD4 T","CD8 T","DC","EC","Fibro","TAM","Mast","NEU","Normal epi",
               "Pericyte","Plasma","Schwann","Tumor"]
    gene_groups = dict(zip(genes,np.repeat(modules,[2,3]+[2]*12)))
    fig,axes = _grid(1,2,17,5.7,gridspec_kw={"width_ratios":[1,3.7]})
    _umap(axes[0,0],cells,"cell_type",MAJOR,alpha=0.25)
    axes[0,0].set_title("Major cell types (NPC Xenium cohort)")
    _dot(axes[0,1],scaled_mean.loc[rows,genes],fraction_pct,gene_groups)
    return fig


#Fig. 1c
def fig_1c(specimens, p_values=None):
    return _single_boxes(specimens, MAJOR_ORDER, "Proportion", p_values)
