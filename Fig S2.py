import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch


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


#Fig. S2a
def fig_s2a(qc_cells):
    metrics = ["total_counts","cell_area","nucleus_area","nucleus_count","n_genes_by_counts"]
    order = [f"{group}_{i:02d}" for group in ["PT-NR","PT-R","RT"] for i in range(1,13)]
    names = qc_cells.batch.replace({r"^PTNR_":"PT-NR_",r"^PTR_":"PT-R_"},regex=True)
    groups = {name:data for name,data in qc_cells.groupby(names,sort=False)}
    fig,axes = _grid(5,1,16,8,sharex=True)
    for ax,metric in zip(axes.flat,metrics):
        values = [groups[name][metric].dropna().to_numpy(float) for name in order]
        pos = np.arange(len(order))
        varied = [i for i,v in enumerate(values) if len(v)>1 and np.ptp(v)>0]
        if varied:
            violins = ax.violinplot([values[i] for i in varied],positions=pos[varied],widths=0.84,
                                    showextrema=False,points=80)
            for body in violins["bodies"]:
                body.set(facecolor="#3979a5",edgecolor="#424242",alpha=0.88)
        for i,v in enumerate(values):
            if not len(v):
                continue
            lower,median,upper = np.percentile(v,[25,50,75])
            ax.vlines(i,v.min(),v.max(),color="0.4",lw=0.7)
            ax.vlines(i,lower,upper,color="0.15",lw=2.1)
            ax.scatter(i,median,s=3,color="white",zorder=3)
        ax.set_ylabel(metric,fontsize=8)
        ax.grid(axis="y",ls="--",alpha=0.2)
    axes[-1,0].set_xticks(range(len(order)),order,rotation=50,ha="right",fontsize=7)
    return fig


#Fig. S2b
def fig_s2b(cells):
    return _atlas(cells,"cell_type",MAJOR)


#Fig. S2c
def fig_s2c(specimens, p_values=None):
    return _single_boxes(specimens,MAJOR_ORDER,"Density (cells/mm²)",p_values)
