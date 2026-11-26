import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


GROUP = {"PT-NR": "#719d60", "PT-R": "#537da1", "RT": "#c7533e"}


def _grid(rows=1, cols=1, width=5, height=3.5, **kwargs):
    return plt.subplots(rows, cols, figsize=(width, height), squeeze=False,
                        layout="constrained", **kwargs)


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


def _subtypes(long_table, lineage, p_values):
    sub = long_table.loc[long_table.major_celltype.eq(lineage)]
    order = list(pd.unique(sub.cell_type))
    wide = sub.pivot(index=["batch","group"],columns="cell_type",values="proportion").reset_index()
    return _single_boxes(wide,order,"Proportion",p_values)


#Fig. S5a
def fig_s5a(specimens, p_values=None):
    return _subtypes(specimens,"CD4T",p_values)


#Fig. S5b
def fig_s5b(specimens, p_values=None):
    return _subtypes(specimens,"CD8T",p_values)


#Fig. S5c
def fig_s5c(specimens, p_values=None):
    return _subtypes(specimens,"DC",p_values)


#Fig. S5d
def fig_s5d(specimens, p_values=None):
    return _subtypes(specimens,"TAM",p_values)


#Fig. S5e
def fig_s5e(specimens, p_values=None):
    return _subtypes(specimens,"Neutrophil",p_values)


#Fig. S5f
def fig_s5f(specimens, p_values=None):
    return _subtypes(specimens,"Fibroblast",p_values)


#Fig. S5g
def fig_s5g(specimens, p_values=None):
    return _subtypes(specimens,"Endothelial",p_values)


#Fig. S5h
def fig_s5h(specimens, p_values=None):
    return _subtypes(specimens,"Pericyte",p_values)
