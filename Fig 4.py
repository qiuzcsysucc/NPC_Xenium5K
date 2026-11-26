import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SE = {"SE1_Immune-ashy": "#e41a1c", "SE2_Immune-insulated": "#984ea3",
      "SE3_Immune-inflamed": "#a65628", "SE4_Immune-cold": "#b3b3b3"}


GROUP = {"PT-NR": "#719d60", "PT-R": "#537da1", "RT": "#c7533e"}


FIVE = [
    ["TAM_FCGBP", "CD4T_eTreg_TNFRSF9", "DC_Langerhans_CD1A", "CD8T_Tex_HAVCR2", "TAM_MKI67"],
    ["Fibro_myCAF-2_CST1", "Fibro_iCAF-2_VEGFA", "Fibro_myCAF-1_TGFB3", "Fibro_proliferation_MKI67", "Pericyte_ECM_THBS1"],
    ["CD8T_Tex_HAVCR2", "CD8T_Trm_ITGAE", "CD8T_ISG_IFIT3", "TAM_MX1", "DC_Langerhans_CD1A"],
    ["TAM_MMP12", "NEU_NETs_IL1B", "DC_Langerhans_CD1A", "TAM_FCGBP", "Fibro_iCAF-2_VEGFA"],
]


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


def _polygons(ax, vertices, facecolors, edgecolors="none", linewidths=0.25):
    poly = PolyCollection(vertices, facecolors=facecolors, edgecolors=edgecolors,
                          linewidths=linewidths, rasterized=True)
    ax.add_collection(poly)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_facecolor("#231916")
    ax.set_xticks([])
    ax.set_yticks([])
    return poly


def _scale(ax, length):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x, y = x0+0.03*(x1-x0), y0+0.06*(y1-y0)
    ax.plot([x,x+length],[y,y],color="white",lw=2,scalex=False,scaley=False)
    ax.annotate(f"{length} μm",(x+length/2,y),xytext=(0,3),textcoords="offset points",
                color="white",ha="center",fontsize=7)


def _cooccurrence(curves, index):
    ecosystem = list(SE)[index]
    fig, axes = _grid(width=5.4,height=4.6)
    ax = axes[0,0]
    for state in FIVE[index]:
        series = curves.loc[curves.state.eq(state)].sort_values("radius")
        ax.plot(series.radius, series.ratio, marker="o", ms=4,lw=1.8,label=state)
    ax.axhline(1,color="0.6",ls="--")
    ax.set(xlabel="Radius (μm)",ylabel="Co-occurrence ratio",title=ecosystem)
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.2),frameon=False,fontsize=8,ncol=2)
    return fig


MAP_COLORS = {
    "SE1_Immune-ashy": "#ef7c1b",
    "CD4T_eTreg_TNFRSF9": "#d62f82",
    "TAM_MKI67": "#0e4e94",
    "DC_Langerhans_CD1A": "#593b91",
    "TAM_FCGBP": "#0e645d",
    "CD8T_Tex_HAVCR2": "#904c9a",
    "Non-focus": "#7b7979",
    "SE2_Immune-insulated": "#b5b5b6",
    "Fibro_myCAF-1_TGFB3": "#74318b",
    "Fibro_myCAF-2_CST1": "#c12583",
    "Pericyte_ECM_THBS1": "#533216",
    "Fibro_proliferation_MKI67": "#7c70a5",
    "Fibro_iCAF-2_VEGFA": "#ad5822",
    "SE4_Immune-cold": "#d62421",
    "NEU_NETs_IL1B": "#f4b064",
    "TAM_MMP12": "#7faccc",
    "SE3_Immune-inflamed": "#229c49",
    "CD8T_ISG_IFIT3": "#8a9cc5",
    "CD8T_Trm_ITGAE": "#dea818",
    "TAM_MX1": "#394e9f",
}


def _ecosystem_map(cells,index,sample):
    ecosystem = list(SE)[index]
    palette = {name: MAP_COLORS[name] for name in [ecosystem, *FIVE[index], "Non-focus"]}
    fill = [palette.get(s,palette["Non-focus"]) for s in cells.state]
    fig, axes = _grid(width=5.1,height=5.1)
    ax = axes[0,0]
    _polygons(ax,cells.vertices.tolist(),fill,"#231916",0.25)
    ax.invert_yaxis()
    _scale(ax,100)
    ax.set_title(f"{ecosystem} ({sample})",fontsize=10)
    _legend(ax,palette,3)
    return fig


#Fig. 4b
def fig_4b(cells):
    fig,axes = _grid(width=5,height=5)
    _umap(axes[0,0],cells,"ecosystem",SE,legend=True,alpha=0.25)
    return fig


#Fig. 4c
def fig_4c(scaled_proportions, lineage_by_state):
    fig,axes = _grid(width=15,height=5.3)
    matrix = scaled_proportions.loc[list(SE)]
    _heat(axes[0,0],matrix,"80 μm neighbors · scaled proportion",(0,1))
    _bands(axes[0,0],[lineage_by_state[s] for s in matrix.columns])
    return fig


#Fig. 4d
def fig_4d(curves):
    return _cooccurrence(curves,0)


#Fig. 4e
def fig_4e(roi_cells):
    return _ecosystem_map(roi_cells,0,"PTR_12")


#Fig. 4f
def fig_4f(curves):
    return _cooccurrence(curves,1)


#Fig. 4g
def fig_4g(roi_cells):
    return _ecosystem_map(roi_cells,1,"PTR_06")


#Fig. 4h
def fig_4h(curves):
    return _cooccurrence(curves,2)


#Fig. 4i
def fig_4i(roi_cells):
    return _ecosystem_map(roi_cells,2,"PTNR_10")


#Fig. 4j
def fig_4j(curves):
    return _cooccurrence(curves,3)


#Fig. 4k
def fig_4k(roi_cells):
    return _ecosystem_map(roi_cells,3,"RT_07")


#Fig. 4l
def fig_4l(specimens, p_values=None):
    return _single_boxes(specimens,list(SE),"Proportion",p_values)


#Fig. 4m
def fig_4m(progeny_scaled):
    fig,axes = _grid(width=8,height=3.4)
    _heat(axes[0,0],progeny_scaled.loc[list(SE)],"Scaled scores",(0,1))
    return fig
