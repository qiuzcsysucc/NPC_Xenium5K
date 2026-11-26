import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from scipy.cluster.hierarchy import dendrogram, leaves_list


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


def _grid(rows=1, cols=1, width=5, height=3.5, **kwargs):
    return plt.subplots(rows, cols, figsize=(width, height), squeeze=False,
                        layout="constrained", **kwargs)


def _legend(ax, palette, ncol=1):
    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=4, label=k)
               for k, c in palette.items()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.02),
              ncol=ncol, frameon=False, fontsize=7, columnspacing=0.8, handletextpad=0.3)


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


def _roi_map(cells, transcripts, pairs, focus_colors, gene_colors, title, niche=None, limits=None):
    n = 1+len(pairs)+(2 if niche else 0)
    fig, axes = _grid(1,n,n*3.4,4.5)
    axes = list(axes[0])
    vertices = cells["vertices"].tolist()
    if niche:
        for ax, key, palette, heading in [(axes[0],"cell_type",MAJOR,"Major cell types"),
                                          (axes[1],"cn",CN,niche)]:
            _polygons(ax,vertices,[palette.get(k,"#aaaaaa") for k in cells[key]],"#231916")
            _legend(ax,{k:v for k,v in palette.items() if k in set(cells[key])}, 2)
            ax.set_title(heading,fontsize=9)
    offset = 2 if niche else 0
    focus = axes[offset]
    _polygons(focus,vertices,"none",[focus_colors.get(k,"none") for k in cells["state"]],0.55)
    for gene,color in gene_colors.items():
        pts = transcripts.loc[transcripts.gene.eq(gene)]
        focus.scatter(pts.x,pts.y,s=2,color=color,lw=0,rasterized=True)
    _legend(focus,{**focus_colors,**gene_colors},2)
    focus.set_title(title,fontsize=9)
    for ax,pair in zip(axes[offset+1:],pairs):
        score = cells[pair].to_numpy(float)
        lo,hi = limits[pair] if limits and pair in limits else (0,np.nanmax(score))
        norm = colors.Normalize(lo,max(hi,lo+1e-12),clip=True)
        mapper = ScalarMappable(norm=norm,cmap="magma")
        _polygons(ax,vertices,mapper.to_rgba(score),"#31282a")
        ax.set_title(f"{pair} interaction score",fontsize=9)
        fig.colorbar(mapper,ax=ax,orientation="horizontal",shrink=0.55,pad=0.03)
    bounds = np.concatenate(vertices)
    for ax in axes:
        ax.set(xlim=(bounds[:,0].min(),bounds[:,0].max()),ylim=(bounds[:,1].max(),bounds[:,1].min()))
    _scale(axes[0],50)
    return fig


#Fig. S6a
def fig_s6a(interaction_scores, fraction_pct, counts, niche_by_pair):
    fig,axes = _grid(width=16,height=5)
    _dot(axes[0,0],interaction_scores.loc[list(CN)],fraction_pct,niche_by_pair,counts,
         color_title="Interaction score")
    fig.suptitle("Top 5 LR interactions")
    return fig


#Fig. S6b
def fig_s6b(roi_cells, roi_transcripts, score_limits=None):
    return _roi_map(roi_cells,roi_transcripts,["TGFA::ERBB2"],
                    {"Normal epithelial cells":"#4299d4"},
                    {"TGFA":"#fde233","ERBB2":"#ee3228"},
                    "Normal epithelial cells (PTR_01)","Normal epithelium (CN1)",score_limits)


#Fig. S6c
def fig_s6c(roi_cells, roi_transcripts, score_limits=None):
    return _roi_map(roi_cells,roi_transcripts,["CCL20::CCR6"],
                    {"Tumor cells":"#5fa9e5","CD4T_eTreg_TNFRSF9":"#df443f","CD8T_Tex_HAVCR2":"#e3cc39","TAM_FCGBP":"#e99139"},
                    {"CCL20":"#80509e","CCR6":"#6ec875"},
                    "States & transcripts (PTR_09)","Tumor boundary (CN2)",score_limits)


#Fig. S6d
def fig_s6d(roi_cells, roi_transcripts, score_limits=None):
    return _roi_map(roi_cells,roi_transcripts,["LAMA3::ITGAV"],
                    {"Tumor cells":"#5fa9e5"},
                    {"LAMA3":"#fa4534","ITGAV":"#fadd2b"},
                    "Tumor cells (RT_09)","Tumor core (CN3)",score_limits)
