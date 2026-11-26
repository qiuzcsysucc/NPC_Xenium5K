import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable


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


#Fig. S7a
def fig_s7a(roi_cells, roi_transcripts, score_limits=None):
    return _roi_map(roi_cells,roi_transcripts,["PLAU::PLAUR"],
                    {"TAM_MMP12":"#62b3e4","NEU_NETs_IL1B":"#fc3c2f","NEU_ISG_IFIT3":"#65ba55"},
                    {"PLAUR":"#4ac6e7","PLAU":"#f5e133"},
                    "Neutrophils & TAMs (RT_08)","Neutrophil enriched (CN4)",score_limits)


#Fig. S7b
def fig_s7b(roi_cells, roi_transcripts, score_limits=None):
    return _roi_map(roi_cells,roi_transcripts,["PECAM1::CD38"],
                    {"Plasma cells":"#5fa9e5","EC_tip-like_PGF":"#ec533c","EC_capillary_IVNS1ABP":"#6fc961"},
                    {"CD38":"#2c78b9","PECAM1":"#eb3731"},
                    "Plasma cells & endothelial cells (PTNR_11)","Plasma enriched (CN7)",score_limits)


#Fig. S7c
def fig_s7c(roi_cells, roi_transcripts, score_limits=None):
    return _roi_map(roi_cells,roi_transcripts,["POSTN::ITGB5"],
                    {"Tumor cells":"#5fa9e5","Fibro_myCAF-1_TGFB3":"#dc4d3f","Fibro_myCAF-2_CST1":"#65b75d","Fibro_myCAF-3_CCN2":"#9b61ac"},
                    {"POSTN":"#ea9024","ITGB5":"#f5cf28"},
                    "Tumor cells & myCAFs (PTR_07)","Stroma enriched (CN8)",score_limits)
