import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import colors
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable


STATE_COLORS = {
    "CD8T_ISG_IFIT3": "#2171aa",
    "CD8T_Tpex_EGR3": "#ef7c22",
    "CD8T_Tem_GZMK": "#97cc86",
    "CD8T_Tex_HAVCR2": "#8a64a9",
    "CD8T_Tgd_TRDC": "#c19b93",
    "CD8T_Tn_LEF1": "#9f9fa0",
    "CD8T_Trm_ITGAE": "#d5d58c",
    "CD8T_proliferation_MKI67": "#9cd3de",
    "DC_cDC3_LAMP3": "#2171aa",
    "DC_cDC2_CLEC10A": "#97cc86",
    "DC_cDC1_CLEC9A": "#89554b",
    "DC_Langerhans_CD1A": "#d9d9da",
    "DC_pDC_LILRA4": "#9cd3de",
    "TAM_MMP12": "#343a75",
    "TAM_CXCL9": "#646ab0",
    "TAM_FCGBP": "#899e52",
    "TAM_FUCA1": "#8a6c31",
    "TAM_LIPA": "#e3c892",
    "TAM_MKI67": "#cd5e68",
    "TAM_MX1": "#9c4f8d",
    "TAM_SLC40A1": "#d29bc4",
    "NEU_ISG_IFIT3": "#73c0c3",
    "NEU_NETs_IL1B": "#eed5bb",
    "Peri_CAF-like_WNT5A": "#317cb2",
    "Peri_ECM_THBS1": "#9bc5db",
    "Peri_FRC-like_CCL19": "#f08940",
    "Peri_ISG_CXCL10": "#f8cda1",
    "Peri_SMC-like_PLN": "#9ed097",
    "Peri_inflamed_STEAP4": "#7067a8",
    "Peri_proliferation_MKI67": "#d9d9e9",
    "Peri_senescence_MDM2": "#b2b3b3",
    "Peri_universal_RGS5": "#e7e7e7",
    "CD4T_ISG_IFIT3": "#363a75",
    "CD4T_Tanergy_EGR3": "#999bcc",
    "CD4T_Tfh_CXCL13": "#b0c96b",
    "CD4T_Tn_GIMAP4": "#e0b654",
    "CD4T_eTreg_TNFRSF9": "#a64949",
    "CD4T_proliferation_MKI67": "#774070",
    "CD4T_rTreg_CCR4": "#d29cc4",
    "Fibro_FDC_CXCL13": "#2171aa",
    "Fibro_FRC_CCL19": "#acc3e2",
    "Fibro_NF_PI16": "#f7b779",
    "Fibro_iCAF-1_IL24": "#97cc86",
    "Fibro_iCAF-2_VEGFA": "#f19493",
    "Fibro_iCAF-3_PAPPA": "#c2aed1",
    "Fibro_ifnCAF_CXCL10": "#89554b",
    "Fibro_myCAF-1_TGFB3": "#ce76ad",
    "Fibro_myCAF-2_CST1": "#9f9fa0",
    "Fibro_myCAF-3_CCN2": "#b5b62c",
    "Fibro_proliferation_MKI67": "#25b3c3",
    "EC_HEV-like_CXCL10": "#343a75",
    "EC_ISG_IFIT1": "#646ab0",
    "EC_arterial_GJA5": "#617739",
    "EC_capillary_IVNS1ABP": "#afc86a",
    "EC_lymphatic_PROX1": "#8a6c31",
    "EC_proliferation_MKI67": "#e3c892",
    "EC_tip-like_PGF": "#a64849",
    "EC_venous-1_NOS3": "#df9298",
    "EC_venous-2_CCL14": "#9c4f8d",
    "EC_venous-3_CSF2RB": "#d29bc4",
    "Pericyte_CAF-like_WNT5A": "#317cb2",
    "Pericyte_ECM_THBS1": "#9bc5db",
    "Pericyte_FRC-like_CCL19": "#f08940",
    "Pericyte_ISG_CXCL10": "#f8cda1",
    "Pericyte_SMC-like_PLN": "#9ed097",
    "Pericyte_inflamed_STEAP4": "#7067a8",
    "Pericyte_proliferation_MKI67": "#d9d9e9",
    "Pericyte_senescence_MDM2": "#b2b3b3",
    "Pericyte_universal_RGS5": "#e7e7e7",
    "Fibro_senescence_MDM2": "#9cd3de",
}


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


#Fig. 3a
def fig_3a(lineage_embeddings, palettes=None):
    lineages = ["CD4 T cells","CD8 T cells","Dendritic cells","Macrophages",
                "Neutrophils","Fibroblasts","Endothelial cells","Pericytes"]
    fig,axes = _grid(2,4,18,10)
    for ax,name in zip(axes.flat,lineages):
        palette = {**STATE_COLORS, **(palettes or {}).get(name, {})}
        _umap(ax,lineage_embeddings[name],"state",palette,legend=True)
        ax.set_title(name)
    return fig


#Fig. 3b
def fig_3b(roe, comparisons=None):
    fig,axes = _grid(width=17,height=5)
    ax = axes[0,0]
    table = roe.loc[["PT_NR","PT_R","RT"]] if "PT_NR" in roe.index else roe.loc[["PT-NR","PT-R","RT"]]
    _heat(ax,table,"Ro/e",(0,2),fmt=".2f")
    if comparisons is not None:
        ax.set_xticks(range(len(table.columns)),[f"{comparisons.get(s,'')} {s}".strip() for s in table.columns],
                      rotation=90,fontsize=7)
    return fig


#Fig. 3c
def fig_3c(roe, tumor_states=(), immune_states=()):
    fig,axes = _grid(width=18,height=6)
    ax = axes[0,0]
    table = roe.loc[list(CN)]
    _heat(ax,table,"Ro/e",(0,2),fmt=".2f")
    for tick in ax.get_xticklabels():
        s = tick.get_text()
        tick.set_color("#cc2222" if s in tumor_states else "#253d87" if s in immune_states else "black")
    for y,states,color in [(1,tumor_states,"#cc2222"),(4,immune_states,"#253d87")]:
        for j,state in enumerate(table.columns):
            if state in states:
                ax.add_patch(Rectangle((j-0.5,y-0.5),1,2,fill=False,ec=color,ls="--",lw=1))
    return fig


#Fig. 3d
def fig_3d(roi_cells, roi_transcripts, score_limits=None):
    return _roi_map(roi_cells,roi_transcripts,["CXCL13::CXCR5"],
                    {"B cells":"#71a9e0","CD4T_Tfh_CXCL13":"#ef534b","Fibro_FDC_CXCL13":"#76bc73"},
                    {"CXCR5":"#fff044","CXCL13":"#ff382a"},
                    "B cells, Tfh & FDC (PTNR_05)","B enriched (CN5)",score_limits)


#Fig. 3e
def fig_3e(roi_cells, roi_transcripts, score_limits=None):
    return _roi_map(roi_cells,roi_transcripts,["CXCL9::CXCR3"],
                    {"TAM_CXCL9":"#55b46b","CD8 T cells":"#75a5df","CD4 T cells":"#ef6548"},
                    {"CXCL9":"#58c8ec","CXCR3":"#f6df32"},
                    "T cells & CXCL9+ TAM (PTNR_09)","T-APC enriched (CN6)",score_limits)
