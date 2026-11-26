import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import colors
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
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


GROUP = {"PT-NR": "#719d60", "PT-R": "#537da1", "RT": "#c7533e"}


LINEAGES = ["Macro_FCGBP", "Macro_SPP1", "Macro_APOE", "Mono_FCN1+FCGR3A+"]


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


def _clinical_boxes(specimens, features, titles, rows, cols, ylabel, p_values):
    fig,axes = _grid(rows,cols,cols*3.0,rows*3.5)
    for ax,feature,title in zip(axes.flat,features,titles):
        _boxes(ax,specimens,[feature],ylabel,p_values,group_col="Group_Label",pair_col="Pair_ID")
        ax.set_xticks([-0.24,0,0.24],list(GROUP),fontsize=8)
        ax.set_title(title,fontsize=9)
        ax.get_legend().remove()
    return fig


def _markers(means, fraction_pct, counts, modules, title, tree=None, transposed=False):
    genes = [g for name,block in modules for g in block]
    groups = {g:name for name,block in modules for g in block}
    fig,axes = _grid(width=max(6.5, len(genes)*0.23+3.5) if not transposed else 7,
                     height=max(4.4,len(means)*0.22+2.9) if not transposed else 9)
    _dot(axes[0,0],means.loc[:,genes],fraction_pct,groups,counts,tree,
         "transposed" if transposed else "normal")
    fig.suptitle(title,fontsize=11)
    return fig


#Fig. S13a
def fig_s13a(cells):
    palette = dict(zip(["B cells","CD4 T cells","CD8 T cells","Dendritic cells","Endothelial cells",
                        "Epithelial cells","Fibroblasts","Mast cells","Mono/Macro","NK/NKT cells","Plasma cells"],
                       ["#215878","#a168a1","#db9fbf","#e16ba3","#961d55","#f4b420",
                        "#ed7773","#2890bc","#c8c1c0","#dce6a9","#76b55e"]))
    fig,axes = _grid(width=5.8,height=5.4)
    _umap(axes[0,0],cells,"cell_type",palette,legend=True)
    axes[0,0].set_title("Major cell types (public scRNA-seq cohort)")
    return fig


#Fig. S13b
def fig_s13b(cells):
    palette = {"CNP0000428":"#2879ad","GSE150825":"#d72b25","GSE162025":"#d178ad","HRA000087":"#1bb7c4"}
    fig,axes = _grid(width=4.8,height=4.8)
    _umap(axes[0,0],cells,"dataset",palette,label=False,legend=True,alpha=0.35)
    axes[0,0].set_title("Datasets (public scRNA-seq cohort)")
    return fig


#Fig. S13c
def fig_s13c(scaled_mean, fraction_pct, counts, row_linkage):
    modules = [("B cells",["MS4A1","CD79A"]),("CD4 T",["CD3E","CD4"]),("CD8 T",["CD8A","CD8B"]),
               ("DC",["IRF8","CSF2RA"]),("EC",["PLVAP","COL4A1"]),("Epithelial",["EPCAM","CLDN1"]),
               ("Fibroblasts",["COL5A1","POSTN"]),("Mast",["KIT","GATA2"]),("Mono/Macro",["CTSL","CD68"]),
               ("NK/NKT",["KLRF1","NCAM1"]),("Plasma",["MZB1","TENT5C"])]
    return _markers(scaled_mean,fraction_pct,counts,modules,"Major cell types",row_linkage)


#Fig. S13d
def fig_s13d(zscored_pearson):
    fig,axes = _grid(width=6,height=4.7)
    ax = axes[0,0]
    _heat(ax,zscored_pearson,"Z-scored scaled r",(-2,2),cmap="coolwarm")
    x = zscored_pearson.columns.get_loc("TAM_FCGBP")
    y = zscored_pearson.index.get_loc("Macro_FCGBP")
    ax.add_patch(Rectangle((x-0.5,y-0.5),1,1,fill=False,ec="black",ls=":",lw=1.5))
    ax.set(xlabel="Xenium cohort",ylabel="Public scRNA-seq cohort",
           title="TAM subtypes expression correlation")
    return fig


#Fig. S13e
def fig_s13e(jaccard):
    fig,axes = _grid(width=6,height=4.7)
    ax = axes[0,0]
    _heat(ax,jaccard,"Jaccard similarity",(0,0.35),cmap="Reds")
    x = jaccard.columns.get_loc("TAM_FCGBP")
    y = jaccard.index.get_loc("Macro_FCGBP")
    ax.add_patch(Rectangle((x-0.5,y-0.5),1,1,fill=False,ec="black",ls=":",lw=1.5))
    ax.set(xlabel="Xenium cohort",ylabel="Public scRNA-seq cohort",
           title="Similarity of top 100 marker genes")
    return fig


#Fig. S13f
def fig_s13f(cells):
    fig,axes = _grid(1,4,14,3.8)
    for i,(ax,lineage) in enumerate(zip(axes.flat,LINEAGES),1):
        dots = ax.scatter(cells.umap1,cells.umap2,c=cells[lineage],s=1,cmap="coolwarm",lw=0,rasterized=True)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"Lineage {i} · {lineage}",fontsize=9)
        fig.colorbar(dots,ax=ax,shrink=0.7,label="Lineage likelihood" if i==4 else None)
    return fig


#Fig. S13g
def fig_s13g(xenium_mean, xenium_pct, scrna_mean, scrna_pct):
    fig,axes = _grid(2,1,6.2,8.2)
    genes = ["IL18","ICOSLG","PVR","PDCD1LG2","CXCL16"]
    for ax,mean,pct,title,ticks in [(axes[0,0],xenium_mean,xenium_pct,"Xenium",(5,10,15,20,25,30)),
                                   (axes[1,0],scrna_mean,scrna_pct,"scRNA-seq",(10,30,50,70,90))]:
        _dot(ax,mean.loc[:,genes],pct,size_ticks=ticks)
        ax.set_title(f"Top 5 ligands ({title} cohort)")
    return fig


#Fig. S13h
def fig_s13h(roi_cells, roi_transcripts, score_limits=None):
    return _roi_map(roi_cells,roi_transcripts,["IL18::IL18R1","IL18::IL18RAP"],
                    {"CD4T_eTreg_TNFRSF9":"#91bbef","TAM_FCGBP":"#80bc78"},
                    {"IL18":"#ff5039","IL18R1":"#f8dd31","IL18RAP":"#a173bd"},
                    "IL18/IL18R1/IL18RAP axis (PTR_01)",None,score_limits)


#Fig. S13k
def fig_s13k(specimens, p_values=None):
    features = ["% TNFRSF9+Treg Positive Cells","% SF3B1+Tum Positive Cells","% FCGBP+TAM Positive Cells"]
    return _clinical_boxes(specimens,features,["TNFRSF9+ Treg cells","SF3B1+ tumor cells","FCGBP+ TAMs"],
                           1,3,"Proportion (%)",p_values)
