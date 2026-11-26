from matplotlib.text import Text
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle, Patch


def _grid(rows=1, cols=1, width=5, height=3.5, **kwargs):
    return plt.subplots(rows, cols, figsize=(width, height), squeeze=False,
                        layout="constrained", **kwargs)


def _lr_bubbles(results,title,cmap):
    fig,axes = _grid(width=6,height=4.8)
    ax = axes[0,0]
    x = list(pd.unique(results["sender_receiver"]))
    y = list(pd.unique(results["interaction"]))
    def area(p):
        return np.interp(-np.log10(np.clip(p,0.001,1)),[0,1,2,3],[8,34,74,120])
    dots = ax.scatter([x.index(k) for k in results.sender_receiver],
                      [y.index(k) for k in results.interaction],s=area(results.padj),
                      c=results.score,cmap=cmap,edgecolors="0.2",linewidths=0.45,vmin=0)
    ax.set_xticks(range(len(x)),x,rotation=50,ha="right",fontsize=7)
    ax.set_yticks(range(len(y)),y,fontsize=8)
    ax.set(xlim=(-0.5,len(x)-0.5),ylim=(len(y)-0.5,-0.5),title=title)
    fig.colorbar(dots,ax=ax,label="Interaction score",shrink=0.65)
    ticks = [0.001,0.01,0.05,0.1,1]
    handles = [ax.scatter([],[],s=area(p),color="0.7",ec="0.3",label=f"{p:g}") for p in ticks]
    ax.legend(handles=handles,title="Adjusted P",loc="upper left",bbox_to_anchor=(1.25,1),
              frameon=False,fontsize=7)
    return fig


def _label_points(ax, coordinates):
    ax.figure.draw_without_rendering()
    renderer = ax.figure.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer)
    occupied = []
    shifts = [(dx, dy) for dy in (5, -7, 14, -17, 24, -27, 34)
              for dx in (4, -4, 14, -14, 24, -24)]
    for x, y, label in coordinates:
        if pd.isna(label) or str(label).strip() in {"", "NA", "nan", "<NA>"} or not np.isfinite(x+y):
            continue
        tag = ax.annotate(str(label), (x, y), xytext=(0, 0), textcoords="offset points",
                          va="center", fontsize=7,
                          arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "0.4"})
        best = None
        for dx, dy in shifts:
            tag.set_position((dx, dy))
            tag.set_ha("left" if dx > 0 else "right")
            tag.update_positions(renderer)
            box = Text.get_window_extent(tag, renderer).expanded(1.07, 1.16)
            outside = max(bounds.x0-box.x0, 0) + max(box.x1-bounds.x1, 0)
            outside += max(bounds.y0-box.y0, 0) + max(box.y1-bounds.y1, 0)
            overlap = sum(max(0, min(box.x1, b.x1)-max(box.x0, b.x0)) *
                          max(0, min(box.y1, b.y1)-max(box.y0, b.y0)) for b in occupied)
            cost = overlap + 10*outside
            if best is None or cost < best[0]:
                best = (cost, dx, dy, box)
            if cost == 0:
                break
        _, dx, dy, box = best
        tag.set_position((dx, dy))
        tag.set_ha("left" if dx > 0 else "right")
        occupied.append(box)


def _volcano(ax, rows, xcol, pcol, labels=None, fold_cut=0, alpha=0.05, lines=True, marker="o",
             xlabel="log2 fold change", ylabel="−log10 adjusted P-value", y_values=None):
    x = pd.to_numeric(rows[xcol], errors="coerce").to_numpy(float)
    p = pd.to_numeric(rows[pcol], errors="coerce").to_numpy(float)
    positive = p[np.isfinite(p) & (p>0)]
    floor = positive.min()*0.5 if len(positive) else np.finfo(float).tiny
    y = -np.log10(np.clip(p,max(floor,np.finfo(float).tiny),1)) if y_values is None else np.asarray(y_values)
    col = np.select([(p<alpha)&(x>0)&(x>=fold_cut),(p<alpha)&(x<0)&(x<=-fold_cut)],
                    ["#d64e3d","#497bb6"],default="#ababab")
    ax.scatter(x,y,c=col,s=12,marker=marker,lw=0.2,edgecolors="0.35",rasterized=True)
    if lines:
        for bound in sorted({-fold_cut,fold_cut}):
            ax.axvline(bound,ls="--",lw=1,c="0.3")
        ax.axhline(-np.log10(alpha),ls="--",lw=1,c="0.3")
    ax.set(xlabel=xlabel,ylabel=ylabel)
    ax.spines[["top","right"]].set_visible(False)
    if labels is not None:
        _label_points(ax, zip(x, y, labels))


#Fig. S12a
def fig_s12a(results):
    return _lr_bubbles(results,"LR interactions (SE2_Immune-insulated)","Oranges")


#Fig. S12b
def fig_s12b(results):
    return _lr_bubbles(results,"LR interactions (SE3_Immune-inflamed)","Blues")


#Fig. S12c
def fig_s12c(results):
    return _lr_bubbles(results,"LR interactions (SE4_Immune-cold)","Purples")


#Fig. S12d
def fig_s12d(pca_scores, variance_pct=(99.6, 0.26)):
    fig,axes = _grid(width=4.5,height=4.4)
    ax = axes[0,0]
    for condition,col in [("siNC","#e97167"),("siSF3B1","#1db6bd")]:
        data = pca_scores.loc[pca_scores.condition.eq(condition)]
        ax.scatter(data.PC1,data.PC2,s=28,c=col,label=condition)
        for j,r in enumerate(data.itertuples()):
            ax.annotate(r.sample,(r.PC1,r.PC2),xytext=(4,(j%3-1)*8),textcoords="offset points",fontsize=7)
    ax.axhline(0,ls="--",lw=0.7,c="0.4")
    ax.axvline(0,ls="--",lw=0.7,c="0.4")
    ax.set(xlabel=f"PC1: {variance_pct[0]:g}% variance",ylabel=f"PC2: {variance_pct[1]:g}% variance",
           title="Sample-level")
    return fig


#Fig. S12e
def fig_s12e(gene_deseq):
    data = gene_deseq.loc[gene_deseq.gene_name.isin(["SF3B1","GAS6"])]
    fig,axes = _grid(width=4.5,height=4)
    _volcano(axes[0,0],data,"log2FoldChange","padj",data.gene_name,fold_cut=0.5,
             y_values=-np.log10(pd.to_numeric(data.padj, errors="coerce").clip(lower=1e-300)))
    axes[0,0].set_title("Gene-level")
    return fig


#Fig. S12f
def fig_s12f(transcript_deseq):
    data = transcript_deseq.loc[transcript_deseq.gene_name.isin(["SF3B1","GAS6"])].dropna(subset=["padj"])
    fig,axes = _grid(width=6.2,height=4.3)
    _volcano(axes[0,0],data,"log2FoldChange","padj",data.transcript_name.where(pd.to_numeric(data.padj, errors="coerce").lt(0.05)),lines=False)
    axes[0,0].set_title("Transcript-level")
    return fig


#Fig. S12g
def fig_s12g(exon_results):
    fig,axes = _grid(width=5,height=3.5)
    ax = axes[0,0]
    _volcano(ax,exon_results,"exon_usage_log2FC","padj",lines=False)
    ax.axvline(0,c="0.35",ls="--",lw=0.8)
    ax.set_title("Exon-level (GAS6)")
    return fig


#Fig. S12h
def fig_s12h(exon_results, fitted_exon_usage):
    data = exon_results.sort_values("exon_bin_index").reset_index(drop=True)
    fig,axes = _grid(2,1,12,5.2,gridspec_kw={"height_ratios":[3,1], "hspace":0.28})
    usage,locus = axes[:,0]
    bins = np.arange(len(data))
    adjusted = pd.to_numeric(data.padj, errors="coerce")
    fitted = fitted_exon_usage.loc[data.featureID, ["siNC", "siSF3B1"]].astype(float)
    for condition, color in [("siNC", "#e64f4f"), ("siSF3B1", "#3278b9")]:
        usage.scatter(bins, fitted[condition], marker="_", s=55, c=color, label=condition)
    usage.set_yscale("function", functions=(lambda x: np.sqrt(np.maximum(x, 0)), np.square))
    upper = max(1, float(np.nanmax(fitted.to_numpy())) * 1.06)
    usage.set_ylim(0, upper)
    ticks = [y for y in (0, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000) if y < upper]
    if len(ticks) >= 3:
        usage.set_yticks(ticks)
    usage.set(xlim=(-0.6,len(bins)-0.4),ylabel="Exon usage",title="ENSG00000183087.17 – GAS6 exon usage")
    usage.set_xticks(bins[::3],data.featureID.iloc[::3],rotation=90,fontsize=7)
    usage.legend(frameon=False,ncol=2)
    start,end = data["genomicData.start"].to_numpy(),data["genomicData.end"].to_numpy()
    locus.hlines(0, start.min(), end.max(), color="0.2")
    for i,(a,b,q) in enumerate(zip(start,end,adjusted)):
        col = "#92288c" if q<0.05 else "#a8a8a8"
        usage.axvline(i,c=col,alpha=0.3,lw=0.65,ls=":",zorder=0)
        locus.add_patch(Rectangle((a,-0.32),max(b-a,1),0.64,ec="black",fc=col,lw=0.5))
        fig.add_artist(ConnectionPatch((i,-0.16),((a+b)/2,0.33),coordsA=usage.get_xaxis_transform(),
                       coordsB=locus.transData,axesA=usage,axesB=locus,color=col,lw=0.45,zorder=0))
    locus.set(xlim=(start.min(),end.max()),ylim=(-0.4,0.4),yticks=[],xlabel="Chromosome 13 position (bp)")
    locus.ticklabel_format(axis="x",style="plain",useOffset=False)
    locus.tick_params(labelsize=7)
    return fig


#Fig. S12i
def fig_s12i(junction_results):
    fig,axes = _grid(width=5.8,height=3.9)
    ax = axes[0,0]
    _volcano(ax,junction_results,"log2FoldChange","padj",junction_results.label,lines=False,
             y_values=junction_results.neglog10padj)
    ax.axvline(0,ls="--",lw=0.8,color="0.35")
    ax.set_title("Junction-level (GAS6)")
    return fig


#Fig. S12j
def fig_s12j(events):
    fig,axes = _grid(width=8.7,height=5.6)
    ax = axes[0,0]
    markers = {"A3SS":"o","A5SS":"^","MXE":"X","RI":"P","SE":"s"}
    for event,part in events.groupby("event_type",sort=False):
        _volcano(ax,part,"deltaPSI","FDR",None,0.10,lines=False,marker=markers[event],
                 xlabel="ΔPSI (siSF3B1 − siNC)",ylabel="−log10 FDR",y_values=part.neglog10FDR)
    for bound in [-0.10,0.10]:
        ax.axvline(bound,c="0.25",ls="--",lw=1)
    ax.axvline(0,c="0.2",lw=1)
    ax.axhline(-np.log10(0.05),c="0.25",ls="--",lw=1)
    ax.set_title("GAS6 rMATS event-level volcano plot")
    handles = [Line2D([],[],ls="",marker=m,c="0.3",label=k) for k,m in markers.items()]
    handles += [Patch(color=c,label=l) for c,l in [("#d64e3d","Higher inclusion"),("#497bb6","Lower inclusion"),("#ababab","NS")]]
    ax.legend(handles=handles,loc="lower center",bbox_to_anchor=(0.5,1.09),ncol=4,fontsize=7,frameon=False)
    _label_points(ax, zip(events.deltaPSI, events.neglog10FDR, events.label))
    return fig


#Fig. S12k
def fig_s12k(irfinder):
    fig,axes = _grid(width=5.5,height=4.2)
    ax = axes[0,0]
    _volcano(ax,irfinder,"delta_IR","DESeq2_padj",irfinder.coord_key,0.10,
             xlabel="ΔIR (siSF3B1 − siNC)",ylabel="−log10 FDR")
    ax.set_title("GAS6 IR events (IRFinder)")
    return fig


#Fig. S12l
def fig_s12l(coverage):
    samples = ["siNC_1","siNC_2","siNC_3","siSF3B1_1","siSF3B1_2","siSF3B1_3"]
    fig,axes = _grid(6,1,9,4.4,sharex=True,sharey=True)
    for i,(ax,sample) in enumerate(zip(axes.flat,samples)):
        track = coverage.loc[coverage["sample"].eq(sample)].sort_values("position")
        ax.fill_between(track.position.to_numpy(float),0,track.coverage.to_numpy(float),
                        color="#4760be" if i<3 else "#f75d3c",lw=0,step="mid")
        ax.set_ylabel(sample,rotation=0,ha="right",va="center",fontsize=7)
        ax.yaxis.tick_right()
        ax.tick_params(labelsize=6)
        ax.spines[["top","left"]].set_visible(False)
    axes[0,0].set_title("chr13:113,818,109–113,866,565")
    axes[-1,0].set(xlim=(113818109,113866565),xlabel="Chromosome 13 position (bp)")
    axes[-1,0].ticklabel_format(axis="x",style="plain",useOffset=False)
    return fig
