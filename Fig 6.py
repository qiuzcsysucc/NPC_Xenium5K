import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
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


def _ptext(p):
    return "<0.001" if float(p) < 0.001 else f"{float(p):.3f}"


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


def _km(ax, steps, at_risk, title, ylabel, p_value=None, ci=True, palette=None, legend=True):
    palette = palette or {"Low":"#3478ab","High":"#e65035"}
    for group,part in steps.groupby("group",sort=False,observed=True):
        part = part.sort_values("time")
        color = palette[group]
        start_n = at_risk.loc[group].iloc[0] if at_risk is not None else None
        label = group if start_n is None else f"{group} (n={int(start_n)})"
        ax.step(part.time,part.survival,where="post",c=color,lw=1.5,label=label)
        if ci:
            ax.fill_between(part.time.to_numpy(float),part.lower.to_numpy(float),part.upper.to_numpy(float),
                            step="post",color=color,alpha=0.12,lw=0)
        if "censored" in part:
            censor = part.loc[part.censored.gt(0)]
            ax.plot(censor.time,censor.survival,ls="",marker="+",ms=3,mew=0.6,color=color)
    ax.set(xlabel="Time (months)",ylabel=ylabel,title=title,ylim=(-0.02,1.04))
    if p_value is not None:
        ax.text(0.03,0.21,f"Log-rank P {'<' if p_value < 0.001 else '='} {_ptext(p_value).lstrip('<')}",
                transform=ax.transAxes,fontsize=8)
    if legend:
        ax.legend(loc="lower left",frameon=False,fontsize=7)
    if at_risk is not None:
        ticks = at_risk.columns.to_numpy(dtype=float)
        ax.set_xticks(ticks)
        ax.set_xlim(min(0,ticks.min()),max(ticks.max(),steps.time.max()))
        risk = ax.inset_axes([0,-0.44,1,0.18])
        risk.set_xlim(ax.get_xlim())
        risk.set_ylim(len(at_risk)-0.5,-0.5)
        for i,(group,row) in enumerate(at_risk.iterrows()):
            for t,n in zip(ticks,row):
                risk.text(t,i,str(int(n)),ha="center",va="center",fontsize=7)
        risk.set_yticks(range(len(at_risk)),at_risk.index,fontsize=7)
        risk.set_title("Number at risk",fontsize=7,loc="left",pad=2)
        risk.set_xticks([])
        risk.tick_params(length=0, pad=10)
        risk.spines[:].set_visible(False)
    ax.spines[["top","right"]].set_visible(False)


def _stacked_recurrence(ax, counts, marker, p_value=None):
    counts = counts.loc[["Low","High"],["NR","R"]]
    bottom = np.zeros(2)
    for level,col in zip(counts.index,["#3b6496","#d94e3a"]):
        pct = 100*counts.loc[level].to_numpy()/counts.sum(axis=0).to_numpy()
        ax.bar([0,1],pct,bottom=bottom,color=col,width=0.65,label=f"{marker}-{level.lower()}")
        bottom += pct
    ax.set_xticks([0,1],["NR","R"])
    ax.set_ylabel("Percent of specimens")
    if p_value is not None:
        ax.set_title(f"P = {p_value:.4g}")
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.12),frameon=False,fontsize=8)


#Fig. 6a
def fig_6a(results):
    return _lr_bubbles(results,"L-R interactions (SE1_Immune-ashy)","Greens")


#Fig. 6b
def fig_6b(roi_cells, roi_transcripts, score_limits=None):
    return _roi_map(roi_cells,roi_transcripts,["GAS6::AXL","GAS6::MERTK"],
                    {"CD4T_eTreg_TNFRSF9":"#92bafa","TAM_FCGBP":"#84c772","SE1_Immune-ashy":"#ffa344"},
                    {"GAS6":"#ff583e","AXL":"#fbd350","MERTK":"#a77ab5","FCGBP":"#73cbeb","TNFRSF9":"#eee6dc"},
                    "GAS6–TAM receptor axis (PTR_09)",None,score_limits)


#Fig. 6f
def fig_6f(events, coverage, junctions, exons):
    order = ["SE_62722","RI_4904","SE_62694","A3SS_17234","A3SS_17235","MXE_18717",
             "A5SS_8599","SE_62695","A3SS_17228","SE_62723","A3SS_17240","SE_62724"]
    table = events.set_index("event_label").loc[order]
    fig = plt.figure(figsize=(10,5.5),layout="constrained")
    gs = fig.add_gridspec(2,2,width_ratios=[1.2,1])
    ax = fig.add_subplot(gs[:,0])
    passed = (table.FDR<0.05)&(table.deltaPSI.abs()>=0.10)
    cols = np.where(passed,np.where(table.deltaPSI>0,"#d7301f","#4975b8"),"#bdbdbd")
    ax.hlines(np.arange(len(table)),0,table.deltaPSI,color=cols)
    ax.scatter(table.deltaPSI,np.arange(len(table)),c=cols,s=26)
    ax.axvline(0,color="black",lw=0.8)
    ax.set_yticks(range(len(table)),table.index,fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="ΔPSI (siSF3B1 − siNC)",title="Top GAS6 splicing events")
    for i,event in enumerate(["SE_62722","RI_4904"]):
        tracks = gs[i,1].subgridspec(3,1,height_ratios=[3,3,0.8],hspace=0.05)
        tracks_axes = []
        for j,(condition,col) in enumerate([("siNC","#f28e21"),("siSF3B1","#d73049")]):
            ax = fig.add_subplot(tracks[j,0],sharex=tracks_axes[0] if tracks_axes else None)
            tracks_axes.append(ax)
            cov = coverage.loc[coverage.event.eq(event)&coverage.condition.eq(condition)].sort_values("position")
            ax.fill_between(cov.position.to_numpy(float),0,cov.coverage.to_numpy(float),color=col,lw=0)
            links = junctions.loc[junctions.event.eq(event)&junctions.condition.eq(condition)]
            ymax = max(float(cov.coverage.max()),1)
            for k,link in enumerate(links.itertuples()):
                a,b = link.start,link.end
                mid = (a+b)/2
                h = ymax*(1.1+0.26*(k%3))
                path = Path([(a,0),(a,h),(mid,h),(b,h),(b,0)],
                            [Path.MOVETO,Path.CURVE3,Path.CURVE3,Path.CURVE3,Path.CURVE3])
                ax.add_patch(PathPatch(path,fill=False,color=col,lw=0.4+0.3*np.log1p(link.count)))
                ax.text(mid,h,str(link.count),ha="center",fontsize=6,color=col)
            ax.set_ylim(0,ymax*2)
            ax.set_ylabel(condition,rotation=0,ha="right",fontsize=7,color=col)
            ax.set_yticks([])
            ax.tick_params(axis="x",bottom=False,labelbottom=False)
            ax.spines[:].set_visible(False)
            if j==0:
                ax.set_title(f"{event} (GAS6)",fontsize=9)
        track = fig.add_subplot(tracks[2,0],sharex=tracks_axes[0])
        spans = exons.loc[exons.event.eq(event)]
        track.hlines(0,spans.start.min(),spans.end.max(),color="0.25",lw=1)
        for span in spans.itertuples():
            track.add_patch(Rectangle((span.start,-0.25),span.end-span.start,0.5,color="0.15"))
        track.set_ylim(-0.35,0.35)
        track.set_axis_off()
    return fig


#Fig. 6g
def fig_6g(exons, expression, p_values=None):
    fig = plt.figure(figsize=(12,7),layout="constrained")
    gs = fig.add_gridspec(2,3,height_ratios=[1.3,1],width_ratios=[1.1,3,3])
    ax = fig.add_subplot(gs[0,:])
    reference_order = ["ENST00000327773.7", "ENST00000480426.5", "ENST00000608763.1",
                       "ENST00000610073.1", "ENST00000998789.1", "ENST00000998790.1",
                       "ENST00001050982.1"]
    observed = list(pd.unique(exons.transcript))
    names = [t for t in reference_order if t in observed] + [t for t in observed if t not in reference_order]
    for i,name in enumerate(names):
        blocks = exons.loc[exons.transcript.eq(name)].sort_values("start")
        ax.hlines(i,blocks.start.min(),blocks.end.max(),color="0.15",lw=1)
        for exon in blocks.itertuples():
            ax.add_patch(Rectangle((exon.start,i-0.23),max(exon.end-exon.start,1),0.46,color="0.12"))
        for left,right in zip(blocks.end.iloc[:-1],blocks.start.iloc[1:]):
            if right>left:
                ax.plot((right+left)/2,i,marker="<" if blocks.strand.iloc[0]=="-" else ">",ms=3,color="black")
        ax.text(1.01,i,str(blocks.annotation.iloc[0]),transform=ax.get_yaxis_transform(),va="center",fontsize=7)
    ax.set_yticks(range(len(names)),names,fontsize=8)
    ax.set_ylim(len(names)-0.5,-0.5)
    ax.set_xticks([])
    ax.set_title("The isoform switch in GAS6 (NC vs siSF3B1)")
    for i,panel in enumerate(["Gene Expression","Isoform Expression","Isoform Usage"]):
        ax = fig.add_subplot(gs[1,i])
        subset = expression.loc[expression.panel.eq(panel)]
        features = [k for k in (["GAS6"] if i==0 else names) if k in set(subset.feature)]
        x = np.arange(len(features))
        for j,(condition,col) in enumerate([("siNC","#416da3"),("siSF3B1","#cd4439")]):
            data = subset.loc[subset.condition.eq(condition)]
            grouped = data.groupby("feature").value
            means,sd = grouped.mean().reindex(features),grouped.std().reindex(features)
            pos = x+0.34*(j-0.5)
            ax.bar(pos,means,width=0.32,color=col,yerr=sd,capsize=2,label=condition)
            for k,f in enumerate(features):
                vals = data.loc[data.feature.eq(f),"value"]
                ax.scatter(np.full(len(vals),pos[k]),vals,c="white",edgecolors="0.25",s=10,linewidths=0.4,zorder=3)
        if p_values is not None:
            for k,f in enumerate(features):
                if (panel,f) in p_values:
                    y = subset.loc[subset.feature.eq(f),"value"].max()
                    ax.annotate(f"{p_values[panel,f]:.2g}",(k,y),xytext=(0,4),textcoords="offset points",
                                ha="center",fontsize=7)
        ax.set_xticks(x,features,rotation=50,ha="right",fontsize=7)
        ax.set_ylabel(panel)
        if i==2:
            ax.legend(fontsize=8)
    return fig


#Fig. 6k
def fig_6k(specimens, fitted_line, spearman_rho, spearman_p):
    fig,axes = _grid(width=4.5,height=4.2)
    ax = axes[0,0]
    rng = np.random.default_rng(6)
    ax.scatter(specimens.SF3B1_IRS+rng.uniform(-0.13,0.13,len(specimens)),
               specimens.GAS6_IRS+rng.uniform(-0.13,0.13,len(specimens)),s=14,color="#48b6d5",alpha=0.45)
    line = fitted_line.sort_values("x")
    ax.fill_between(line.x.to_numpy(float),line.lower.to_numpy(float),line.upper.to_numpy(float),color="#d9eff2")
    ax.plot(line.x,line["fit"],color="#222222")
    ax.text(0.03,0.96,f"Spearman ρ = {spearman_rho:.3f}\nP "+("< 0.001" if spearman_p<0.001 else f"= {spearman_p:.3g}"),
            va="top",transform=ax.transAxes)
    ax.set(xlabel="SF3B1 IRS",ylabel="GAS6 IRS")
    return fig


#Fig. 6l
def fig_6l(recurrence_counts, association_p, km_steps, at_risk, logrank_p):
    fig,axes = _grid(1,2,8.5,4.5)
    _stacked_recurrence(axes[0,0],recurrence_counts,"GAS6",association_p)
    _km(axes[0,1],km_steps,at_risk,"GAS6 expression","LRRFS probability",logrank_p,ci=False)
    return fig
