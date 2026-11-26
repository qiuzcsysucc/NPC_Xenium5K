from matplotlib.text import Text
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.cm import ScalarMappable


SE = {"SE1_Immune-ashy": "#e41a1c", "SE2_Immune-insulated": "#984ea3",
      "SE3_Immune-inflamed": "#a65628", "SE4_Immune-cold": "#b3b3b3"}


MP = {"MP_1": "#df2118", "MP_2": "#4daf4a", "MP_3": "#ff7f00",
      "MP_4": "#a65628", "MP_5": "#b3b3b3"}


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


def _ptext(p):
    return "<0.001" if float(p) < 0.001 else f"{float(p):.3f}"


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


#Fig. 5a
def fig_5a(program_similarity, program_groups):
    matrix = program_similarity.loc[program_similarity.columns,program_similarity.columns]
    fig,axes = _grid(width=7.6,height=5.5)
    ax = axes[0,0]
    im = ax.imshow(matrix.to_numpy(),origin="lower",cmap="magma_r",interpolation="nearest",vmin=0,vmax=25)
    ax.set_xticks([])
    groups = pd.Series(program_groups).loc[matrix.index]
    labels = {"MP1":"MP1 RNA splicing\nSF3B1, SUGP2, SRSF2","MP2":"MP2 Hypoxia\nPKM, LDHA, ALDOA",
              "MP3":"MP3 IFN-response\nIFI44L, IFIT3, IFI16","MP4":"MP4 Cell cycling\nMKI67, STMN1, PCNA",
              "MP5":"MP5 EMT\nMMP14, ITGB4, LAMB3"}
    keys = list(pd.unique(groups))
    ax.set_yticks([np.flatnonzero(groups.eq(k)).mean() for k in keys],[labels.get(k,k) for k in keys],fontsize=8)
    fig.colorbar(im,ax=ax,shrink=0.6,label="Program similarity")
    return fig


#Fig. 5b
def fig_5b(enrichment):
    terms = enrichment.sort_values("FDR",kind="stable")
    x = -np.log10(terms.FDR.clip(lower=np.finfo(float).tiny))
    fig,axes = _grid(width=6.5,height=4.8)
    ax = axes[0,0]
    ax.scatter(x,np.arange(len(terms)),s=terms["Count"]*21,c="#2572ae")
    ax.set_yticks(range(len(terms)),terms["term"],fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="−log10(FDR)",title="MP1 GO:BP enrichment")
    sizes = [3,8,10,12]
    ax.legend(handles=[ax.scatter([],[],s=k*21,c="#2572ae",label=str(k)) for k in sizes],
              title="Count",frameon=False,fontsize=7)
    return fig


#Fig. 5c
def fig_5c(cells, mp_proportions):
    fig = plt.figure(figsize=(10,5.2),layout="constrained")
    gs = fig.add_gridspec(2,3,width_ratios=[1.6,1,1])
    ax = fig.add_subplot(gs[:,0])
    _umap(ax,cells,"program",MP,label=False,legend=True)
    ax.set_title("Meta-programs (NPC Xenium cohort)")
    for i,se in enumerate(SE):
        ax = fig.add_subplot(gs[i//2,1+i%2])
        values = mp_proportions.loc[se,list(MP)]
        ax.pie(values,colors=list(MP.values()),autopct="%1.1f%%",startangle=90,counterclock=False,
               wedgeprops={"edgecolor":"white"},textprops={"fontsize":8})
        ax.set_title(se,fontsize=9)
    return fig


#Fig. 5d
def fig_5d(km_steps, at_risk, logrank_p):
    fig,axes = _grid(width=5,height=4.8)
    _km(axes[0,0],km_steps,at_risk,"MP1 RNA splicing","Progression-free survival",logrank_p)
    return fig


#Fig. 5e
def fig_5e(progeny_scores):
    scores = progeny_scores.sort_values()
    fig,axes = _grid(width=4,height=5.2)
    ax = axes[0,0]
    norm = colors.CenteredNorm(vcenter=0,halfrange=max(abs(scores.min()),abs(scores.max()),1e-12))
    mapper = ScalarMappable(norm=norm,cmap="coolwarm")
    ax.barh(scores.index,scores,color=mapper.to_rgba(scores))
    ax.set(xlabel="Score",title="PROGENy (MP1 vs. MP2–5)")
    fig.colorbar(mapper,ax=ax,shrink=0.55)
    return fig


#Fig. 5f
def fig_5f(deseq, mp1_genes):
    fig,axes = _grid(1,2,9,4.9,gridspec_kw={"width_ratios":[1.7,1]})
    names = ["NLRP2","HDAC5","TRPA1","CD177","SEMA3D","PGF","SF3B1","CABP4","BCHE","BAD"]
    _volcano(axes[0,0],deseq,"log2FoldChange","padj",deseq.gene.where(deseq.gene.isin(names)),fold_cut=0.5)
    axes[0,0].set_xlabel("log₂ fold change · PT-NR ←  → PT-R")
    adjusted = pd.to_numeric(deseq.padj, errors="coerce")
    fold_change = pd.to_numeric(deseq.log2FoldChange, errors="coerce")
    up = set(deseq.loc[(adjusted < 0.05) & (fold_change > 0.5), "gene"])
    mp1 = set(mp1_genes)
    ax = axes[0,1]
    for center,radius,color in [((0,0.62),0.53,"#69ae67"),((0,-0.27),0.43,"#df6e70")]:
        ax.add_patch(Circle(center,radius,facecolor=color,alpha=0.72))
    ax.text(0,0.7,f"MP1 gene list\n{len(mp1-up)}",ha="center")
    ax.text(0,-0.33,f"PT-R up-DEGs\n{len(up-mp1)}",ha="center")
    ax.annotate(f"{len(up & mp1)}  "+", ".join(sorted(up & mp1)),(0,0.12),xytext=(0.24,0.19),
                fontsize=8,arrowprops={"arrowstyle":"-","color":"0.4"})
    ax.set(xlim=(-0.6,0.85),ylim=(-0.75,1.2),aspect="equal",title="Overlap with MP1")
    ax.set_axis_off()
    return fig


#Fig. 5i
def fig_5i(recurrence_counts, fisher_p):
    fig,axes = _grid(1,2,6.8,3.8)
    _stacked_recurrence(axes[0,0],recurrence_counts,"SF3B1",fisher_p)
    ax = axes[0,1]
    counts = recurrence_counts.loc[["Low","High"],["NR","R"]]
    ax.table(cellText=counts.to_numpy(),rowLabels=counts.index,colLabels=counts.columns,
             loc="center",cellLoc="center")
    ax.set_title("SF3B1 expression")
    ax.set_axis_off()
    return fig


#Fig. 5j
def fig_5j(km_steps, at_risk, logrank_p):
    fig,axes = _grid(width=4.2,height=4.5)
    _km(axes[0,0],km_steps,at_risk,"SF3B1 expression","LRRFS probability",logrank_p,
        ci=False,palette={"Low":"#15b3b4","High":"#ec625d"})
    return fig
