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


def _grid(rows=1, cols=1, width=5, height=3.5, **kwargs):
    return plt.subplots(rows, cols, figsize=(width, height), squeeze=False,
                        layout="constrained", **kwargs)


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


#Fig. S11a
def fig_s11a(cnv_by_celltype, cnv_by_group, chromosomes, cell_types, clinical_groups):
    fig,axes = _grid(2,1,14,5.6,gridspec_kw={"height_ratios":[3,1]})
    chrom = pd.Series(chromosomes).loc[cnv_by_celltype.columns].to_numpy()
    ends = np.r_[0,np.flatnonzero(chrom[1:]!=chrom[:-1])+1,len(chrom)]
    for i,(ax,table,annotations,preset) in enumerate(zip(axes.flat,[cnv_by_celltype,cnv_by_group],
                                                       [cell_types,clinical_groups],[MAJOR,GROUP])):
        table = table.loc[:,cnv_by_celltype.columns]
        tags = pd.Series(annotations).loc[table.index].replace({"PT_NR":"PT-NR","PT_R":"PT-R"})
        index = pd.Categorical(tags,categories=list(preset),ordered=True).argsort()
        tags = tags.iloc[index].to_numpy()
        im = ax.imshow(table.iloc[index].to_numpy(float),aspect="auto",interpolation="nearest",
                       cmap="bwr",vmin=-0.2,vmax=0.2,rasterized=True)
        for boundary in ends[1:-1]:
            ax.axvline(boundary-0.5,lw=0.55,c="#31638c")
        for boundary in np.flatnonzero(tags[:-1]!=tags[1:])+0.5:
            ax.axhline(boundary,lw=0.7,c="0.15")
        strip = ax.inset_axes([-0.012,0,0.01,1])
        strip.imshow(np.array([colors.to_rgba(preset[v]) for v in tags])[:,None,:],aspect="auto",interpolation="nearest")
        strip.set_axis_off()
        ax.set_yticks([])
        if i==0:
            ax.set_xticks((ends[:-1]+ends[1:]-1)/2,chrom[ends[:-1]],rotation=90,fontsize=7)
            ax.xaxis.tick_top()
        else:
            ax.set_xticks([])
        ax.legend(handles=[Patch(color=preset[k],label=k) for k in pd.unique(tags)],loc="center left",
                  bbox_to_anchor=(1,0.5),frameon=False,fontsize=7,ncol=2)
    fig.colorbar(im,ax=axes.flat[:].tolist(),location="bottom",shrink=0.4,label="Normalized copy-number deviation")
    return fig


#Fig. S11b
def fig_s11b(curves_by_program, risk_by_program, logrank_p):
    fig,axes = _grid(1,4,14,4.2)
    programs = {"MP_2":"MP2 Hypoxia","MP_3":"MP3 IFN-response","MP_4":"MP4 Cell cycling","MP_5":"MP5 EMT"}
    for ax,(key,title) in zip(axes.flat,programs.items()):
        _km(ax,curves_by_program[key],risk_by_program[key],title,"Progression-free survival",logrank_p[key])
    return fig
