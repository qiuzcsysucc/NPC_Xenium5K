import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, leaves_list


GROUP = {"PT-NR": "#719d60", "PT-R": "#537da1", "RT": "#c7533e"}


LINEAGES = ["Macro_FCGBP", "Macro_SPP1", "Macro_APOE", "Mono_FCN1+FCGR3A+"]


MACRO = {"Macro_FCGBP": "#ff7f0e", "Macro_SPP1": "#8c564b", "Macro_APOE": "#1f77b4",
         "Mono_FCN1+FCGR3A+": "#e377c2", "Mono_FCN1+SELL+": "#17becf",
         "Mono_FCN1+IL1B+": "#bcbd22", "Macro_NR4A3": "#d62728"}


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


def _roc(ax, curves, legend=True):
    styles = ["-","--","-.",":","--","-.","-",":","--",":"]
    for i,(name,part) in enumerate(curves.groupby("model",sort=False,observed=True)):
        part = part.sort_values(["fpr","tpr"],kind="stable")
        ax.step(part.fpr,part.tpr,where="post",lw=1.7,ls=styles[i % len(styles)],
                label=f"{name} (AUC = {part.auc.iloc[0]:.3f})")
    ax.plot([0,1],[0,1],"--",lw=0.8,color="0.55",label="Chance (AUC = 0.500)")
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel="1 − specificity",ylabel="Sensitivity",aspect="equal")
    if legend:
        ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.18),frameon=False,fontsize=7,ncol=2)


def _clinical_boxes(specimens, features, titles, rows, cols, ylabel, p_values):
    fig,axes = _grid(rows,cols,cols*3.0,rows*3.5)
    for ax,feature,title in zip(axes.flat,features,titles):
        _boxes(ax,specimens,[feature],ylabel,p_values,group_col="Group_Label",pair_col="Pair_ID")
        ax.set_xticks([-0.24,0,0.24],list(GROUP),fontsize=8)
        ax.set_title(title,fontsize=9)
        ax.get_legend().remove()
    return fig


#Fig. 7a
def fig_7a(scaled_mean, fraction_pct):
    genes = ["TYRO3","AXL","MERTK","FCGBP","CTNND1","NDRG2","MKI67","TOP2A","PCNA"]
    groups = dict(zip(genes,np.repeat(["GAS6–TAM","TAM_FCGBP","TAM_MKI67"],3)))
    fig,axes = _grid(width=5.8,height=4.7)
    _dot(axes[0,0],scaled_mean.loc[:,genes],fraction_pct,groups,size_ticks=(10,20,30,40,50))
    return fig


#Fig. 7b
def fig_7b(cells):
    fig,axes = _grid(width=5,height=4.8)
    _umap(axes[0,0],cells,"state",MACRO)
    axes[0,0].set_title("Mono/Macro (public scRNA-seq cohort)")
    return fig


#Fig. 7c
def fig_7c(scaled_mean, fraction_pct):
    genes = ["CD14","FCN1","SELL","FCGR3A","IL1B","APOE","SPP1","NR4A3","FCGBP","CTNND1","NDRG2","TYRO3","AXL","MERTK"]
    groups = dict(zip(genes,np.repeat(["Monocyte","Macrophage","GAS6–TAM"],[5,6,3])))
    fig,axes = _grid(width=7.5,height=5)
    _dot(axes[0,0],scaled_mean.loc[:,genes],fraction_pct,groups)
    return fig


#Fig. 7d
def fig_7d(cells, velocity_grid, branch_ribbons):
    fig,axes = _grid(1,4,15.5,4.2)
    for idx,key,cmap,title in [(0,"cytotrace2","plasma","CytoTRACE2 relative"),(2,"pseudotime","viridis_r","Pseudotime")]:
        ax = axes[0,idx]
        dots = ax.scatter(cells.umap1,cells.umap2,c=cells[key],s=1,cmap=cmap,rasterized=True,lw=0)
        fig.colorbar(dots,ax=ax,shrink=0.65)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_axis_off()
    _umap(axes[0,1],cells,"state",MACRO)
    axes[0,1].set_title("TAM subtypes")
    for ax in axes[0,1:3]:
        ax.streamplot(velocity_grid["x"],velocity_grid["y"],velocity_grid["u"],velocity_grid["v"],
                      density=0.85,color="#333333",arrowsize=0.65,linewidth=0.45)
    ax = axes[0,3]
    for name,ribbon in branch_ribbons.groupby("state",sort=False):
        ribbon = ribbon.sort_values("pseudotime")
        ax.fill_between(ribbon.pseudotime.to_numpy(float),ribbon.lower.to_numpy(float),
                        ribbon.upper.to_numpy(float),color=MACRO[name])
        k = (ribbon.upper-ribbon.lower).to_numpy().argmax()
        anchor = ribbon.iloc[k]
        ax.text(anchor.pseudotime,(anchor.lower+anchor.upper)/2,name,ha="center",color=MACRO[name],
                fontsize=7,path_effects=[pe.withStroke(linewidth=2,foreground="white")])
    ax.set(title="Trajectory",xlabel="Pseudotime",xticks=[0,1],yticks=[])
    ax.spines[["left","right","top"]].set_visible(False)
    return fig


#Fig. 7e
def fig_7e(smoothed_expression):
    genes = ["AXL","MERTK","IL18","ICOSLG","PVR","PDCD1LG2","CXCL16","FCGBP"]
    fig,axes = _grid(2,4,11,5.7)
    for ax,gene in zip(axes.flat,genes):
        data = smoothed_expression.loc[smoothed_expression.gene.eq(gene)]
        for state,col in zip(LINEAGES,["#21295d","#2cbdf5","#f5cf00","#921c18"]):
            curve = data.loc[data.lineage.eq(state)].sort_values("pseudotime")
            ax.plot(curve.pseudotime,curve.expression,color=col,label=state)
        ax.set_title(gene,fontsize=9)
        ax.set_xticks([0,1])
        ax.spines[["top","right"]].set_visible(False)
    fig.supxlabel("Pseudotime")
    fig.supylabel("Expression")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=4,fontsize=8,frameon=False)
    return fig


#Fig. 7g
def fig_7g(specimens, p_values=None):
    radii = [20,40,60,80]
    return _clinical_boxes(specimens,[f"ternary_ratio_{r}um" for r in radii],
                           [f"RAIN ratio ({r} μm)" for r in radii],2,2,"Ratio",p_values)


#Fig. 7h
def fig_7h(roc_curves):
    fig,axes = _grid(width=6.2,height=5.2)
    _roc(axes[0,0],roc_curves)
    axes[0,0].set_title("ROC curves for predicting recurrence")
    return fig


#Fig. 7i
def fig_7i(curves_by_marker, risk_by_marker, one_sided_p):
    fig,axes = _grid(3,1,5.5,11.8)
    for ax,marker in zip(axes.flat,["SF3B1+Tumor","TNFRSF9+Treg","FCGBP+TAM"]):
        _km(ax,curves_by_marker[marker],risk_by_marker[marker],marker,"LRFS probability",one_sided_p[marker])
    return fig


#Fig. 7j
def fig_7j(km_steps, at_risk, two_sided_p):
    fig,axes = _grid(width=5.2,height=4.8)
    _km(axes[0,0],km_steps,at_risk,"RAIN ratio (40 μm)","LRFS probability",two_sided_p)
    return fig


#Fig. 7k
def fig_7k(cox_results):
    fig,axes = _grid(1,2,8,3.4,gridspec_kw={"width_ratios":[1.6,1]})
    ax,table = axes[0]
    y = np.arange(len(cox_results))
    hr,lo,hi = (cox_results[k].to_numpy(float) for k in ("HR","lower","upper"))
    ax.errorbar(hr,y,xerr=[hr-lo,hi-hr],fmt="s",ms=4,color="#277ab5",capsize=2)
    ax.axvline(1,ls="--",color="#277ab5",lw=1)
    ax.set(xscale="log",xlabel="Adjusted hazard ratio",ylim=(len(y)-0.5,-0.5))
    ax.set_yticks(y,cox_results.variable,fontsize=8)
    ax.set_xticks([0.5,1,2],["0.5","1","2"])
    table.set(xlim=(0,1),ylim=ax.get_ylim())
    table.set_axis_off()
    table.set_title("HR (95% CI)                 P",fontsize=9)
    for i,r in enumerate(cox_results.itertuples()):
        table.text(0,i,f"{r.HR:.2f} ({r.lower:.2f}–{r.upper:.2f})",va="center",fontsize=8)
        table.text(0.95,i,_ptext(r.p),ha="right",va="center",fontsize=8)
    fig.suptitle("Multivariable Cox regression analysis")
    return fig
