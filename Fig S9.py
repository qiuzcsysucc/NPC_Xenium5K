import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


GROUP = {"PT-NR": "#719d60", "PT-R": "#537da1", "RT": "#c7533e"}


RADII = (60, 70, 90, 100)


def _grid(rows=1, cols=1, width=5, height=3.5, **kwargs):
    return plt.subplots(rows, cols, figsize=(width, height), squeeze=False,
                        layout="constrained", **kwargs)


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


#Fig. S9a
def fig_s9a(scaled_by_radius, lineage_by_state):
    fig,axes = _grid(4,1,16,8)
    states = scaled_by_radius[RADII[0]].columns
    for i,(ax,radius) in enumerate(zip(axes.flat,RADII)):
        table = scaled_by_radius[radius].loc[:,states]
        im = _heat(ax,table,"Scaled proportion",(0,1),bar=False)
        if i<3:
            ax.tick_params(labelbottom=False)
        if i==0:
            _bands(ax,[lineage_by_state[s] for s in states])
        ax.set_ylabel(f"{radius} μm",fontsize=8)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.5,label="Scaled proportion")
    return fig


#Fig. S9b
def fig_s9b(specimens, p_values_by_radius=None):
    fig,axes = _grid(2,2,12,7)
    for ax,radius in zip(axes.flat,RADII):
        data = specimens.loc[specimens.radius_um.eq(radius)]
        p = None if p_values_by_radius is None else p_values_by_radius[radius]
        _boxes(ax,data,["SE1","SE2","SE3","SE4"],"Tumor subtype proportion (%)",p,scale=100)
        ax.set_title(f"{radius} μm")
    return fig


#Fig. S9c
def fig_s9c(curves_by_state, risk_by_state, one_sided_p):
    states = ["TAM_FCGBP","CD4T_eTreg_TNFRSF9","Fibro_myCAF_2_CST1","Fibro_iCAF_2_VEGFA",
              "CD8T_Tex_HAVCR2","CD8T_Trm_ITGAE","TAM_MMP12","NEU_NETs_IL1B",
              "SE1_immune_ashy","SE2_immune_insulated","SE3_immune_inflamed","SE4_immune_cold"]
    fig,axes = _grid(3,4,16,11.5)
    for ax,state in zip(axes.flat,states):
        _km(ax,curves_by_state[state],risk_by_state[state],state,"PFS",one_sided_p[state],ci=False,legend=False)
    fig.legend(handles=[Line2D([],[],color="#3478ab",label="Low abundance"),
                        Line2D([],[],color="#e65035",label="High abundance")],
               loc="outside upper center",ncol=2,frameon=False)
    return fig
