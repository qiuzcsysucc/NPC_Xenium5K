#Fig.5a https://github.com/tiroshlab/3ca/blob/main/ITH_hallmarks/Generating_MPs/Generate_Meta_Programs.R


#Fig.5b
fig, ax = plt.subplots(figsize=(5,4))

if "X_umap" in adata.obsm:
    sc.pl.umap(
        adata,
        color="MP",
        show=False,
        ax=ax
    )

plt.tight_layout()
plt.savefig("Fig4b.pdf")
plt.close()


#Fig.5c
prop = (
    adata.obs
    .groupby("batch")["MP"]
    .value_counts(normalize=True)
    .rename("proportion")
    .reset_index()
)

fig, ax = plt.subplots(figsize=(6,4))

sns.boxplot(
    data=prop,
    x="MP",
    y="proportion",
    hue="group",
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=prop,
    x="MP",
    y="proportion",
    hue="group",
    dodge=True,
    ax=ax,
    size=3,
    legend=False
)

plt.tight_layout()
plt.savefig("Fig4c.pdf")
plt.close()


#Fig.5d
survival = pd.read_csv("MP1_survival.csv")

from lifelines import KaplanMeierFitter

fig, ax = plt.subplots(figsize=(5,4))

for label, group in survival.groupby("MP1_group"):
    km = KaplanMeierFitter()
    km.fit(
        group["time"],
        group["event"],
        label=label
    )
    km.plot(ax=ax)

plt.tight_layout()
plt.savefig("Fig5d.pdf")
plt.close()


#Fig.5e
dc.pl.barplot(data=pw_acts, name="MP1.vs.Others", figsize=(3, 3))


#Fig.5f

import adjustText as at
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from decoupler._Plotter import Plotter
from decoupler.pp.net import _validate_net


def volcano(
    data: pd.DataFrame,
    x: str,
    y: str,
    net: pd.DataFrame | None = None,
    name: str | None = None,
    thr_stat: float = 0.5,
    thr_sign: float = 0.05,
    max_stat: float | None = None,
    max_sign: float | None = None,
    color_pos: str = "#D62728",
    color_neg: str = "#1F77B4",
    color_null: str = "gray",
    highlight: str | list[str] | None = None,
    highlight_color: str = "orange",
    highlight_size: int = 60,
    figsize: tuple[float, float] | list[float] | None = None,
    dpi: int = 300,
    show: bool = True,
    expand_text: tuple[float, float] = (1.8, 2.0),
    expand_points: tuple[float, float] = (1.6, 1.8),
    only_move: dict | None = None,
    adjust_lim: int = 1000,
    adjust_precision: float = 0.01,
    show_group_marker: bool = False,
    group_left_label: str = "MP1_low",
    group_right_label: str = "MP1_high",
    group_marker_mode: str = "newline",
    group_marker_pad: float = 0.12,
    group_marker_color: str = "black",
    group_marker_fontsize: float | None = None,
    group_marker_bottom_margin: float | None = None,
    **kwargs,
) -> None | Figure:

    m = f"data must be a pd.DataFrame containing the columns {x} and {y}"
    assert isinstance(data, pd.DataFrame), m
    assert {x, y}.issubset(data.columns), m
    assert (net is None) == (name is None), "net and name must be both defined or both None"
    assert isinstance(thr_stat, (int, float)) and thr_stat > 0, "thr_stat must be numeric and > 0"
    assert isinstance(thr_sign, (int, float)) and thr_sign > 0, "thr_sign must be numeric and > 0"

    if max_stat is None:
        max_stat = np.inf
    if max_sign is None:
        max_sign = np.inf

    assert isinstance(max_stat, (int, float)) and max_stat > 0, "max_stat must be None, or numeric and > 0"
    assert isinstance(max_sign, (int, float)) and max_sign > 0, "max_sign must be None, or numeric and > 0"
    assert isinstance(color_pos, str), "color_pos must be str"
    assert isinstance(color_neg, str), "color_neg must be str"
    assert isinstance(color_null, str), "color_null must be str"

    if highlight is not None:
        assert isinstance(highlight, (str, list)), "highlight must be a str, list[str], or None"
        if isinstance(highlight, list):
            assert all(isinstance(g, str) for g in highlight), "highlight list must contain strings"

    if figsize is not None:
        assert (
            isinstance(figsize, (tuple, list))
            and len(figsize) == 2
            and all(isinstance(v, (int, float)) and v > 0 for v in figsize)
        ), "figsize must be a (width, height) tuple/list of positive numbers"

    assert isinstance(dpi, int) and dpi > 0, "dpi must be a positive integer"
    assert isinstance(expand_text, (tuple, list)) and len(expand_text) == 2, "expand_text must be length-2 tuple/list"
    assert isinstance(expand_points, (tuple, list)) and len(expand_points) == 2, "expand_points must be length-2 tuple/list"

    if only_move is None:
        only_move = {"points": "y", "text": "xy"}

    assert isinstance(only_move, dict), "only_move must be a dict like {'points':'y','text':'xy'}"
    assert isinstance(adjust_lim, int) and adjust_lim > 0, "adjust_lim must be positive int"
    assert isinstance(adjust_precision, (int, float)) and adjust_precision > 0, "adjust_precision must be positive number"
    assert group_marker_mode in ("newline", "under"), "group_marker_mode must be 'newline' or 'under'"

    bp = Plotter(**kwargs)

    if figsize is not None:
        bp.fig.set_size_inches(figsize[0], figsize[1], forward=True)

    bp.fig.set_dpi(dpi)

    if show_group_marker and group_marker_mode == "under" and (group_marker_bottom_margin is not None):
        bottom = float(group_marker_bottom_margin)

        if bottom > 1:
            fig_h_in = bp.fig.get_size_inches()[1]
            if fig_h_in <= 0:
                fig_h_in = 1.0
            bottom = bottom / fig_h_in

        top_now = bp.fig.subplotpars.top if bp.fig.subplotpars.top is not None else 0.9
        bottom = max(0.05, min(bottom, top_now - 0.05))
        bp.fig.subplots_adjust(bottom=bottom)

    thr_sign = -np.log10(thr_sign)

    df = data.copy()
    df["stat"] = df[x]

    nonzero = df[y] != 0
    non_zero_min = df[y][nonzero].min() if nonzero.any() else np.nextafter(0, 1)
    df["pval"] = -np.log10(df[y].clip(lower=non_zero_min, upper=1))

    if net is not None:
        vnet = _validate_net(net)
        snet = vnet[vnet["source"] == name]
        assert snet.shape[0] > 0, f'name={name} must be in net["source"]'
        df = df[df.index.isin(snet["target"])]

    msk_stat = np.abs(df["stat"]) < np.abs(max_stat)
    msk_sign = df["pval"] < np.abs(max_sign)
    df = df.loc[msk_stat & msk_sign]

    df["weight"] = color_null

    up_msk = (df["stat"] >= thr_stat) & (df["pval"] >= thr_sign)
    dw_msk = (df["stat"] <= -thr_stat) & (df["pval"] >= thr_sign)

    df.loc[up_msk, "weight"] = color_pos
    df.loc[dw_msk, "weight"] = color_neg

    df.plot.scatter(
        x="stat",
        y="pval",
        c="weight",
        sharex=False,
        ax=bp.ax
    )

    bp.ax.set_axisbelow(True)

    bp.ax.axvline(x=thr_stat, linestyle="--", color="black")
    bp.ax.axvline(x=-thr_stat, linestyle="--", color="black")
    bp.ax.axhline(y=thr_sign, linestyle="--", color="black")

    bp.ax.set_title(name)
    bp.ax.set_ylabel(rf"$-\log_{{10}}({y})$")

    marker_text = f"← {group_left_label}    {group_right_label} →"

    if show_group_marker and group_marker_mode == "newline":
        bp.ax.set_xlabel(f"{x}\n{marker_text}")
    else:
        bp.ax.set_xlabel(x)

        if show_group_marker and group_marker_mode == "under":
            fs = group_marker_fontsize or (bp.ax.xaxis.get_label().get_size() * 0.9)

            bp.ax.text(
                0.5,
                -abs(group_marker_pad),
                marker_text,
                transform=bp.ax.transAxes,
                ha="center",
                va="top",
                fontsize=fs,
                color=group_marker_color,
            )

            if group_marker_bottom_margin is None:
                top_now = bp.fig.subplotpars.top if bp.fig.subplotpars.top is not None else 0.9
                bottom_auto = min(0.22, top_now - 0.05)
                current_bottom = bp.fig.subplotpars.bottom if bp.fig.subplotpars.bottom is not None else 0.11

                if bottom_auto > current_bottom:
                    bp.fig.subplots_adjust(bottom=bottom_auto)

    texts = []

    if highlight is not None:
        genes = [highlight] if isinstance(highlight, str) else list(dict.fromkeys(highlight))
        genes_present = [g for g in genes if g in df.index]

        for g in genes_present:
            row = df.loc[g]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            bp.ax.scatter(
                row["stat"],
                row["pval"],
                s=highlight_size,
                facecolors="none",
                edgecolors=highlight_color,
                linewidths=1.5,
                zorder=3,
            )

            texts.append(
                bp.ax.text(
                    row["stat"],
                    row["pval"],
                    g,
                    color=highlight_color,
                    fontweight="bold"
                )
            )

    if len(texts) > 0:
        at.adjust_text(
            texts,
            arrowprops={"arrowstyle": "-", "color": "black"},
            ax=bp.ax,
            expand_text=tuple(expand_text),
            expand_points=tuple(expand_points),
            only_move=only_move,
            lim=adjust_lim,
            precision=adjust_precision,
        )

    if not show:
        return bp.fig

    return bp._return()



#Fig.5g graphpad


#Fig.5h IHC


#Fig.5j
expression = pd.read_csv("SF3B1_expression_survival.csv")

fig, ax = plt.subplots(figsize=(5,4))

for label, group in expression.groupby("SF3B1_group"):
    km = KaplanMeierFitter()
    km.fit(
        group["time"],
        group["event"],
        label=label
    )
    km.plot(ax=ax)

plt.tight_layout()
plt.savefig("Fig4j.pdf")
plt.close()
