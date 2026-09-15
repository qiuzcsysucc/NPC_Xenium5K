#Fig.5a biorender

#Fig.5b
fig, ax = plt.subplots(figsize=(6,5))

sc.pl.umap(
    adata,
    color="tumor_spatial_subtype",
    palette="tab10",
    show=False,
    ax=ax
)

plt.tight_layout()
plt.savefig("Fig5b.pdf")
plt.close()


#Fig.5c
score_matrix = (
    adata.obs
    .groupby("tumor_spatial_subtype")[celltype]
    .mean()
)

score_matrix = score_matrix.reindex(ss_order)

fig, ax = plt.subplots(figsize=(8,4))
sns.heatmap(
    score_matrix,
    cmap="RdBu_r",
    center=0,
    ax=ax
)

plt.tight_layout()
plt.savefig("Fig5c.pdf")
plt.close()


#Fig.5d, f, h, j, l
import numpy as np
import matplotlib.pyplot as plt


def plot_many_conds_to_one_exp_from_groupstats(
        group_stats: dict,
        group_name: str,
        exp: str,
        cond_list: list | None = None,
        topK: int | None = None,
        k_for_rank: int = 0,
        use_log2: bool = False,
        with_sem: bool = True,
        figsize=(4.5, 4),
        palette: list | None = None,
        legend_loc: str = "lower center"
):
    if group_name not in group_stats:
        raise KeyError(f"组 {group_name} 不在 group_stats 中。")

    st = group_stats[group_name]
    cats = list(st["cats"])
    if exp not in cats:
        raise KeyError(f"{exp} 不在该组的 cats 中。")

    j = cats.index(exp)
    x = st["radius_mid"]

    ykey = "log2_mean" if use_log2 else "mean"
    ekey = "log2_sem" if use_log2 else "sem"
    ylabel = "log2 fold-change" if use_log2 else "Co-occurrence ratio"
    baseline = 0.0 if use_log2 else 1.0

    mean_occ = st[ykey]
    sem_occ = st[ekey]

    if cond_list is None and topK is not None:
        if k_for_rank < 0 or k_for_rank >= len(x):
            raise IndexError("k_for_rank 超出半径分箱范围。")

        vals = np.asarray(mean_occ[:, j, k_for_rank], float).copy()

        if 0 <= j < vals.size:
            vals[j] = np.nan

        order = np.argsort(vals)[::-1]
        cond_list = [
            cats[i]
            for i in order
            if np.isfinite(vals[i])
        ][:topK]

    if not cond_list:
        raise ValueError("请提供 cond_list 或设置 topK。")

    cond_list = [
        c for c in cond_list
        if (c in cats) and (c != exp)
    ]

    if len(cond_list) == 0:
        raise ValueError("可用的 cond_list 为空（可能都不在 cats 中或等于 exp）。")

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    if palette is not None:
        ax.set_prop_cycle(color=list(palette))

    for cond in cond_list:
        i = cats.index(cond)
        y = np.asarray(mean_occ[i, j, :], float)

        ax.plot(
            x,
            y,
            lw=2,
            marker="o",
            label=f"{cond} ↔ {exp}"
        )

        if with_sem:
            ye = np.asarray(sem_occ[i, j, :], float)

            if np.isfinite(ye).any():
                ax.fill_between(
                    x,
                    y - ye,
                    y + ye,
                    alpha=0.25
                )

    ax.axhline(
        baseline,
        ls="--",
        lw=1,
        color="gray"
    )

    ax.set_xlabel("Radius")
    ax.set_ylabel(ylabel)

    top_txt = (
        f"Top{topK}"
        if (cond_list is None and topK is not None)
        else f"N={len(cond_list)}"
    )

    ax.set_title(
        f"{top_txt} co-occurrence to {exp} ({group_name})"
    )

    ax.legend(
        ncol=1,
        fontsize=9,
        loc=legend_loc,
        bbox_to_anchor=(1, 1.0)
    )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, ax


def combine_all_slices(group_stats: dict, eps: float = 1e-9):
    stacks = []
    st0 = None

    for g, st in group_stats.items():
        arr = st.get("per_slice", None)

        if arr is None:
            continue

        if st0 is None:
            st0 = st

        stacks.append(arr.astype(float))

    if len(stacks) == 0:
        raise ValueError("没有找到可用的 per_slice，无法合并。")

    arr_all = np.concatenate(
        stacks,
        axis=0
    )

    cats = list(st0["cats"])
    radius_mid = st0["radius_mid"]

    mean = np.nanmean(
        arr_all,
        axis=0
    )

    if arr_all.shape[0] > 1:
        sem = (
            np.nanstd(
                arr_all,
                axis=0,
                ddof=1
            ) / np.sqrt(arr_all.shape[0])
        )
    else:
        sem = np.zeros_like(mean)

    log2_arr = np.log2(
        np.maximum(arr_all, eps)
    )

    log2_mean = np.nanmean(
        log2_arr,
        axis=0
    )

    if arr_all.shape[0] > 1:
        log2_sem = (
            np.nanstd(
                log2_arr,
                axis=0,
                ddof=1
            ) / np.sqrt(arr_all.shape[0])
        )
    else:
        log2_sem = np.zeros_like(log2_mean)

    combined_stats = {
        "ALL": dict(
            mean=mean,
            sem=sem,
            log2_mean=log2_mean,
            log2_sem=log2_sem,
            radius_mid=radius_mid,
            cats=cats,
            per_slice=arr_all
        )
    }

    return combined_stats, arr_all.shape[0]


def _recompute_from_per_slice(st, eps=1e-9):
    arr = st["per_slice"]

    mean = np.nanmean(
        arr,
        axis=0
    )

    if arr.shape[0] > 1:
        sem = (
            np.nanstd(
                arr,
                axis=0,
                ddof=1
            ) / np.sqrt(arr.shape[0])
        )
    else:
        sem = np.zeros_like(mean)

    log2_arr = np.log2(
        np.maximum(arr, eps)
    )

    log2_mean = np.nanmean(
        log2_arr,
        axis=0
    )

    if log2_arr.shape[0] > 1:
        log2_sem = (
            np.nanstd(
                log2_arr,
                axis=0,
                ddof=1
            ) / np.sqrt(log2_arr.shape[0])
        )
    else:
        log2_sem = np.zeros_like(log2_mean)

    st["mean"] = mean
    st["sem"] = sem
    st["log2_mean"] = log2_mean
    st["log2_sem"] = log2_sem


#Fig.5e, g, i, k, m
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

def plot_cell_black_subset(
    adata,
    group=None,
    subset=None,
    col_map=None,
    boundary_col='boundary',
    cell_edge_width=1,
    cell_face_alpha=1,
    show_scalebar=True,
    scale_length_um=100.0,
    max_cell_types_legend=12,
    figure_size=(6, 6),
    dpi=300,
    title=None,
    non_focus_color=(0.2, 0.2, 0.2, 0.8),
    non_focus_label="Non-focus",
    output=None
):

    def _as_rgba(color_like, alpha_mult=1.0):
        r, g, b, a = to_rgba(color_like)
        a = max(0.0, min(1.0, a * alpha_mult))
        return (r, g, b, a)

    obs = adata.obs.copy()
    if boundary_col not in obs.columns:
        raise ValueError(f"adata.obs 中找不到 '{boundary_col}' 列。")

    fig, ax_main = plt.subplots(figsize=figure_size, dpi=dpi)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.18)

    if title is not None:
        fig.suptitle(title, fontsize=14, color="black")

    ax_main.set_facecolor("black")

    xs_all, ys_all = [], []

    has_group = (group is not None) and (group in obs.columns)
    use_subset = (subset is not None) and has_group

    non_focus_rgba = _as_rgba(
        non_focus_color,
        alpha_mult=cell_face_alpha
    )

    default_rgba = _as_rgba(
        (0.7, 0.7, 0.7, 0.9),
        alpha_mult=cell_face_alpha
    )

    for cell_id, row in obs.iterrows():
        b = row[boundary_col]

        if pd.isna(b):
            continue

        try:
            coords = ast.literal_eval(b) if isinstance(b, str) else b
        except Exception:
            continue

        if coords is None or len(coords) == 0:
            continue

        xy = np.asarray(coords)

        xs_all.extend(xy[:, 0])
        ys_all.extend(xy[:, 1])

        if has_group:
            gval = row[group]

            if use_subset:
                if (
                    (col_map is not None)
                    and (gval in subset)
                    and (gval in col_map)
                ):
                    face_rgba = _as_rgba(
                        col_map[gval],
                        alpha_mult=cell_face_alpha
                    )
                else:
                    face_rgba = non_focus_rgba
            else:
                if (
                    (col_map is not None)
                    and (gval in col_map)
                ):
                    face_rgba = _as_rgba(
                        col_map[gval],
                        alpha_mult=cell_face_alpha
                    )
                else:
                    face_rgba = default_rgba
        else:
            face_rgba = default_rgba

        patch = Polygon(
            xy,
            closed=True,
            fill=True,
            facecolor=face_rgba,
            edgecolor="black",
            linewidth=cell_edge_width,
            zorder=2
        )

        ax_main.add_patch(patch)

    ax_main.set_aspect("equal")

    if len(xs_all) == 0 or len(ys_all) == 0:
        ax_main.axis("off")

        if output is not None:
            fig.savefig(output, facecolor="white")
            plt.close(fig)
        else:
            plt.show()

        return

    x_min, x_max = min(xs_all), max(xs_all)
    y_min, y_max = min(ys_all), max(ys_all)

    width_um = max(x_max - x_min, 1e-6)
    height_um = max(y_max - y_min, 1e-6)

    bg_rect = Rectangle(
        (x_min, y_min),
        width_um,
        height_um,
        facecolor="black",
        edgecolor="none",
        zorder=0
    )

    ax_main.add_patch(bg_rect)

    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_max, y_min)
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    for spine in ax_main.spines.values():
        spine.set_visible(False)

    fig.canvas.draw()

    mid_y = 0.5 * (y_min + y_max)

    disp_left = ax_main.transData.transform(
        (x_min, mid_y)
    )

    disp_right = ax_main.transData.transform(
        (x_max, mid_y)
    )

    left_fig, _ = fig.transFigure.inverted().transform(
        disp_left
    )

    right_fig, _ = fig.transFigure.inverted().transform(
        disp_right
    )

    mid_x = 0.5 * (x_min + x_max)

    disp_bottom = ax_main.transData.transform(
        (mid_x, y_max)
    )

    _, bottom_fig = fig.transFigure.inverted().transform(
        disp_bottom
    )

    base_y = max(bottom_fig - 0.04, 0.02)
    scale_y = min(base_y + 0.02, bottom_fig - 0.01)

    legend_offset = 0.02
    legend_y = scale_y + legend_offset

    bar_start = left_fig
    bar_end = left_fig + 0.0

    if show_scalebar and scale_length_um > 0:
        frac = min(scale_length_um / width_um, 0.6)

        bar_start = left_fig
        bar_end = left_fig + frac * (right_fig - left_fig)

        line = Line2D(
            [bar_start, bar_end],
            [scale_y, scale_y],
            transform=fig.transFigure,
            color="black",
            linewidth=2.0,
            solid_capstyle="butt"
        )

        fig.add_artist(line)

        fig.text(
            (bar_start + bar_end) / 2.0,
            scale_y - 0.015,
            f"{int(scale_length_um)} µm",
            transform=fig.transFigure,
            ha="center",
            va="top",
            fontsize=10,
            color="black"
        )
    else:
        bar_end = left_fig + 0.4 * (right_fig - left_fig)

    handles_cell = []

    if has_group:
        present_types = list(pd.unique(obs[group]))

        if use_subset:
            focus_types = [
                ct for ct in subset
                if ct in present_types
            ]

            if (
                max_cell_types_legend is not None
                and max_cell_types_legend > 1
            ):
                focus_types = focus_types[
                    :max_cell_types_legend - 1
                ]

            non_focus_types = [
                ct for ct in present_types
                if ct not in focus_types
            ]
        else:
            focus_types = present_types
            non_focus_types = []

        for ct in focus_types:
            if col_map is not None and ct in col_map:
                c = _as_rgba(
                    col_map[ct],
                    alpha_mult=cell_face_alpha
                )
            else:
                c = default_rgba

            h = Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor=c,
                markeredgecolor="black",
                markeredgewidth=0.3,
                label=str(ct)
            )

            handles_cell.append(h)

        if use_subset and len(non_focus_types) > 0:
            h_nf = Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor=non_focus_rgba,
                markeredgecolor="black",
                markeredgewidth=0.3,
                label=str(non_focus_label)
            )

            handles_cell.append(h_nf)

    if handles_cell:
        legend_x = (
            bar_end
            + 0.02 * (right_fig - left_fig)
        )

        leg_cell = fig.legend(
            handles=handles_cell,
            loc="upper left",
            bbox_to_anchor=(legend_x, legend_y),
            bbox_transform=fig.transFigure,
            frameon=False,
            fontsize=7,
            ncol=2,
            columnspacing=0.8,
            handletextpad=0.4,
            borderpad=0.2
        )

        for text in leg_cell.get_texts():
            text.set_color("black")

    if output is not None:
        fig.savefig(
            output,
            dpi=dpi,
            facecolor="white"
        )
        plt.show()
    else:
        plt.show()


custom_color_mapping = {
    'SS1-Tum': '#e41a1c',
    'SS2-Tum': '#4daf4a',
    'SS3-Tum': '#ff7f00',
    'SS4-Tum': '#a65628',
    'SS5-Tum': '#999999',

    'NEU_NETs_IL1B': '#fdb462',
    'TAM_MMP12': '#80b1d3',
    'TAM_FCGBP': '#01665e',
    'DC_Langerhans_CD1A': '#5e3c99',
    'Fibro_iCAF-2_VEGFA': '#b35806',

    'CD8T_ISG_IFIT3': '#8da0cb',
    'CD8T_Tex_HAVCR2': '#984ea3',
    'TAM_CXCL9': '#66c2a5',
    'CD4T_Tn_GIMAP4': '#e6ab02',
    'TAM_MX1': '#1b9e77',

    'CD4T_eTreg_CCR8': '#e7298a',
    'TAM_MKI67': '#08519c',
    'CD4T_Tanergy_EGR3': '#a6761d',

    'Fibro_myCAF-1_TGFB3': '#7b3294',
    'Fibro_myCAF-2_CST1': '#d01c8b',
    'Fibro_myCAF-3_CCN2': '#4d9221',
    'Fibro_proliferation_MKI67': '#8073ac',
    'Fibro_senescence_MDM2': '#543005',

    'Pericyte_universal_RGS5': '#2166ac',
    'Pericyte_ECM_THBS1': '#762a83',
    'Pericyte_senescence_MDM2': '#bf812d',
    'EC_capillary_IVNS1ABP': '#35978f',
    'Pericyte_proliferation_MKI67': '#f781bf',
}


center_x = 8519
center_y = 2239

half_size = 200.0

x_min = center_x - half_size
x_max = center_x + half_size
y_min = center_y - half_size
y_max = center_y + half_size

print("ROI x范围:", x_min, x_max)
print("ROI y范围:", y_min, y_max)

cells_roi_mask = (
    (spyrrow_adata.obs["x_centroid"] >= x_min)
    & (spyrrow_adata.obs["x_centroid"] <= x_max)
    & (spyrrow_adata.obs["y_centroid"] >= y_min)
    & (spyrrow_adata.obs["y_centroid"] <= y_max)
)

adata_roi = spyrrow_adata[cells_roi_mask].copy()

print("ROI 内细胞数量:", adata_roi.n_obs)

tr = spyrrow_adata.uns["transcripts"]

tr_roi_mask = (
    (tr["x_location"] >= x_min)
    & (tr["x_location"] <= x_max)
    & (tr["y_location"] >= y_min)
    & (tr["y_location"] <= y_max)
)

tr_roi = tr[tr_roi_mask].copy()

print("ROI 内转录本数量:", tr_roi.shape[0])

adata_roi.uns["transcripts"] = tr_roi


plot_cell_black_subset(
    adata=adata_roi,
    group='cell_type',
    subset=[
        'SS1-Tum',
        'Fibro_myCAF-1_TGFB3',
        'Fibro_myCAF-2_CST1',
        'Fibro_myCAF-3_CCN2',
        'Fibro_proliferation_MKI67',
        'Fibro_senescence_MDM2'
    ],
    col_map=custom_color_mapping,
    boundary_col='boundary',
    cell_edge_width=1,
    cell_face_alpha=1,
    show_scalebar=True,
    scale_length_um=100,
    figure_size=(4, 5),
    title='SS1-Tum and top 5 co-occurring types\n(PTR_10)',
    output='ss1_xe.pdf'
)



#Fig.5n
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statannotations.Annotator import Annotator
prop_df = (adata.obs
           .groupby('batch')['cell_type']
           .value_counts(normalize=True)
           .rename("proportion")
           .reset_index())

conditions = [
    prop_df['batch'].str.startswith('F'),
    prop_df['batch'].str.startswith('P'),
    prop_df['batch'].str.startswith('R'),
]
choices = ['PT_NR', 'PT_R', 'RT']
prop_df['group'] = np.select(conditions, choices, default=np.nan)
group_order = ['PT_NR', 'PT_R', 'RT']

def make_pair_id(s):
    if isinstance(s, str) and len(s) >= 2:
        return s[1:]      return np.nan

prop_df["pair_id"] = prop_df["batch"].apply(make_pair_id)


dfp = prop_df.loc[:, ["batch", "pair_id", "cell_type", "group", "proportion"]].copy()
dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()

x_order = [
    "MP2_G2/M",
    "MP3_Squamous",
    "MP5_Immune-invasion",
    "MP6_S-phase",
    "MP7_RNA-splicing",
    "MP8_IFN-response",
    "MP9_Hypoxia",
    "MP12_Progenitor",
    "MP13_Senescence-like",
]
x_order = [x for x in x_order if x in set(dfp["cell_type"])]
hue_order = [g for g in group_order if g in dfp["group"].unique().tolist()]

plt.figure(figsize=(8, 6))
ax = sns.boxplot(
    data=dfp,
    x="cell_type",
    y="proportion",
    hue="group",
    order=x_order,
    hue_order=hue_order,
    palette=["#4ea64a", "#3777ac", "#d5231d"][:len(hue_order)],
    showfliers=False
)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Proportion")
plt.xlabel("cell_type")
plt.tight_layout()

sns.stripplot(
    data=dfp,
    x="cell_type",
    y="proportion",
    hue="group",
    order=x_order,
    hue_order=hue_order,
    dodge=True,
    jitter=False,                     alpha=0.7,
    size=4,
    linewidth=0.3,
    edgecolor="black",
    palette=["black"] * len(hue_order),
    ax=ax,
    legend=False,                )

pairs, labels, pvalues = [], [], []

for ct in x_order:
    sub = dfp[dfp["cell_type"] == ct]
    if set(['PT_NR', 'PT_R']).issubset(set(sub["group"].unique())):
        x1 = sub.loc[sub["group"] == 'PT_NR', "proportion"].values
        x2 = sub.loc[sub["group"] == 'PT_R',  "proportion"].values
        if (len(x1) >= 1) and (len(x2) >= 1):
            try:
                mw = stats.mannwhitneyu(x1, x2, alternative='two-sided', method='auto')
                p_mw = float(mw.pvalue)
                pairs.append(((ct, 'PT_NR'), (ct, 'PT_R')))
                labels.append(f"{p_mw:.3f}")
                pvalues.append(p_mw)
            except Exception:
                pass
    sub_pr = sub[sub["group"].isin(['PT_R', 'RT'])].dropna(subset=["pair_id"]).copy()
    if not sub_pr.empty:
        piv = sub_pr.pivot_table(index="pair_id", columns="group",
                                 values="proportion", aggfunc='mean')
        piv = piv.dropna(subset=['PT_R', 'RT'], how='any')
        if piv.shape[0] >= 1:
            try:
                w = stats.wilcoxon(piv['PT_R'].values, piv['RT'].values,
                                   alternative='two-sided', zero_method='wilcox')
                p_w = float(w.pvalue)
                pairs.append(((ct, 'PT_R'), (ct, 'RT')))
                labels.append(f"{p_w:.3f}")
                pvalues.append(p_w)
            except Exception:
                pass
annotator = Annotator(
    ax,
    pairs,
    data=dfp,
    x="cell_type",
    y="proportion",
    hue="group",
    order=x_order,
    hue_order=hue_order,
)
annotator.configure(
    test=None,
    show_test_name=False,
    comparisons_correction=None,
    loc="inside",
    line_width=1.2
)
annotator.set_custom_annotations(labels)
annotator.annotate()
def get_xpos(ct, grp, x_order, hue_order, width=0.8):
    i = x_order.index(ct)
    j = hue_order.index(grp)
    n = len(hue_order)
    return i - width/2 + (j + 0.5) * width / n

for ct in x_order:
    sub = dfp[dfp["cell_type"] == ct]
    sub_pr = sub[sub["group"].isin(['PT_R', 'RT'])].dropna(subset=["pair_id"]).copy()
    if sub_pr.empty:
        continue
    piv = sub_pr.pivot_table(index="pair_id", columns="group",
                             values="proportion", aggfunc="mean")
    piv = piv.dropna(subset=['PT_R', 'RT'], how='any')
    if piv.shape[0] == 0:
        continue

    x_pr = get_xpos(ct, 'PT_R', x_order, hue_order)
    x_rt = get_xpos(ct, 'RT',   x_order, hue_order)

    for pid, row in piv.iterrows():
        y1, y2 = float(row['PT_R']), float(row['RT'])
        ax.plot([x_pr, x_rt], [y1, y2],
                color="grey", linewidth=1.2, alpha=0.9, zorder=5)
plt.xlabel('Metaprograms')
plt.legend(title="Group", bbox_to_anchor=(1, 1), loc="upper right")
plt.tight_layout()
plt.savefig('Proportion_MPtumor_with_points_pairs.pdf', bbox_inches='tight')
plt.show()

#Fig.5o
mp7 = pd.read_csv("MP7_ternary_fraction.csv")

fig, axes = plt.subplots(
    1,
    4,
    figsize=(12,3)
)

for ax, radius in zip(
    axes,
    ["20μm","40μm","60μm","80μm"]
):
    sub = mp7[mp7["radius"] == radius]

    sns.boxplot(
        data=sub,
        x="group",
        y="MP7_fraction",
        ax=ax,
        showfliers=False
    )

    sns.stripplot(
        data=sub,
        x="group",
        y="MP7_fraction",
        ax=ax,
        size=3
    )

    ax.set_title(radius)

plt.tight_layout()
plt.savefig("Fig5o.pdf")
plt.close()
