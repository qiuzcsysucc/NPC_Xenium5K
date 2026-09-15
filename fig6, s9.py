#Fig.6a
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import cm, colors
from matplotlib.lines import Line2D
from typing import Optional, Union, List


cmap_own = cm.get_cmap("magma_r", 256)
newcolors = cmap_own(np.linspace(0, 0.75, 256))

greys_cmap = cm.get_cmap("Greys_r", 256)
newcolors[:10, :] = greys_cmap(
    np.linspace(0.8125, 0.8725, 10)
)

pos_cmap = colors.ListedColormap(newcolors)


def _compute_bubble_sizes_log10(
    p_values: np.ndarray,
    bubble_size: float,
    n_permutations: int = 1000
) -> np.ndarray:

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    min_p = 1.0 / (n_permutations + 1)

    p_clipped = np.clip(
        p_values,
        min_p,
        1.0
    )

    neg_log10_p = -np.log10(p_clipped)
    max_neg_log10 = -np.log10(min_p)

    if max_neg_log10 <= 0:
        return np.full(
            len(p_values),
            bubble_size
        )

    normalized = neg_log10_p / max_neg_log10

    sizes = bubble_size * (
        0.1 + 0.9 * normalized
    )

    return sizes


def _create_pvalue_legend_on_ax(
    legend_ax,
    bubble_size: float,
    n_permutations: int = 1000,
    title: str = r"$P$ adjust",
    title_fontsize: int = 12,
    label_fontsize: int = 10
):

    legend_ax.axis("off")

    min_p = 1.0 / (n_permutations + 1)
    max_neg_log10 = -np.log10(min_p)

    candidate_p_values = [
        min_p,
        0.001,
        0.01,
        0.05,
        0.10,
        1.00
    ]

    p_values_legend = sorted({
        p for p in candidate_p_values
        if min_p <= p <= 1
    })

    legend_handles = []

    for p_value in p_values_legend:

        neg_log10_p = -np.log10(p_value)

        if max_neg_log10 > 0:
            normalized = (
                neg_log10_p / max_neg_log10
            )
        else:
            normalized = 1

        size = bubble_size * (
            0.1 + 0.9 * normalized
        )

        if np.isclose(p_value, min_p):
            label = (
                rf"$P$ ≤ {min_p:.3f}"
            )
        elif p_value < 0.01:
            label = (
                rf"$P$ = {p_value:.3f}"
            )
        else:
            label = (
                rf"$P$ = {p_value:.2f}"
            )

        handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="white",
            markerfacecolor="gray",
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=np.sqrt(size),
            label=label
        )

        legend_handles.append(handle)

    legend = legend_ax.legend(
        handles=legend_handles,
        title=title,
        loc="upper left",
        bbox_to_anchor=(0, 1),
        frameon=False,
        borderaxespad=0,
        labelspacing=1.2,
        handletextpad=1.2,
        fontsize=label_fontsize
    )

    legend.get_title().set_fontsize(
        title_fontsize
    )

    return legend


def plotCCCDotPlot_on_ax(
    laris_results: pd.DataFrame,
    interactions_to_plot: List[str],
    ax,
    cbar_ax,
    legend_ax,
    senders: Optional[List[str]] = None,
    receivers: Optional[List[str]] = None,
    sender_receiver_pairs: Optional[List[str]] = None,
    delimiter_pair: str = "-->",
    n_top: int = 3000,
    cmap: Union[str, colors.Colormap] = None,
    bubble_size: float = 500,
    p_value_col: str = "p_value_fdr",
    threshold: float = 0.05,
    filter_by_interaction_score: bool = True,
    threshold_interaction_score: float = 0.01,
    filter_significant: bool = True,
    n_permutations: int = 1000,
    legend_fontsize: int = 12,
    tick_fontsize: int = 10,
    show_grid: bool = True,
    title: Optional[str] = None,
    panel_label: Optional[str] = None
):

    if cmap is None:
        cmap = pos_cmap

    required_columns = {
        "sender",
        "receiver",
        "interaction_name",
        "interaction_score"
    }

    missing_columns = (
        required_columns
        - set(laris_results.columns)
    )

    if missing_columns:
        raise ValueError(
            "laris_results缺少必要列："
            + ", ".join(sorted(missing_columns))
        )

    if sender_receiver_pairs is not None:

        senders = []
        receivers = []

        for pair in sender_receiver_pairs:

            parts = pair.split(
                delimiter_pair
            )

            if len(parts) != 2:
                raise ValueError(
                    f"无法解析细胞对：{pair}\n"
                    f"正确格式应为：sender"
                    f"{delimiter_pair}receiver"
                )

            senders.append(
                parts[0].strip()
            )

            receivers.append(
                parts[1].strip()
            )

    if senders is None or receivers is None:
        raise ValueError(
            "必须提供senders和receivers，"
            "或者提供sender_receiver_pairs。"
        )

    if len(senders) != len(receivers):
        raise ValueError(
            "senders与receivers长度不一致："
            f"{len(senders)} vs {len(receivers)}"
        )

    df = laris_results.copy()

    for column in [
        "sender",
        "receiver",
        "interaction_name"
    ]:
        df[column] = (
            df[column]
            .astype(str)
        )

    df["interaction_score"] = pd.to_numeric(
        df["interaction_score"],
        errors="coerce"
    )

    if filter_significant:

        if p_value_col in df.columns:

            df[p_value_col] = pd.to_numeric(
                df[p_value_col],
                errors="coerce"
            )

            df = df.loc[
                df[p_value_col] < threshold
            ].copy()

        elif "significant" in df.columns:

            df = df.loc[
                df["significant"].astype(bool)
            ].copy()

        else:
            print(
                f"Warning: 不存在{p_value_col}或"
                "significant列，跳过显著性过滤。"
            )

    if filter_by_interaction_score:

        df = df.loc[
            df["interaction_score"]
            > threshold_interaction_score
        ].copy()

    elif n_top is not None:

        df = (
            df.sort_values(
                "interaction_score",
                ascending=False
            )
            .head(n_top)
            .copy()
        )

    mask_cell_pairs = np.zeros(
        len(df),
        dtype=bool
    )

    for sender, receiver in zip(
        senders,
        receivers
    ):

        current_mask = (
            df["sender"].eq(sender)
            & df["receiver"].eq(receiver)
        )

        mask_cell_pairs |= (
            current_mask.to_numpy()
        )

    df_filtered = df.loc[
        mask_cell_pairs
        & df["interaction_name"].isin(
            interactions_to_plot
        )
    ].copy()

    df_filtered = df_filtered.loc[
        df_filtered["interaction_score"]
        .notna()
    ].copy()

    all_cell_pairs = [
        f"{sender} → {receiver}"
        for sender, receiver
        in zip(senders, receivers)
    ]

    df_filtered["cell_type_pair"] = (
        df_filtered["sender"]
        + " → "
        + df_filtered["receiver"]
    )

    df_filtered["cell_type_pair"] = (
        pd.Categorical(
            df_filtered["cell_type_pair"],
            categories=all_cell_pairs,
            ordered=True
        )
    )

    df_filtered["interaction_name"] = (
        pd.Categorical(
            df_filtered["interaction_name"],
            categories=interactions_to_plot,
            ordered=True
        )
    )

    if p_value_col in df_filtered.columns:

        bubble_p_col = p_value_col

    elif "p_value" in df_filtered.columns:

        bubble_p_col = "p_value"

    else:

        bubble_p_col = None

    if bubble_p_col is not None:

        df_filtered[bubble_p_col] = (
            pd.to_numeric(
                df_filtered[bubble_p_col],
                errors="coerce"
            )
        )

        p_values_for_size = (
            df_filtered[bubble_p_col]
            .fillna(1)
            .to_numpy()
        )

        df_filtered["bubble_size_plot"] = (
            _compute_bubble_sizes_log10(
                p_values=p_values_for_size,
                bubble_size=bubble_size,
                n_permutations=n_permutations
            )
        )

    else:

        df_filtered["bubble_size_plot"] = (
            bubble_size
        )

    df_nonzero = df_filtered.loc[
        df_filtered["interaction_score"] > 0
    ].copy()

    ax.set_xticks(
        range(len(all_cell_pairs))
    )

    ax.set_xticklabels(
        all_cell_pairs,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=tick_fontsize
    )

    ax.set_yticks(
        range(len(interactions_to_plot))
    )

    ax.set_yticklabels(
        interactions_to_plot,
        fontsize=tick_fontsize + 1
    )

    ax.set_xlim(
        -0.5,
        len(all_cell_pairs) - 0.5
    )

    ax.set_ylim(
        -0.5,
        len(interactions_to_plot) - 0.5
    )

    ax.set_xlabel(
        "Cell type pairs",
        fontsize=12
    )

    ax.set_ylabel(
        "Ligand–receptor interactions",
        fontsize=12
    )

    if show_grid:

        ax.grid(
            visible=True,
            axis="both",
            linestyle="-",
            linewidth=0.5,
            alpha=0.25,
            zorder=0
        )

    else:

        ax.grid(False)

    if not df_nonzero.empty:

        max_score = (
            df_nonzero["interaction_score"]
            .max()
        )

        if not np.isfinite(max_score) or max_score <= 0:
            max_score = 1

        scatter = ax.scatter(
            x=df_nonzero[
                "cell_type_pair"
            ].cat.codes,
            y=df_nonzero[
                "interaction_name"
            ].cat.codes,
            c=df_nonzero[
                "interaction_score"
            ],
            s=df_nonzero[
                "bubble_size_plot"
            ],
            cmap=cmap,
            vmin=0,
            vmax=max_score,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            zorder=3
        )

        colorbar = ax.figure.colorbar(
            scatter,
            cax=cbar_ax
        )

        colorbar.set_label(
            "Interaction score",
            fontsize=legend_fontsize
        )

        colorbar.ax.tick_params(
            labelsize=legend_fontsize - 2
        )

    else:

        scatter = None

        cbar_ax.axis("off")

        ax.text(
            0.5,
            0.5,
            "No matching interactions",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14
        )

    if bubble_p_col is not None:

        _create_pvalue_legend_on_ax(
            legend_ax=legend_ax,
            bubble_size=bubble_size,
            n_permutations=n_permutations,
            title=r"$P$ adjust",
            title_fontsize=legend_fontsize,
            label_fontsize=legend_fontsize - 2
        )

    else:

        legend_ax.axis("off")

    if title is not None:

        ax.set_title(
            title,
            fontsize=15,
            fontweight="bold",
            pad=12
        )

    if panel_label is not None:

        ax.text(
            -0.17,
            1.10,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=22,
            fontweight="bold"
        )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    return {
        "scatter": scatter,
        "data": df_filtered,
        "bubble_p_col": bubble_p_col
    }


plot_configs = [

    {
        "panel_label": "A",
        "title": (
            "LR interactions among SS1-Tum "
            "and top 5 co-occurring types"
        ),
        "interactions_to_plot": [
            "POSTN::ITGB5",
            "WNT5A::FZD1",
            "TGFB3::TGFBR1",
            "TGFB3::TGFBR2",
            "PDGFC::PDGFRA",
            "THBS1::SDC4",
            "THBS2::CD47",
            "POSTN::ITGAV",
            "FGF2::FGFR1",
            "PTGES3::PTGER3"
        ],
        "senders": [
            "Fibro_myCAF-3_CCN2",
            "Fibro_myCAF-1_TGFB3",
            "Fibro_myCAF-1_TGFB3",
            "Fibro_myCAF-3_CCN2",
            "Fibro_myCAF-1_TGFB3",
            "SS1-Tum",
            "SS1-Tum",
            "SS1-Tum"
        ],
        "receivers": [
            "Fibro_myCAF-3_CCN2",
            "Fibro_myCAF-1_TGFB3",
            "Fibro_senescence_MDM2",
            "SS1-Tum",
            "SS1-Tum",
            "Fibro_myCAF-1_TGFB3",
            "Fibro_myCAF-2_CST1",
            "Fibro_myCAF-3_CCN2"
        ],
        "cmap": "Reds",
        "threshold_interaction_score": 0.0001
    },

    {
        "panel_label": "B",
        "title": (
            "LR interactions among SS2-Tum "
            "and top 5 co-occurring types"
        ),
        "interactions_to_plot": [
            "COL4A1::ITGA1",
            "COL4A2::ITGA1",
            "PDGFA::PDGFRB",
            "PDGFB::PDGFRB",
            "JAG1::NOTCH3",
            "COL4A1::SDC4",
            "COL4A2::SDC4",
            "HSPG2::DAG1",
            "APP::TNFRSF21",
            "SDC2::ADGRA2"
        ],
        "senders": [
            "Pericyte_ECM_THBS1",
            "Pericyte_universal_RGS5",
            "EC_capillary_IVNS1ABP",
            "Pericyte_ECM_THBS1",
            "EC_capillary_IVNS1ABP",
            "SS2-Tum"
        ],
        "receivers": [
            "Pericyte_universal_RGS5",
            "Pericyte_ECM_THBS1",
            "Pericyte_universal_RGS5",
            "SS2-Tum",
            "SS2-Tum",
            "Pericyte_universal_RGS5"
        ],
        "cmap": "Oranges",
        "threshold_interaction_score": 0.0001
    },

    {
        "panel_label": "C",
        "title": (
            "LR interactions among SS3-Tum "
            "and top 5 co-occurring types"
        ),
        "interactions_to_plot": [
            "LGALS9::HAVCR2",
            "CD274::PDCD1",
            "PDCD1LG2::PDCD1",
            "PVR::TIGIT",
            "CXCL16::CXCR6",
            "ICOSLG::CTLA4",
            "GAS6::AXL",
            "GAS6::MERTK",
            "IL18::IL18RAP",
            "IL18::IL18R1"
        ],
        "senders": [
            "TAM_FCGBP",
            "TAM_FCGBP",
            "TAM_FCGBP",
            "SS3-Tum",
            "SS3-Tum",
            "SS3-Tum",
            "TAM_FCGBP"
        ],
        "receivers": [
            "CD8T_Tex_HAVCR2",
            "CD4T_Tanergy_EGR3",
            "CD4T_eTreg_TNFRSF9",
            "CD4T_eTreg_TNFRSF9",
            "CD4T_Tanergy_EGR3",
            "TAM_FCGBP",
            "SS3-Tum"
        ],
        "cmap": "Greens",
        "threshold_interaction_score": 0.0000001
    },

    {
        "panel_label": "D",
        "title": (
            "LR interactions among SS4-Tum "
            "and top 5 co-occurring types"
        ),
        "interactions_to_plot": [
            "LCK::CD8A",
            "CXCL9::CXCR3",
            "CXCL10::CXCR3",
            "IL16::CD4",
            "ICAM1::ITGB2",
            "VCAM1::ITGB2",
            "F11R::ITGAL",
            "F11R::ITGB2",
            "SEMA4D::PLXNB2",
            "CCL8::CCR1"
        ],
        "senders": [
            "CD8T_ISG_IFIT3",
            "TAM_CXCL9",
            "TAM_MX1",
            "TAM_CXCL9",
            "TAM_MX1",
            "CD4T_Tn_GIMAP4",
            "TAM_CXCL9",
            "CD8T_Tex_HAVCR2",
            "SS4-Tum",
            "SS4-Tum",
            "CD8T_Tex_HAVCR2"
        ],
        "receivers": [
            "CD8T_ISG_IFIT3",
            "CD8T_ISG_IFIT3",
            "CD8T_ISG_IFIT3",
            "CD8T_Tex_HAVCR2",
            "CD8T_Tex_HAVCR2",
            "TAM_MX1",
            "TAM_CXCL9",
            "TAM_MX1",
            "CD8T_Tex_HAVCR2",
            "TAM_MX1",
            "SS4-Tum"
        ],
        "cmap": "Blues",
        "threshold_interaction_score": 0.0001
    },

    {
        "panel_label": "E",
        "title": (
            "LR interactions among SS5-Tum "
            "and top 5 co-occurring types"
        ),
        "interactions_to_plot": [
            "PLAU::PLAUR",
            "IL1B::IL1RAP",
            "CXCL5::CXCR1",
            "CXCL2::CXCR1",
            "ICAM1::ITGAX",
            "CD55::ADGRE5",
            "POSTN::ITGB5",
            "ANGPTL4::ITGA5",
            "THBS1::SDC4",
            "LAMB3::CD44"
        ],
        "senders": [
            "NEU_NETs_IL1B",
            "NEU_NETs_IL1B",
            "TAM_MMP12",
            "SS5-Tum",
            "Fibro_iCAF-2_VEGFA",
            "Fibro_iCAF-2_VEGFA",
            "SS5-Tum",
            "SS5-Tum",
            "SS5-Tum"
        ],
        "receivers": [
            "NEU_NETs_IL1B",
            "TAM_MMP12",
            "NEU_NETs_IL1B",
            "NEU_NETs_IL1B",
            "Fibro_iCAF-2_VEGFA",
            "SS5-Tum",
            "SS5-Tum",
            "TAM_FCGBP",
            "DC_Langerhans_CD1A"
        ],
        "cmap": "Purples",
        "threshold_interaction_score": 0.0001
    }
]


required_res_columns = {
    "sender",
    "receiver",
    "interaction_name",
    "interaction_score"
}

missing_res_columns = (
    required_res_columns
    - set(res_LARIS.columns)
)

if missing_res_columns:
    raise ValueError(
        "res_LARIS缺少以下必要列："
        + ", ".join(
            sorted(missing_res_columns)
        )
    )


fig = plt.figure(
    figsize=(8, 30)
)

outer_grid = gridspec.GridSpec(
    nrows=5,
    ncols=1,
    figure=fig,
    left=0.27,
    right=0.98,
    bottom=0.03,
    top=0.985,
    hspace=0.58
)


plot_results = {}

for row_index, config in enumerate(
    plot_configs
):

    inner_grid = outer_grid[
        row_index
    ].subgridspec(
        nrows=1,
        ncols=3,
        width_ratios=[
            1.00,
            0.030,
            0.20
        ],
        wspace=0.16
    )

    ax = fig.add_subplot(
        inner_grid[0, 0]
    )

    cbar_ax = fig.add_subplot(
        inner_grid[0, 1]
    )

    legend_ax = fig.add_subplot(
        inner_grid[0, 2]
    )

    current_result = plotCCCDotPlot_on_ax(
        laris_results=res_LARIS,
        interactions_to_plot=(
            config[
                "interactions_to_plot"
            ]
        ),
        senders=config["senders"],
        receivers=config["receivers"],
        ax=ax,
        cbar_ax=cbar_ax,
        legend_ax=legend_ax,
        bubble_size=500,
        cmap=config["cmap"],
        filter_significant=False,
        p_value_col="p_value_fdr",
        filter_by_interaction_score=True,
        threshold_interaction_score=(
            config[
                "threshold_interaction_score"
            ]
        ),
        n_permutations=1000,
        legend_fontsize=12,
        tick_fontsize=9,
        show_grid=True,
        title=config["title"],
        panel_label=config["panel_label"]
    )

    plot_results[
        config["panel_label"]
    ] = current_result

    print(
        f"Panel {config['panel_label']}: "
        f"{len(current_result['data'])} "
        "matching interaction records"
    )


output_pdf = (
    "SS1_SS5_integrated_CCC_dotplots_"
    "1col_5rows.pdf"
)

fig.savefig(
    output_pdf,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"Integrated PDF saved to: "
    f"{output_pdf}"
)


output_png = (
    "SS1_SS5_integrated_CCC_dotplots_"
    "1col_5rows.png"
)

fig.savefig(
    output_png,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"Integrated PNG saved to: "
    f"{output_png}"
)

plt.show()



#Fig.6b
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D


def plot_RNA_on_cell_black(
    adata,
    RNA_name,
    group=None,
    subset=None,
    col_map=None,
    RNA_name_col='feature_name',
    RNA_x='x_location',
    RNA_y='y_location',
    boundary_col='boundary',
    RNA_cell_id='cell_id',
    RNA_color=None,
    RNA_size=8,
    cell_edge_width=0.4,
    show_scalebar=True,
    scale_length_um=100.0,
    add_gene_legend=True,
    add_cell_legend=False,
    max_cell_types_legend=8,
    figure_size=(6, 6),
    dpi=300,
    title=None,
    output=None
):

    if 'transcripts' not in adata.uns:
        raise ValueError("adata.uns 中找不到 'transcripts'。")

    transcripts = adata.uns['transcripts'].copy()

    if isinstance(RNA_name, str):
        genes = [RNA_name]
    else:
        genes = list(RNA_name)

    if genes:
        transcripts = transcripts[
            transcripts[RNA_name_col].isin(genes)
        ]

    obs = adata.obs.copy()

    if subset is not None:
        if group is None:
            raise ValueError(
                "subset 已设置，但 group 为 None，请先指定 group 列名。"
            )

        mask_cells = obs[group].isin(subset)
        obs = obs[mask_cells]

    if (
        subset is not None
        and group is not None
        and RNA_cell_id in transcripts.columns
    ):
        if RNA_cell_id in obs.columns:
            keep_ids = obs[RNA_cell_id].unique()
        else:
            keep_ids = obs.index.astype(str)

        transcripts = transcripts[
            transcripts[RNA_cell_id].isin(keep_ids)
        ]

    fig, ax_main = plt.subplots(
        figsize=figure_size,
        dpi=dpi
    )

    fig.patch.set_facecolor("white")

    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.935,
        bottom=0.18
    )

    if title is not None:
        fig.suptitle(
            title,
            fontsize=14,
            color="black"
        )

    ax_main.set_facecolor("black")

    xs_all, ys_all = [], []

    if boundary_col not in obs.columns:
        raise ValueError(
            f"adata.obs 中找不到 '{boundary_col}' 列。"
        )

    for cell_id, row in obs.iterrows():

        b = row[boundary_col]

        if pd.isna(b):
            continue

        try:
            coords = (
                ast.literal_eval(b)
                if isinstance(b, str)
                else b
            )
        except Exception:
            continue

        if coords is None or len(coords) == 0:
            continue

        xy = np.asarray(coords)

        xs_all.extend(xy[:, 0])
        ys_all.extend(xy[:, 1])

        if (
            group is not None
            and col_map is not None
            and row.get(group) in col_map
        ):
            edge_c = col_map[row[group]]
        else:
            edge_c = (
                0.7,
                0.7,
                0.7,
                0.7
            )

        patch = Polygon(
            xy,
            closed=True,
            fill=False,
            edgecolor=edge_c,
            linewidth=cell_edge_width,
            zorder=2
        )

        ax_main.add_patch(patch)

    if RNA_color is None:

        default_colors = [
            "#00FFFF",
            "#FF00FF",
            "#FFFF00",
            "#00FF00",
            "#FF0000",
            "#FFA500",
            "#1E90FF",
            "#FF1493"
        ]

        genes_to_plot = (
            genes
            if genes
            else list(
                transcripts[
                    RNA_name_col
                ].unique()
            )
        )

        RNA_color = {
            g: default_colors[
                i % len(default_colors)
            ]
            for i, g in enumerate(
                genes_to_plot
            )
        }

    else:

        genes_to_plot = (
            genes
            if genes
            else list(
                RNA_color.keys()
            )
        )

    for g in genes_to_plot:

        df_g = transcripts[
            transcripts[RNA_name_col] == g
        ]

        if df_g.empty:
            continue

        ax_main.scatter(
            df_g[RNA_x].values,
            df_g[RNA_y].values,
            s=RNA_size,
            c=RNA_color.get(
                g,
                "#FFFFFF"
            ),
            linewidths=0,
            alpha=1.0,
            zorder=3
        )

        xs_all.extend(
            df_g[RNA_x].values.tolist()
        )

        ys_all.extend(
            df_g[RNA_y].values.tolist()
        )

    ax_main.set_aspect("equal")

    if len(xs_all) == 0 or len(ys_all) == 0:

        ax_main.axis("off")

        if output is not None:
            fig.savefig(
                output,
                facecolor="white"
            )
            plt.close(fig)
        else:
            plt.show()

        return

    x_min, x_max = (
        min(xs_all),
        max(xs_all)
    )

    y_min, y_max = (
        min(ys_all),
        max(ys_all)
    )

    width_um = max(
        x_max - x_min,
        1e-6
    )

    height_um = max(
        y_max - y_min,
        1e-6
    )

    bg_rect = Rectangle(
        (x_min, y_min),
        width_um,
        height_um,
        facecolor="black",
        edgecolor="none",
        zorder=0
    )

    ax_main.add_patch(bg_rect)

    ax_main.set_xlim(
        x_min,
        x_max
    )

    ax_main.set_ylim(
        y_max,
        y_min
    )

    ax_main.set_xticks([])
    ax_main.set_yticks([])

    for spine in ax_main.spines.values():
        spine.set_visible(False)

    fig.canvas.draw()

    mid_y = 0.5 * (
        y_min + y_max
    )

    disp_left = ax_main.transData.transform(
        (x_min, mid_y)
    )

    disp_right = ax_main.transData.transform(
        (x_max, mid_y)
    )

    left_fig, _ = (
        fig.transFigure.inverted()
        .transform(disp_left)
    )

    right_fig, _ = (
        fig.transFigure.inverted()
        .transform(disp_right)
    )

    mid_x = 0.5 * (
        x_min + x_max
    )

    disp_bottom = ax_main.transData.transform(
        (mid_x, y_max)
    )

    _, bottom_fig = (
        fig.transFigure.inverted()
        .transform(disp_bottom)
    )

    base_y = max(
        bottom_fig - 0.04,
        0.02
    )

    scale_y = min(
        base_y + 0.02,
        bottom_fig - 0.01
    )

    legend_offset = 0.02
    legend_y = (
        scale_y + legend_offset
    )

    bar_start = left_fig
    bar_end = left_fig + 0.0

    if (
        show_scalebar
        and scale_length_um > 0
    ):

        frac = min(
            scale_length_um / width_um,
            0.6
        )

        bar_start = left_fig

        bar_end = (
            left_fig
            + frac
            * (right_fig - left_fig)
        )

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

        bar_end = (
            left_fig
            + 0.4
            * (right_fig - left_fig)
        )

    handles_gene = []
    handles_cell = []

    if (
        add_gene_legend
        and len(genes_to_plot) > 0
    ):

        for g in genes_to_plot:

            if g not in RNA_color:
                continue

            h = Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor=RNA_color[g],
                markeredgecolor="black",
                markeredgewidth=0.3,
                label=g
            )

            handles_gene.append(h)

    if (
        add_cell_legend
        and group is not None
        and col_map is not None
    ):

        present_types = list(
            pd.unique(obs[group])
        )

        if subset is not None:

            cell_types = [
                ct
                for ct in subset
                if ct in present_types
            ]

        else:

            cell_types = present_types

        cell_types = cell_types[
            :max_cell_types_legend
        ]

        for ct in cell_types:

            c = col_map.get(
                ct,
                (
                    0.7,
                    0.7,
                    0.7,
                    0.7
                )
            )

            h = Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor="none",
                markeredgecolor=c,
                markeredgewidth=1.5,
                label=str(ct)
            )

            handles_cell.append(h)

    if handles_gene:

        leg_gene = fig.legend(
            handles=handles_gene,
            loc="upper right",
            bbox_to_anchor=(
                right_fig,
                legend_y
            ),
            bbox_transform=fig.transFigure,
            frameon=False,
            fontsize=9,
            handletextpad=0.4,
            borderpad=0.2
        )

        for text in leg_gene.get_texts():
            text.set_color("black")

    if handles_cell:

        mid_fig_x = (
            bar_end + right_fig
        ) / 2.0

        leg_cell = fig.legend(
            handles=handles_cell,
            loc="upper center",
            bbox_to_anchor=(
                mid_fig_x,
                legend_y
            ),
            bbox_transform=fig.transFigure,
            frameon=False,
            fontsize=9,
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


import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable


def plot_obs_black(
    adata,
    value_key="CXCL9::CXCR3",
    boundary_col="boundary",
    cell_edge_width=1,
    show_scalebar=True,
    scale_length_um=100.0,
    vmin=None,
    vmax=None,
    norm_method="none",
    figure_size=(6, 6),
    dpi=300,
    title=None,
    output=None
):

    obs = adata.obs.copy()

    if boundary_col not in obs.columns:
        raise ValueError(
            f"adata.obs 中找不到 '{boundary_col}' 列。"
        )

    if value_key not in obs.columns:
        raise ValueError(
            f"adata.obs 中找不到 '{value_key}' 列。"
        )

    vals_raw = pd.to_numeric(
        obs[value_key],
        errors="coerce"
    ).values.astype(float)

    finite_mask = np.isfinite(
        vals_raw
    )

    if not np.any(finite_mask):
        raise ValueError(
            f"{value_key} 中没有有效的数值。"
        )

    vals_norm = vals_raw.copy()
    label_suffix = ""

    nm = norm_method.lower()

    if nm == "minmax":

        min_v = float(
            np.nanmin(
                vals_raw[finite_mask]
            )
        )

        max_v = float(
            np.nanmax(
                vals_raw[finite_mask]
            )
        )

        if max_v == min_v:
            max_v = min_v + 1e-9

        vals_norm = (
            vals_raw - min_v
        ) / (
            max_v - min_v + 1e-9
        )

        label_suffix = " (min-max)"

    elif nm == "zscore":

        mean_v = float(
            np.nanmean(
                vals_raw[finite_mask]
            )
        )

        std_v = float(
            np.nanstd(
                vals_raw[finite_mask]
            )
        )

        if std_v == 0:
            std_v = 1e-9

        vals_norm = (
            vals_raw - mean_v
        ) / (
            std_v + 1e-9
        )

        label_suffix = " (z-score)"

    elif nm == "none":

        label_suffix = ""

    else:

        raise ValueError(
            "norm_method 必须是 'none', "
            "'minmax' 或 'zscore'。"
        )

    val_map = pd.Series(
        vals_norm,
        index=obs.index
    )

    finite_mask_norm = np.isfinite(
        vals_norm
    )

    if vmin is None:

        vmin = float(
            np.nanmin(
                vals_norm[
                    finite_mask_norm
                ]
            )
        )

    if vmax is None:

        vmax = float(
            np.nanmax(
                vals_norm[
                    finite_mask_norm
                ]
            )
        )

    if vmin == vmax:

        vmin -= 1e-9
        vmax += 1e-9

    cmap = get_cmap("magma")

    norm = Normalize(
        vmin=vmin,
        vmax=vmax
    )

    fig, ax_main = plt.subplots(
        figsize=figure_size,
        dpi=dpi
    )

    fig.patch.set_facecolor("white")

    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.935,
        bottom=0.18
    )

    if title is not None:

        fig.suptitle(
            title,
            fontsize=14,
            color="black"
        )

    ax_main.set_facecolor("black")

    xs_all, ys_all = [], []

    for cell_id, row in obs.iterrows():

        b = row[boundary_col]

        if pd.isna(b):
            continue

        try:

            coords = (
                ast.literal_eval(b)
                if isinstance(b, str)
                else b
            )

        except Exception:

            continue

        if coords is None or len(coords) == 0:
            continue

        xy = np.asarray(coords)

        xs_all.extend(
            xy[:, 0]
        )

        ys_all.extend(
            xy[:, 1]
        )

        val = val_map.loc[cell_id]

        if np.isfinite(val):

            face_c = cmap(
                norm(val)
            )

        else:

            face_c = (
                0.3,
                0.3,
                0.3,
                1.0
            )

        patch = Polygon(
            xy,
            closed=True,
            fill=True,
            facecolor=face_c,
            edgecolor="black",
            linewidth=cell_edge_width,
            alpha=1.0,
            zorder=2
        )

        ax_main.add_patch(patch)

    ax_main.set_aspect("equal")

    if len(xs_all) == 0 or len(ys_all) == 0:

        ax_main.axis("off")

        if output is not None:

            fig.savefig(
                output,
                facecolor="white"
            )

            plt.close(fig)

        else:

            plt.show()

        return

    x_min, x_max = (
        min(xs_all),
        max(xs_all)
    )

    y_min, y_max = (
        min(ys_all),
        max(ys_all)
    )

    width_um = max(
        x_max - x_min,
        1e-6
    )

    height_um = max(
        y_max - y_min,
        1e-6
    )

    bg_rect = Rectangle(
        (x_min, y_min),
        width_um,
        height_um,
        facecolor="black",
        edgecolor="none",
        zorder=0
    )

    ax_main.add_patch(bg_rect)

    ax_main.set_xlim(
        x_min,
        x_max
    )

    ax_main.set_ylim(
        y_max,
        y_min
    )

    ax_main.set_xticks([])
    ax_main.set_yticks([])

    for spine in ax_main.spines.values():
        spine.set_visible(False)

    fig.canvas.draw()

    mid_y = 0.5 * (
        y_min + y_max
    )

    disp_left = ax_main.transData.transform(
        (x_min, mid_y)
    )

    disp_right = ax_main.transData.transform(
        (x_max, mid_y)
    )

    left_fig, _ = (
        fig.transFigure.inverted()
        .transform(disp_left)
    )

    right_fig, _ = (
        fig.transFigure.inverted()
        .transform(disp_right)
    )

    mid_x = 0.5 * (
        x_min + x_max
    )

    disp_bottom = ax_main.transData.transform(
        (mid_x, y_max)
    )

    _, bottom_fig = (
        fig.transFigure.inverted()
        .transform(disp_bottom)
    )

    base_y = max(
        bottom_fig - 0.04,
        0.02
    )

    scale_y = min(
        base_y + 0.02,
        bottom_fig - 0.01
    )

    bar_start = left_fig
    bar_end = left_fig + 0.0

    if (
        show_scalebar
        and scale_length_um > 0
    ):

        frac = min(
            scale_length_um / width_um,
            0.6
        )

        bar_start = left_fig

        bar_end = (
            left_fig
            + frac
            * (right_fig - left_fig)
        )

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

        bar_end = (
            left_fig
            + 0.4
            * (right_fig - left_fig)
        )

    cbar_width = (
        0.25
        * (right_fig - left_fig)
    )

    cbar_height = 0.018

    cbar_left = (
        right_fig - cbar_width
    )

    cbar_bottom = max(
        scale_y - cbar_height / 2,
        0.02
    )

    cax = fig.add_axes(
        [
            cbar_left,
            cbar_bottom,
            cbar_width,
            cbar_height
        ]
    )

    sm = ScalarMappable(
        norm=norm,
        cmap=cmap
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal"
    )

    cbar.set_label(
        value_key + label_suffix,
        fontsize=9,
        color="black"
    )

    cbar.ax.tick_params(
        labelsize=8,
        colors="black"
    )

    for t in cbar.ax.get_xticklabels():

        t.set_color("black")

    if output is not None:

        fig.savefig(
            output,
            dpi=dpi,
            facecolor="white"
        )

        plt.show()

    else:

        plt.show()


import numpy as np
import pandas as pd

center_x = 3108
center_y = 8143

half_size = 100.0

x_min = center_x - half_size
x_max = center_x + half_size
y_min = center_y - half_size
y_max = center_y + half_size

print(
    "ROI x范围:",
    x_min,
    x_max
)

print(
    "ROI y范围:",
    y_min,
    y_max
)

cells_roi_mask = (
    (spyrrow_adata.obs["x_centroid"] >= x_min)
    & (spyrrow_adata.obs["x_centroid"] <= x_max)
    & (spyrrow_adata.obs["y_centroid"] >= y_min)
    & (spyrrow_adata.obs["y_centroid"] <= y_max)
)

adata_roi = spyrrow_adata[
    cells_roi_mask
].copy()

print(
    "ROI 内细胞数量:",
    adata_roi.n_obs
)

tr = spyrrow_adata.uns[
    "transcripts"
]

tr_roi_mask = (
    (tr["x_location"] >= x_min)
    & (tr["x_location"] <= x_max)
    & (tr["y_location"] >= y_min)
    & (tr["y_location"] <= y_max)
)

tr_roi = tr[
    tr_roi_mask
].copy()

print(
    "ROI 内转录本数量:",
    tr_roi.shape[0]
)

adata_roi.uns[
    "transcripts"
] = tr_roi


plot_RNA_on_cell_black(
    adata=adata_roi,
    RNA_name=[
        'GAS6',
        'AXL',
        'MERTK',
        'FCGBP',
        'TNFRSF9'
    ],
    group='cell_type',
    subset=[
        'CD4T_eTreg_CCR8',
        'TAM_FCGBP',
        'SS3-Tum'
    ],
    RNA_color={
        'GAS6': '#FF3B30',
        'AXL': '#FFD60A',
        'MERTK': '#BF5AF2',
        'FCGBP': '#00FFFF',
        'TNFRSF9': '#FFFFFF'
    },
    col_map={
        'CD4T_eTreg_CCR8': '#1F77B4',
        'TAM_FCGBP': '#2CA02C',
        'SS3-Tum': '#FF7F0E'
    },
    RNA_name_col='feature_name',
    RNA_x='x_location',
    RNA_y='y_location',
    boundary_col='boundary',
    RNA_cell_id='cell_id',
    RNA_size=10,
    cell_edge_width=1,
    show_scalebar=True,
    scale_length_um=50.0,
    add_cell_legend=True,
    figure_size=(6, 6),
    title='GAS6-TAM receptor axis (PT-NR_09)',
    output='/home/qiuzc/GAS6_AXL_MERTK_FCGBP_TNFRSF9_1.pdf'
)


import numpy as np
import pandas as pd
from scipy.sparse import issparse

gene = "GAS6::AXL"

if gene not in adata_lr.var_names:
    raise ValueError(
        f"{gene} 不在 adata_lr.var_names 中"
    )

g_idx = adata_lr.var_names.get_loc(
    gene
)

if issparse(adata_lr.X):

    vals_lr = (
        adata_lr.X[:, g_idx]
        .toarray()
        .ravel()
    )

else:

    vals_lr = np.asarray(
        adata_lr.X[:, g_idx]
    ).ravel()

lr_map = pd.Series(
    vals_lr,
    index=adata_lr.obs['cell_id']
).to_dict()

adata_roi.obs[gene] = (
    adata_roi.obs['cell_id']
    .map(lr_map)
)


plot_obs_black(
    adata=adata_roi,
    value_key="GAS6::AXL",
    boundary_col="boundary",
    scale_length_um=50.0,
    title="GAS6::AXL interaction score",
    figure_size=(6, 6),
    vmax=5,
    output="/home/qiuzc/GAS6_AXL_MERTK_FCGBP_TNFRSF9_2.pdf"
)


import numpy as np
import pandas as pd
from scipy.sparse import issparse

gene = "GAS6::MERTK"

if gene not in adata_lr.var_names:
    raise ValueError(
        f"{gene} 不在 adata_lr.var_names 中"
    )

g_idx = adata_lr.var_names.get_loc(
    gene
)

if issparse(adata_lr.X):

    vals_lr = (
        adata_lr.X[:, g_idx]
        .toarray()
        .ravel()
    )

else:

    vals_lr = np.asarray(
        adata_lr.X[:, g_idx]
    ).ravel()

lr_map = pd.Series(
    vals_lr,
    index=adata_lr.obs['cell_id']
).to_dict()

adata_roi.obs[gene] = (
    adata_roi.obs['cell_id']
    .map(lr_map)
)


plot_obs_black(
    adata=adata_roi,
    value_key="GAS6::MERTK",
    boundary_col="boundary",
    scale_length_um=50.0,
    title="GAS6::MERTK interaction score",
    figure_size=(6, 6),
    vmax=5,
    output="/home/qiuzc/GAS6_AXL_MERTK_FCGBP_TNFRSF9_3.pdf"
)


#Fig.6c
mp = pd.read_csv("SS3_MP_composition.csv")

fig, ax = plt.subplots(figsize=(5,5))

ax.pie(
    mp["fraction"],
    labels=mp["metaprogram"],
    autopct="%1.1f%%"
)

plt.tight_layout()
plt.savefig("Fig6c.pdf")
plt.close()


#Fig.6d
corr = pd.read_csv("SF3B1_GAS6_correlation.csv")

fig, ax = plt.subplots(figsize=(5,4))

sns.scatterplot(
    data=corr,
    x="SF3B1",
    y="GAS6",
    hue="group",
    ax=ax
)

plt.tight_layout()
plt.savefig("Fig6d.pdf")
plt.close()


#Fig.6e grahpad


#Fig.6f
import matplotlib.image as mpimg

fig, axes = plt.subplots(1,2,figsize=(8,3))

for ax, image in zip(
    axes,
    ["HK1_WB.png","C6661_WB.png"]
):
    ax.imshow(mpimg.imread(image))
    ax.axis("off")

plt.tight_layout()
plt.savefig("Fig6f.pdf")
plt.close()


#Fig.6g, h, i RNA-seq R script


#Fig.6j WB


#Fig.6k IHC

#Fig.6l
corr = pd.read_csv("SF3B1_GAS6_correlation_IHC.csv")

fig, ax = plt.subplots(figsize=(5,4))

sns.scatterplot(
    data=corr,
    x="SF3B1",
    y="GAS6",
    hue="group",
    ax=ax
)

plt.tight_layout()
plt.savefig("Fig6l.pdf")
plt.close()


#Fig.6m
from lifelines import KaplanMeierFitter

surv = pd.read_csv("GAS6_survival.csv")

fig, ax = plt.subplots(figsize=(5,4))

for group, data in surv.groupby("group"):
    km = KaplanMeierFitter()
    km.fit(
        data["time"],
        data["event"],
        label=group
    )
    km.plot(ax=ax)

plt.tight_layout()
plt.savefig("Fig6l.pdf")
plt.close()

