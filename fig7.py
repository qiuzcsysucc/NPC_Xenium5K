#Fig.7a
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

adata = sc.read("GAS6_TAM_public_scRNA.h5ad")

markers = [
    "TYRO3",
    "AXL",
    "MERTK",
    "FCGBP",
    "CTNND1",
    "NDRG2",
    "MKI67",
    "TOP2A",
    "PCNA"
]

dot = sc.get.rank_genes_groups_df(
    adata,
    group=None
)

fig, ax = plt.subplots(figsize=(6,5))

sc.pl.dotplot(
    adata,
    markers,
    groupby="TAM_subtype",
    show=False,standard_scale="var",
    ax=ax
)

plt.tight_layout()
plt.savefig("Fig7a.pdf")
plt.close()


#Fig.7b
fig, ax = plt.subplots(figsize=(5,4))

sc.pl.umap(
    adata,
    color="TAM_subtype",
    show=False,
    ax=ax
)

plt.tight_layout()
plt.savefig("Fig7b.pdf")
plt.close()


#Fig.7c

from matplotlib import patheffects
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.reset_orig()

plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 12})

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

small_marker_dict = {
    'Monocyte': ['CD14', 'FCN1', 'SELL', 'FCGR3A', 'IL1B'],
    'Macrophage': ['APOE', 'SPP1', 'NR4A3', 'FCGBP', 'CTNND1', 'NDRG2'],
    'GAS6-TAM': ['TYRO3', 'AXL', 'MERTK'],
}

sc.pl.dotplot(
    adata,
    groupby="celltype",
    var_names=small_marker_dict,
    cmap='coolwarm',
    dendrogram=False,
    standard_scale="var",
    show=False
)

plt.savefig('dotplot_tam_sc.pdf', bbox_inches='tight')
plt.show()



#Fig.7d
fig, ax = plt.subplots(figsize=(4, 4))
sc.pl.embedding(adata,
                basis='X_umap',cmap='plasma',
                color=['CytoTRACE2_Relative'],
                frameon=False,
                show=False,ax=ax,title='CytoTRACE2_Relative')
for coll in ax.collections:
    coll.set_rasterized(True)
plt.savefig('umap_cytotrace_sc.pdf', bbox_inches='tight')
plt.show()
from omicverse.external import VIA
fig, ax = VIA.core.via_streamplot(
    via_object=v0,
    embedding=stavia_embedding,
    dpi=100,
    density_grid=1.0,
    density_stream=2.5,
    scatter_size=18,
    scatter_alpha=0.28,
    linewidth=0.8,cmap='tab10'
)
for collection in ax.collections:
    collection.set_rasterized(True)
fig.set_size_inches(3, 3)
plt.savefig(
    "streamplot_celltype.pdf"
)
plt.show()
fig, ax = VIA.core.via_streamplot(
    via_object=v0,
    embedding=stavia_embedding,
    dpi=100,
    density_grid=1.0,
    density_stream=2.5,
    scatter_size=18,
    scatter_alpha=0.28,
    linewidth=0.8,
    color_scheme="time",
    min_mass=1,
    cutoff_perc=5,
    marker_edgewidth=0.1,
    smooth_transition=1,
    smooth_grid=0.5,
)
for collection in ax.collections:
    collection.set_rasterized(True)
fig.set_size_inches(3, 3)
plt.savefig(
    "streamplot_time.pdf"
)
plt.show()
fig, ax = ov.pl.branch_streamplot(
    adata,
    group_key='celltype',
    pseudotime_key='stavia_pseudotime',
    show=False,n_branches=4,figsize=(3,4)
)
plt.savefig('time2.pdf',bbox_inches='tight')
plt.show()

#Fig.7e
fig, axs = VIA.core.get_gene_expression(
    via_object=v0,
    gene_exp=df_magic[stavia_marker_genes],
    marker_genes=stavia_marker_genes,
    marker_lineages=[12, 22, 23, 33],
    dpi=80,
    figsize=(7, 4),
    ncols=3,
    legend_loc="right",
)
plt.savefig('xenium_tam_ligand_gene_expression.pdf',bbox_inches='tight')
plt.show()




#Fig.7f IL18 RNA-seq



#Fig.7g mIF

#Fig.7h
```
from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import font_manager
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests


INPUT_FILE = Path("halo_result.xlsx")
OUTPUT_DIR = Path("halo_boxplot_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_COL = "Group_Label"
PAIR_COL = "Pair_ID"

GROUP_ORDER = ["PT-NR", "PT-R", "RT"]

COLORS = {
    "PT-NR": "#4ea64a",
    "PT-R": "#3777ac",
    "RT": "#d5231d",
}

METRICS = [
    "ternary_ratio_20um",
    "ternary_ratio_40um",
    "ternary_ratio_60um",
    "ternary_ratio_80um",
    "% SF3B1+Tum Positive Cells",
    "% FCGBP+TAM Positive Cells",
    "% TNFRSF9+Treg Positive Cells",
]

PANEL_TITLES = {
    "ternary_ratio_20um": "Ternary ratio (20 μm)",
    "ternary_ratio_40um": "Ternary ratio (40 μm)",
    "ternary_ratio_60um": "Ternary ratio (60 μm)",
    "ternary_ratio_80um": "Ternary ratio (80 μm)",
    "% SF3B1+Tum Positive Cells": "SF3B1+ Tumor cells (%)",
    "% FCGBP+TAM Positive Cells": "FCGBP+ TAM cells (%)",
    "% TNFRSF9+Treg Positive Cells": "TNFRSF9+ Treg cells (%)",
}


def configure_arial() -> str:
    """
    查找并注册 Arial 字体。

    若 Arial 字体文件不在系统默认字体目录，可设置环境变量：
    ARIAL_FONT_FILE=/path/to/arial.ttf

    字体文件只在本地使用，不会复制到输出目录。
    """
    env_font = os.environ.get("ARIAL_FONT_FILE")

    candidate_paths = [
        env_font,

        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\Arial.ttf",

        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",

        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Regular.ttf",
    ]

    for font_path in candidate_paths:
        if font_path and Path(font_path).is_file():
            font_manager.fontManager.addfont(font_path)

    installed_names = {
        font.name.lower()
        for font in font_manager.fontManager.ttflist
    }

    if "arial" not in installed_names:
        raise RuntimeError(
            "未检测到 Arial 字体。请在系统中安装 Arial，或将环境变量 "
            "ARIAL_FONT_FILE 设置为本地 Arial 字体文件路径。"
        )

    return "Arial"


PLOT_FONT = configure_arial()

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [PLOT_FONT],
    "font.size": 5,
    "axes.titlesize": 5,
    "axes.labelsize": 5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "axes.linewidth": 0.55,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
})


def fmt_p(p: float) -> str:
    """将 P 值格式化为适合论文图片的形式。"""
    if not np.isfinite(p):
        return "P=NA"

    if p < 1e-4:
        return "P<0.0001"

    if p < 0.001:
        return f"P={p:.4f}"

    return f"P={p:.3f}"


def add_p_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    h: float,
    label: str,
) -> None:
    """绘制组间比较横线及 P 值。"""
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        color="black",
        linewidth=0.5,
        clip_on=False,
    )

    ax.text(
        (x1 + x2) / 2,
        y + h * 1.18,
        label,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=5,
        clip_on=False,
    )


def iqr_summary(x: pd.Series) -> tuple:
    """返回中位数、Q1、Q3和有效样本量。"""
    x = pd.to_numeric(x, errors="coerce").dropna()

    if x.empty:
        return np.nan, np.nan, np.nan, 0

    return (
        x.median(),
        x.quantile(0.25),
        x.quantile(0.75),
        len(x),
    )


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"未找到输入文件：{INPUT_FILE.resolve()}"
    )

df = pd.read_excel(INPUT_FILE)

required_columns = [
    GROUP_COL,
    PAIR_COL,
    *METRICS,
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise KeyError(
        f"Excel 中缺少以下必要列：{missing_columns}"
    )

df = df[
    df[GROUP_COL].isin(GROUP_ORDER)
].copy()

for metric in METRICS:
    df[metric] = pd.to_numeric(
        df[metric],
        errors="coerce",
    )

pair_counts = (
    df.loc[
        df[GROUP_COL].isin(["PT-R", "RT"])
    ]
    .dropna(subset=[PAIR_COL])
    .groupby([PAIR_COL, GROUP_COL])
    .size()
    .unstack(fill_value=0)
)

if "PT-R" not in pair_counts.columns:
    raise ValueError("数据中不存在可配对的 PT-R 样本。")

if "RT" not in pair_counts.columns:
    raise ValueError("数据中不存在可配对的 RT 样本。")

valid_pair_ids = pair_counts.index[
    (pair_counts["PT-R"] == 1)
    & (pair_counts["RT"] == 1)
]

if len(valid_pair_ids) == 0:
    raise ValueError(
        "Pair_ID 中没有找到一一对应的 PT-R/RT 配对样本。"
    )

print("各组样本量：")
print(df[GROUP_COL].value_counts().reindex(GROUP_ORDER))

print(f"\n有效 PT-R/RT 配对数：{len(valid_pair_ids)}")


results = []

for metric in METRICS:
    pt_nr = df.loc[
        df[GROUP_COL] == "PT-NR",
        metric,
    ].dropna()

    pt_r = df.loc[
        df[GROUP_COL] == "PT-R",
        metric,
    ].dropna()

    if pt_nr.empty or pt_r.empty:
        mw_u = np.nan
        mw_p = np.nan
    else:
        mw_result = mannwhitneyu(
            pt_nr,
            pt_r,
            alternative="two-sided",
            method="asymptotic",
        )

        mw_u = float(mw_result.statistic)
        mw_p = float(mw_result.pvalue)

    paired = (
        df.loc[
            df[PAIR_COL].isin(valid_pair_ids)
            & df[GROUP_COL].isin(["PT-R", "RT"]),
            [PAIR_COL, GROUP_COL, metric],
        ]
        .pivot(
            index=PAIR_COL,
            columns=GROUP_COL,
            values=metric,
        )
        .dropna(subset=["PT-R", "RT"])
        .sort_index()
    )

    if paired.empty:
        wilcoxon_w = np.nan
        wilcoxon_p = np.nan

    else:
        differences = (
            paired["PT-R"] - paired["RT"]
        ).to_numpy()

        if np.allclose(
            differences,
            0,
            equal_nan=False,
        ):
            wilcoxon_w = 0.0
            wilcoxon_p = 1.0

        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                wilcoxon_result = wilcoxon(
                    paired["PT-R"],
                    paired["RT"],
                    alternative="two-sided",
                    zero_method="wilcox",
                    correction=False,
                    method="auto",
                )

            wilcoxon_w = float(
                wilcoxon_result.statistic
            )
            wilcoxon_p = float(
                wilcoxon_result.pvalue
            )

    result_row = {
        "metric": metric,
        "MWU_U": mw_u,
        "MWU_p": mw_p,
        "Wilcoxon_W": wilcoxon_w,
        "Wilcoxon_p": wilcoxon_p,
        "paired_n": len(paired),
    }

    for group in GROUP_ORDER:
        median, q1, q3, n = iqr_summary(
            df.loc[
                df[GROUP_COL] == group,
                metric,
            ]
        )

        result_row[f"{group}_n"] = n
        result_row[f"{group}_median"] = median
        result_row[f"{group}_Q1"] = q1
        result_row[f"{group}_Q3"] = q3

    results.append(result_row)


results_df = pd.DataFrame(results)

valid_mwu = results_df["MWU_p"].notna()
results_df["MWU_q_BH"] = np.nan

if valid_mwu.any():
    results_df.loc[
        valid_mwu,
        "MWU_q_BH",
    ] = multipletests(
        results_df.loc[valid_mwu, "MWU_p"],
        method="fdr_bh",
    )[1]

valid_wilcoxon = results_df["Wilcoxon_p"].notna()
results_df["Wilcoxon_q_BH"] = np.nan

if valid_wilcoxon.any():
    results_df.loc[
        valid_wilcoxon,
        "Wilcoxon_q_BH",
    ] = multipletests(
        results_df.loc[
            valid_wilcoxon,
            "Wilcoxon_p",
        ],
        method="fdr_bh",
    )[1]


results_df.to_csv(
    OUTPUT_DIR / "halo_nonparametric_statistics.csv",
    index=False,
    encoding="utf-8-sig",
)

results_df.to_excel(
    OUTPUT_DIR / "halo_nonparametric_statistics.xlsx",
    index=False,
)


rng = np.random.default_rng(20260805)

fig, axes = plt.subplots(
    2,
    4,
    figsize=(7.2, 4.05),
    constrained_layout=False,
)

axes = axes.ravel()


for panel_index, (ax, metric) in enumerate(
    zip(axes, METRICS)
):
    group_data = [
        df.loc[
            df[GROUP_COL] == group,
            metric,
        ]
        .dropna()
        .to_numpy()
        for group in GROUP_ORDER
    ]

    boxplot_result = ax.boxplot(
        group_data,
        positions=[0, 1, 2],
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={
            "color": "black",
            "linewidth": 0.75,
        },
        whiskerprops={
            "color": "black",
            "linewidth": 0.55,
        },
        capprops={
            "color": "black",
            "linewidth": 0.55,
        },
        boxprops={
            "edgecolor": "black",
            "linewidth": 0.55,
        },
    )

    for box_patch, group in zip(
        boxplot_result["boxes"],
        GROUP_ORDER,
    ):
        box_patch.set_facecolor(COLORS[group])
        box_patch.set_alpha(0.78)

    x_pt_nr = rng.normal(
        loc=0,
        scale=0.055,
        size=len(group_data[0]),
    )

    ax.scatter(
        x_pt_nr,
        group_data[0],
        s=5.5,
        facecolor=COLORS["PT-NR"],
        edgecolor="white",
        linewidth=0.18,
        alpha=0.62,
        zorder=3,
    )

    paired_plot = (
        df.loc[
            df[PAIR_COL].isin(valid_pair_ids)
            & df[GROUP_COL].isin(["PT-R", "RT"]),
            [PAIR_COL, GROUP_COL, metric],
        ]
        .pivot(
            index=PAIR_COL,
            columns=GROUP_COL,
            values=metric,
        )
        .dropna(subset=["PT-R", "RT"])
        .sort_index()
    )

    x_pt_r = np.full(
        len(paired_plot),
        1.0,
    )

    x_rt = np.full(
        len(paired_plot),
        2.0,
    )

    for x1, x2, y1, y2 in zip(
        x_pt_r,
        x_rt,
        paired_plot["PT-R"],
        paired_plot["RT"],
    ):
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#8a8a8a",
            linewidth=0.32,
            alpha=0.34,
            zorder=1,
        )

    ax.scatter(
        x_pt_r,
        paired_plot["PT-R"],
        s=6.0,
        facecolor=COLORS["PT-R"],
        edgecolor="white",
        linewidth=0.18,
        alpha=0.72,
        zorder=3,
    )

    ax.scatter(
        x_rt,
        paired_plot["RT"],
        s=6.0,
        facecolor=COLORS["RT"],
        edgecolor="white",
        linewidth=0.18,
        alpha=0.72,
        zorder=3,
    )

    statistical_row = results_df.loc[
        results_df["metric"] == metric
    ].iloc[0]

    finite_arrays = [
        values[np.isfinite(values)]
        for values in group_data
        if len(values) > 0
    ]

    if not finite_arrays:
        raise ValueError(
            f"指标 {metric} 没有可用于绘图的有效数值。"
        )

    finite_values = np.concatenate(finite_arrays)

    data_min = float(np.min(finite_values))
    data_max = float(np.max(finite_values))

    value_span = data_max - data_min

    if value_span <= 0:
        value_span = max(
            abs(data_max),
            1.0,
        )

    bracket_height = 0.025 * value_span

    first_bracket_y = (
        data_max + 0.075 * value_span
    )

    second_bracket_y = (
        data_max + 0.205 * value_span
    )

    add_p_bracket(
        ax=ax,
        x1=0,
        x2=1,
        y=first_bracket_y,
        h=bracket_height,
        label=fmt_p(
            statistical_row["MWU_p"]
        ),
    )

    add_p_bracket(
        ax=ax,
        x1=1,
        x2=2,
        y=second_bracket_y,
        h=bracket_height,
        label=fmt_p(
            statistical_row["Wilcoxon_p"]
        ),
    )

    if data_min >= 0:
        lower_limit = 0
    else:
        lower_limit = (
            data_min - 0.06 * value_span
        )

    ax.set_ylim(
        lower_limit,
        data_max + 0.34 * value_span,
    )

    ax.set_xlim(-0.48, 2.48)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(GROUP_ORDER)

    ax.set_title(
        PANEL_TITLES[metric],
        pad=2.2,
        fontweight="bold",
    )

    ax.set_ylabel("Value")

    ax.grid(
        axis="y",
        color="#d9d9d9",
        linewidth=0.35,
        alpha=0.65,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="x",
        pad=1.2,
    )

    ax.tick_params(
        axis="y",
        pad=1.2,
    )

    ax.text(
        -0.14,
        1.035,
        chr(ord("A") + panel_index),
        transform=ax.transAxes,
        fontsize=5.5,
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="bottom",
    )


axes[-1].axis("off")


legend_handles = [
    Patch(
        facecolor=COLORS[group],
        edgecolor="black",
        linewidth=0.45,
        label=group,
    )
    for group in GROUP_ORDER
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.997),
    handlelength=1.2,
    handleheight=0.8,
    columnspacing=1.7,
)


fig.subplots_adjust(
    left=0.075,
    right=0.992,
    bottom=0.105,
    top=0.91,
    wspace=0.46,
    hspace=0.48,
)

fig.savefig(
    OUTPUT_DIR / "halo_boxplots.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_DIR / "halo_boxplots.svg",
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_DIR / "halo_boxplots.png",
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_DIR / "halo_boxplots.tiff",
    dpi=600,
    bbox_inches="tight",
    pil_kwargs={
        "compression": "tiff_lzw"
    },
)

plt.close(fig)


display_columns = [
    "metric",
    "MWU_U",
    "MWU_p",
    "MWU_q_BH",
    "Wilcoxon_W",
    "Wilcoxon_p",
    "Wilcoxon_q_BH",
    "paired_n",
]

print(
    results_df[display_columns].to_string(
        index=False
    )
)

print(
    f"\n所有结果已保存至："
    f"{OUTPUT_DIR.resolve()}"
)
```

```
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from sklearn.metrics import auc, roc_curve



INPUT_FILE = Path("halo_auc.xlsx")
SHEET_NAME = "Sheet1"

OUT_FIGURE = Path("halo_auc_roc.pdf")
OUT_RESULTS = Path("halo_auc_roc_results.csv")

LABEL_COLUMN = "Group_Label"
POSITIVE_LABEL = "PT-R"
NEGATIVE_LABEL = "PT-NR"

FEATURES = [
    "ternary_ratio_20um",
    "ternary_ratio_40um",
    "ternary_ratio_60um",
    "ternary_ratio_80um",
    "% SF3B1+Tum Positive Cells",
    "% FCGBP+TAM Positive Cells",
    "% TNFRSF9+Treg Positive Cells",
    "EBV-DNA",
    "T stage",
    "N stage",
    "TNM",
]

AUTO_REVERSE = True



FONT_SIZE = 5


FIGURE_SIZE = (2.5,2.5)


def set_publication_style():
    """设置适用于期刊出版的Matplotlib全局样式。"""

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "Liberation Sans",
            "DejaVu Sans",
        ],
        "font.size": FONT_SIZE,

        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",

        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,

        "legend.fontsize": FONT_SIZE,
        "legend.title_fontsize": FONT_SIZE,

        "axes.linewidth": 0.55,

        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "lines.linewidth": 0.9,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "svg.fonttype": "none",

        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    })


def get_youden_cutoff(fpr, tpr, thresholds):
    """
    根据Youden指数选择最佳截点。

    Youden J
        = sensitivity + specificity - 1
        = tpr - fpr
    """

    finite_mask = np.isfinite(thresholds)

    if not finite_mask.any():
        return np.nan, np.nan, np.nan, np.nan

    finite_indexes = np.flatnonzero(finite_mask)
    youden_values = tpr[finite_mask] - fpr[finite_mask]

    best_local_index = int(np.argmax(youden_values))
    best_index = finite_indexes[best_local_index]

    cutoff = float(thresholds[best_index])
    sensitivity = float(tpr[best_index])
    specificity = float(1 - fpr[best_index])
    youden_j = float(youden_values[best_local_index])

    return cutoff, sensitivity, specificity, youden_j


def main():

    set_publication_style()


    df = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
    )


    required_columns = [LABEL_COLUMN] + FEATURES

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Excel中缺少以下列：{missing_columns}"
        )


    labels = (
        df[LABEL_COLUMN]
        .astype("string")
        .str.strip()
    )

    observed_labels = set(
        labels.dropna().unique()
    )

    expected_labels = {
        POSITIVE_LABEL,
        NEGATIVE_LABEL,
    }

    unexpected_labels = (
        observed_labels - expected_labels
    )

    if unexpected_labels:
        raise ValueError(
            f"{LABEL_COLUMN}中存在非预期标签："
            f"{sorted(unexpected_labels)}"
        )

    y_all = labels.map({
        POSITIVE_LABEL: 1,
        NEGATIVE_LABEL: 0,
    })

    print(f"总样本数：{len(df)}")
    print(f"PT-R：{int((y_all == 1).sum())}")
    print(f"PT-NR：{int((y_all == 0).sum())}")


    results = []
    curves = []

    for feature in FEATURES:

        x_all = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        valid_mask = (
            y_all.notna()
            & x_all.notna()
        )

        y = (
            y_all.loc[valid_mask]
            .astype(int)
            .to_numpy()
        )

        x = (
            x_all.loc[valid_mask]
            .astype(float)
            .to_numpy()
        )

        if np.unique(y).size != 2:
            raise ValueError(
                f"{feature}的有效数据中未同时包含"
                f"{POSITIVE_LABEL}和{NEGATIVE_LABEL}"
            )

        if np.unique(x).size < 2:
            raise ValueError(
                f"{feature}的有效数值少于2种，"
                "无法计算ROC曲线"
            )

        raw_fpr, raw_tpr, _ = roc_curve(
            y,
            x,
            pos_label=1,
        )

        raw_auc = float(
            auc(raw_fpr, raw_tpr)
        )

        reversed_direction = (
            AUTO_REVERSE
            and raw_auc < 0.5
        )

        if reversed_direction:
            score = -x
        else:
            score = x

        fpr, tpr, thresholds = roc_curve(
            y,
            score,
            pos_label=1,
        )

        displayed_auc = float(
            auc(fpr, tpr)
        )

        cutoff, sensitivity, specificity, youden_j = (
            get_youden_cutoff(
                fpr,
                tpr,
                thresholds,
            )
        )

        if reversed_direction:
            original_cutoff = -cutoff
            direction = "数值越低越倾向PT-R"
            decision_rule = (
                f"{feature} <= "
                f"{original_cutoff:.6g}"
            )
        else:
            original_cutoff = cutoff
            direction = "数值越高越倾向PT-R"
            decision_rule = (
                f"{feature} >= "
                f"{original_cutoff:.6g}"
            )

        results.append({
            "feature": feature,
            "n": int(valid_mask.sum()),
            "PT-R_n": int(y.sum()),
            "PT-NR_n": int((1 - y).sum()),
            "raw_auc": raw_auc,
            "displayed_auc": displayed_auc,
            "direction": direction,
            "best_cutoff": original_cutoff,
            "decision_rule_for_PT-R": decision_rule,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "youden_j": youden_j,
        })

        curves.append({
            "feature": feature,
            "fpr": fpr,
            "tpr": tpr,
            "auc": displayed_auc,
        })

    curves = sorted(
        curves,
        key=lambda item: item["auc"],
        reverse=True,
    )


    fig, ax = plt.subplots(
        figsize=FIGURE_SIZE,
    )

    tab20 = plt.get_cmap("tab20")

    color_indexes = [
        0, 2, 4, 6, 8, 10,
        12, 14, 16, 18, 1,
    ]

    colors = [
        tab20(index)
        for index in color_indexes
    ]

    line_styles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (5, 1.5)),
        (0, (3, 1.2)),
        (0, (1, 1)),
        (0, (5, 1, 1, 1)),
        (0, (3, 1, 1, 1)),
        (0, (4, 1, 1, 1, 1, 1)),
        (0, (2, 1)),
    ]

    for index, curve in enumerate(curves):

        ax.plot(
            curve["fpr"],
            curve["tpr"],
            color=colors[index],
            linestyle=line_styles[index],
            linewidth=0.9,
            solid_capstyle="round",
            zorder=3,
            label=(
                f'{curve["feature"]} '
                f'(AUC = {curve["auc"]:.3f})'
            ),
        )

    ax.plot(
        [0, 1],
        [0, 1],
        color="0.35",
        linestyle=(0, (3, 2)),
        linewidth=0.7,
        zorder=2,
        label="Chance (AUC = 0.500)",
    )


    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    try:
        ax.set_box_aspect(1)
    except AttributeError:
        ax.set_aspect(
            "equal",
            adjustable="box",
        )

    ax.set_xlabel(
        "False-positive rate (1 − specificity)",
        fontsize=FONT_SIZE,
        labelpad=2.5,
    )

    ax.set_ylabel(
        "True-positive rate (sensitivity)",
        fontsize=FONT_SIZE,
        labelpad=2.5,
    )

    ax.set_title(
        "ROC curves for predicting PT-R",
        fontsize=FONT_SIZE,
        fontweight="bold",
        loc="left",
        pad=4,
    )

    ax.xaxis.set_major_locator(
        MultipleLocator(0.2)
    )

    ax.yaxis.set_major_locator(
        MultipleLocator(0.2)
    )

    ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=FONT_SIZE,
        width=0.5,
        length=2.2,
        pad=1.5,
        direction="out",
        top=False,
        right=False,
    )

    ax.set_axisbelow(True)

    ax.grid(
        visible=True,
        which="major",
        linestyle=(0, (1.2, 2.0)),
        linewidth=0.35,
        color="0.82",
        alpha=1.0,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("black")


    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        fontsize=FONT_SIZE,
        frameon=False,
        handlelength=3.0,
        handletextpad=0.6,
        labelspacing=0.55,
        borderaxespad=0,
        columnspacing=1.0,
    )

    for text in legend.get_texts():
        text.set_fontsize(FONT_SIZE)

    fig.subplots_adjust(
        left=0.10,
        right=0.63,
        bottom=0.14,
        top=0.91,
    )


    fig.savefig(
        OUT_FIGURE,
        format="pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={
            "Title": "ROC curves for predicting PT-R",
            "Subject": "Receiver operating characteristic analysis",
            "Creator": "Python Matplotlib",
        },
    )

    plt.show(fig)


    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="displayed_auc",
        ascending=False,
    )

    results_df.to_csv(
        OUT_RESULTS,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )


    print("\nROC计算结果：")

    print(
        results_df[
            [
                "feature",
                "n",
                "raw_auc",
                "displayed_auc",
                "best_cutoff",
                "sensitivity",
                "specificity",
                "youden_j",
                "decision_rule_for_PT-R",
            ]
        ]
        .round({
            "raw_auc": 3,
            "displayed_auc": 3,
            "best_cutoff": 4,
            "sensitivity": 3,
            "specificity": 3,
            "youden_j": 3,
        })
        .to_string(index=False)
    )

    print(
        f"\nROC矢量图已保存至："
        f"{OUT_FIGURE.resolve()}"
    )

    print(
        f"统计结果已保存至："
        f"{OUT_RESULTS.resolve()}"
    )


if __name__ == "__main__":
    main()
```





#Fig.7k, l
```

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FixedFormatter

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test



FILE = "halo_auc.xlsx"
SHEET = "Sheet1"

OUT_PREFIX = "ternary_ratio_40um_survival_cox"

FONT_SIZE = 5

FIG_WIDTH = 7.09
FIG_HEIGHT = 2.80



arial_available = any(
    f.name.lower() == "arial"
    for f in font_manager.fontManager.ttflist
)

if not arial_available:
    warnings.warn(
        "Arial was not found in the current Python environment. "
        "Matplotlib will use a fallback font. "
        "Install Arial if the journal strictly requires it."
    )

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FONT_SIZE,

    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.linewidth": 0.6,

    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,

    "legend.fontsize": FONT_SIZE,
    "legend.frameon": False,

    "lines.linewidth": 0.8,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.bbox": "tight",
})



df = pd.read_excel(
    FILE,
    sheet_name=SHEET
)

required_cols = [
    "ternary_ratio_40um",
    "EBV-DNA",
    "T stage",
    "Status",
    "Time",
    "Sex",
    "Age",
]

missing_cols = [x for x in required_cols if x not in df.columns]

if missing_cols:
    raise ValueError(
        f"Missing required columns: {missing_cols}"
    )



dat = df[required_cols].copy()

for col in required_cols:
    dat[col] = pd.to_numeric(
        dat[col],
        errors="coerce"
    )

dat = dat.dropna().copy()

if not set(dat["Status"].unique()).issubset({0, 1}):
    raise ValueError(
        "Status must be coded as 0/1."
    )

if not set(dat["Sex"].unique()).issubset({0, 1}):
    warnings.warn(
        "Sex is not strictly coded as 0/1. "
        "Please check the coding."
    )

if (dat["Time"] <= 0).any():
    raise ValueError(
        "Survival time must be > 0."
    )


print("=" * 60)
print("Dataset summary")
print("=" * 60)
print(f"N = {len(dat)}")
print(f"Events = {int(dat['Status'].sum())}")
print(f"Censored = {int((dat['Status'] == 0).sum())}")



median_cutoff = dat["ternary_ratio_40um"].median()

dat["ternary_group"] = np.where(
    dat["ternary_ratio_40um"] > median_cutoff,
    "High",
    "Low"
)

print(f"\nMedian ternary_ratio_40um = {median_cutoff:.6f}")

group_summary = (
    dat.groupby("ternary_group", observed=True)
    .agg(
        N=("Status", "size"),
        Events=("Status", "sum")
    )
)

print("\nKM groups:")
print(group_summary)



low = dat[
    dat["ternary_group"] == "Low"
].copy()

high = dat[
    dat["ternary_group"] == "High"
].copy()

lr = logrank_test(
    low["Time"],
    high["Time"],
    event_observed_A=low["Status"],
    event_observed_B=high["Status"],
)

logrank_p = lr.p_value


def format_p(p):
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"


print(
    f"\nLog-rank P = {logrank_p:.6g}"
)



cox_dat = dat[
    [
        "Time",
        "Status",
        "ternary_ratio_40um",
        "EBV-DNA",
        "T stage",
        "Sex",
        "Age",
    ]
].copy()


cox_dat["ternary_per_0.1"] = (
    cox_dat["ternary_ratio_40um"] / 0.1
)

cox_dat["log10_EBV_DNA"] = np.log10(
    cox_dat["EBV-DNA"] + 1
)

cox_dat["T_stage"] = cox_dat["T stage"]

cox_dat["Sex_1_vs_0"] = cox_dat["Sex"]

cox_dat["Age_per_10y"] = (
    cox_dat["Age"] / 10
)


cox_vars = [
    "ternary_per_0.1",
    "log10_EBV_DNA",
    "T_stage",
    "Sex_1_vs_0",
    "Age_per_10y",
]


cox_model_dat = cox_dat[
    ["Time", "Status"] + cox_vars
].copy()



cph = CoxPHFitter()

cph.fit(
    cox_model_dat,
    duration_col="Time",
    event_col="Status",
)

cox_summary = cph.summary.loc[
    cox_vars,
    [
        "coef",
        "exp(coef)",
        "exp(coef) lower 95%",
        "exp(coef) upper 95%",
        "p",
    ]
].copy()


cox_summary.columns = [
    "coef",
    "HR",
    "CI_low",
    "CI_high",
    "P",
]


variable_labels = {
    "ternary_per_0.1":
        "Ternary ratio\n(per 0.1 increase)",

    "log10_EBV_DNA":
        "EBV-DNA\n(per 10-fold increase)",

    "T_stage":
        "T stage\n(per stage)",

    "Sex_1_vs_0":
        "Sex\n(1 vs 0)",

    "Age_per_10y":
        "Age\n(per 10 years)",
}


print("\n" + "=" * 60)
print("Multivariable Cox model")
print("=" * 60)

print(
    cox_summary[
        ["HR", "CI_low", "CI_high", "P"]
    ].round(4)
)



max_time = dat["Time"].max()

risk_times = np.arange(
    0,
    np.floor(max_time / 12) * 12 + 1,
    12
)

if len(risk_times) < 3:
    risk_times = np.linspace(
        0,
        max_time,
        5
    )



fig = plt.figure(
    figsize=(FIG_WIDTH, FIG_HEIGHT)
)

outer = gridspec.GridSpec(
    nrows=2,
    ncols=2,
    figure=fig,
    width_ratios=[1.05, 1.10],
    height_ratios=[4.1, 1.15],
    wspace=0.36,
    hspace=0.04,
)



ax_km = fig.add_subplot(
    outer[0, 0]
)

ax_risk = fig.add_subplot(
    outer[1, 0],
    sharex=ax_km
)


kmf_low = KaplanMeierFitter()
kmf_high = KaplanMeierFitter()



kmf_low.fit(
    durations=low["Time"],
    event_observed=low["Status"],
    label=(
        f"Low (≤ median, n={len(low)})"
    ),
)

kmf_low.plot_survival_function(
    ax=ax_km,
    ci_show=True,
    ci_alpha=0.10,
    show_censors=True,
    linewidth=0.9,
    linestyle="--",
    censor_styles={
        "marker": "|",
        "ms": 3.0,
        "mew": 0.6,
    },
)



kmf_high.fit(
    durations=high["Time"],
    event_observed=high["Status"],
    label=(
        f"High (> median, n={len(high)})"
    ),
)

kmf_high.plot_survival_function(
    ax=ax_km,
    ci_show=True,
    ci_alpha=0.10,
    show_censors=True,
    linewidth=0.9,
    linestyle="-",
    censor_styles={
        "marker": "|",
        "ms": 3.0,
        "mew": 0.6,
    },
)



ax_km.set_ylim(
    0,
    1.03
)

ax_km.set_xlim(
    0,
    max(risk_times[-1], max_time)
)

ax_km.set_ylabel(
    "Survival probability"
)

ax_km.set_xlabel("")

ax_km.set_yticks(
    [0, 0.25, 0.50, 0.75, 1.00]
)

ax_km.set_yticklabels(
    ["0", "0.25", "0.50", "0.75", "1.00"]
)

ax_km.tick_params(
    axis="x",
    labelbottom=False
)

ax_km.spines["top"].set_visible(False)
ax_km.spines["right"].set_visible(False)

ax_km.legend(
    loc="lower left",
    frameon=False,
    handlelength=2.2,
    borderaxespad=0.3,
    labelspacing=0.25,
)



p_text = (
    f"Log-rank $P$ {format_p(logrank_p)}"
    if logrank_p < 0.001
    else f"Log-rank $P$ = {format_p(logrank_p)}"
)

ax_km.text(
    0.97,
    0.96,
    p_text,
    transform=ax_km.transAxes,
    ha="right",
    va="top",
    fontsize=FONT_SIZE,
)


ax_km.text(
    0.97,
    0.87,
    f"Median cutoff = {median_cutoff:.4f}",
    transform=ax_km.transAxes,
    ha="right",
    va="top",
    fontsize=FONT_SIZE,
)


ax_km.text(
    -0.16,
    1.06,
    "A",
    transform=ax_km.transAxes,
    fontsize=6,
    fontweight="bold",
    ha="left",
    va="bottom",
)



ax_risk.set_ylim(
    -0.05,
    2.1
)

ax_risk.set_yticks([])

ax_risk.set_xlabel(
    "Time (months)"
)

ax_risk.set_xticks(
    risk_times
)

ax_risk.tick_params(
    axis="x",
    length=2,
    pad=1
)


for spine in [
    "left",
    "right",
    "top"
]:
    ax_risk.spines[spine].set_visible(False)

ax_risk.spines["bottom"].set_linewidth(0.5)



ax_risk.text(
    -0.04,
    0.97,
    "No. at risk",
    transform=ax_risk.transAxes,
    ha="right",
    va="center",
    fontsize=FONT_SIZE,
    clip_on=False,
)

ax_risk.text(
    -0.04,
    0.60,
    "Low",
    transform=ax_risk.transAxes,
    ha="right",
    va="center",
    fontsize=FONT_SIZE,
    clip_on=False,
)

ax_risk.text(
    -0.04,
    0.23,
    "High",
    transform=ax_risk.transAxes,
    ha="right",
    va="center",
    fontsize=FONT_SIZE,
    clip_on=False,
)



risk_y = {
    "Low": 1.20,
    "High": 0.42,
}

for group_name, group_df in [
    ("Low", low),
    ("High", high),
]:

    for t in risk_times:

        n_risk = np.sum(
            group_df["Time"] >= t
        )

        ax_risk.text(
            t,
            risk_y[group_name],
            str(int(n_risk)),
            ha="center",
            va="center",
            fontsize=FONT_SIZE,
        )




right_grid = outer[:, 1].subgridspec(
    1,
    3,
    width_ratios=[
        1.40,
        2.05,
        1.75
    ],
    wspace=0.04,
)

ax_var = fig.add_subplot(
    right_grid[0, 0]
)

ax_forest = fig.add_subplot(
    right_grid[0, 1]
)

ax_stat = fig.add_subplot(
    right_grid[0, 2]
)



forest_df = cox_summary.copy()

forest_df["label"] = [
    variable_labels[x]
    for x in forest_df.index
]

n_forest = len(forest_df)

y_positions = np.arange(
    n_forest,
    0,
    -1
)



ax_var.set_xlim(
    0,
    1
)

ax_var.set_ylim(
    0.35,
    n_forest + 1.05
)

ax_var.axis("off")


ax_var.text(
    0,
    n_forest + 0.72,
    "Variable",
    ha="left",
    va="center",
    fontweight="bold",
)


for y, label in zip(
    y_positions,
    forest_df["label"]
):
    ax_var.text(
        0,
        y,
        label,
        ha="left",
        va="center",
        linespacing=0.95,
    )


ax_var.text(
    -0.10,
    1.015,
    "B",
    transform=ax_var.transAxes,
    fontsize=6,
    fontweight="bold",
    ha="left",
    va="bottom",
)



HR = forest_df["HR"].values
LOW = forest_df["CI_low"].values
HIGH = forest_df["CI_high"].values


lower_error = HR - LOW
upper_error = HIGH - HR


ax_forest.errorbar(
    HR,
    y_positions,
    xerr=[
        lower_error,
        upper_error
    ],
    fmt="s",
    markersize=3.0,
    capsize=1.8,
    capthick=0.6,
    elinewidth=0.7,
    markeredgewidth=0.5,
)


ax_forest.axvline(
    1,
    linestyle="--",
    linewidth=0.6,
    zorder=0,
)


ax_forest.set_xscale(
    "log"
)

ax_forest.set_ylim(
    0.35,
    n_forest + 1.05
)

ax_forest.set_yticks([])

ax_forest.set_xlabel(
    "Adjusted hazard ratio"
)



ci_min = np.nanmin(LOW)
ci_max = np.nanmax(HIGH)

xmin = max(
    0.1,
    ci_min * 0.65
)

xmax = min(
    20,
    ci_max * 1.40
)

xmin = min(
    xmin,
    0.7
)

xmax = max(
    xmax,
    1.5
)

ax_forest.set_xlim(
    xmin,
    xmax
)


candidate_ticks = [
    0.1,
    0.2,
    0.5,
    1,
    2,
    5,
    10,
    20,
]

forest_ticks = [
    t for t in candidate_ticks
    if xmin <= t <= xmax
]

ax_forest.xaxis.set_major_locator(
    FixedLocator(forest_ticks)
)

ax_forest.xaxis.set_major_formatter(
    FixedFormatter(
        [
            f"{x:g}"
            for x in forest_ticks
        ]
    )
)


ax_forest.spines["top"].set_visible(False)
ax_forest.spines["right"].set_visible(False)
ax_forest.spines["left"].set_visible(False)


ax_forest.text(
    0.50,
    1.005,
    "Adjusted HR",
    transform=ax_forest.transAxes,
    ha="center",
    va="bottom",
    fontweight="bold",
)



ax_stat.set_xlim(
    0,
    1
)

ax_stat.set_ylim(
    0.35,
    n_forest + 1.05
)

ax_stat.axis("off")


ax_stat.text(
    0.00,
    n_forest + 0.72,
    "HR (95% CI)",
    ha="left",
    va="center",
    fontweight="bold",
)

ax_stat.text(
    1.00,
    n_forest + 0.72,
    "$P$",
    ha="right",
    va="center",
    fontweight="bold",
)


for y, (_, row) in zip(
    y_positions,
    forest_df.iterrows()
):

    hr_text = (
        f"{row['HR']:.2f} "
        f"({row['CI_low']:.2f}–"
        f"{row['CI_high']:.2f})"
    )

    p = row["P"]

    if p < 0.001:
        pval_text = "<0.001"
    else:
        pval_text = f"{p:.3f}"

    ax_stat.text(
        0.00,
        y,
        hr_text,
        ha="left",
        va="center",
    )

    ax_stat.text(
        1.00,
        y,
        pval_text,
        ha="right",
        va="center",
    )



fig.subplots_adjust(
    left=0.085,
    right=0.985,
    bottom=0.17,
    top=0.94,
)



pdf_file = f"{OUT_PREFIX}.pdf"
tiff_file = f"{OUT_PREFIX}.tiff"
png_file = f"{OUT_PREFIX}.png"


fig.savefig(
    pdf_file,
    format="pdf",
    bbox_inches="tight",
)

fig.savefig(
    tiff_file,
    format="tiff",
    dpi=600,
    bbox_inches="tight",
    pil_kwargs={
        "compression": "tiff_lzw"
    },
)

fig.savefig(
    png_file,
    format="png",
    dpi=600,
    bbox_inches="tight",
)


plt.show()


print("\n" + "=" * 60)
print("Files saved")
print("=" * 60)
print(pdf_file)
print(tiff_file)
print(png_file)



cox_export = forest_df[
    [
        "HR",
        "CI_low",
        "CI_high",
        "P",
    ]
].copy()

cox_export.insert(
    0,
    "Variable",
    [
        variable_labels[x].replace("\n", " ")
        for x in cox_export.index
    ]
)

cox_export.to_csv(
    "multivariable_cox_results.csv",
    index=False
)

print(
    "\nCox results saved to: "
    "multivariable_cox_results.csv"
)
```


