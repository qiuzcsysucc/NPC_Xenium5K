#Fig.3a
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
import omicverse as ov

ov.plot_set(font_path="Arial")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

celltype_files = [
    ("xenium_cd4.h5ad","CD4 T cells"),
    ("xenium_cd8.h5ad","CD8 T cells"),
    ("xenium_dc.h5ad","Dendritic cells"),
    ("xenium_tam.h5ad","Macrophages"),
    ("xenium_neu.h5ad","Neutrophils"),
    ("xenium_fibro.h5ad","Fibroblasts"),
    ("xenium_endo.h5ad","Endothelial cells"),
    ("xenium_pericyte.h5ad","Pericytes")
]

fig, axes = plt.subplots(2,4,figsize=(16,8))

for ax,(file,title) in zip(axes.flatten(),celltype_files):
    adata=sc.read(file)
    ov.pl.embedding(
        adata,
        basis="X_umap",
        color=["cell_type"],
        show=False,
        frameon="small",
        ax=ax,
        title=title
    )

plt.tight_layout()
plt.savefig("Fig3a.pdf",bbox_inches="tight")
plt.close()


#Fig.3b
import pandas as pd
import seaborn as sns

sns.heatmap(
    prop.pivot_table(
        index="cell_type",
        columns="batch",
        values="roe",
        fill_value=0
    ),
    cmap="RdBu_r",
    ax=ax
)

plt.tight_layout()
plt.savefig("Fig3b.pdf",bbox_inches="tight")
plt.close()


#Fig.3c
roe_data=pd.read_csv("cellular_niche_roe.csv",index_col=0)

fig,ax=plt.subplots(figsize=(12,5))

sns.heatmap(
    roe_data,
    cmap="RdBu_r",
    center=0,
    ax=ax
)

plt.tight_layout()
plt.savefig("Fig3c.pdf",bbox_inches="tight")
plt.close()


#Fig.3d,e
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable

def plot_combined_1x4(
    adata,
    group_1='major_celltype',
    col_map_1=None,
    group_2='Niche',
    col_map_2=None,
    group_3='cell_type_plot',
    subset_3=None,
    col_map_3=None,
    rna_list=['CXCL9', 'CXCR3'],
    rna_colors={'CXCL9': "#4daf4a", 'CXCR3': "#ffff33"},
    value_key_4='CXCL9::CXCR3',
    boundary_col='boundary',
    scale_length_um=50.0,
    figure_size=(20, 6),
    dpi=300,
    wspace=0.025,
    legend_ncol=2,
    bottom_margin=0.2,
    output=None
):
    fig, axes = plt.subplots(1, 4, figsize=figure_size, dpi=dpi)
    fig.patch.set_facecolor("white")

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.92,
        bottom=bottom_margin,
        wspace=wspace
    )

    obs = adata.obs.copy()

    def get_fig_coords(ax, x, y):
        disp = ax.transData.transform((x, y))
        return fig.transFigure.inverted().transform(disp)

    def draw_scalebar(ax, x_min, x_max, y_min, y_max, width_um):
        mid_y = 0.5 * (y_min + y_max)
        mid_x = 0.5 * (x_min + x_max)

        left_fig, _ = get_fig_coords(ax, x_min, mid_y)
        right_fig, _ = get_fig_coords(ax, x_max, mid_y)
        _, bottom_fig = get_fig_coords(ax, mid_x, y_max)

        scale_y = bottom_fig - 0.02

        bar_start = left_fig
        frac = min(scale_length_um / width_um, 0.6)
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

        return bar_end, scale_y, right_fig, bottom_fig

    ax1 = axes[0]
    ax1.set_title("By major celltype", fontsize=14, color="black")
    ax1.set_facecolor("black")

    xs1, ys1 = [], []

    for _, row in obs.iterrows():
        b = row[boundary_col]

        if pd.isna(b):
            continue

        try:
            coords = ast.literal_eval(b) if isinstance(b, str) else b
        except:
            continue

        xy = np.asarray(coords)
        xs1.extend(xy[:, 0])
        ys1.extend(xy[:, 1])

        c = (
            col_map_1.get(
                row.get(group_1),
                (0.7, 0.7, 0.7, 0.9)
            )
            if col_map_1
            else (0.7, 0.7, 0.7, 0.9)
        )

        patch = Polygon(
            xy,
            closed=True,
            fill=True,
            facecolor=c,
            edgecolor="black",
            linewidth=1,
            alpha=1,
            zorder=2
        )

        ax1.add_patch(patch)

    if xs1:
        x_min, x_max = min(xs1), max(xs1)
        y_min, y_max = min(ys1), max(ys1)

        ax1.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor="black",
                zorder=0
            )
        )

        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_max, y_min)
        ax1.axis("off")

        bar_end, scale_y, right_fig, _ = draw_scalebar(
            ax1,
            x_min,
            x_max,
            y_min,
            y_max,
            x_max - x_min
        )

        handles = []

        if col_map_1:
            types = [
                t for t in col_map_1.keys()
                if t in obs[group_1].unique()
            ]

            for t in types:
                handles.append(
                    Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="None",
                        color=col_map_1[t],
                        label=t
                    )
                )

        if handles:
            fig.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(bar_end + 0.01, scale_y + 0.01),
                bbox_transform=fig.transFigure,
                frameon=False,
                fontsize=8,
                ncol=legend_ncol
            )

    ax2 = axes[1]
    ax2.set_title("By cellular niche", fontsize=14, color="black")
    ax2.set_facecolor("black")

    xs2, ys2 = [], []

    for _, row in obs.iterrows():
        b = row[boundary_col]

        if pd.isna(b):
            continue

        try:
            coords = ast.literal_eval(b) if isinstance(b, str) else b
        except:
            continue

        xy = np.asarray(coords)
        xs2.extend(xy[:, 0])
        ys2.extend(xy[:, 1])

        c = (
            col_map_2.get(
                row.get(group_2),
                (0.7, 0.7, 0.7, 0.9)
            )
            if col_map_2
            else (0.7, 0.7, 0.7, 0.9)
        )

        patch = Polygon(
            xy,
            closed=True,
            fill=True,
            facecolor=c,
            edgecolor="black",
            linewidth=1,
            alpha=1,
            zorder=2
        )

        ax2.add_patch(patch)

    if xs2:
        x_min, x_max = min(xs2), max(xs2)
        y_min, y_max = min(ys2), max(ys2)

        ax2.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor="black",
                zorder=0
            )
        )

        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_max, y_min)
        ax2.axis("off")

        bar_end, scale_y, right_fig, _ = draw_scalebar(
            ax2,
            x_min,
            x_max,
            y_min,
            y_max,
            x_max - x_min
        )

        handles = []

        if col_map_2:
            types = [
                t for t in col_map_2.keys()
                if t in obs[group_2].unique()
            ]

            for t in types:
                handles.append(
                    Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="None",
                        color=col_map_2[t],
                        label=t
                    )
                )

        if handles:
            fig.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(bar_end + 0.01, scale_y + 0.01),
                bbox_transform=fig.transFigure,
                frameon=False,
                fontsize=8,
                ncol=legend_ncol
            )

    ax3 = axes[2]
    ax3.set_title("By subtype & transcript", fontsize=14, color="black")
    ax3.set_facecolor("black")

    obs_sub_3 = obs.copy()

    if subset_3 and group_3:
        obs_sub_3 = obs_sub_3[
            obs_sub_3[group_3].isin(subset_3)
        ]

    xs3, ys3 = [], []

    for _, row in obs_sub_3.iterrows():
        b = row[boundary_col]

        if pd.isna(b):
            continue

        try:
            coords = ast.literal_eval(b) if isinstance(b, str) else b
        except:
            continue

        xy = np.asarray(coords)
        xs3.extend(xy[:, 0])
        ys3.extend(xy[:, 1])

        edge_c = (
            col_map_3.get(
                row.get(group_3),
                (0.7, 0.7, 0.7, 0.7)
            )
            if col_map_3
            else "gray"
        )

        patch = Polygon(
            xy,
            closed=True,
            fill=False,
            edgecolor=edge_c,
            linewidth=1,
            zorder=2
        )

        ax3.add_patch(patch)

    if "transcripts" in adata.uns:
        transcripts = adata.uns["transcripts"].copy()

        transcripts = transcripts[
            transcripts["feature_name"].isin(rna_list)
        ]

        keep_ids = obs_sub_3.index.astype(str)

        if "cell_id" in transcripts.columns:
            transcripts = transcripts[
                transcripts["cell_id"].isin(keep_ids)
            ]

        for g in rna_list:
            df_g = transcripts[
                transcripts["feature_name"] == g
            ]

            if df_g.empty:
                continue

            ax3.scatter(
                df_g["x_location"].values,
                df_g["y_location"].values,
                s=10,
                c=rna_colors.get(g, "white"),
                linewidths=0,
                alpha=1.0,
                zorder=3
            )

    if xs3:
        x_min, x_max = min(xs3), max(xs3)
        y_min, y_max = min(ys3), max(ys3)

        ax3.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor="black",
                zorder=0
            )
        )

        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_max, y_min)
        ax3.axis("off")

        bar_end, scale_y, right_fig, _ = draw_scalebar(
            ax3,
            x_min,
            x_max,
            y_min,
            y_max,
            x_max - x_min
        )

        h_cells = []

        if subset_3:
            for t in subset_3:
                c = col_map_3.get(t, "gray")

                h_cells.append(
                    Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="None",
                        markerfacecolor="none",
                        markeredgecolor=c,
                        label=t
                    )
                )

        h_genes = []

        for g in rna_list:
            h_genes.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="None",
                    color=rna_colors.get(g, "white"),
                    label=g
                )
            )

        mid_x = (bar_end + right_fig) / 2

        if h_cells:
            fig.legend(
                handles=h_cells,
                loc="upper center",
                bbox_to_anchor=(mid_x, scale_y + 0.01),
                bbox_transform=fig.transFigure,
                frameon=False,
                fontsize=8
            )

        if h_genes:
            fig.legend(
                handles=h_genes,
                loc="upper right",
                bbox_to_anchor=(right_fig, scale_y + 0.01),
                bbox_transform=fig.transFigure,
                frameon=False,
                fontsize=8
            )

    ax4 = axes[3]
    ax4.set_title(
        f"{value_key_4} interaction score",
        fontsize=14,
        color="black"
    )
    ax4.set_facecolor("black")

    obs_sub_4 = obs.copy()

    if subset_3 and group_3:
        obs_sub_4 = obs_sub_4[
            obs_sub_4[group_3].isin(subset_3)
        ]

    vals_raw = pd.to_numeric(
        obs_sub_4[value_key_4],
        errors="coerce"
    ).values.astype(float)

    if len(vals_raw) > 0 and np.any(np.isfinite(vals_raw)):
        vmin = np.nanmin(vals_raw)
        vmax = np.nanmax(vals_raw)
    else:
        vmin, vmax = 0, 1

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = get_cmap("magma")

    val_map = pd.Series(
        vals_raw,
        index=obs_sub_4.index
    )

    xs4, ys4 = [], []

    for cell_id, row in obs_sub_4.iterrows():
        b = row[boundary_col]

        if pd.isna(b):
            continue

        try:
            coords = ast.literal_eval(b) if isinstance(b, str) else b
        except:
            continue

        xy = np.asarray(coords)
        xs4.extend(xy[:, 0])
        ys4.extend(xy[:, 1])

        val = val_map.loc[cell_id]

        face_c = (
            cmap(norm(val))
            if np.isfinite(val)
            else (0.3, 0.3, 0.3, 1.0)
        )

        patch = Polygon(
            xy,
            closed=True,
            fill=True,
            facecolor=face_c,
            edgecolor="black",
            linewidth=1,
            zorder=2
        )

        ax4.add_patch(patch)

    if xs4:
        x_min, x_max = min(xs4), max(xs4)
        y_min, y_max = min(ys4), max(ys4)

        ax4.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor="black",
                zorder=0
            )
        )

        ax4.set_xlim(x_min, x_max)
        ax4.set_ylim(y_max, y_min)
        ax4.axis("off")

        bar_end, scale_y, right_fig, _ = draw_scalebar(
            ax4,
            x_min,
            x_max,
            y_min,
            y_max,
            x_max - x_min
        )

        cbar_width = 0.3 * (right_fig - bar_end)
        cbar_height = 0.015
        cbar_left = right_fig - cbar_width
        cbar_bottom = scale_y - cbar_height / 2

        cax = fig.add_axes([
            cbar_left,
            cbar_bottom,
            cbar_width,
            cbar_height
        ])

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
            value_key_4,
            fontsize=8,
            color="black"
        )

        cbar.ax.tick_params(
            labelsize=6,
            colors="black"
        )

    if output:
        fig.savefig(
            output,
            dpi=dpi,
            facecolor="white",
            bbox_inches=None
        )
        print(f"Figure saved to {output}")

    plt.show()


plot_combined_1x4(
    adata=adata_roi,
    group_1="major_celltype",
    col_map_1=major_celltype_col_map,
    group_2="Niche",
    col_map_2=niche_col_map,
    group_3="cell_type_plot",
    subset_3=[
        "TAM_CXCL9",
        "CD8 T cells",
        "CD4 T cells"
    ],
    col_map_3={
        "CD8 T cells": "#377eb8",
        "CD4 T cells": "#e41a1c",
        "TAM_CXCL9": "#4daf4a"
    },
    rna_list=[
        "CXCL9",
        "CXCR3"
    ],
    rna_colors={
        "CXCL9": "#00E5FF",
        "CXCR3": "#ffff33"
    },
    value_key_4="CXCL9::CXCR3",
    boundary_col="boundary",
    scale_length_um=50.0,
    wspace=0.025,
    legend_ncol=2,
    bottom_margin=0.2,
    output="/home/qiuzc/Spyrrow/fig/CN6.pdf"
)

