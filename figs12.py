#Fig.S12a
import scanpy as sc
import matplotlib.pyplot as plt

adata = sc.read("public_scRNAseq_cohort.h5ad")

sc.pl.umap(
    adata,
    color="major_celltype",
    show=False
)

plt.savefig("FigS10a.pdf", bbox_inches="tight")
plt.close()


#Fig.S12b
sc.pl.umap(
    adata,
    color="dataset",
    show=False
)

plt.savefig("FigS10b.pdf", bbox_inches="tight")
plt.close()


#Fig.S12c
import seaborn as sns
import pandas as pd

marker = [
"MS4A1",
"CD79A",
"CD3E",
"CD4",
"CD8A",
"CD8B",
"IRF8",
"CSF2RA",
"PLVAP",
"COL4A1",
"EPCAM",
"CLDN1",
"COL5A1",
"POSTN",
"KIT",
"GATA2",
"CTSL",
"CD68",
"KLRF1",
"NCAM1",
"MZB1",
"TENT5C"
]

sc.pl.dotplot(
    adata,
    marker,
    groupby="major_celltype",
    show=False
)

plt.savefig("FigS10c.pdf", bbox_inches="tight")
plt.close()


#Fig.S12d
corr = pd.read_csv("TAM_subtype_expression_correlation.csv", index_col=0)

sns.heatmap(
    corr,
    cmap="RdBu_r",
    center=0
)

plt.savefig("FigS10d.pdf", bbox_inches="tight")
plt.close()


#Fig.S12e
similarity = pd.read_csv("TAM_marker_similarity.csv", index_col=0)

sns.heatmap(
    similarity,
    cmap="Reds",
    vmin=0,
    vmax=0.35
)

plt.savefig("FigS10e.pdf", bbox_inches="tight")
plt.close()

#Fig.S12f-g
tam_recruit_activation_receptors = {
    'Top 5 ligand':['IL18','ICOSLG','PVR','PDCD1LG2','CXCL16','IL10']

}
sc.pl.dotplot(
    adata,
    var_names=tam_recruit_activation_receptors,
    groupby="celltype",              standard_scale="var",
    cmap="coolwarm",
    show=False,
)
#plt.savefig('xenium_tam_dotplot_ligand.pdf',bbox_inches='tight')
plt.show()



#Fig.S12h
import sys
import re
import inspect
import textwrap
import importlib

import numpy as np
import matplotlib.pyplot as plt


def find_original_via_plotting_module():

    candidate_modules = []

    for module_name, module in list(sys.modules.items()):

        if module is None:
            continue

        required_names = [
            "plot_sc_lineage_probability",
            "_make_knn_embeddedspace",
            "plot_sc_pb",
        ]

        if not all(hasattr(module, name) for name in required_names):
            continue

        func = getattr(module, "plot_sc_lineage_probability")

        if not callable(func):
            continue

        try:
            source = textwrap.dedent(inspect.getsource(func))
        except (OSError, TypeError):
            continue

        if "fig_ncols" in source and "n_terminal_clusters" in source:
            candidate_modules.append(
                (module_name, module, func, source)
            )

    if len(candidate_modules) == 0:

        possible_module_names = [
            "pyVIA.plotting_via",
            "VIA.plotting_via",
            "via.plotting_via",
            "pyVIA.core",
        ]

        for module_name in possible_module_names:
            try:
                module = importlib.import_module(module_name)
            except (ImportError, ModuleNotFoundError):
                continue

            required_names = [
                "plot_sc_lineage_probability",
                "_make_knn_embeddedspace",
                "plot_sc_pb",
            ]

            if not all(hasattr(module, name) for name in required_names):
                continue

            func = getattr(module, "plot_sc_lineage_probability")

            try:
                source = textwrap.dedent(inspect.getsource(func))
            except (OSError, TypeError):
                continue

            if "fig_ncols" in source and "n_terminal_clusters" in source:
                candidate_modules.append(
                    (module_name, module, func, source)
                )

    if len(candidate_modules) == 0:
        raise RuntimeError(
            "未能找到 VIA 原始绘图模块。"
        )

    for result in candidate_modules:
        source = result[3]

        if re.search(
            r"fig_ncols\s*=\s*min\(\s*3\s*,\s*n_terminal_clusters\s*\)",
            source
        ):
            return result

    for result in candidate_modules:
        source = result[3]

        if re.search(
            r"fig_ncols\s*=\s*min\(\s*4\s*,\s*n_terminal_clusters\s*\)",
            source
        ):
            return result

    return candidate_modules[0]


module_name, plotting_module, original_func, original_source = (
    find_original_via_plotting_module()
)

print(f"找到 VIA 绘图模块：{module_name}")


patched_source, replace_count = re.subn(
    pattern=(
        r"fig_ncols\s*=\s*min\(\s*3\s*,\s*"
        r"n_terminal_clusters\s*\)"
    ),
    repl="fig_ncols = min(4, n_terminal_clusters)",
    string=original_source,
    count=1
)

if replace_count == 0:

    already_four_columns = re.search(
        r"fig_ncols\s*=\s*min\(\s*4\s*,\s*n_terminal_clusters\s*\)",
        original_source
    )

    if already_four_columns:
        patched_func = original_func
        print("当前函数已经设置为最大4列。")

    else:
        raise RuntimeError(
            "找到了 VIA 绘图函数，但没有找到需要修改的代码。"
        )

else:
    exec(
        compile(
            patched_source,
            filename="<VIA_plot_sc_lineage_probability_4cols>",
            mode="exec"
        ),
        plotting_module.__dict__
    )

    patched_func = plotting_module.plot_sc_lineage_probability

    print("已将最大列数由3列修改为4列。")


VIA.core.plot_sc_lineage_probability = patched_func


marker_lineages = [12, 22, 23, 33]

fig, axs = VIA.core.plot_sc_lineage_probability(
    via_object=v0,
    embedding=stavia_embedding,
    marker_lineages=marker_lineages,
    dpi=90,
    cmap_name="coolwarm"
)


for ax in fig.axes:

    for collection in ax.collections:
        collection.set_rasterized(True)

    for image in ax.images:
        image.set_rasterized(True)

    ax.grid(False)


fig.set_size_inches(12, 3)

fig.subplots_adjust(
    left=0.01,
    right=0.99,
    bottom=0.02,
    top=0.94,
    wspace=0.12
)


fig.savefig(
    "lineage_1row_4cols.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.02,
    dpi=600
)

plt.show()


#Fig.S12i

import numpy as np
import pandas as pd
from scipy.sparse import issparse

center_x = 9052
center_y = 19196

half_size = 100

x_min = center_x - half_size
x_max = center_x + half_size
y_min = center_y - half_size
y_max = center_y + half_size

print("ROI x范围:", x_min, x_max)
print("ROI y范围:", y_min, y_max)

cells_roi_mask = (
    (spyrrow_adata.obs["x_centroid"] >= x_min) &
    (spyrrow_adata.obs["x_centroid"] <= x_max) &
    (spyrrow_adata.obs["y_centroid"] >= y_min) &
    (spyrrow_adata.obs["y_centroid"] <= y_max)
)

adata_roi = spyrrow_adata[cells_roi_mask].copy()
print("ROI 内细胞数量:", adata_roi.n_obs)

tr = spyrrow_adata.uns["transcripts"]

tr_roi_mask = (
    (tr["x_location"] >= x_min) &
    (tr["x_location"] <= x_max) &
    (tr["y_location"] >= y_min) &
    (tr["y_location"] <= y_max)
)

tr_roi = tr[tr_roi_mask].copy()
print("ROI 内转录本数量:", tr_roi.shape[0])

adata_roi.uns["transcripts"] = tr_roi

plot_RNA_on_cell_black(
    adata=adata_roi,
    RNA_name=[
        'IL18',
        'IL18R1',
        'IL18RAP',
    ],
    group='cell_type',
    subset=[
        'CD4T_eTreg_TNFRSF9',
        'TAM_FCGBP',
    ],
    RNA_color={
        'IL18': '#FF3B30',
        'IL18R1': '#FFD60A',
        'IL18RAP': '#BF5AF2',
    },
    col_map={
        'CD4T_eTreg_TNFRSF9': '#1F77B4',
        'TAM_FCGBP': '#2CA02C',
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
    title='IL18/IL18R1/IL18RAP axis (PT-NR_01)',
    output='/home/qiuzc/IL18_1.pdf'
)

gene = "IL18::IL18R1"

if gene not in adata_lr.var_names:
    raise ValueError(f"{gene} 不在 adata_lr.var_names 中")

g_idx = adata_lr.var_names.get_loc(gene)

if issparse(adata_lr.X):
    vals_lr = adata_lr.X[:, g_idx].toarray().ravel()
else:
    vals_lr = np.asarray(adata_lr.X[:, g_idx]).ravel()

lr_map = pd.Series(vals_lr, index=adata_lr.obs['cell_id']).to_dict()

adata_roi.obs[gene] = adata_roi.obs['cell_id'].map(lr_map)

plot_obs_black(
    adata=adata_roi,
    value_key="IL18::IL18R1",
    boundary_col="boundary",
    scale_length_um=50.0,
    title="IL18::IL18R1 interaction score",
    figure_size=(6, 6),
    output="/home/qiuzc/IL18_2.pdf"
)

gene = "IL18::IL18RAP"

if gene not in adata_lr.var_names:
    raise ValueError(f"{gene} 不在 adata_lr.var_names 中")

g_idx = adata_lr.var_names.get_loc(gene)

if issparse(adata_lr.X):
    vals_lr = adata_lr.X[:, g_idx].toarray().ravel()
else:
    vals_lr = np.asarray(adata_lr.X[:, g_idx]).ravel()

lr_map = pd.Series(vals_lr, index=adata_lr.obs['cell_id']).to_dict()

adata_roi.obs[gene] = adata_roi.obs['cell_id'].map(lr_map)

plot_obs_black(
    adata=adata_roi,
    value_key="IL18::IL18RAP",
    boundary_col="boundary",
    scale_length_um=50.0,
    title="IL18::IL18RAP interaction score",
    figure_size=(6, 6),
    output="/home/qiuzc/IL18_3.pdf"
)

#Fig.S12j WB