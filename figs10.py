#Fig.S10a
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


#Fig.S10b
sc.pl.umap(
    adata,
    color="dataset",
    show=False
)

plt.savefig("FigS10b.pdf", bbox_inches="tight")
plt.close()


#Fig.S10c
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


#Fig.S10d
corr = pd.read_csv("TAM_subtype_expression_correlation.csv", index_col=0)

sns.heatmap(
    corr,
    cmap="RdBu_r",
    center=0
)

plt.savefig("FigS10d.pdf", bbox_inches="tight")
plt.close()


#Fig.S10e
similarity = pd.read_csv("TAM_marker_similarity.csv", index_col=0)

sns.heatmap(
    similarity,
    cmap="Reds",
    vmin=0,
    vmax=0.35
)

plt.savefig("FigS10e.pdf", bbox_inches="tight")
plt.close()


#Fig.S10f
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

#Fig.S10g


#Fig.S10i
import matplotlib.image as mpimg

fig, axes = plt.subplots(2,4,figsize=(10,6))

for ax, img in zip(
    axes.flatten(),
    [
        "MERTK.png",
        "AXL.png",
        "FCGBP.png",
        "Tubulin.png",
        "p_MERTK.png",
        "p_AXL.png",
        "pro_IL18.png",
        "mature_IL18.png"
    ]
):
    ax.imshow(mpimg.imread(img))
    ax.axis("off")

plt.tight_layout()
plt.savefig("FigS10i.pdf", bbox_inches="tight")
plt.close()
