#Fig.S1a
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

qc = pd.read_csv("xenium_QC_metrics.csv")

metrics = [
    "total_counts",
    "cell_area",
    "nucleus_area",
    "nucleus_count",
    "n_genes_by_counts"
]

fig, axes = plt.subplots(
    len(metrics),
    1,
    figsize=(10,12)
)

for ax, metric in zip(axes, metrics):
    sns.violinplot(
        data=qc,
        x="sample",
        y=metric,
        ax=ax,
        inner="box",
        color="lightgrey"
    )
    ax.set_xlabel("")
    ax.tick_params(axis="x",rotation=90)

plt.tight_layout()
plt.savefig("FigS1a.pdf")
plt.close()


#Fig.S1b
import scanpy as sc

adata = sc.read("xenium_all_samples.h5ad")

sample_order = [
    "PTNR_01","PTNR_02","PTNR_03","PTNR_04",
    "PTNR_05","PTNR_06","PTNR_07","PTNR_08",
    "PTNR_09","PTNR_10","PTNR_11","PTNR_12",
    "PTR_01","PTR_02","PTR_03","PTR_04",
    "PTR_05","PTR_06","PTR_07","PTR_08",
    "PTR_09","PTR_10","PTR_11","PTR_12",
    "RT_01","RT_02","RT_03","RT_04",
    "RT_05","RT_06","RT_07","RT_08",
    "RT_09","RT_10","RT_11","RT_12"
]

fig, axes = plt.subplots(
    3,
    12,
    figsize=(18,8)
)

for ax,sample in zip(
    axes.flatten(),
    sample_order
):
    sub=adata[adata.obs["sample"]==sample]
    sc.pl.spatial(
        sub,
        color="major_celltype",
        show=False,
        ax=ax
    )
    ax.set_title(sample,fontsize=6)

plt.tight_layout()
plt.savefig("FigS1b.pdf")
plt.close()


#Fig.S1c
density = pd.read_csv("celltype_density.csv")

fig, ax = plt.subplots(figsize=(10,5))

sns.boxplot(
    data=density,
    x="celltype",
    y="density",
    hue="group",
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=density,
    x="celltype",
    y="density",
    hue="group",
    dodge=True,
    size=3,
    ax=ax,
    legend=False
)

ax.tick_params(axis="x",rotation=60)

plt.tight_layout()
plt.savefig("FigS1c.pdf")
plt.close()
