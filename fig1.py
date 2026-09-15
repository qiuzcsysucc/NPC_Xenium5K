#Fig.1a biorender

#Fig.1b-c
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
import omicverse as ov
from matplotlib import patheffects

ov.plot_set(font_path="Arial")
mpl.rcParams["pdf.fonttype"]=42
mpl.rcParams["ps.fonttype"]=42

adata = sc.read("xenium.h5ad")

fig, ax = plt.subplots(figsize=(4,4))
ov.pl.embedding(
    adata,
    basis="X_umap",
    color=["major_celltype"],
    show=False,
    frameon="small",
    legend_fontoutline=2,
    ax=ax,
    title="Major celltypes\n(NPC Xenium cohort)"
)

ov.pl.embedding_adjust(
    adata,
    groupby="major_celltype",
    basis="X_umap",
    ax=ax,
    adjust_kwargs=dict(arrowprops=dict(arrowstyle="-",color="black")),
    text_kwargs=dict(
        fontsize=10,
        weight="bold",
        path_effects=[patheffects.withStroke(linewidth=2,foreground="white")]
    )
)

plt.savefig("Fig1b.pdf",bbox_inches="tight")
plt.close()

marker_dict={
"Fibroblasts":["COL5A1","POSTN"],
"Pericytes":["PDGFRB","RGS5"],
"Schwann cells":["SOX10","MPZ"],
"Endothelial cells":["PLVAP","COL4A1"],
"Tumor cells":["EPCAM","CLDN1"],
"Normal epithelial cells":["FOXJ1","MUC5B"],
"Mast cells":["KIT","GATA2"],
"Neutrophils":["HCAR2","HCAR3"],
"Dendritic cells":["IRF8","CSF2RA"],
"Macrophages":["CTSL","CD68"],
"B cells":["MS4A1","CD79A"],
"Plasma cells":["MZB1","TENT5C"],
"CD4 T cells":["CD2","CD3E","CD4"],
"CD8 T cells":["CD8A","CD8B"]
}

ov.pl.dotplot(
    adata,
    groupby="major_celltype",
    var_names=marker_dict,
    cmap="coolwarm",
    dendrogram=True,
    standard_scale="var",
    show=False
)

plt.savefig("Fig1b2.pdf",bbox_inches="tight")
plt.close()

#Fig.1c
import pandas as pd
import numpy as np
import seaborn as sns

group_order=["PT-NR","PT-R","RT"]

adata.obs["group"]=pd.Categorical(
    adata.obs["group"],
    categories=group_order,
    ordered=True
)

prop_df=(
    adata.obs
    .groupby("batch")["major_celltype"]
    .value_counts(normalize=True)
    .rename("proportion")
    .reset_index()
)

fig, ax=plt.subplots(figsize=(18,6))

sns.boxplot(
    data=prop_df,
    x="major_celltype",
    y="proportion",
    ax=ax,
    showfliers=False
)

sns.stripplot(
    data=prop_df,
    x="major_celltype",
    y="proportion",
    ax=ax,
    size=4
)

ax.tick_params(axis="x",rotation=45)
ax.set_ylabel("Proportion")
ax.set_xlabel("Major celltypes")

plt.tight_layout()
plt.savefig("Fig1c.pdf",bbox_inches="tight")
plt.close()
