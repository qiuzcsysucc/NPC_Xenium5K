#Fig.2a
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cn_order = [
    "CN1_Normal_epithelium",
    "CN2_Tumor_boundary",
    "CN3_Tumor_core",
    "CN4_Neutrophil_enriched",
    "CN5_B_enriched",
    "CN6_T-APC_enriched",
    "CN7_Plasma_enriched",
    "CN8_Stroma_enriched"
]

adata = sc.read("xenium_cellular_niche.h5ad")

if "cellular_niche" in adata.obs:
    niche = adata.obs["cellular_niche"].astype(str)
else:
    niche = adata.obs["niche"].astype(str)

celltypes = adata.obs["major_celltype"].astype(str)

ct_order = list(celltypes.unique())

table = pd.crosstab(
    celltypes,
    niche,
    normalize="index"
)

table = table.reindex(columns=[x for x in cn_order if x in table.columns])

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(
    table,
    cmap="coolwarm",
    center=0.5,
    ax=ax
)
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("Fig2a.pdf")
plt.close()


#Fig.2b
prop = (
    adata.obs
    .groupby("batch")["cellular_niche"]
    .value_counts(normalize=True)
    .rename("proportion")
    .reset_index()
)

fig, ax = plt.subplots(figsize=(8,4))

sns.boxplot(
    data=prop,
    x="cellular_niche",
    y="proportion",
    hue="group",
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=prop,
    x="cellular_niche",
    y="proportion",
    hue="group",
    dodge=True,
    size=3,
    ax=ax,
    legend=False
)

ax.set_xlabel("")
ax.set_ylabel("Proportion")
ax.tick_params(axis="x",rotation=45)
plt.tight_layout()
plt.savefig("Fig2b.pdf")
plt.close()


#Fig.2c
samples = list(adata.obs["batch"].unique())

fig, axes = plt.subplots(
    3,
    12,
    figsize=(18,6)
)

for ax, sample in zip(axes.flatten(), samples):
    sub = adata[adata.obs["batch"] == sample]
    sc.pl.spatial(
        sub,
        color="cellular_niche",
        spot_size=10,
        show=False,
        ax=ax
    )
    ax.set_title(sample,fontsize=6)

for ax in axes.flatten()[len(samples):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("Fig2c.pdf")
plt.close()


#Fig.2d
sample_id = "PTR_03"

sub = adata[adata.obs["batch"] == sample_id]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(10,7)
)

sc.pl.spatial(
    sub,
    color="cellular_niche",
    show=False,
    ax=axes[0,0]
)

sc.pl.spatial(
    sub,
    color="major_celltype",
    show=False,
    ax=axes[0,1]
)

sc.pl.spatial(
    sub,
    color="major_celltype",
    show=False,
    ax=0,2
)

sc.pl.spatial(
    sub,
    color="cellular_niche",
    show=False,
    ax=axes[1,0]
)

sc.pl.spatial(
    sub,
    color="major_celltype",
    show=False,
    ax=axes[1,1]
)

axes[1,2].axis("off")

plt.tight_layout()
plt.savefig("Fig2d.pdf")
plt.close()
