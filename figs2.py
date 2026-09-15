#Fig.S2a
import scanpy as sc
import matplotlib.pyplot as plt

adata = sc.read("xenium_spatial_niche.h5ad")

samples = [
    "PTR_01",
    "PTR_10"
]

fig, axes = plt.subplots(2,4,figsize=(12,6))

for row, sample in enumerate(samples):
    sub = adata[adata.obs["sample"] == sample]

    sc.pl.spatial(
        sub,
        color="cellular_niche",
        show=False,
        ax=axes[row,0]
    )

    sc.pl.spatial(
        sub,
        color="cellular_niche",
        show=False,
        ax=axes[row,1]
    )

    sc.pl.spatial(
        sub,
        color="major_celltype",
        show=False,
        ax=axes[row,2]
    )

    sc.pl.spatial(
        sub,
        color="major_celltype",
        show=False,
        ax=axes[row,3]
    )

plt.tight_layout()
plt.savefig("FigS2a.pdf")
plt.close()


#Fig.S2b
sample = "PTR_10"

sub = adata[adata.obs["sample"] == sample]

fig, axes = plt.subplots(1,4,figsize=(12,3))

sc.pl.spatial(
    sub,
    color="cellular_niche",
    show=False,
    ax=axes[0]
)

sc.pl.spatial(
    sub,
    color="cellular_niche",
    show=False,
    ax=axes[1]
)

sc.pl.spatial(
    sub,
    color="major_celltype",
    show=False,
    ax=axes[2]
)

sc.pl.spatial(
    sub,
    color="major_celltype",
    show=False,
    ax=axes[3]
)

plt.tight_layout()
plt.savefig("FigS2b.pdf")
plt.close()


#Fig.S2c
samples = [
    "RT_09"
]

fig, axes = plt.subplots(1,4,figsize=(12,3))

for sample in samples:
    sub = adata[adata.obs["sample"] == sample]

    sc.pl.spatial(
        sub,
        color="major_celltype",
        show=False,
        ax=axes[0]
    )

    sc.pl.spatial(
        sub,
        color="major_celltype",
        show=False,
        ax=axes[1]
    )

    sc.pl.spatial(
        sub,
        color="cellular_niche",
        show=False,
        ax=axes[2]
    )

    sc.pl.spatial(
        sub,
        color="cellular_niche",
        show=False,
        ax=axes[3]
    )

plt.tight_layout()
plt.savefig("FigS2c.pdf")
plt.close()
