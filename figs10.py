#Fig.S7a
import scanpy as sc
import pandas as pd
import infercnvpy as cnv
import numpy as np
import os
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
from scipy.sparse import csr_matrix, issparse
from scvelo import logging as logg
from scvelo import settings
from scvelo.preprocessing.neighbors import get_connectivities, get_n_neighs, neighbors, verify_neighbors
from scvelo.preprocessing.utils import normalize_per_cell, not_yet_normalized
import infercnvpy as cnv
import matplotlib
from scipy.sparse import csr_matrix
cnv.pl.chromosome_heatmap(adata_all, groupby="major_celltype")
cnv.pl.chromosome_heatmap(adata_all, groupby="group")

#Fig.S7b
from lifelines import KaplanMeierFitter

mp_survival = pd.read_csv("metaprogram_survival.csv")

fig, axes = plt.subplots(2,4,figsize=(12,6))

for ax, mp in zip(axes.flatten(), mp_order):
    sub = mp_survival[mp_survival["metaprogram"] == mp]
    for group, data in sub.groupby("group"):
        km = KaplanMeierFitter()
        km.fit(
            data["time"],
            data["event"],
            label=group
        )
        km.plot(ax=ax)
    ax.set_title(mp)
    ax.set_xlabel("PFS time (months)")
    ax.set_ylabel("PFS probability")

plt.tight_layout()
plt.savefig("FigS7b.pdf")
plt.close()


#Fig.S7c, d graphpad


#Fig.S7e
import matplotlib.image as mpimg

fig, ax = plt.subplots(figsize=(6,3))

img = mpimg.imread("SF3B1_gammaH2AX_WB.png")
ax.imshow(img)
ax.axis("off")

plt.tight_layout()
plt.savefig("FigS7e.pdf")
plt.close()
