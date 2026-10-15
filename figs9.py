#Fig.S8a biorender


#Fig.S8b

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

merged = pd.read_csv("xenium_visium_nmf_factor.csv", index_col=0)

plot_df = merged.T

plt.figure(figsize=(25, 5))

sns.heatmap(
    plot_df,
    cmap="coolwarm",
    cbar_kws={"label": "fraction", "pad": 0.01},
)

plt.title('Identifying spatial co-occurrence of cell types', fontsize=16)
plt.xlabel("Cell types (Visium cohort)")
plt.ylabel("")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    'Identifying spatial co-occurrence of cell types.pdf',
    bbox_inches='tight'
)

plt.show()


#Fig.S8c-f


import os
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt


prefix = "q05cell_abundance_w_sf_"

out_dir = r"C:\Users\Lenovo\Desktop\xenium_c2l_plots_c1_0525"
os.makedirs(out_dir, exist_ok=True)

for sample_id, adata0 in adatas.items():
    print(f"Plotting sample: {sample_id}")

    adata = adata0.copy()

    if "q05_cell_abundance_w_sf" not in adata.obsm.keys():
        print(f"  [WARN] {sample_id} 中没有 obsm['q05_cell_abundance_w_sf']，跳过")
        continue

    q05 = adata.obsm["q05_cell_abundance_w_sf"]

    for ct in target_ct:
        full_col = prefix + ct

        if full_col not in q05.columns:
            continue

        adata.obs[ct] = q05[full_col].values

    ct_to_plot = [ct for ct in target_ct if ct in adata.obs.columns]

    if len(ct_to_plot) == 0:
        print(f"  [WARN] {sample_id} 没有可画的 target_ct，跳过")
        continue

    with mpl.rc_context({
        "figure.figsize": [6, 6],
        "axes.facecolor": "black",
    }):
        sc.pl.spatial(
            adata,
            color=ct_to_plot,
            cmap="magma",
            ncols=6,
            size=1.3,
            vmin=0,
            vmax="p99.2",
            show=False,
            img_key=None,
        )

        fig = plt.gcf()

        for ax in fig.axes:
            if ax.get_xlabel() == "spatial1" and ax.get_ylabel() == "spatial2":
                ax.set_facecolor("black")

        fig.suptitle(sample_id, y=0.99, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        png_path = os.path.join(out_dir, f"{sample_id}_cell2loc_spatial.png")
        pdf_path = os.path.join(out_dir, f"{sample_id}_cell2loc_spatial.pdf")

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")

        plt.close(fig)
