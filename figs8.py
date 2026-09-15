#Fig.S8a biorender


#Fig.S8b
cooccur = pd.read_csv("visium_celltype_cooccurrence.csv")

fig, ax = plt.subplots(figsize=(10,5))

sns.heatmap(
    cooccur.pivot(
        index="celltype1",
        columns="celltype2",
        values="fraction"
    ),
    cmap="RdBu_r",
    ax=ax
)

plt.tight_layout()
plt.savefig("FigS8b.pdf")
plt.close()


#Fig.S8c
adata = sc.read("visium_SS1.h5ad")

genes = [
"MP12_Progenitor",
"Fibro_myCAF-1_TGFB3",
"Fibro_myCAF-2_CST1",
"Fibro_myCAF-3_CCN2",
"Fibro_proliferation_MKI67",
"Fibro_senescence_MDM2"
]

fig, axes = plt.subplots(1,6,figsize=(18,3))

for ax,gene in zip(axes,genes):
    sc.pl.spatial(
        adata,
        color=gene,
        show=False,
        ax=ax
    )

plt.tight_layout()
plt.savefig("FigS8c.pdf")
plt.close()


#Fig.S8d
adata = sc.read("visium_SS2.h5ad")

genes = [
"MP2_G2M",
"Pericyte_ECM_THBS1",
"Pericyte_proliferation_MKI67",
"Pericyte_senescence_MDM2",
"Pericyte_universal_RGS5",
"EC_capillary_IVNS1ABP"
]

fig, axes = plt.subplots(1,6,figsize=(18,3))

for ax,gene in zip(axes,genes):
    sc.pl.spatial(
        adata,
        color=gene,
        show=False,
        ax=ax
    )

plt.tight_layout()
plt.savefig("FigS8d.pdf")
plt.close()


#Fig.S8e
adata = sc.read("visium_SS3.h5ad")

genes = [
"MP7_RNA-splicing",
"TAM_FCGBP",
"CD4T_eTreg_TNFRSF9",
"CD8T_Tex_HAVCR2",
"TAM_MKI67",
"CD4T_Tanergy_EGR3"
]

fig, axes = plt.subplots(1,6,figsize=(18,3))

for ax,gene in zip(axes,genes):
    sc.pl.spatial(
        adata,
        color=gene,
        show=False,
        ax=ax
    )

plt.tight_layout()
plt.savefig("FigS8e.pdf")
plt.close()


#Fig.S8f
adata = sc.read("visium_SS4.h5ad")

genes = [
"MP8_IFN-response",
"CD8T_Tex_HAVCR2",
"TAM_MX1",
"TAM_CXCL9",
"CD4T_Tn_GIMAP4",
"CD8T_ISG_IFIT3"
]

fig, axes = plt.subplots(1,6,figsize=(18,3))

for ax,gene in zip(axes,genes):
    sc.pl.spatial(
        adata,
        color=gene,
        show=False,
        ax=ax
    )

plt.tight_layout()
plt.savefig("FigS8f.pdf")
plt.close()


#Fig.S8g
adata = sc.read("visium_SS5.h5ad")

genes = [
"MP9_Hypoxia",
"TAM_MMP12",
"NEU_NETs_IL1B",
"DC_Langerhans_CD1A",
"TAM_FCGBP",
"Fibro_iCAF-2_VEGFA"
]

fig, axes = plt.subplots(1,6,figsize=(18,3))

for ax,gene in zip(axes,genes):
    sc.pl.spatial(
        adata,
        color=gene,
        show=False,
        ax=ax
    )

plt.tight_layout()
plt.savefig("FigS8g.pdf")
plt.close()
