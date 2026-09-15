#Fig.S3a
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

adata = sc.read("xenium_subtype_annotation.h5ad")

marker_sets = {
"CD4_T": [
"IFIT3","IFIT1","MX1","EGR3","EGR2","TNF","NR4A1","NR4A3",
"CXCL13","PDCD1","CD200","CD40LG","GIMAP4","GIMAP5",
"SELL","TCF7","LEF1","TNFRSF9","CTLA4","CCR8",
"CCR6","IL2RA","MKI67","FOXP3","CCR4"
]
}

sc.pl.dotplot(
    adata,
    marker_sets,
    groupby="CD4_subtype",
    show=False
)

plt.savefig("FigS3a.pdf",bbox_inches="tight")
plt.close()


#Fig.S3b
score = adata.obs[
[
"Quiescence_score",
"Regulating_score",
"Proliferation_score",
"Helper1_2_score",
"Helper17_score",
"Fhelper_score",
"Cytotoxicity_score",
"Progenitor_exhaustion_score",
"Terminal_exhaustion_score",
"Senescence_score"
]]

sns.heatmap(
    score.groupby(adata.obs["CD4_subtype"]).mean(),
    cmap="RdBu_r"
)

plt.savefig("FigS3b.pdf",bbox_inches="tight")
plt.close()


#Fig.S3c
sc.pl.dotplot(
    adata,
    [
"MX1","IFI44L","IFIT1","IFIT2","IFIT3",
"EGR3","EGR2","IFNG","TNF","FASLG",
"TNFSF9","NR4A1","NR4A3","GZMK",
"KLRG1","TNFRSF9","GZMB","LAG3",
"HAVCR2","TRDC","KLRC1","KLRK1",
"KLRD1","SELL","TCF7","LEF1",
"ITGAE","ZNF683","CXCR6","KLRB1","MKI67"
],
groupby="CD8_subtype",
show=False
)

plt.savefig("FigS3c.pdf",bbox_inches="tight")
plt.close()


#Fig.S3d
sns.heatmap(
    adata.obs.groupby("CD8_subtype").mean(numeric_only=True),
    cmap="RdBu_r"
)

plt.savefig("FigS3d.pdf",bbox_inches="tight")
plt.close()


#Fig.S3e
sc.pl.dotplot(
    adata,
    [
"FSCN1","CD83","LAMP3","FCGR2A",
"CLEC10A","CD14","WDFY4",
"CLEC9A","CPVL","FCER1A",
"CD1A","CD207","GZMB",
"LILRA4","IL3RA"
],
groupby="DC_subtype",
show=False
)

plt.savefig("FigS3e.pdf",bbox_inches="tight")
plt.close()


#Fig.S3f
sc.pl.dotplot(
    adata,
    [
"MMP12","CXCL9","FCGBP","FUCA1",
"LIPA","MKI67","MX1","IFIT3",
"SLC40A1","STAB1","MRC1"
],
groupby="TAM_subtype",
show=False
)

plt.savefig("FigS3f.pdf",bbox_inches="tight")
plt.close()


#Fig.S3g
sns.heatmap(
    adata.obs.groupby("TAM_subtype").mean(numeric_only=True),
    cmap="RdBu_r"
)

plt.savefig("FigS3g.pdf",bbox_inches="tight")
plt.close()


#Fig.S3h
sc.pl.dotplot(
    adata,
    [
"CXCL10","STAT1","CXCL9","IFIT3",
"IL1B","PLAUR","HCAR2","PDE4B",
"BCL2A1","TREM1"
],
groupby="NEU_subtype",
show=False
)

plt.savefig("FigS3h.pdf",bbox_inches="tight")
plt.close()


#Fig.S3i
sc.pl.dotplot(
    adata,
    [
"CXCL13","CR1","CR2","FCER2",
"CCL19","ABI3BP","PI16","IGF1",
"IL24","CXCL6","SERPINE1",
"VEGFA","PAPPA","TWIST2",
"CXCL10","POSTN","MMP11",
"CCN2","CDKN1A"
],
groupby="Fibro_subtype",
show=False
)

plt.savefig("FigS3i.pdf",bbox_inches="tight")
plt.close()


#Fig.S3j
sc.pl.dotplot(
    adata,
    [
"CXCL9","CXCL10","CXCL11","MX1",
"IFIT3","IFIT1","GJA5","VEGFC",
"MECOM","KDR","PROX1","PDPN",
"MKI67","ESM1","APLN","NOS3"
],
groupby="EC_subtype",
show=False
)

plt.savefig("FigS3j.pdf",bbox_inches="tight")
plt.close()


#Fig.S3k
sc.pl.dotplot(
    adata,
    [
"WNT5A","THBS1","COL4A1","COL4A2",
"COL5A1","COL5A2","CCL19","CXCL12",
"CXCL9","MX1","GBP1","CXCL10",
"PLN","STEAP4","MKI67","MDM2",
"CDKN1A","RGS5"
],
groupby="Pericyte_subtype",
show=False
)

plt.savefig("FigS3k.pdf",bbox_inches="tight")
plt.close()
