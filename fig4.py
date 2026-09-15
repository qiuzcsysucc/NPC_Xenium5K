#Fig.4a https://github.com/tiroshlab/3ca/blob/main/ITH_hallmarks/Generating_MPs/Generate_Meta_Programs.R


#Fig.4b
fig, ax = plt.subplots(figsize=(5,4))

if "X_umap" in adata.obsm:
    sc.pl.umap(
        adata,
        color="MP",
        show=False,
        ax=ax
    )

plt.tight_layout()
plt.savefig("Fig4b.pdf")
plt.close()


#Fig.4c
prop = (
    adata.obs
    .groupby("batch")["MP"]
    .value_counts(normalize=True)
    .rename("proportion")
    .reset_index()
)

fig, ax = plt.subplots(figsize=(6,4))

sns.boxplot(
    data=prop,
    x="MP",
    y="proportion",
    hue="group",
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=prop,
    x="MP",
    y="proportion",
    hue="group",
    dodge=True,
    ax=ax,
    size=3,
    legend=False
)

plt.tight_layout()
plt.savefig("Fig4c.pdf")
plt.close()


#Fig.4d
survival = pd.read_csv("MP7_survival.csv")

from lifelines import KaplanMeierFitter

fig, ax = plt.subplots(figsize=(5,4))

for label, group in survival.groupby("MP7_group"):
    km = KaplanMeierFitter()
    km.fit(
        group["time"],
        group["event"],
        label=label
    )
    km.plot(ax=ax)

plt.tight_layout()
plt.savefig("Fig4d.pdf")
plt.close()


#Fig.4e decoupler barplot


#Fig.4f
de = pd.read_csv("MP7_DEG.csv")

fig, ax = plt.subplots(figsize=(5,5))

sns.scatterplot(
    data=de,
    x="log2FoldChange",
    y="-log10padj",
    ax=ax
)

plt.tight_layout()
plt.savefig("Fig4f.pdf")
plt.close()


#Fig.4g graphpad


#Fig.4h IHC


#Fig.4j
expression = pd.read_csv("SF3B1_expression_survival.csv")

fig, ax = plt.subplots(figsize=(5,4))

for label, group in expression.groupby("SF3B1_group"):
    km = KaplanMeierFitter()
    km.fit(
        group["time"],
        group["event"],
        label=label
    )
    km.plot(ax=ax)

plt.tight_layout()
plt.savefig("Fig4j.pdf")
plt.close()
