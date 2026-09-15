#Fig.S4a
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

prop = pd.read_csv("CD4_subtype_proportion.csv")

fig, ax = plt.subplots(figsize=(8,4))

sns.boxplot(
    data=prop,
    x="subtype",
    y="proportion",
    hue="group",
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=prop,
    x="subtype",
    y="proportion",
    hue="group",
    dodge=True,
    size=3,
    ax=ax,
    legend=False
)

ax.tick_params(axis="x",rotation=60)
plt.tight_layout()
plt.savefig("FigS4a.pdf")
plt.close()


#Fig.S4b
prop = pd.read_csv("CD8_subtype_proportion.csv")

fig, ax = plt.subplots(figsize=(8,4))

sns.boxplot(
    data=prop,
    x="subtype",
    y="proportion",
    hue="group",
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=prop,
    x="subtype",
    y="proportion",
    hue="group",
    dodge=True,
    size=3,
    ax=ax,
    legend=False
)

ax.tick_params(axis="x",rotation=60)
plt.tight_layout()
plt.savefig("FigS4b.pdf")
plt.close()


#Fig.S4c
prop = pd.read_csv("DC_subtype_proportion.csv")

fig, ax = plt.subplots(figsize=(7,4))

sns.boxplot(
    data=prop,
    x="subtype",
    y="proportion",
    hue="group",
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=prop,
    x="subtype",
    y="proportion",
    hue="group",
    dodge=True,
    size=3,
    ax=ax,
    legend=False
)

plt.tight_layout()
plt.savefig("FigS4c.pdf")
plt.close()


#Fig.S4d
prop = pd.read_csv("TAM_subtype_proportion.csv")

fig, ax = plt.subplots(figsize=(8,4))

sns.boxplot(
    data=prop,
    x="subtype",
    y="proportion",
    hue="group",
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=prop,
    x="subtype",
    y="proportion",
    hue="group",
    dodge=True,
    size=3,
    ax=ax,
    legend=False
)

ax.tick_params(axis="x",rotation=60)
plt.tight_layout()
plt.savefig("FigS4d.pdf")
plt.close()


#Fig.S4e
prop = pd.read_csv("NEU_subtype_proportion.csv")

fig, ax = plt.subplots(figsize=(5,4))

sns.boxplot(
    data=prop,
    x="subtype",
    y="proportion",
    hue="group",
    showfliers=False,
    ax=ax
)

plt.tight_layout()
plt.savefig("FigS4e.pdf")
plt.close()


#Fig.S4f
prop = pd.read_csv("Fibroblast_subtype_proportion.csv")

fig, ax = plt.subplots(figsize=(10,4))

sns.boxplot(
    data=prop,
    x="subtype",
    y="proportion",
    hue="group",
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=prop,
    x="subtype",
    y="proportion",
    hue="group",
    dodge=True,
    size=2,
    ax=ax,
    legend=False
)

ax.tick_params(axis="x",rotation=70)

plt.tight_layout()
plt.savefig("FigS4f.pdf")
plt.close()


#Fig.S4g
prop = pd.read_csv("Endothelial_subtype_proportion.csv")

fig, ax = plt.subplots(figsize=(10,4))

sns.boxplot(
    data=prop,
    x="subtype",
    y="proportion",
    hue="group",
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=prop,
    x="subtype",
    y="proportion",
    hue="group",
    dodge=True,
    size=2,
    ax=ax,
    legend=False
)

ax.tick_params(axis="x",rotation=70)

plt.tight_layout()
plt.savefig("FigS4g.pdf")
plt.close()


#Fig.S4h
prop = pd.read_csv("Pericyte_subtype_proportion.csv")

fig, ax = plt.subplots(figsize=(8,4))

sns.boxplot(
    data=prop,
    x="subtype",
    y="proportion",
    hue="group",
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=prop,
    x="subtype",
    y="proportion",
    hue="group",
    dodge=True,
    size=3,
    ax=ax,
    legend=False
)

ax.tick_params(axis="x",rotation=60)

plt.tight_layout()
plt.savefig("FigS4h.pdf")
plt.close()
