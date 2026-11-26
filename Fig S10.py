import numpy as np
import matplotlib.pyplot as plt


SE = {"SE1_Immune-ashy": "#e41a1c", "SE2_Immune-insulated": "#984ea3",
      "SE3_Immune-inflamed": "#a65628", "SE4_Immune-cold": "#b3b3b3"}


FIVE = [
    ["TAM_FCGBP", "CD4T_eTreg_TNFRSF9", "DC_Langerhans_CD1A", "CD8T_Tex_HAVCR2", "TAM_MKI67"],
    ["Fibro_myCAF-2_CST1", "Fibro_iCAF-2_VEGFA", "Fibro_myCAF-1_TGFB3", "Fibro_proliferation_MKI67", "Pericyte_ECM_THBS1"],
    ["CD8T_Tex_HAVCR2", "CD8T_Trm_ITGAE", "CD8T_ISG_IFIT3", "TAM_MX1", "DC_Langerhans_CD1A"],
    ["TAM_MMP12", "NEU_NETs_IL1B", "DC_Langerhans_CD1A", "TAM_FCGBP", "Fibro_iCAF-2_VEGFA"],
]


def _grid(rows=1, cols=1, width=5, height=3.5, **kwargs):
    return plt.subplots(rows, cols, figsize=(width, height), squeeze=False,
                        layout="constrained", **kwargs)


def _heat(ax, matrix, label, limits=None, cmap="coolwarm", fmt=None, bar=True):
    values = matrix.to_numpy(dtype=float)
    lo, hi = limits if limits is not None else (np.nanmin(values), np.nanmax(values))
    im = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=lo, vmax=hi)
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=7)
    ax.tick_params(length=0, pad=2)
    if fmt:
        for (r, c), val in np.ndenumerate(values):
            if np.isfinite(val):
                rgb = im.cmap(im.norm(val))[:3]
                ink = "white" if np.dot(rgb, [0.299, 0.587, 0.114]) < 0.5 else "#222222"
                ax.text(c, r, format(val, fmt), color=ink, fontsize=6, ha="center", va="center")
    if bar:
        ax.figure.colorbar(im, ax=ax, label=label, shrink=0.7, pad=0.012)
    return im


def _visium(spots, sample, index):
    names = [list(SE)[index]] + FIVE[index]
    order = [[0,1,2,4,5,3],[0,3,1,5,4,2],[0,1,2,3,4,5],[0,1,2,3,4,5]][index]
    fig,axes = _grid(1,6,16.5,3)
    for ax,j in zip(axes.flat,order):
        key = names[j]
        dot = ax.scatter(spots.x,spots.y,c=spots[key],s=5,marker="h",lw=0,cmap="magma",vmin=0,rasterized=True)
        ax.set(xticks=[],yticks=[],aspect="equal",facecolor="#231916",xlabel="spatial1")
        ax.invert_yaxis()
        ax.set_title(key,fontsize=8)
        fig.colorbar(dot,ax=ax,pad=0.012,shrink=0.75)
    axes[0,0].set_ylabel("spatial2")
    fig.suptitle(sample)
    return fig


#Fig. S10b
def fig_s10b(nmf_loadings):
    fig,axes = _grid(width=16,height=5.3)
    matrix = nmf_loadings.loc[["Normal_Visium","SE4_Visium","SE3_Visium","SE1_Visium","SE2_Visium"]]
    _heat(axes[0,0],matrix,"Fraction",(0,1))
    axes[0,0].set_title("Identifying spatial co-occurrence of cell types")
    return fig


#Fig. S10c
def fig_s10c(section_spots):
    return _visium(section_spots,"GSM6248648",0)


#Fig. S10d
def fig_s10d(section_spots):
    return _visium(section_spots,"GSM6248647",1)


#Fig. S10e
def fig_s10e(section_spots):
    return _visium(section_spots,"GSM6248650",2)


#Fig. S10f
def fig_s10f(section_spots):
    return _visium(section_spots,"HRS1925483",3)
