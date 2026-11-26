import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, leaves_list


STATE_COLORS = {
    "CD8T_ISG_IFIT3": "#2171aa",
    "CD8T_Tpex_EGR3": "#ef7c22",
    "CD8T_Tem_GZMK": "#97cc86",
    "CD8T_Tex_HAVCR2": "#8a64a9",
    "CD8T_Tgd_TRDC": "#c19b93",
    "CD8T_Tn_LEF1": "#9f9fa0",
    "CD8T_Trm_ITGAE": "#d5d58c",
    "CD8T_proliferation_MKI67": "#9cd3de",
    "DC_cDC3_LAMP3": "#2171aa",
    "DC_cDC2_CLEC10A": "#97cc86",
    "DC_cDC1_CLEC9A": "#89554b",
    "DC_Langerhans_CD1A": "#d9d9da",
    "DC_pDC_LILRA4": "#9cd3de",
    "TAM_MMP12": "#343a75",
    "TAM_CXCL9": "#646ab0",
    "TAM_FCGBP": "#899e52",
    "TAM_FUCA1": "#8a6c31",
    "TAM_LIPA": "#e3c892",
    "TAM_MKI67": "#cd5e68",
    "TAM_MX1": "#9c4f8d",
    "TAM_SLC40A1": "#d29bc4",
    "NEU_ISG_IFIT3": "#73c0c3",
    "NEU_NETs_IL1B": "#eed5bb",
    "Peri_CAF-like_WNT5A": "#317cb2",
    "Peri_ECM_THBS1": "#9bc5db",
    "Peri_FRC-like_CCL19": "#f08940",
    "Peri_ISG_CXCL10": "#f8cda1",
    "Peri_SMC-like_PLN": "#9ed097",
    "Peri_inflamed_STEAP4": "#7067a8",
    "Peri_proliferation_MKI67": "#d9d9e9",
    "Peri_senescence_MDM2": "#b2b3b3",
    "Peri_universal_RGS5": "#e7e7e7",
    "CD4T_ISG_IFIT3": "#363a75",
    "CD4T_Tanergy_EGR3": "#999bcc",
    "CD4T_Tfh_CXCL13": "#b0c96b",
    "CD4T_Tn_GIMAP4": "#e0b654",
    "CD4T_eTreg_TNFRSF9": "#a64949",
    "CD4T_proliferation_MKI67": "#774070",
    "CD4T_rTreg_CCR4": "#d29cc4",
    "Fibro_FDC_CXCL13": "#2171aa",
    "Fibro_FRC_CCL19": "#acc3e2",
    "Fibro_NF_PI16": "#f7b779",
    "Fibro_iCAF-1_IL24": "#97cc86",
    "Fibro_iCAF-2_VEGFA": "#f19493",
    "Fibro_iCAF-3_PAPPA": "#c2aed1",
    "Fibro_ifnCAF_CXCL10": "#89554b",
    "Fibro_myCAF-1_TGFB3": "#ce76ad",
    "Fibro_myCAF-2_CST1": "#9f9fa0",
    "Fibro_myCAF-3_CCN2": "#b5b62c",
    "Fibro_proliferation_MKI67": "#25b3c3",
    "EC_HEV-like_CXCL10": "#343a75",
    "EC_ISG_IFIT1": "#646ab0",
    "EC_arterial_GJA5": "#617739",
    "EC_capillary_IVNS1ABP": "#afc86a",
    "EC_lymphatic_PROX1": "#8a6c31",
    "EC_proliferation_MKI67": "#e3c892",
    "EC_tip-like_PGF": "#a64849",
    "EC_venous-1_NOS3": "#df9298",
    "EC_venous-2_CCL14": "#9c4f8d",
    "EC_venous-3_CSF2RB": "#d29bc4",
    "Pericyte_CAF-like_WNT5A": "#317cb2",
    "Pericyte_ECM_THBS1": "#9bc5db",
    "Pericyte_FRC-like_CCL19": "#f08940",
    "Pericyte_ISG_CXCL10": "#f8cda1",
    "Pericyte_SMC-like_PLN": "#9ed097",
    "Pericyte_inflamed_STEAP4": "#7067a8",
    "Pericyte_proliferation_MKI67": "#d9d9e9",
    "Pericyte_senescence_MDM2": "#b2b3b3",
    "Pericyte_universal_RGS5": "#e7e7e7",
    "Fibro_senescence_MDM2": "#9cd3de",
}


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


def _bands(ax, groups, top=1.02):
    labels = list(groups)
    start = 0
    while start < len(labels):
        stop = start + 1
        while stop < len(labels) and labels[stop] == labels[start]:
            stop += 1
        trans = ax.get_xaxis_transform()
        ax.plot([start-0.45, stop-0.55], [top, top], transform=trans, clip_on=False, color="0.2", lw=2)
        ax.text((start+stop-1)/2, top+0.02, str(labels[start]), transform=trans, rotation=90,
                fontsize=7, ha="center", va="bottom")
        start = stop


def _dot(ax, means, fractions, gene_groups=None, counts=None, tree=None, orientation="normal",
         color_title="Scaled mean expression", size_ticks=(20, 40, 60, 80, 100)):
    fractions = fractions.loc[means.index, means.columns]
    if tree is not None:
        order = leaves_list(tree)
        means, fractions = means.iloc[order], fractions.iloc[order]
    if orientation == "transposed":
        display, sizes = means.T, fractions.T
    else:
        display, sizes = means, fractions
    y, x = np.indices(display.shape)
    dots = ax.scatter(x, y, s=95*sizes.to_numpy(dtype=float)/100, c=display.to_numpy(dtype=float),
                      cmap="coolwarm", vmin=0, vmax=1, edgecolors="#666666", linewidths=0.25)
    ax.set(xlim=(-0.5, display.shape[1]-0.5), ylim=(display.shape[0]-0.5, -0.5))
    ax.set_xticks(range(display.shape[1]), display.columns, rotation=90, fontsize=7)
    if orientation == "transposed":
        ax.xaxis.tick_top()
    ax.set_yticks(range(display.shape[0]), display.index, fontsize=7)
    ax.tick_params(length=0)
    if gene_groups is not None:
        groups = [gene_groups[g] for g in means.columns]
        edges = np.flatnonzero(np.asarray(groups[1:]) != np.asarray(groups[:-1])) + 0.5
        for edge in edges:
            if orientation == "transposed":
                ax.axhline(edge, color="0.25", lw=1)
            else:
                ax.axvline(edge, color="0.25", lw=1)
        if orientation != "transposed":
            _bands(ax, groups)
    row_colors = [STATE_COLORS.get(str(k), "#bbbbbb") for k in means.index]
    if orientation == "transposed":
        ax.bar(range(len(means)), [0.018]*len(means), bottom=1.0, width=1,
               transform=ax.get_xaxis_transform(), color=row_colors, clip_on=False)
    else:
        ax.barh(range(len(means)), [0.016]*len(means), left=-0.017, height=1,
                transform=ax.get_yaxis_transform(), color=row_colors, clip_on=False)
    fig = ax.figure
    handles = [ax.scatter([], [], s=95*t/100, color="0.65", label=f"{t}%") for t in size_ticks]
    ax.legend(handles=handles, title="Fraction of cells", fontsize=7, title_fontsize=7,
              frameon=False, ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5,-0.42))
    fig.colorbar(dots, ax=ax, label=color_title, shrink=0.6, pad=(0.32 if counts is not None and orientation != "transposed" else 0.03))
    if counts is not None:
        values = counts.loc[means.index]
        if orientation == "transposed":
            side = ax.inset_axes([0, -0.23, 1, 0.12])
            side.bar(range(len(values)), values, color="#e9b468")
            side.set_xlim(ax.get_xlim())
            side.invert_yaxis()
            side.set_xticks([])
            side.set_ylabel("Count", fontsize=6)
        else:
            side = ax.inset_axes([1.015, 0, 0.14, 1])
            side.barh(range(len(values)), values, color="#e9b468")
            side.set_ylim(ax.get_ylim())
            side.set_yticks([])
            side.set_xlabel("Count", fontsize=6)
        side.tick_params(labelsize=6)
    if tree is not None:
        if orientation == "transposed":
            branch = ax.inset_axes([0, -0.32, 1, 0.08])
            dendrogram(tree, ax=branch, orientation="bottom", no_labels=True, above_threshold_color="0.25",
                       color_threshold=0)
        else:
            branch = ax.inset_axes([1.17, 0, 0.12, 1])
            dendrogram(tree, ax=branch, orientation="right", no_labels=True,
                       above_threshold_color="0.25", color_threshold=0)
            branch.invert_yaxis()
        branch.set_axis_off()


def _markers(means, fraction_pct, counts, modules, title, tree=None, transposed=False):
    genes = [g for name,block in modules for g in block]
    groups = {g:name for name,block in modules for g in block}
    fig,axes = _grid(width=max(6.5, len(genes)*0.23+3.5) if not transposed else 7,
                     height=max(4.4,len(means)*0.22+2.9) if not transposed else 9)
    _dot(axes[0,0],means.loc[:,genes],fraction_pct,groups,counts,tree,
         "transposed" if transposed else "normal")
    fig.suptitle(title,fontsize=11)
    return fig


def _score_heat(matrix,title):
    fig,axes = _grid(width=max(4.4,len(matrix.columns)*0.45+2.7),height=4)
    _heat(axes[0,0],matrix,"Scaled scores",(0,1))
    axes[0,0].set_title(title,fontsize=10)
    return fig


#Fig. S4a
def fig_s4a(scaled_mean, fraction_pct, counts, row_linkage):
    modules = [("ISG", "IFIT3 IFIT1 MX1".split()),
               ("Tanergy", "EGR3 EGR2 TNF NR4A1 NR4A3 KDM6B".split()),
               ("Tfh", "CXCL13 PDCD1 CD200 CD40LG".split()),
               ("Tn", "GIMAP4 GIMAP5 SELL TCF7 LEF1".split()),
               ("eTreg", "TNFRSF9 CTLA4 CCR8 CCR6 IL2RA".split()),
               ("Proliferation", "MKI67".split()),
               ("rTreg", "FOXP3 CCR4 RTKN2".split())]
    return _markers(scaled_mean,fraction_pct,counts,modules,"CD4 T cells",row_linkage)


#Fig. S4b
def fig_s4b(scaled_scores):
    return _score_heat(scaled_scores,"CD4 T cells")


#Fig. S4c
def fig_s4c(scaled_mean, fraction_pct, counts, row_linkage):
    modules = [("ISG", "MX1 IFI44L IFIT1 IFIT2 IFIT3".split()),
               ("Tpex", "EGR3 EGR2 IFNG TNF FASLG TNFSF9 NR4A1 NR4A3".split()),
               ("Tem", "GZMK KLRG1 AOAH".split()),
               ("Tex", "TNFRSF9 GZMB LAG3 HAVCR2".split()),
               ("Tgd", "TRDC KLRC1 KLRK1 KLRD1 CD300A".split()),
               ("Tn", "SELL TCF7 LEF1".split()),
               ("Trm", "ITGAE ZNF683 CXCR6 KLRB1".split()),
               ("Proliferation", "MKI67".split())]
    return _markers(scaled_mean,fraction_pct,counts,modules,"CD8 T cells",row_linkage)


#Fig. S4d
def fig_s4d(scaled_scores):
    return _score_heat(scaled_scores,"CD8 T cells")


#Fig. S4e
def fig_s4e(scaled_mean, fraction_pct, counts):
    modules = [("cDC3", "FSCN1 CD83 LAMP3".split()),
               ("cDC2", "FCGR2A CLEC10A CD14".split()),
               ("cDC1", "WDFY4 CLEC9A CPVL".split()),
               ("Langerhans", "FCER1A CD1A CD207".split()),
               ("pDC", "GZMB LILRA4 IL3RA".split())]
    return _markers(scaled_mean,fraction_pct,counts,modules,"Dendritic cells")


#Fig. S4f
def fig_s4f(scaled_mean, fraction_pct, counts, row_linkage):
    modules = [("MMP12", "MMP12 PLAUR ADM SLC2A3".split()),
               ("CXCL9", "CXCL9 CXCL10 GBP1".split()),
               ("FCGBP", "FCGBP CTNND1 NDRG2".split()),
               ("FUCA1", "FUCA1 MMP9 ENPP2".split()),
               ("LIPA", "LIPA PLA2G7 CCL18".split()),
               ("Proliferation", "MKI67 PCNA STMN1".split()),
               ("MX1", "MX1 IFIT3 IFIT2 IFIT1".split()),
               ("SLC40A1", "SLC40A1 STAB1 MRC1".split())]
    return _markers(scaled_mean,fraction_pct,counts,modules,"Macrophages",row_linkage)


#Fig. S4g
def fig_s4g(scaled_scores, row_linkage):
    fig,axes = _grid(1,2,5.3,3.8,gridspec_kw={"width_ratios":[4,1]})
    order = leaves_list(row_linkage)
    im = _heat(axes[0,0],scaled_scores.iloc[order],"Scaled scores",(0,1),bar=False)
    dendrogram(row_linkage,ax=axes[0,1],orientation="right",no_labels=True,
               color_threshold=0,above_threshold_color="0.35")
    axes[0,1].invert_yaxis()
    axes[0,1].set_axis_off()
    fig.colorbar(im,ax=axes[0,0],location="bottom",shrink=0.7,pad=0.04,label="Scaled scores")
    fig.suptitle("TAMs")
    return fig


#Fig. S4h
def fig_s4h(scaled_mean, fraction_pct, counts):
    modules = [("ISG", "CXCL10 STAT1 CXCL9 IFIT3".split()),
               ("NETs", "IL1B PLAUR HCAR2 PDE4B BCL2A1 TREM1".split())]
    return _markers(scaled_mean,fraction_pct,counts,modules,"Neutrophils")


#Fig. S4i
def fig_s4i(scaled_mean, fraction_pct, counts):
    modules = [("FDC", "CXCL13 CR1 CR2 FCER2".split()),
               ("FRC", "CCL19 ABI3BP".split()),
               ("NF", "PI16 IGF1".split()),
               ("iCAF-1", "IL24 CXCL6".split()),
               ("iCAF-2", "SERPINE1 VEGFA".split()),
               ("iCAF-3", "PAPPA MASP1 TWIST2".split()),
               ("ifnCAF", "CXCL10 CXCL9 GBP1 IFIT3".split()),
               ("myCAF-1", "POSTN TGFB3".split()),
               ("myCAF-2", "CST1 CST2 CST4".split()),
               ("myCAF-3", "MMP11 CCN2".split()),
               ("Proliferation", "MKI67".split()),
               ("Senescence", "MDM2 CDKN1A".split())]
    return _markers(scaled_mean,fraction_pct,counts,modules,"Fibroblasts")


#Fig. S4j
def fig_s4j(scaled_mean, fraction_pct, counts):
    modules = [("HEV", "CXCL9 CXCL10 CXCL11".split()),
               ("ISG", "MX1 IFIT3 IFIT1".split()),
               ("Arterial", "GJA5 VEGFC MECOM".split()),
               ("Capillary", "IVNS1ABP INSR KDR".split()),
               ("Lymphatic", "PROX1 PDPN TFPI".split()),
               ("Proliferation", "MKI67".split()),
               ("Tip-like", "PGF ESM1 APLN".split()),
               ("Venous 1", "SELP PRCP NOS3".split()),
               ("Venous 2", "CCL14 CCL19 IL33".split()),
               ("Venous 3", "CSF2RB TSPAN7 SELE".split())]
    return _markers(scaled_mean,fraction_pct,counts,modules,"Endothelial cells")


#Fig. S4k
def fig_s4k(scaled_mean, fraction_pct, counts, row_linkage):
    modules = [("CAF-like", "WNT5A MMP14 MMP11".split()),
               ("ECM", "THBS1 COL4A1 COL4A2 COL5A1 COL5A2".split()),
               ("FRC-like", "CCL19 CXCL12".split()),
               ("ISG", "CXCL9 MX1 GBP1 CXCL10 IFIT3".split()),
               ("SMC-like", "PLN LBH".split()),
               ("Inflamed", "STEAP4".split()),
               ("Proliferation", "MKI67".split()),
               ("Senescence", "MDM2 CDKN1A".split()),
               ("Universal", "RGS5".split())]
    return _markers(scaled_mean,fraction_pct,counts,modules,"Pericytes",row_linkage,transposed=True)
