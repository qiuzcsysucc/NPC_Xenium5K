#figs7a

from pathlib import Path
import os

os.environ['NUMBA_CACHE_DIR'] = '/tmp/niche_umap_numba'
os.environ['MPLCONFIGDIR'] = '/tmp/niche_umap_mpl'

for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'

import h5py
import numpy as np
import pandas as pd
import json
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
from config import OUTPUT_DIR, SOURCE_DIR

O = OUTPUT_DIR / 'umap'
O.mkdir(parents=True, exist_ok=True)
S = SOURCE_DIR

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none'
})

fig, axs = plt.subplots(1, 4, figsize=(12, 3.1))
fig.subplots_adjust(left=0.025, right=0.99, bottom=0.12, top=0.9, wspace=0.12)

qc = []
palette = None

for ax, r in zip(axs, [60, 70, 90, 100]):
    with h5py.File(S / f'tumor_niche_0929_{r}.h5ad') as f:
        umap = f['obsm/X_umap'][:]
        g = f['obs/cell_type']
        cats = g['categories'].asstr()[:]
        codes = g['codes'][:]

        assert np.isfinite(umap).all() and (codes >= 0).all()

        if r == 60:
            q1, q3 = np.quantile(umap, [0.25, 0.75], axis=0)
            iqr = q3 - q1

            keep = (
                (umap >= q1 - 3 * iqr) &
                (umap <= q3 + 3 * iqr)
            ).all(axis=1)

            ids = f['obs'][f['obs'].attrs['_index']].asstr()[:]

            pd.DataFrame({
                'cell_id': ids[~keep],
                'cell_type': cats[codes[~keep]],
                'UMAP1': umap[~keep, 0],
                'UMAP2': umap[~keep, 1]
            }).to_csv(
                O / '60um_excluded_plot_outliers.csv',
                index=False
            )

            print('60um excluded', int((~keep).sum()), flush=True)

            umap = umap[keep]
            codes = codes[keep]

        stored = f['uns/cell_type_colors'].asstr()[:].tolist()

        print(r, cats, stored, flush=True)

        if palette is None:
            palette = {
                name.split('_')[0]: color
                for name, color in zip(cats, stored)
            }

        data = ad.AnnData(
            obs=pd.DataFrame({
                'cell_type': pd.Categorical.from_codes(
                    codes,
                    categories=cats
                )
            })
        )

        data.obsm['X_umap'] = umap
        data.uns['cell_type_colors'] = [
            palette[name.split('_')[0]]
            for name in cats
        ]

        sc.pl.umap(
            data,
            color='cell_type',
            legend_loc='on data',
            legend_fontsize=7,
            legend_fontweight='normal',
            legend_fontoutline=1.5,
            size=0.15,
            alpha=0.7,
            frameon=False,
            title=f'{r} μm',
            ax=ax,
            show=False,
            sort_order=False
        )

        for coll in ax.collections:
            coll.set_rasterized(True)

        ax.set_aspect('equal', adjustable='box', anchor='C')

        qc.append({
            'radius_um': r,
            'n_cells': len(codes),
            'coordinates': 'stored obsm/X_umap; not recomputed',
            'categories': cats.tolist(),
            'colors': data.uns['cell_type_colors']
        })

    for text in ax.texts:
        text.set_fontfamily('Arial')

        if r == 60 and text.get_text().startswith(('SS2_', 'SS3_')):
            x, y = text.get_position()
            span = ax.get_xlim()[1] - ax.get_xlim()[0]

            text.set_position((
                x + (
                    0.045 if text.get_text().startswith('SS2_')
                    else -0.045
                ) * span,
                y
            ))

for ext in ['pdf', 'svg', 'png']:
    fig.savefig(
        O / f'cell_type_UMAP_1x4_on_data.{ext}',
        dpi=300,
        facecolor='white',
        bbox_inches='tight'
    )

plt.close(fig)

json.dump(
    {
        'panels': qc,
        'legend_loc': 'on data',
        'all_cells_plotted': False,
        '60um_rule': 'Exclude coordinates outside Q1-3*IQR or Q3+3*IQR on either UMAP axis; plot-only exclusion; original h5ad unchanged',
        'other_radii': 'all cells retained',
        'font': 'Arial 7 pt',
        'points_rasterized_in_vector_outputs': True
    },
    open(O / 'plot_parameters.json', 'w'),
    indent=2
)

print('DONE', flush=True)



#figs7b
from pathlib import Path
import os

os.environ['MPLCONFIGDIR'] = '/tmp/niche_similarity_mpl'

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import DATA_DIR, OUTPUT_DIR

O = DATA_DIR
D = OUTPUT_DIR / 'heatmap_64types'
D.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none'
})

for name, root, title in [
    ('64types', O, '64 non-tumor cell types; renormalized')
]:
    d = pd.read_csv(
        root / 'Pearson_20x20_patient_balanced.csv',
        index_col=0
    )

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.subplots_adjust(left=0.15, right=0.9, bottom=0.18, top=0.89)

    lo = float(d.values.min())
    im = ax.imshow(d.values, vmin=lo, vmax=1, cmap='coolwarm')

    labels = [v.replace('um_', ' μm · ') for v in d.index]

    ax.set_xticks(range(20), labels, rotation=90)
    ax.set_yticks(range(20), labels)

    for i in range(20):
        for j in range(20):
            v = d.iloc[i, j]
            ax.text(
                j, i, f'{v:.3f}',
                fontsize=7,
                ha='center',
                va='center',
                color='white'
                if (v - lo) / (1 - lo) < 0.15
                or (v - lo) / (1 - lo) > 0.85
                else 'black'
            )

    for pos in [3.5, 7.5, 11.5, 15.5]:
        ax.axvline(pos, color='white', lw=1.5)
        ax.axhline(pos, color='white', lw=1.5)

    ax.set_title(
        title + '\nPatient-balanced Pearson correlation',
        pad=30
    )

    for i, r in enumerate([60, 70, 80, 90, 100]):
        ax.text(
            i * 4 + 1.5,
            -0.95,
            f'{r} μm',
            ha='center',
            va='bottom',
            fontsize=7,
            clip_on=False
        )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.025,
        label='Pearson r',
        shrink=0.8
    )

    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(
            D / f'Pearson_20x20_{name}_coolwarm_Arial7pt.{ext}',
            dpi=400,
            facecolor='white'
        )

    plt.close(fig)

print('Created the 64-type coolwarm heatmap.')

#figs8b


from pathlib import Path
import os

os.environ['MPLCONFIGDIR'] = '/tmp/niche_group_mpl'

import h5py
import pandas as pd
import numpy as np
import json
from scipy.stats import mannwhitneyu, wilcoxon, PermutationMethod

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from config import OUTPUT_DIR, SOURCE_DIR

O = OUTPUT_DIR / 'group_proportions'
O.mkdir(parents=True, exist_ok=True)
S = SOURCE_DIR

radii = [60, 70, 90, 100]
subs = ['SS1', 'SS2', 'SS3', 'SS4']
groups = ['PT_NR', 'PT_R', 'RT']

colors = {
    'PT_NR': '#4EA64A',
    'PT_R': '#3777AC',
    'RT': '#D5231D'
}

records = []

for r in radii:
    with h5py.File(S / f'tumor_niche_0929_{r}.h5ad') as f:
        b = f['obs/batch']
        s = f['obs/cell_type']

        bc = b['codes'][:].astype(np.int64)
        sc = s['codes'][:].astype(np.int64)

        bn = b['categories'].asstr()[:]
        sn = s['categories'].asstr()[:]

        assert (bc >= 0).all() and (sc >= 0).all()

        ct = np.bincount(
            bc * len(sn) + sc,
            minlength=len(bn) * len(sn)
        ).reshape(len(bn), len(sn))

        assert ct.sum() == len(bc)

        for i, batch in enumerate(bn):
            if ct[i].sum() == 0:
                continue

            group = {
                'F': 'PT_NR',
                'P': 'PT_R',
                'R': 'RT'
            }[batch[0]]

            for ss in subs:
                j = [
                    j for j, v in enumerate(sn)
                    if v.startswith(ss + '_')
                ]

                assert len(j) == 1

                records.append({
                    'radius_um': r,
                    'batch': batch,
                    'patient': ('NR_' if batch.startswith('F') else 'PAIR_') + batch[1:],
                    'group': group,
                    'subtype': ss,
                    'n_subtype': int(ct[i, j[0]]),
                    'n_total_tumor': int(ct[i].sum()),
                    'proportion': ct[i, j[0]] / ct[i].sum()
                })

d = pd.DataFrame(records)

assert np.allclose(
    d.groupby(['radius_um', 'batch']).proportion.sum(),
    1
)

d.to_csv(
    O / 'sample_subtype_proportions.csv',
    index=False
)

stats = []

for r in radii:
    for ss in subs:
        z = d[
            (d.radius_um == r) &
            (d.subtype == ss)
        ]

        f = z[z.group == 'PT_NR'].proportion.to_numpy()
        p = z[z.group == 'PT_R'].set_index('patient').proportion
        rt = z[z.group == 'RT'].set_index('patient').proportion

        assert len(f) == len(p) == len(rt) == 12
        assert set(p.index) == set(rt.index)

        rt = rt.reindex(p.index)

        u = mannwhitneyu(
            p.to_numpy(),
            f,
            alternative='two-sided',
            method='asymptotic',
            use_continuity=True
        )

        delta = p.to_numpy() - rt.to_numpy()

        w = (
            wilcoxon(
                delta,
                alternative='two-sided',
                zero_method='wilcox',
                method=PermutationMethod(n_resamples=np.inf),
                correction=False
            )
            if np.any(delta != 0)
            else None
        )

        stats.append({
            'radius_um': r,
            'subtype': ss,
            'comparison': 'PT_R vs PT_NR',
            'test': 'Mann-Whitney U (two-sided, asymptotic, continuity correction)',
            'n_1': 12,
            'n_2': 12,
            'statistic': u.statistic,
            'P_unadjusted': u.pvalue,
            'median_group1': p.median(),
            'median_group2': np.median(f),
            'median_paired_difference': np.nan
        })

        stats.append({
            'radius_um': r,
            'subtype': ss,
            'comparison': 'PT_R vs RT',
            'test': 'Wilcoxon signed-rank (two-sided, exhaustive sign permutations, zero differences excluded)',
            'n_1': 12,
            'n_2': 12,
            'statistic': w.statistic if w else 0,
            'P_unadjusted': w.pvalue if w else 1,
            'median_group1': p.median(),
            'median_group2': rt.median(),
            'median_paired_difference': np.median(delta)
        })

t = pd.DataFrame(stats)

t.to_csv(
    O / 'nonparametric_tests_unadjusted.csv',
    index=False
)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.linewidth': 0.6
})

rng = np.random.default_rng(20261013)


def panel(ax, r):
    z = d[d.radius_um == r]
    top = 0

    for i, ss in enumerate(subs):
        q = z[z.subtype == ss]
        heights = []

        for j, group in enumerate(groups):
            vals = q[q.group == group].proportion.to_numpy() * 100
            pos = i + (j - 1) * 0.24

            heights.extend(vals)

            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=0.19,
                patch_artist=True,
                showfliers=False,
                whis=1.5,
                manage_ticks=False,
                medianprops={'color': 'black', 'linewidth': 0.8},
                boxprops={'linewidth': 0.6},
                whiskerprops={'linewidth': 0.6},
                capprops={'linewidth': 0.6}
            )

            bp['boxes'][0].set_facecolor(colors[group])
            bp['boxes'][0].set_alpha(0.7)

            ax.scatter(
                pos + rng.uniform(-0.05, 0.05, len(vals)),
                vals,
                c=colors[group],
                s=10,
                edgecolors='white',
                linewidths=0.25,
                zorder=3
            )

        y = max(heights) + 6

        for k, (x1, x2, comp) in enumerate([
            (i - 0.24, i, 'PT_R vs PT_NR'),
            (i, i + 0.24, 'PT_R vs RT')
        ]):
            yy = y + k * 10

            pval = float(
                t.loc[
                    (t.radius_um == r) &
                    (t.subtype == ss) &
                    (t.comparison == comp),
                    'P_unadjusted'
                ].iloc[0]
            )

            ax.plot(
                [x1, x1, x2, x2],
                [yy - 1.5, yy, yy, yy - 1.5],
                color='black',
                lw=0.6
            )

            ax.text(
                (x1 + x2) / 2,
                yy + 1,
                f'P={pval:.3g}',
                ha='center',
                va='bottom',
                fontsize=7
            )

            top = max(top, yy + 8)

    ax.set_xticks(range(4), subs)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0, 126)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel('Tumor subtype proportion (%)')
    ax.set_title(f'{r} μm', pad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.15)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(
            O / f'{name}.{ext}',
            dpi=500,
            facecolor='white',
            bbox_inches='tight'
        )

    plt.close(fig)


fig, axs = plt.subplots(2, 2, figsize=(9, 7))

fig.subplots_adjust(
    left=0.08,
    right=0.98,
    bottom=0.08,
    top=0.9,
    wspace=0.22,
    hspace=0.28
)

for ax, r in zip(axs.flat, radii):
    panel(ax, r)

fig.legend(
    handles=[
        Patch(facecolor=colors[g], label=g)
        for g in groups
    ],
    loc='upper center',
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.985)
)

save(fig, 'subtype_proportions_60_70_90_100um_2x2')

for r in radii:
    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        bottom=0.12,
        top=0.83
    )

    panel(ax, r)

    fig.legend(
        handles=[
            Patch(facecolor=colors[g], label=g)
            for g in groups
        ],
        loc='upper center',
        ncol=3,
        frameon=False
    )

    save(fig, f'subtype_proportions_{r}um')

json.dump(
    {
        'denominator': 'all tumor cells in each batch; includes cells with empty neighborhoods; no minimum subtype count filter',
        'patients': 'F01–F12 independent; P01/R01 paired, etc.',
        'n_per_group': 12,
        'multiple_testing': 'none',
        'test_methods': t.test.unique().tolist(),
        'boxplot': 'median, Q1/Q3, whiskers 1.5 IQR; all 12 sample points shown',
        'colors': colors
    },
    open(O / 'methods.json', 'w'),
    indent=2
)

print(
    t[
        [
            'radius_um',
            'subtype',
            'comparison',
            'P_unadjusted',
            'median_group1',
            'median_group2'
        ]
    ].to_string(index=False)
)

