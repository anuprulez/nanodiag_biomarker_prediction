"""Publication figures rendered from saved patient predictions and raw beta values."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 13,
    'axes.labelsize': 16, 'axes.titlesize': 18, 'axes.linewidth': 1.,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
}
PANEL_STYLE = {
    'seeds_only': ('Seeds only', '#0072B2', '-'),
    'candidates_only': ('Candidates only', '#D55E00', '-'),
    'seeds_plus_candidates': ('Seeds + candidates', '#7B2CBF', '-'),
    'all_common_likely_positive': ('All common likely positive', '#009E73', '-'),
    'all_common_predicted_negative': ('Common predicted negatives', '#8C564B', '-'),
}


def save_publication(fig, out, stem):
    """Keep curves/text vector-based; rasterize only the dense patient points."""
    out = Path(out)/'plots'
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for extension in ('pdf', 'svg', 'png'):
            fig.savefig(out/f'{stem}.{extension}', dpi=300, bbox_inches='tight',
                        facecolor='white', transparent=False)
    plt.close(fig)


def performance_figure(labels, predictions, counts, cohort, disease=None):
    """ROC and empirical PR curves only; no refitting or probability changes."""
    labels = np.asarray(labels, dtype=int)
    if cohort == 'external_transfer_selected_cohort':
        predictions = {name: predictions[name] for name in ('seeds_only', 'candidates_only')}
    with plt.rc_context(STYLE):
        many=len(predictions)>3
        fig, axes = plt.subplots(1, 2, figsize=(14.4, 8.5 if many else 6.7))
        # Baselines sit underneath data curves and do not compete with the legend.
        axes[0].plot([0, 1], [0, 1], color='#a0a0a0', lw=1.2, ls=':', zorder=1)
        axes[1].axhline(labels.mean(), color='#a0a0a0', lw=1.2, ls=':', zorder=1)
        for name in predictions:
            label,color,linestyle=PANEL_STYLE[name]
            p = np.asarray(predictions[name], dtype=float)
            if len(p) != len(labels) or not np.isfinite(p).all():
                raise ValueError(f'Invalid predictions for {name}')
            fpr, tpr, _ = roc_curve(labels, p)
            precision, recall, _ = precision_recall_curve(labels, p)
            caption = f'{label} ({counts[name]} CpGs)'
            axes[0].plot(fpr, tpr, color=color, ls=linestyle, lw=2.8, zorder=3,
                         label=(f'{caption} · AUROC = {roc_auc_score(labels, p):.3f}' if many else
                                f'{caption}\nAUROC = {roc_auc_score(labels, p):.3f}'))
            # Average precision is reported explicitly, not relabeled as trapezoid AUC.
            axes[1].step(recall, precision, where='post', color=color,
                         ls=linestyle, lw=2.8, zorder=3,
                         label=(f'{caption} · AP = {average_precision_score(labels, p):.3f}' if many else
                                f'{caption}\nAP = {average_precision_score(labels, p):.3f}'))
            # Mark the true full-recall endpoint; do not truncate or extend the empirical curve.
            axes[1].plot(recall[0], precision[0], 'o', color=color, ms=5, zorder=4)
        axes[0].set(xlabel='1 − specificity', ylabel='Sensitivity', title='A   Receiver operating characteristic')
        axes[1].set(xlabel='Recall', ylabel='Precision', title='B   Precision–recall')
        for ax in axes:
            ax.set_xlim(-.025, 1.025); ax.set_ylim(-.025, 1.025)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks(np.linspace(0., 1., 6))
            ax.set_yticks(np.linspace(0., 1., 6))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.tick_params(direction='out', length=5, pad=6)
            ax.set_axisbelow(True); ax.grid(color='#e8e8e8', linewidth=.65)
            ax.spines[['top', 'right']].set_visible(False)
            ax.title.set_position((.5, 1.035))
        for ax, location in zip(axes, ['lower right', 'lower left']):
            if many:
                ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),fontsize=10.5,
                          frameon=False,handlelength=2.7,labelspacing=.8)
            else:
                ax.legend(loc=location, fontsize=10.5, frameon=True, facecolor='white',
                          edgecolor='#dddddd', framealpha=.96, handlelength=3.3,
                          borderpad=.8, labelspacing=.8)
        title = 'Original cohort · five-fold patient cross-validation' if cohort == 'original_patient_CV' else 'External cohort · original-trained models'
        if disease:
            title = ('Breast cancer' if disease == 'BC' else disease) + ' | ' + title
        fig.suptitle(title, fontsize=19, y=.985)
        fig.subplots_adjust(left=.07, right=.985, bottom=.32 if many else .13,
                            top=.86 if many else .82, wspace=.24)
    return fig


def external_distribution_figure(panel, matrix, metadata, disease, group_label='candidate'):
    """One figure for a complete CpG group; preserve all measured patients."""
    cpgs = list(panel.CpG_ID)
    if not cpgs:
        raise ValueError('Cannot plot an empty candidate panel')
    metadata = metadata.loc[matrix.index]
    labels = metadata.label.astype(int)
    colors = ['#0072B2', '#D55E00']
    disease_name = 'Breast cancer' if disease == 'BC' else disease
    ncols = 12 if len(cpgs) > 144 else 8 if len(cpgs) > 60 else 6 if len(cpgs) > 24 else 5
    nrows = int(np.ceil(len(cpgs)/ncols))
    rng = np.random.default_rng(20261031)
    is_bc_candidates = disease == 'BC' and group_label == 'candidate' and len(cpgs) == 33
    is_aml_candidates = disease == 'AML' and group_label == 'candidate' and len(cpgs) == 17
    is_fixed_candidates = is_bc_candidates or is_aml_candidates
    font_scale = 1.2 if is_fixed_candidates else 1.0
    style = {key: value * font_scale if key.endswith('size') and isinstance(value, (int, float)) else value
             for key, value in STYLE.items()}
    with plt.rc_context(style):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.05*ncols, 2.85*nrows+1.3),
                                 sharey=False, squeeze=False)
        for index, (ax, cpg) in enumerate(zip(axes.flat, cpgs)):
            values = [matrix.loc[labels.eq(k), cpg].dropna().to_numpy(float) for k in (0, 1)]
            if any(len(group) == 0 for group in values):
                raise ValueError(f'{cpg}: missing external disease/control measurements')
            for x, (group, color) in enumerate(zip(values, colors), 1):
                # Keep large grids vector-based: repeated per-axis rasterization
                # would allocate and scan a full-page bitmap for every CpG.
                ax.scatter(x+rng.uniform(-.16, .16, len(group)), group, s=5,
                           color=color, alpha=.22, linewidths=0, rasterized=(len(cpgs) <= 60), zorder=2)
            box = ax.boxplot(values, positions=[1, 2], widths=.48, patch_artist=True,
                             showfliers=False, zorder=3,
                             medianprops={'color':'#202020', 'linewidth':1.4},
                             whiskerprops={'color':'#404040', 'linewidth':1.},
                             capprops={'color':'#404040', 'linewidth':1.})
            for patch, color in zip(box['boxes'], colors):
                patch.set(facecolor=matplotlib.colors.to_rgba(color, .18), edgecolor=color, linewidth=1.2)
            delta = values[1].mean()-values[0].mean()
            ax.set_title(f'{cpg}\nΔβ = {delta:+.3f}', fontsize=11.5*font_scale, pad=9*font_scale)
            ax.set_xticks([1, 2], ['Healthy', disease])
            ax.set_ylim(-.035, 1.035); ax.set_xlim(.5, 2.5)
            ax.yaxis.set_major_locator(MultipleLocator(.25))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='both', labelsize=10*font_scale, direction='out', length=3)
            ax.tick_params(axis='y', labelleft=(index % ncols == 0))
            ax.grid(axis='y', color='#e8e8e8', lw=.6); ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
        for ax in list(axes.flat)[len(cpgs):]:
            ax.set_visible(False)
        for row in range(nrows):
            axes[row, 0].set_ylabel('Methylation β', fontsize=12*font_scale)
        healthy_studies = ', '.join(sorted(metadata.loc[labels.eq(0), 'series'].unique()))
        disease_studies = ', '.join(sorted(metadata.loc[labels.eq(1), 'series'].unique()))
        legend = [Patch(facecolor=matplotlib.colors.to_rgba(colors[0], .3), edgecolor=colors[0],
                        label=f'Healthy · {healthy_studies} · n={labels.eq(0).sum()}'),
                  Patch(facecolor=matplotlib.colors.to_rgba(colors[1], .3), edgecolor=colors[1],
                        label=f'{disease_name} · {disease_studies} · n={labels.eq(1).sum()}')]
        title = ('Breast cancer: External cohort methylation distribution | 33 candiate CpGs'
                 if is_bc_candidates else
                 'AML: External cohort methylation distribution | 17 candidate CpGs'
                 if is_aml_candidates else
                 f'{disease_name}: external methylation distributions | {len(cpgs)} {group_label} CpGs')
        fig.suptitle(title, fontsize=20*font_scale, y=.99)
        fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .965),
                   ncol=2, fontsize=12*font_scale, frameon=False, columnspacing=2.)
        fig.subplots_adjust(left=.055, right=.99, bottom=.035,
                            top=.865 if is_aml_candidates else .90,
                            wspace=.28, hspace=.80 if is_fixed_candidates else .68)
    return fig


def forest_figure(panel, disease):
    """Paired cohort effects and pointwise Welch intervals, with no footer."""
    from matplotlib.ticker import MaxNLocator
    n = len(panel)
    if n == 0:
        raise ValueError('Cannot draw an empty forest plot')
    y = np.arange(n)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11.8, max(8., n*.43+1.7)))
        for row in range(n):
            if row % 2 == 0:
                ax.axhspan(row-.5, row+.5, color='#f3f5f7', zorder=0, linewidth=0)
        handles = []
        for cohort, offset, color, marker, label in [
            ('Original', -.16, '#0072B2', 'o', 'Original cohort'),
            ('External', .16, '#D55E00', 'D', 'External cohort'),
        ]:
            effect = panel[f'{cohort}_patient_delta'].to_numpy(float)
            low = panel[f'{cohort}_CI95_low'].to_numpy(float)
            high = panel[f'{cohort}_CI95_high'].to_numpy(float)
            if not np.isfinite(np.r_[effect, low, high]).all():
                raise ValueError(f'{cohort}: nonfinite forest plot effect or interval')
            if np.any(low > effect) or np.any(high < effect):
                raise ValueError(f'{cohort}: invalid confidence intervals')
            handles.append(ax.errorbar(effect, y+offset,
                xerr=np.vstack([effect-low, high-effect]), fmt=marker,
                markersize=6, markeredgecolor='white', markeredgewidth=.5,
                capsize=3, capthick=1.2, elinewidth=1.7,
                color=color, label=label, zorder=3))
        ax.axvline(0, color='#777777', lw=1.2, ls='--', zorder=1)
        ax.set_yticks(y, panel.CpG_ID, fontsize=11.5)
        ax.set_ylim(n-.5, -.5)
        lower = min(0., panel[['Original_CI95_low', 'External_CI95_low']].min().min())
        upper = max(0., panel[['Original_CI95_high', 'External_CI95_high']].max().max())
        margin = max(.025, (upper-lower)*.06)
        ax.set_xlim(lower-margin, upper+margin)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xlabel('Δβ (disease − healthy; pointwise 95% Welch CI)', labelpad=12)
        ax.set_ylabel('CpG site', labelpad=14)
        ax.tick_params(axis='x', direction='out', length=5, pad=7)
        ax.tick_params(axis='y', length=0, pad=9)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#dddddd', lw=.65)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        name = 'Breast cancer' if disease == 'BC' else disease
        fig.suptitle(f'{name} | Original and external methylation effects', fontsize=20, y=.985)
        fig.legend(handles=handles, labels=['Original cohort', 'External cohort'],
                   loc='upper center', bbox_to_anchor=(.57, .957), ncol=2,
                   fontsize=13, frameon=False, handlelength=2.7, columnspacing=3.)
        fig.subplots_adjust(left=.19, right=.975, bottom=.085, top=.91)
    return fig
