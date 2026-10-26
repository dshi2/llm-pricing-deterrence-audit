"""Rebuild the README table and figure from bundled, hashed analysis exports."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'deepseek/deepseek-v4.1-flash-1017': 'DeepSeek', 'qwen/qwen3.8-flash': 'Qwen'}


def build():
    source = ROOT / 'results/exports'
    manifest = json.loads((ROOT / 'results/export_manifest.json').read_text())
    for name, record in manifest['files'].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != record['bundled_sha256']:
            raise ValueError(f'Export hash mismatch: {name}')
    games = pd.read_csv(source / 'experiment1_games.csv')
    effects = pd.concat([pd.read_csv(source / f'experiment2_{name}_summary.csv')
                         for name in ('deepseek', 'qwen')])
    effects = effects[(effects.contrast == 'replacement_memory') &
                      (effects.outcome == 'mean_price') &
                      (effects.subgroup == 'all_calm_checkpoints')]
    rows = ['| Model | Eligible pricing games / completed | Removal effect | 98.75% interval | Calm pairs |',
            '|---|---:|---:|---:|---:|']
    for row in effects.itertuples():
        g = games[games.model == row.model]
        rows.append(f'| {NAMES[row.model]} | {(g.status == "eligible").sum()}/{len(g)} | '
                    f'{row.mean:+.4f} | [{row.low:.4f}, {row.high:.4f}] | {row.n_pairs} |')
    text = '\n'.join(rows)
    (ROOT / 'results/README_TABLE.md').write_text(text + '\n')
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1.6, 1]})
    eligible = games[games.status == 'eligible']
    for row in eligible.itertuples():
        label = f'{NAMES[row.model]} · game {row.seed}'
        axes[0].plot(range(4), [row.G00, row.G10, row.G01, row.G11], 'o-',
                     label=label, linewidth=1.5, markersize=4)
    axes[0].axhline(0, color='#555555', linewidth=.8, linestyle='--')
    axes[0].set(xticks=range(4), xticklabels=['Neither', 'Rival 1', 'Rival 2', 'Both'],
                xlabel='Rivals allowed to respond', ylabel='Gain from prescribed price cut',
                title='Individual and joint responses')
    axes[0].legend(fontsize=10, loc='lower left', frameon=False)
    for i, row in enumerate(effects.itertuples()):
        axes[1].errorbar(row.mean, i, xerr=[[row.mean-row.low], [row.high-row.mean]],
                         fmt='o', color='#245c80', capsize=4)
    axes[1].axvline(0, color='#555555', linewidth=.8, linestyle='--')
    axes[1].set(yticks=range(len(effects)), yticklabels=[NAMES[m] for m in effects.model],
                ylim=(-.7, len(effects)-.3), xlabel='Mean-price difference: removed − retained',
                title='Pricing after retaliation removal')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=2)
    for ext in ('png', 'svg'):
        fig.savefig(ROOT / f'figures/results_summary.{ext}', dpi=180)
    plt.close(fig)
    readme = ROOT / 'README.md'
    content = readme.read_text()
    start, end = '<!-- RESULTS:START -->', '<!-- RESULTS:END -->'
    if start in content:
        before, rest = content.split(start, 1)
        _, after = rest.split(end, 1)
        readme.write_text(before + start + '\n' + text + '\n' + end + after)


if __name__ == '__main__':
    build()
