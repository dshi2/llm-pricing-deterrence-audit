"""Report design figures; no live data or collection code is modified."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/collusion-report-mpl')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path('reports/designs')
INK = '#162b3b'
BLUE = '#235b83'
TEAL = '#16756f'
PALE = '#edf4f9'
GREEN = '#edf7f5'
GRAY = '#536471'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})


def canvas(height, bottom=0):
    fig, ax = plt.subplots(figsize=(10, height - bottom))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 10), ylim=(bottom, height))
    ax.axis('off')
    return fig, ax


def label(ax, x, y, text, size=12, color=INK, weight='normal', ha='center'):
    return ax.text(x, y, text, ha=ha, va='center', fontsize=size,
                   color=color, weight=weight, linespacing=1.35)


def box(ax, x, y, w, h, text, color=BLUE, fill=PALE, size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.025,rounding_size=0.08',
                               facecolor=fill, edgecolor=color, linewidth=1.25))
    label(ax, x+w/2, y+h/2, text, size)


def arrow(ax, a, b, color=GRAY):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=12,
                                 linewidth=1.3, color=color))


def save(fig, name):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white')
    plt.close(fig)


def pricing():
    fig, ax = canvas(8.5, bottom=.5)
    label(ax, 5, 8.16, 'Two complementary pricing experiments', 19, weight='bold')
    box(ax, .3, 7.5, 9.4, .38,
        'Before study collection: scripted controls, endpoint checks and canaries', size=12)
    label(ax, 2.55, 7.12, 'EXPERIMENT 1\nIndividual and joint deterrence', 13, BLUE, 'bold')
    label(ax, 7.45, 7.12, 'EXPERIMENT 2\nPricing after retaliation removal', 13, TEAL, 'bold')
    ax.plot([5, 5], [.65, 7.2], color='#d6dfe4', lw=1)
    box(ax, .35, 6.12, 4.4, .65, '3 LLM sellers • 150 parent rounds\nSave the parent checkpoint')
    box(ax, 5.25, 6.12, 4.4, .65, '1 LLM seller + scripted rival\n100 parent rounds; fixed checkpoint', TEAL, GREEN)
    arrow(ax, (2.55, 6.12), (2.55, 5.91))
    arrow(ax, (7.45, 6.12), (7.45, 5.91))
    box(ax, .35, 5.12, 4.4, .75, '20-round all-LLM reference\nTest the prescribed cut against\nreplayed rival prices')
    box(ax, 5.25, 5.12, 4.4, .75, 'Fresh fixed-rival baseline for every\ncompleted parent; only calm rival\nstates enter removal branches', TEAL, GREEN)
    arrow(ax, (2.55, 5.12), (2.55, 4.86))
    arrow(ax, (7.45, 5.12), (7.45, 4.86))
    label(ax, 2.55, 4.62, 'If eligible: select seller A; restart\nall branches at the parent checkpoint', 12, BLUE)
    label(ax, 7.45, 4.62, 'At calm checkpoints: restart\nall branches from the same history', 12, TEAL)
    label(ax, 2.55, 4.09, 'LIVE RIVALS × A’S PRICE PATH', 11, BLUE, 'bold')
    label(ax, 7.45, 4.09, 'RIVAL POLICY × SELLER CONTEXT', 11, TEAL, 'bold')
    # Main matrix: every response condition has a cut and no-cut branch.
    xs = [.45, 1.53, 2.61, 3.69]
    for x, text in zip(xs, ['Neither\n00', 'B only\n10', 'C only\n01', 'Both\n11']):
        box(ax, x, 3.35, .96, .55, text, size=11)
        box(ax, x, 2.83, .96, .38, 'No cut', size=11)
        box(ax, x, 2.34, .96, .38, 'Cut', size=11)
    label(ax, 2.55, 1.92, '8 branches × 20 rounds + A/A repeat\nA’s cut: 10% for the first 10 rounds', 11.5)
    # Experiment 2 matrix: the seller is not told about the change; disclosure branches are extra.
    label(ax, 7.33, 3.68, 'Kept', 11, TEAL, 'bold')
    label(ax, 8.83, 3.68, 'Reset', 11, TEAL, 'bold')
    label(ax, 5.9, 3.15, 'Retained', 11)
    label(ax, 5.9, 2.63, 'Removed', 11)
    for x in [6.66, 8.16]:
        for y in [2.94, 2.42]:
            box(ax, x, y, 1.34, .4, 'No disclosure', TEAL, GREEN, 10.5)
    label(ax, 7.45, 1.92, '+ 2 disclosure branches + A/A repeat\nEach path lasts 20 rounds', 11.5)
    arrow(ax, (2.55, 1.61), (2.55, 1.42))
    arrow(ax, (7.45, 1.61), (7.45, 1.42))
    box(ax, .35, .69, 4.4, .68, 'Within-game cut gains: G00, G10, G01, G11\nPrimary summaries: S, J, M and S − J', size=11.5)
    box(ax, 5.25, .69, 4.4, .68, 'Primary: removed − retained mean price\nwith context kept (paired within parent)', TEAL, GREEN, 11.5)
    save(fig, 'report1_pricing_design')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('reports/designs'))
    args = parser.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    pricing()
    print(f'Saved PNG, SVG and PDF diagram to {OUT}')
