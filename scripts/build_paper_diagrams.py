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


def canvas(height):
    fig, ax = plt.subplots(figsize=(10, height))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 10), ylim=(0, height))
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
    fig, ax = canvas(8.5)
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
    # F1 matrix uses four neutral-text cells, plus disclosure diagnostics.
    label(ax, 7.33, 3.68, 'Kept', 11, TEAL, 'bold')
    label(ax, 8.83, 3.68, 'Reset', 11, TEAL, 'bold')
    label(ax, 5.9, 3.15, 'Retained', 11)
    label(ax, 5.9, 2.63, 'Removed', 11)
    for x in [6.66, 8.16]:
        for y in [2.94, 2.42]:
            box(ax, x, y, 1.34, .4, 'Neutral text', TEAL, GREEN, 10.5)
    label(ax, 7.45, 1.92, '+ 2 disclosure branches + A/A repeat\nEach path lasts 20 rounds', 11.5)
    arrow(ax, (2.55, 1.61), (2.55, 1.42))
    arrow(ax, (7.45, 1.61), (7.45, 1.42))
    box(ax, .35, .69, 4.4, .68, 'Within-game cut gains: G00, G10, G01, G11\nPrimary summaries: S, J, M and S − J', size=11.5)
    box(ax, 5.25, .69, 4.4, .68, 'Primary: removed − retained mean price\nwith context kept (paired within parent)', TEAL, GREEN, 11.5)
    label(ax, 5, .27, 'Keep ineligible, non-calm and incomplete games in accounting. Branches are not independent samples.', 10.5, GRAY)
    save(fig, 'report1_pricing_design')


def evidence():
    fig, ax = canvas(9.4)
    label(ax, 5, 9.06, 'What does the supplied evidence establish?', 19, weight='bold')
    box(ax, .35, 8.14, 9.3, .58,
        'Scripted sender + recipient: 5 mechanism types in each of 20 evaluation blocks\nUsed; ignored; ignored but matching; decoding failed; coincidence', size=12)
    label(ax, 5, 7.73, 'Each case is shown in four views, in separate auditor calls', 12, GRAY)
    for x in [1.45, 3.82, 6.18, 8.55]:
        arrow(ax, (x, 7.52), (x, 7.19))
    specs=[(.35,'PUBLIC','Price and\naction trace'),
           (2.72,'DECODER','Public trace +\ndecoding record'),
           (5.08,'PARTIAL EXECUTION','Decoder view +\npartial policy trials'),
           (7.45,'COMPLETE','Decoder view +\nall policy trials')]
    for x,title,body in specs:
        box(ax, x, 6.05, 2.2, 1.1, '', size=11)
        label(ax,x+1.1,6.87,title,10.2,BLUE,'bold')
        label(ax,x+1.1,6.45,body,12)
        arrow(ax,(x+1.1,6.05),(x+1.1,5.68))
    box(ax, .35, 4.61, 9.3, 1.02,
        'LLM auditor returns two separate judgments\n1  Was the target recovered?     2  Does action depend on decoded input?\nFor each: supported / contradicted / insufficient evidence', size=12)
    arrow(ax,(5,4.61),(5,4.39))
    box(ax, .35, 3.01, 9.3, 1.32,
        'MATCHED HARD NEGATIVE: SAME DECODER EVIDENCE\n'
        'Both recover target T and produce action T. Dependence is still insufficiently evidenced.\n'
        'Dependent policy: another input changes the action. Ignored-but-matching: all inputs yield T.\n'
        'Complete trials distinguish the two policies; the observed match alone cannot.',
        TEAL, GREEN, 11.2)
    arrow(ax,(5,3.01),(5,2.69))
    box(ax,.35,1.82,9.3,.82,
        'Code-based reference: grade only what the visible records establish\nMatched ambiguous cases prevent hidden truth from rewarding an unjustified guess\nScripted positive/negative controls + baselines on the same evaluation cases', TEAL,GREEN,11.5)
    arrow(ax,(5,1.82),(5,1.52))
    box(ax,.35,.69,9.3,.77,
        'Report coverage, resolved accuracy and unsupported conclusions\nRetain missing replies; compare evidence views within matched blocks\n400 planned cases/model • 2 judgments/case • uncertainty clustered by 20 blocks',size=11.5)
    label(ax,5,.28,'Supplied-record auditing—not discovery of hidden channels, intent or real-world collusion.',10.8,GRAY)
    save(fig,'report2_evidence_audit_design')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('reports/designs'))
    args = parser.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    pricing()
    evidence()
    print(f'Saved PNG, SVG and PDF diagrams to {OUT}')
