"""Paper tables from a labelled snapshot of exported F2 judgments; no API calls."""
import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scripts.check_covert_controls import baseline
from scripts.run_saved_report import environment
from sprint import covert_analyze
from sprint.covert import cases

BASELINES = ('evidence_level_only', 'parallel_rise', 'always_abstain', 'always_accuse')
VIEWS = ('public', 'decoder', 'partial_execution', 'complete')


def markdown(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str, r))+' |' for r in rows])


def build(inputs, out, label):
    if out.exists():
        raise FileExistsError(out)
    before = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    table = pd.concat([pd.read_csv(p) for p in inputs], ignore_index=True)
    if table.duplicated(['model', 'case_id', 'question']).any():
        raise ValueError('Overlapping model/case judgments: do not mix original-cap and recovery snapshots')
    if (table.groupby('model').setup.nunique() != 1).any():
        raise ValueError('Multiple configurations for one model; report separately')
    for (model, scheme), group in table.groupby(['model', 'scheme']):
        generated = {c['id']: c for c in cases(sorted(map(int, group.block.unique())), scheme)}
        for r in group.itertuples():
            case = generated[r.case_id]
            assert r.expected == case['expected'][r.question], (model, r.case_id, 'reference mismatch')
            assert bool(r.correct) == (r.predicted == r.expected)
            for name in BASELINES:
                correct = baseline(name, case['record'])[r.question] == r.expected
                assert correct == bool(getattr(r, name+'_correct')), (model, r.case_id, name)
    scores = covert_analyze.summary(table)
    paired = covert_analyze.paired_comparisons(table)
    out.mkdir(parents=True)
    scores.to_csv(out/'metrics.csv', index=False)
    paired.to_csv(out/'same_case_comparisons.csv', index=False)
    account = table.drop_duplicates(['model', 'case_id']).groupby(['model', 'status']).size()
    account.rename('cases').to_csv(out/'accounting.csv')
    baseline_rows = []
    for (model, view, question), group in table.groupby(['model', 'view', 'question']):
        for name in BASELINES:
            values = group.groupby('block')[name+'_correct'].mean()
            baseline_rows.append({'model': model, 'view': view, 'question': question, 'baseline': name,
                                      'numerator': int(group[name+'_correct'].sum()), 'denominator': len(group),
                                      'blocks': len(values), 'mean_block_accuracy': values.mean()})
    baselines = pd.DataFrame(baseline_rows)
    baselines.to_csv(out/'same_case_baseline_scores.csv', index=False)
    models = list(table.model.unique())
    short = {'deepseek/deepseek-v4.1-flash-1017':'DeepSeek', 'qwen/qwen3.8-flash':'Qwen',
             'openai/gpt-oss-120b':'GPT-OSS', 'mistralai/mistral-large-2512':'Mistral'}
    headers = ['View / judgment'] + [short.get(m,m) for m in models]
    text = [f'# F2 tables — {label}',
            'Snapshot of exported judgments, not a fresh raw-journal reconstruction. Reference labels and all four baseline predictions are checked against regenerated case records. Rates and intervals below are percentages; n/d gives the applicable raw numerator/denominator. Rates are means of within-block rates, with descriptive 95% block-bootstrap intervals. Zero-denominator metrics are not estimable. Degenerate intervals do not establish population certainty.']
    for metric in ('correct_per_planned_judgment', 'unsupported_positive', 'unsupported_negative', 'resolved_accuracy', 'answer_coverage'):
        rows = []
        for view in VIEWS:
            for question in ('recovered', 'dependent'):
                row = [view+' / '+question]
                for model in models:
                    r = scores[(scores.model==model)&(scores.view==view)&(scores.question==question)&(scores.metric==metric)].iloc[0]
                    row.append('— (0 applicable)' if r.denominator==0 else
                               f'{100*r.mean_block_rate:.1f}% [{100*r.ci_low:.1f}, {100*r.ci_high:.1f}]; {int(r.numerator)}/{int(r.denominator)}')
                rows.append(row)
        text += ['## '+metric, markdown(headers, rows)]
    rows = []
    for view in VIEWS:
        for question in ('recovered', 'dependent'):
            row = [view+' / '+question]
            for model in models:
                r = paired[(paired.model==model)&(paired.question==question)&(paired.contrast==view+'_minus_evidence_level_only')].iloc[0]
                row.append(f'{100*r.mean_accuracy_difference:+.1f} [{100*r.ci_low:+.1f}, {100*r.ci_high:+.1f}]')
            rows.append(row)
    text += ['## Same-case comparison: model minus evidence-level-only baseline',
             'Percentage-point differences with descriptive, unadjusted 95% block-bootstrap intervals. All planned judgments enter; unanswered model judgments count as incorrect. Comparisons with all four baselines are in the CSV.',
             markdown(headers, rows),
             'Public-view reference labels are always insufficient evidence; success there measures appropriate abstention. These completed-data tables do not replace original-cap analyses or Mistral partial-data accounting.']
    (out/'PAPER_TABLES.md').write_text('\n\n'.join(text)+'\n')
    assert before == {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (out/'manifest.json').write_text(json.dumps({
        'label':label, 'created_utc':datetime.now(UTC).isoformat(), 'inputs':before,
        'analysis_environment':environment(Path(covert_analyze.__file__).resolve().parents[1]),
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'bootstrap':{'seed':0,'draws':10000,'unit':'within-block metric or paired-difference mean'},
        'checks':'Reference labels and all four baseline predictions regenerated for every supplied judgment.'
    },indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--label', required=True, help='Name the collection snapshot/cap explicitly')
    args = parser.parse_args()
    build(args.inputs, args.out, args.label)
