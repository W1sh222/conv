"""Recommend an evaluated checkpoint subject to a LongBench retention gate.

No model loading, training or weight replacement. Prefer held-out validation
scores for checkpoint selection; keep final benchmark evaluation independent.
"""
import argparse
import csv
import json
import math
from pathlib import Path

LB_TASKS = ('2wikimqa gov_report hotpotqa lcc lsht multi_news multifieldqa_en '
            'musique narrativeqa qasper qmsum repobench-p samsum trec triviaqa vcsum').split()
RU_TASKS = ('niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 '
            'niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2').split()
EXCLUDED = {'niah_multikey_2', 'niah_multikey_3', 'qa_1'}


def finite_score(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError(f'Invalid task score: {value}')
    return value


def longbench_mean(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    scores=[]
    for task in LB_TASKS:
        values=[finite_score(v) for k,v in data.items() if k == task or k.startswith(task+'-')]
        if len(values)!=1:
            raise ValueError(f'Missing or duplicate LongBench task: {task}')
        scores.extend(values)
    return math.fsum(scores)/16


def ruler_mean(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        rows={r[0]:r[1:] for r in csv.reader(f) if r}
    if len(rows['Tasks'])!=13 or set(rows['Tasks'])!=set(RU_TASKS) or len(rows['Score'])!=13:
        raise ValueError('Need all 13 task scores; exclusions are applied by this script')
    scores=dict(zip(rows['Tasks'],map(finite_score,rows['Score'])))
    return math.fsum(v for t,v in scores.items() if t not in EXCLUDED)/10


def choose(baseline_lb, baseline_ru, candidates, max_lb_drop=0, min_ruler_gain=0.1):
    eligible=[r for r in candidates if r['longbench'] >= baseline_lb-max_lb_drop-1e-9
              and r['ruler10'] >= baseline_ru+min_ruler_gain-1e-9]
    return max(eligible,key=lambda r:(r['ruler10'],r['longbench'])) if eligible else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-longbench',required=True)
    p.add_argument('--baseline-ruler128k',required=True)
    p.add_argument('--candidates',required=True,help='JSON list: weight, longbench, ruler128k, top_p, selector')
    p.add_argument('--topp',type=float,default=0.92)
    p.add_argument('--max-longbench-drop',type=float,default=0.0,help='Score points; default requires no mean regression')
    p.add_argument('--min-ruler-gain',type=float,default=0.1)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    if not 0<args.topp<=1 or any(not math.isfinite(x) or x<0 for x in
                              (args.max_longbench_drop,args.min_ruler_gain)):
        p.error('Invalid threshold or score tolerances')
    entries=json.loads(Path(args.candidates).read_text(encoding='utf-8'))
    measured=[]
    for r in entries:
        if float(r['top_p'])!=args.topp or r['selector']!='positive_mass_v1':
            raise ValueError('All candidates must use the requested Top-p and positive_mass_v1')
        measured.append(dict(weight=r['weight'],longbench=longbench_mean(r['longbench']),
                             ruler10=ruler_mean(r['ruler128k'])))
    lb,ru=longbench_mean(args.baseline_longbench),ruler_mean(args.baseline_ruler128k)
    best=choose(lb,ru,measured,args.max_longbench_drop,args.min_ruler_gain)
    result=dict(top_p=args.topp,selector='positive_mass_v1',baseline_longbench=lb,
                baseline_ruler10=ru,max_longbench_drop=args.max_longbench_drop,
                min_ruler_gain=args.min_ruler_gain,candidates=measured,
                recommendation=best or 'keep_input_anchor')
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
