"""Reference serialization and task-balanced loss summaries (CPU only)."""
import math

TASKS = ('qa_2', 'fwe', 'niah_single_1', 'niah_multikey_1')
SIZES = (1, 3, 5, 7, 9, 11)


def convert(row, task):
    if task not in TASKS:
        raise ValueError('Unsupported task: ' + task)
    prompt, answers = row.get('input'), row.get('outputs')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Missing RULER input')
    if not isinstance(answers, list) or not answers or any(not isinstance(x, str) or not x.strip() for x in answers):
        raise ValueError('Missing RULER reference answers')
    # QA outputs are alternatives; extraction outputs are jointly required items.
    label = answers[0] if task == 'qa_2' else ', '.join(answers)
    return {'prompt': prompt, 'label': label, 'task': task,
            'ruler_outputs': answers, 'ruler_index': row.get('index'),
            'label_rule': 'first reference; no EOS' if task == 'qa_2' else
                          'all outputs in generator order, comma-space separated; no EOS'}


def summarize(rows, tasks, policies):
    """Equal-weight input means within each task, then equal-weight task means."""
    cells = {}
    for task in tasks:
        for policy in policies:
            group = [r for r in rows if r['task'] == task and r['policy'] == policy]
            if not group or any(not math.isfinite(r['label_loss']) for r in group):
                raise ValueError('Missing/nonfinite task-policy cell')
            ids = [r['sample_index'] for r in group]
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicated input-policy observation')
            cells[(task, policy)] = group
        baseline_ids = {r['sample_index'] for r in cells[(task, policies[0])]}
        if any({r['sample_index'] for r in cells[(task, p)]} != baseline_ids for p in policies):
            raise ValueError('Unpaired policies within task')
    by_task = {t: {p: sum(r['label_loss'] for r in cells[(t,p)]) / len(cells[(t,p)])
                   for p in policies} for t in tasks}
    macro = {p: sum(by_task[t][p] for t in tasks) / len(tasks) for p in policies}
    return {'tasks': list(tasks), 'policies': list(policies), 'task_mean_nll': by_task,
            'macro_mean_nll': macro,
            'samples_per_task': {t: len(cells[(t,policies[0])]) for t in tasks},
            'macro_nll_decrease_from_1x1': {p: macro['fixed_1x1'] - macro[p] for p in policies},
            'definition': 'mean answer-token NLL within input; mean over inputs within task; equal-weight mean over tasks',
            'status': 'complete'}
