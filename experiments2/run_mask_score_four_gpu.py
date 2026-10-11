#!/usr/bin/env python3
"""Four pre-specified all-row mask-score sweeps, one process per visible GPU.

Run from any directory; data/model paths saved as relative paths are resolved
against the repository root. Reuses run_mask_score.py without changing its NLL.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = [(6, 2), (14, 16), (22, 24), (30, 4)]
VALUE_OPTIONS = ('sample_index', 'ratio', 'stride', 'selector', 'conv_weights',
                 'background', 'dtype', 'seed', 'rope_factor',
                 'rope_original_length', 'max_position_embeddings',
                 'max_prompt_tokens', 'max_label_tokens', 'loss_tolerance')


def absolute(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def parse_pairs(value):
    try:
        pairs = [tuple(map(int, pair.split(':'))) for pair in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('Use layer:head,layer:head,layer:head,layer:head') from error
    if len(pairs) != 4 or any(len(p) != 2 or min(p) < 0 for p in pairs) or len(set(pairs)) != 4:
        raise argparse.ArgumentTypeError('Specify four distinct nonnegative layer:head pairs')
    return pairs


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=absolute,
                   default=ROOT/'output/ruler_observation/mask_score_layer14_head8/experiment.json',
                   help='Existing experiment.json (or its directory); inherit input/settings')
    p.add_argument('--model', type=absolute, help='Override a moved model directory')
    p.add_argument('--data', type=absolute, help='Override the location of the SAME saved VT JSONL')
    p.add_argument('--pairs', type=parse_pairs, default=DEFAULT_PAIRS,
                   help='Default 6:2,14:16,22:24,30:4; zero-based query-head indices')
    p.add_argument('--gpus', default='0,1,2,3',
                   help='Four indices into the CURRENT visible GPU list; default 0,1,2,3')
    p.add_argument('--output-root', type=absolute,
                   default=ROOT/'output/ruler_observation/mask_score_four_gpu')
    p.add_argument('--figures-root', type=absolute,
                   default=ROOT/'acl_revision/Figs/mask_score_four_gpu')
    p.add_argument('--dry-run', action='store_true', help='Print commands; no CUDA/model/file writes')
    return p


def build_plan(args):
    reference = args.reference/'experiment.json' if args.reference.is_dir() else args.reference
    meta = json.loads(reference.read_text(encoding='utf-8'))
    if meta.get('status') != 'complete':
        raise ValueError('Reference must be a completed experiment')
    settings = meta['arguments'].copy()
    settings['model'] = str(args.model or absolute(settings['model']))
    settings['data'] = str(args.data or absolute(settings['data']))
    if settings.get('conv_weights'):
        settings['conv_weights'] = str(absolute(settings['conv_weights']))
    try:
        indices = [int(g) for g in args.gpus.split(',')]
    except ValueError as error:
        raise ValueError('--gpus must contain four visible GPU indices') from error
    if len(indices) != 4 or len(set(indices)) != 4 or min(indices) < 0:
        raise ValueError('--gpus requires four distinct nonnegative visible GPU indices')
    parent_visibility = os.environ.get('CUDA_VISIBLE_DEVICES')
    if parent_visibility is not None:
        visible = [v.strip() for v in parent_visibility.split(',') if v.strip()]
        if not visible or max(indices) >= len(visible):
            raise ValueError('Requested index is outside inherited CUDA_VISIBLE_DEVICES')
        gpu_ids = [visible[i] for i in indices]
    else:
        gpu_ids = [str(i) for i in indices]
    tasks = []
    for (layer, head), index, gpu in zip(args.pairs, indices, gpu_ids):
        name = f'layer{layer}_head{head}'
        output = args.output_root/name
        command = [sys.executable, '-u', str(ROOT/'experiments2/run_mask_score.py'),
                   '--model', settings['model'], '--data', settings['data'],
                   '--layer', str(layer), '--head', str(head), '--query-block', 'last',
                   '--device-map', 'cuda:0', '--output', str(output)]
        for option in VALUE_OPTIONS:
            if settings.get(option) is not None:
                command += ['--'+option.replace('_', '-'), str(settings[option])]
        if settings.get('chat_template'):
            command += ['--chat-template']
        prior = output/'experiment.json'
        complete = False
        if prior.exists():
            saved = json.loads(prior.read_text(encoding='utf-8'))
            complete = saved.get('status') == 'complete'
            expected = {**settings, 'layer': layer, 'head': head}
            for key in ('model', 'data', 'layer', 'head', 'chat_template', *VALUE_OPTIONS):
                actual = saved['arguments'].get(key)
                wanted = expected.get(key)
                if key in ('model', 'data', 'conv_weights') and actual:
                    actual = str(absolute(actual))
                if actual != wanted:
                    raise ValueError(f'{name}: existing result has different {key}; use a new output root')
            if not complete:
                command += ['--resume']
        elif output.exists() and any(output.iterdir()):
            raise ValueError(f'{output} is nonempty without metadata; use a new output root')
        plot = [sys.executable, '-u', str(ROOT/'experiments2/replot_mask_score.py'),
                '--input-dir', str(output), '--output-prefix',
                str(args.figures_root/f'mask_score_{name}'), '--dpi', '600']
        tasks.append({'name': name, 'layer': layer, 'head': head,
                      'visible_gpu_index': index, 'cuda_visible_devices': gpu,
                      'output': str(output), 'command': command,
                      'plot_command': plot, 'already_complete': complete})
    identity = meta.get('identity', {})
    expected_prompt = identity.get('prompt_sha256', meta.get('prompt_sha256'))
    expected_label = identity.get('label_ids', meta.get('label_ids'))
    return {'reference': str(reference), 'settings': settings,
            'expected_prompt_sha256': expected_prompt, 'expected_label_ids': expected_label,
            'design': 'Pre-specified layers/heads; same saved input and settings; no outcome-based selection',
            'tasks': tasks}


def validate_complete(task, plan):
    meta = json.loads((Path(task['output'])/'experiment.json').read_text(encoding='utf-8'))
    if meta.get('status') != 'complete' or meta['completed_trials'] != meta['total_trials']:
        raise ValueError(f"{task['name']}: incomplete sweep")
    identity = meta['identity']
    for field, expected in [('prompt_sha256', plan['expected_prompt_sha256']),
                            ('label_ids', plan['expected_label_ids'])]:
        if expected is not None and identity[field] != expected:
            raise ValueError(f"{task['name']}: {field} differs from reference; refusing comparison")
    return {'layer': task['layer'], 'head': task['head'],
            'prompt_tokens': meta['prompt_token_count'], 'answer_tokens': meta['label_token_count'],
            'baseline_nll': meta['baseline']['label_loss'],
            'tested_candidates': meta['completed_trials'], 'improved_candidates': meta['improved_count'],
            'improved_fraction_of_tested': meta['improved_fraction'],
            'effective_tolerance': meta['effective_improvement_tolerance'],
            'baseline_repeat_drift': meta['baseline_repeat_drift'], 'output': task['output']}


def main():
    args = parser().parse_args()
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    for key in ('model', 'data'):
        if not Path(plan['settings'][key]).exists():
            raise FileNotFoundError(f'{key}: {plan["settings"][key]}')
    config = json.loads((Path(plan['settings']['model'])/'config.json').read_text(encoding='utf-8'))
    for layer, head in args.pairs:
        if layer >= config['num_hidden_layers'] or head >= config['num_attention_heads']:
            raise ValueError(f'Layer/head {layer}/{head} is outside model configuration')
    probe = subprocess.run([sys.executable, '-c', 'import torch; print(torch.cuda.device_count())'],
                           check=True, capture_output=True, text=True)
    count = int(probe.stdout.strip().splitlines()[-1])
    if max(t['visible_gpu_index'] for t in plan['tasks']) >= count:
        raise RuntimeError(f'Only {count} CUDA devices visible; four assigned GPUs are required')
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = args.output_root/'.launcher.lock'
    # One launcher per root; a hard-killed process may leave a lock to inspect.
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, f'{os.getpid()}\n'.encode()); os.close(fd)
    running, handles, results, failures = [], [], [], []
    manifest = args.output_root/'four_gpu_manifest.json'
    plan['data_file_sha256'] = hashlib.sha256(Path(plan['settings']['data']).read_bytes()).hexdigest()

    def write_manifest(status):
        data = {**plan, 'status': status, 'results': results, 'failures': failures}
        temporary = manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        temporary.replace(manifest)

    def finish(task):
        results.append(validate_complete(task, plan))
        subprocess.run(task['plot_command'], cwd=ROOT, check=True)
        print(f"[{task['name']}] complete; figure: {args.figures_root}/mask_score_{task['name']}.pdf", flush=True)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    old_sigterm = signal.signal(signal.SIGTERM, interrupt)
    try:
        write_manifest('running')
        for task in plan['tasks']:
            if task['already_complete']:
                try:
                    finish(task)
                except Exception as error:
                    failures.append({'name': task['name'], 'error': str(error)})
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=task['cuda_visible_devices'],
                       PYTHONUNBUFFERED='1')
            log = open(args.output_root/f"{task['name']}.log", 'a', encoding='utf-8')
            handles.append(log)
            process = subprocess.Popen(task['command'], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            running.append((task, process))
            print(f"[{task['name']}] GPU {task['cuda_visible_devices']}; PID {process.pid}; "
                  f"log {args.output_root}/{task['name']}.log", flush=True)
        while running:
            for task, process in running[:]:
                code = process.poll()
                if code is None:
                    continue
                running.remove((task, process))
                try:
                    if code:
                        raise RuntimeError(f'Worker exited {code}; inspect {task["name"]}.log')
                    finish(task)
                except Exception as error:
                    failures.append({'name': task['name'], 'error': str(error)})
                    print(f"[{task['name']}] FAILED: {error}", file=sys.stderr, flush=True)
                write_manifest('running')
            if running:
                time.sleep(2)
        write_manifest('failed' if failures else 'complete')
        summary = args.output_root/'four_gpu_summary.csv'
        if results:
            with summary.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(results[0]))
                writer.writeheader(); writer.writerows(sorted(results, key=lambda r: (r['layer'], r['head'])))
        print(f'Summary: {summary}; manifest: {manifest}', flush=True)
        return 1 if failures else 0
    except KeyboardInterrupt:
        write_manifest('interrupted')
        print('Interrupted; rerun the same command to resume missing trials.', file=sys.stderr)
        return 130
    finally:
        for _, process in running:
            if process.poll() is None:
                process.terminate()
        for _, process in running:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        for handle in handles:
            handle.close()
        lock.unlink(missing_ok=True)
        signal.signal(signal.SIGTERM, old_sigterm)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
