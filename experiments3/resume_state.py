"""Atomic checkpoints for expensive diagnostics; no CUDA dependency."""
import hashlib
import json
import os
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_run(out, kind, args):
    out = Path(out)
    ignored = {'resume', 'dry_run', 'no_plots', 'output'}
    identity = {'version': 1, 'experiment': kind,
                'arguments': {k: v for k, v in vars(args).items() if k not in ignored},
                'data_sha256': file_hash(args.data),
                'weight_sha256': file_hash(args.conv_weights)}
    source_dir = Path(__file__).resolve().parent
    identity['engine_sha256'] = {name: file_hash(source_dir / name) for name in
                                ('experiment_common.py', 'analysis_core.py', 'resume_state.py')}
    identity['model_metadata_sha256'] = {name: file_hash(Path(args.model) / name) for name in
        ('config.json', 'tokenizer_config.json', 'tokenizer.json') if (Path(args.model) / name).is_file()}
    manifest = out / 'run_manifest.json'
    if out.exists() and any(out.iterdir()):
        if not args.resume:
            raise FileExistsError('Use a new/empty output directory or --resume')
        if not manifest.is_file():
            raise ValueError('No resume manifest: legacy results cannot be resumed safely; use a new output directory')
        if json.loads(manifest.read_text(encoding='utf-8')) != identity:
            raise ValueError('Resume configuration/data/weight mismatch; use the original settings or a new output directory')
    else:
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest, identity)
    return identity


def trial_key(record):
    fields = ('policy', 'trial_type', 'removed', 'added', 'key_block')
    return tuple((k, record[k]) for k in fields if k in record)


class TrialCheckpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.state = json.loads(self.path.read_text(encoding='utf-8')) if self.path.is_file() else {'trials': []}
        self.by_key = {trial_key(r): r for r in self.state['trials']}
        if len(self.by_key) != len(self.state['trials']):
            raise ValueError('Duplicate trial checkpoint keys')

    def check_baseline(self, result, tolerance):
        previous = self.state.get('baseline')
        if previous is not None and abs(previous['label_loss'] - result['label_loss']) > tolerance:
            raise ValueError('Resumed baseline NLL changed beyond tolerance; refusing to mix trials')
        if previous is None:
            self.state['baseline'] = result
            atomic_json(self.path, self.state)
        return self.state['baseline']

    def get(self, record):
        cached = self.by_key.get(trial_key(record))
        if cached is not None and any(cached.get(k) != v for k, v in record.items()):
            raise ValueError('Resumed trial scores/selection changed; refusing to mix trials')
        return cached

    def save(self, record):
        key = trial_key(record)
        if key in self.by_key:
            raise ValueError('Trial already checkpointed')
        self.state['trials'].append(record)
        atomic_json(self.path, self.state)
        self.by_key[key] = record
