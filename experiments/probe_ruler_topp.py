"""Instrument one RULER prompt without changing its attention selection.

Run on the GPU evaluation host. Arguments after -- are passed to call_api.py.
Use a fresh --save_dir to avoid reusing previous predictions. Only the first
prompt is processed by default; normal answer generation is retained.
"""
from pathlib import Path
import argparse
import ast
import json
import inspect
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PROBE_VERSION = '2026-10-03.4'


def main():
    print(f'[Top-p Probe] version={PROBE_VERSION} file={Path(__file__).resolve()} python={sys.executable}', flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', help='Check script and source files without loading PyTorch or the model.')
    parser.add_argument('--audit-output', type=Path, default=ROOT/'output/ruler_topp_audit/actual_layers.jsonl')
    parser.add_argument('--dump-layer', type=int, default=16)
    parser.add_argument('--max-samples', type=int, default=1)
    parser.add_argument('--positive-temperature', type=float, default=.015)
    args, forwarded = parser.parse_known_args()
    if args.self_check:
        required = {
            ROOT/'xattn/src/Conv.py': {'apply_conv2d_block_map', 'conv_estimate', '_sanitize_block_sparse_mask'},
            ROOT/'xattn/src/conv_topp.py': {'select_conv_topp_blocks', 'select_conv_topp_mask'},
            ROOT/'eval/RULER/scripts/pred/call_api.py': {'main', 'get_llm'},
        }
        for path, functions in required.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            found = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            missing = functions-found
            if missing:
                raise RuntimeError(f'{path}: missing expected functions {sorted(missing)}')
            print(f'[Top-p Probe] source OK: {path}', flush=True)
        print('[Top-p Probe] self-check PASS; no model loaded', flush=True)
        return
    if forwarded and forwarded[0] == '--':
        forwarded = forwarded[1:]
    if not forwarded:
        parser.error('Pass call_api.py arguments after --')
    if args.max_samples < 1 or args.positive_temperature <= 0:
        parser.error('max-samples and positive-temperature must be positive')
    print('[Top-p Probe] loading PyTorch and Conv module', flush=True)
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(ROOT))
    from xattn.src import Conv

    captured = {}
    seen = set()
    original_refine = Conv.apply_conv2d_block_map
    original_estimate = Conv.conv_estimate
    estimate_signature = inspect.signature(original_estimate)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)

    with args.audit_output.open('w', encoding='utf-8', buffering=1) as handle:
        print(f'[Top-p Probe] audit output: {args.audit_output.resolve()}', flush=True)
        def refine(*inputs, **kwargs):
            scores = original_refine(*inputs, **kwargs)
            layer = kwargs.get('layer_idx')
            if layer not in seen:
                captured[layer] = scores.detach()
            return scores

        def estimate(*inputs, **kwargs):
            result = original_estimate(*inputs, **kwargs)
            layer = kwargs.get('layer_idx')
            if layer not in captured or layer in seen:
                return result
            seen.add(layer)
            call = estimate_signature.bind(*inputs, **kwargs)
            call.apply_defaults()
            params = call.arguments
            query_tokens = int(params['query_states'].shape[-2])
            key_tokens = int(params['key_states'].shape[-2])
            block_size = int(params['block_size'])
            real_q = (query_tokens+block_size-1)//block_size
            real_k = (key_tokens+block_size-1)//block_size
            full_scores = captured.pop(layer).float()
            estimator_shape = list(full_scores.shape)
            if real_q > estimator_shape[-2] or real_k > estimator_shape[-1]:
                raise ValueError('Actual block dimensions exceed the estimated map')
            raw_scores = full_scores[..., :real_q, :real_k]
            nonfinite_count = int((~torch.isfinite(raw_scores)).sum())
            scores = torch.nan_to_num(raw_scores, nan=0, posinf=1e4, neginf=-1e4)
            # Match the mask cleanup immediately before block_sparse_attn_func.
            # This is a diagnostic view; the returned production result is unchanged.
            mask = Conv._sanitize_block_sparse_mask(
                result[1], real_q, real_k, causal=params['causal'],
                keep_sink=params['keep_sink'], keep_recent=params['keep_recent'],
            )
            b, h, q, k = scores.shape
            offset = k-q
            valid = torch.arange(k, device=scores.device)[None, :] <= (offset+torch.arange(q, device=scores.device))[:, None]
            if not params['causal']:
                valid = torch.ones_like(valid)
            valid4 = valid[None, None]
            positive = scores.clamp_min(0).masked_fill(~valid4, 0)
            softplus = (F.softplus(scores/args.positive_temperature)*args.positive_temperature).masked_fill(~valid4, 0)

            def coverage(mass):
                total = mass.sum(-1)
                selected = (mass*mask).sum(-1)
                # Zero positive mass has undefined coverage, not 100% coverage.
                return selected/total.clamp_min(1e-30), total > 0

            (poscov, posdefined), (spcov, spdefined) = coverage(positive), coverage(softplus)
            counts = mask.sum(-1).float()
            threshold = params['threshold']
            p = float(threshold) if not torch.is_tensor(threshold) else float(threshold.float().mean())
            denom = valid.sum()*b*h

            def coverage_summary(values, defined, prefix):
                items = values[defined]
                return {
                    prefix+'_coverage_min': float(items.min()) if items.numel() else None,
                    prefix+'_coverage_mean': float(items.mean()) if items.numel() else None,
                    prefix+'_rows_below_p_fraction': float((items+1e-6<p).float().mean()) if items.numel() else None,
                    prefix+'_zero_mass_rows_fraction': float((~defined).float().mean()),
                }

            def last_mean(values, defined):
                items = values[..., -1][defined[..., -1]]
                return float(items.mean()) if items.numel() else None

            clean_full = torch.nan_to_num(full_scores, nan=0, posinf=1e4, neginf=-1e4)
            real_query_abs_mass = clean_full[..., :real_q, :].abs().sum()
            record = {
                'probe_version': PROBE_VERSION,
                'layer': layer, 'shape': list(scores.shape), 'estimator_shape': estimator_shape,
                'query_tokens': query_tokens, 'key_tokens': key_tokens, 'block_size': block_size,
                'real_query_blocks': real_q, 'real_key_blocks': real_k,
                'padding_query_blocks': estimator_shape[-2]-real_q,
                'padding_key_blocks': estimator_shape[-1]-real_k,
                'chunk_size': int(params['chunk_size']),
                'diagnostic_mask_stage': 'post-sanitize', 'nonfinite_real_score_count': nonfinite_count,
                'threshold': p,
                'topk_ratio': params['topk_ratio'], 'fixed_topk': params['fixed_topk'],
                'conv_safe_topk': params['conv_safe_topk'],
                'conv_topp_selector': os.environ.get('CONV_TOPP_SELECTOR', 'positive') if
                    params['topk_ratio'] is None and params['fixed_topk'] is None and not params['conv_safe_topk'] else 'not_used',
                'causal_density': float(mask.sum()/denom),
                'negative_valid_fraction': float(((scores<0)&valid4).sum()/denom),
                'future_abs_score_fraction': float(scores.abs().masked_fill(valid4, 0).sum()/scores.abs().sum().clamp_min(1e-30)),
                'padding_key_abs_score_fraction': float(clean_full[..., :real_q, real_k:].abs().sum()/real_query_abs_mass.clamp_min(1e-30)),
                **coverage_summary(poscov, posdefined, 'positive'),
                **coverage_summary(spcov, spdefined, 'softplus'),
                'last_row_query_block_index': real_q-1,
                'last_row_mean_kept': float(counts[..., -1].mean()),
                'last_row_min_kept': int(counts[..., -1].min()),
                'last_row_visible_blocks': int(valid[-1].sum()),
                'last_row_positive_mass_mean': float(positive[..., -1, :].sum(-1).mean()),
                'last_row_positive_coverage_mean': last_mean(poscov, posdefined),
                'last_row_positive_zero_mass_heads': int((~posdefined[..., -1]).sum()),
                'last_row_softplus_coverage_mean': last_mean(spcov, spdefined),
            }
            handle.write(json.dumps(record, allow_nan=False)+'\n')
            print('[Top-p Audit]', json.dumps(record), flush=True)
            if layer == args.dump_layer:
                output = args.audit_output.parent/f'layer{layer}_refined_scores.pt'
                # Preserve the full map for reproducing chunked selection.
                torch.save(full_scores.cpu(), output)
                metadata = output.with_suffix('.metadata.json')
                metadata.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
                torch.save(mask.cpu(), output.with_name(f'layer{layer}_actual_mask.pt'))
                print('[Top-p Audit] saved score map', output, flush=True)
            return result

        Conv.apply_conv2d_block_map = refine
        Conv.conv_estimate = estimate
        entry = ROOT/'eval/RULER/scripts/pred/call_api.py'
        sys.path.insert(0, str(entry.parent))
        sys.argv = [str(entry), *forwarded]
        namespace = runpy.run_path(str(entry), run_name='ruler_topp_probe')
        globals_ = namespace['main'].__globals__
        runtime_args = globals_['args']
        if runtime_args.metric != 'conv' or runtime_args.threshold is None:
            raise ValueError('This probe requires --metric conv and an explicit --threshold.')
        pred_file = runtime_args.save_dir/f'{runtime_args.task}.jsonl'
        if pred_file.exists():
            raise FileExistsError(f'Use a fresh --save_dir; prediction file already exists: {pred_file}')
        original_read = globals_['read_manifest']
        task_file = runtime_args.data_dir/runtime_args.task/f'{runtime_args.subset}.jsonl'

        def read_manifest(path, *inputs, **kwargs):
            records = original_read(path, *inputs, **kwargs)
            return list(records)[:args.max_samples] if Path(path).resolve() == task_file.resolve() else records

        globals_['read_manifest'] = read_manifest
        try:
            print(f'[Top-p Probe] evaluating up to {args.max_samples} sample(s) from {task_file}', flush=True)
            namespace['main']()
        finally:
            Conv.apply_conv2d_block_map = original_refine
            Conv.conv_estimate = original_estimate
    print('Saved unchanged-selector diagnostics:', args.audit_output)


if __name__ == '__main__':
    main()
