"""Shared prompt-only block intervention engine; reuses the original label-NLL runner."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
from analysis_core import compare_sets, row_swap, ranking_summary, input_cluster_summary, fixed_vertical_diagonal_kernel, validate_checkpoint_layout
from resume_state import atomic_json, prepare_run, TrialCheckpoint

ROOT=Path(__file__).resolve().parents[1]
SWAP=ROOT/'experiments/block_label_swap'
sys.path.insert(0,str(SWAP))
spec=importlib.util.spec_from_file_location('existing_swap_runner',SWAP/'run_experiment.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
WEIGHT=ROOT/'xattn/qwen_weights/conv_qwen3_t065_longbench_stage4_v2/conv_kernel_7x7_qwen3_t065_longbench_replay_native_8k64k_s8_bf16_ema_step5000.pt'
POLICIES=['initial','positive_1x1','fixed_vertical_diagonal','learned_5000']


def policies(args):
    return POLICIES[:3]+[args.learned_policy_name]


def parser(kind):
    p=argparse.ArgumentParser(description=f'Experiment {kind}: equal-budget block utility analysis')
    p.add_argument('--model',default='/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Qwen3-8B')
    p.add_argument('--data',required=True,help='JSONL with prompt and label; can reuse observation.jsonl')
    p.add_argument('--conv-weights',default=str(WEIGHT))
    p.add_argument('--learned-policy-name',default='learned_5000',help='Output key identifying the learned checkpoint; Llama launcher uses learned_llama_replay')
    p.add_argument('--sample-indices',default='0',help='Comma-separated held-out input indices')
    p.add_argument('--layers',default='16',help='Comma-separated zero-based layers')
    p.add_argument('--heads',default='8',help='Comma-separated query heads')
    p.add_argument('--query-blocks',default='last')
    p.add_argument('--ratio',type=float,default=.65)
    p.add_argument('--stride',type=int,default=8)
    p.add_argument('--background',choices=['sparse','dense'],default='sparse')
    p.add_argument('--device-map',default='auto')
    p.add_argument('--dtype',choices=['bfloat16','float16'],default='bfloat16')
    p.add_argument('--chat-template',action='store_true')
    p.add_argument('--rope-factor',type=float)
    p.add_argument('--rope-original-length',type=int,default=32768)
    p.add_argument('--max-position-embeddings',type=int)
    p.add_argument('--max-prompt-tokens',type=int,default=0)
    p.add_argument('--max-label-tokens',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--loss-tolerance',type=float,default=1e-5)
    p.add_argument('--max-candidates',type=int,default=0,help='0=all; otherwise a seeded random candidate subset, never chosen by loss')
    p.add_argument('--top-sizes',default='5,10,20')
    p.add_argument('--all-pairs',action='store_true',help='Experiment4: exhaustive removed x added pairs; can be costly')
    p.add_argument('--output',required=True)
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--resume',action='store_true',help='Reuse completed cases/trials with identical configuration, data and weights')
    p.add_argument('--dry-run',action='store_true')
    return p


def write_json(path,value):
    atomic_json(path,value)


def csv_file(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


class Controller(old.MaskController):
    def __init__(self,args,prompt_length,torch,adapter,conv,score_fn,weights,apply):
        self.current_layer=None;self.maps={};self.rows={};self.record_policy='initial'
        self.row_override=None;self.weights=weights;self.apply=apply
        self.raw_score_fn=score_fn
        super().__init__(args,prompt_length,torch,adapter,conv,self.estimate)

    def estimate(self,*a,**kw):
        initial,meta=self.raw_score_fn(*a,**kw)
        layer=self.current_layer
        need_all=layer==self.args.layer
        policy_names=policies(self.args) if need_all else [self.record_policy]
        computed={}
        for policy in policy_names:
            if policy in ('initial','positive_1x1'):
                energy=initial  # positive 1x1 gain=1: exact Top-k invariance control
            elif policy==self.args.learned_policy_name:
                energy=self.apply(initial,self.weights[layer].to(initial.device))
            else:
                h=initial.shape[1]
                kernel=self.torch.as_tensor(fixed_vertical_diagonal_kernel(),device=initial.device)
                kernel=kernel[None].expand(h,-1,-1).contiguous()
                energy=self.apply(initial,kernel)  # overlap counted once; no trained weights
            computed[policy]=energy
            real=energy[:,:,:self.nblocks,:self.nblocks]
            valid=self.torch.ones(self.nblocks,self.nblocks,device=energy.device,dtype=self.torch.bool).tril()
            if not self.torch.isfinite(real[...,valid]).all():
                raise RuntimeError(f'Non-finite causal scores: layer={layer}, policy={policy}')
            if need_all:
                self.maps[policy]=energy[0,self.args.head,:self.nblocks,:self.nblocks].float().cpu().numpy().copy()
                mask=self.conv._topk_ratio_mask_from_scores(energy,self.args.ratio,offset=0,causal=True)
                mask=self.conv._sanitize_block_sparse_mask(mask,self.nblocks,self.nblocks,
                                                          causal=True,keep_sink=False,keep_recent=False)
                qb=self.nblocks-1 if self.args.query_block=='last' else int(self.args.query_block)
                self.rows[policy]=mask[0,self.args.head,qb].cpu().numpy().copy()
        return computed[self.record_policy],meta

    def prefill(self,attn,q,k,v,attention_mask):
        self.current_layer=attn.layer_idx
        original=self.frozen_masks.get(attn.layer_idx)
        if not self.recording and self.row_override is not None and attn.layer_idx==self.args.layer:
            temporary=original.clone()
            qb=self.plan['query_block']
            override=np.asarray(self.row_override,dtype=bool)
            if override.shape!=(self.nblocks,) or override[qb+1:].any():
                raise ValueError('Invalid intervention row')
            if override.sum()!=original[0,self.args.head,qb].sum().item():
                raise ValueError('Intervention changed block budget')
            temporary[0,self.args.head,qb]=self.torch.as_tensor(override)
            self.frozen_masks[attn.layer_idx]=temporary
        try:
            return super().prefill(attn,q,k,v,attention_mask)
        finally:
            if not self.recording and original is not None:
                self.frozen_masks[attn.layer_idx]=original


def setup(a):
    import torch
    import transformers
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required; no experiment results have been computed')
    if not transformers.__version__.startswith('4.51.'):
        raise RuntimeError('This runner uses the Transformers 4.51 adapter; activate the matching evaluation environment')
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ft_scripts/sparse_ruler'))
    from xattn.src import Conv as conv,load_transformers_451 as adapter
    from conv_sparse_ops import kernels_conv_block_scores_infer_full,apply_conv_energy
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=AutoConfig.from_pretrained(a.model)
    if config.model_type not in ('qwen3','llama'): raise ValueError('Only Qwen3 and Llama backbones are supported')
    if config.model_type=='llama' and a.learned_policy_name=='learned_5000':
        a.learned_policy_name='learned_checkpoint'
    if getattr(config,'use_sliding_window',False): raise ValueError('Sliding-window attention is unsupported')
    if a.rope_factor is not None:
        config.rope_scaling={'rope_type':'yarn','factor':a.rope_factor,'original_max_position_embeddings':a.rope_original_length}
        config.max_position_embeddings=math.ceil(a.rope_factor*a.rope_original_length)
    if a.max_position_embeddings: config.max_position_embeddings=a.max_position_embeddings
    weight=torch.load(a.conv_weights,map_location='cpu',weights_only=True).float()
    if weight.ndim==5 and weight.shape[2]==1: weight=weight[:,:,0]
    validate_checkpoint_layout(config.model_type,config.num_hidden_layers,config.num_attention_heads,weight.shape)
    if not torch.isfinite(weight).all(): raise ValueError('Checkpoint contains non-finite convolution weights')
    tokenizer=AutoTokenizer.from_pretrained(a.model,use_fast=True)
    dm=a.device_map if a.device_map in {'auto','balanced','balanced_low_0','sequential'} else {'':a.device_map}
    model=AutoModelForCausalLM.from_pretrained(a.model,config=config,torch_dtype=getattr(torch,a.dtype),
                                            device_map=dm,attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    return model,tokenizer,config,torch,adapter,conv,kernels_conv_block_scores_infer_full,weight,apply_conv_energy


def evaluate(model,prompt_ids,label_ids,torch,controller,row=None):
    controller.row_override=row
    try: return old.label_nll(model,prompt_ids,label_ids,torch)
    finally: controller.row_override=None


def run_case(kind,a,env,case_dir):
    model,tokenizer,config,torch,adapter,conv,score_fn,weight,apply=env
    example=old.read_example(a.data,a.sample_index)
    prompt,label,prompt_ids,label_ids=old.prepare_tokens(tokenizer,example,a)
    if len(prompt_ids)+len(label_ids)>config.max_position_embeddings:
        raise ValueError('Prompt+label exceeds context; never silently truncate. Supply matching RoPE settings.')
    if not 0<=a.layer<config.num_hidden_layers or not 0<=a.head<config.num_attention_heads:
        raise ValueError('Layer/head outside model layout')
    c=Controller(a,len(prompt_ids),torch,adapter,conv,score_fn,weight,apply)
    learned_policy=a.learned_policy_name
    fast=adapter.BaseFastPrefillConfig(metric='conv',stride=a.stride,block_topk_ratio=a.ratio,print_detail=False)
    original=adapter._run_prefill; saved=[]
    adapter._run_prefill=c.prefill
    for layer in model.model.layers:
        attn=layer.self_attn;saved.append((attn,attn.forward,getattr(attn,'fastprefillconfig',None)))
        attn.fastprefillconfig=fast;attn.forward=adapter.forward_eval_451.__get__(attn,type(attn))
    meta={'arguments':dict(vars(a)),'input_id':hashlib.sha256(prompt.encode()).hexdigest(),
          'label_sha256':hashlib.sha256(label.encode()).hexdigest(),
          'label_tokens':len(label_ids),'prompt_tokens':len(prompt_ids),
          'loss_definition':'Mean teacher-forced label-token NLL; prompt-only sparse prefill; fresh KV each trial; no EOS',
          'score_definition':'Repository antidiagonal estimator; initial baseline is not the complete XAttention policy',
          'status':'running','scope':'single layer/head/query row with all other masks frozen' if kind!=5 else
          ('all sparse layers: each policy estimates masks on its own actual activations' if a.background=='sparse' else 'target head, other attention dense'),
          'weight_sha256':hashlib.sha256(Path(a.conv_weights).read_bytes()).hexdigest(),
          'versions':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,
                      'numpy':np.__version__},
          'gpu':torch.cuda.get_device_name(),
          'model_type':config.model_type,'model_path':a.model,'weight_path':str(Path(a.conv_weights).resolve()),
          'learned_policy':learned_policy,'topk_ratio':a.ratio}
    case_dir.mkdir(parents=True,exist_ok=True);write_json(case_dir/'experiment.json',meta)
    checkpoint=TrialCheckpoint(case_dir/'checkpoint.json')
    trials=[]
    try:
        baseline=evaluate(model,prompt_ids,label_ids,torch,c)
        baseline=checkpoint.check_baseline(baseline,a.loss_tolerance)
        c.recording=False
        if c.plan is None: raise RuntimeError('Target row was not visited')
        qb=c.plan['query_block'];base=c.rows['initial'].copy();learned=c.rows[learned_policy].copy()
        maps={k:v.copy() for k,v in c.maps.items()}
        if (case_dir/'block_maps.npz').is_file():
            with np.load(case_dir/'block_maps.npz') as prior:
                if (int(prior['query_block'])!=qb or not np.array_equal(prior['initial_mask'],base)
                    or not np.array_equal(prior['learned_mask'],learned)
                    or any(not np.array_equal(prior[k],v) for k,v in maps.items())):
                    raise ValueError('Resumed score maps/masks differ; refusing to mix trials')
        with (case_dir/'block_maps.npz.tmp').open('wb') as f:
            np.savez_compressed(f,query_block=qb,initial_mask=base,learned_mask=learned,**maps)
        (case_dir/'block_maps.npz.tmp').replace(case_dir/'block_maps.npz')
        def trial(row,record):
            cached=checkpoint.get(record)
            if cached is not None:
                trials.append(cached)
                print(case_dir.name,'reusing trial',dict(record),flush=True)
                return cached
            result=evaluate(model,prompt_ids,label_ids,torch,c,row)
            record.update(label_loss=result['label_loss'],token_losses=result['token_losses'],
                          baseline_loss=baseline['label_loss'],utility=baseline['label_loss']-result['label_loss'])
            checkpoint.save(record);trials.append(record)
            print(case_dir.name,record.get('policy',record.get('key_block',record.get('added'))),result['label_loss'],flush=True)
            return result
        if kind==3:
            candidates=list(c.plan['candidate_key_blocks'])
            initial_order=sorted(candidates,key=lambda k:(-maps['initial'][qb,k],k))
            learned_order=sorted(candidates,key=lambda k:(-maps[learned_policy][qb,k],k))
            initial_rank={k:i+1 for i,k in enumerate(initial_order)}
            learned_rank={k:i+1 for i,k in enumerate(learned_order)}
            if a.max_candidates and len(candidates)>a.max_candidates:
                candidates=sorted(random.Random(a.seed+a.sample_index+a.layer*100+a.head).sample(candidates,a.max_candidates))
            for k in candidates:
                trial(row_swap(base,qb,c.plan['removed_key_block'],k),dict(key_block=k,
                      initial_candidate_rank=initial_rank[k],learned_candidate_rank=learned_rank[k],
                      removed=c.plan['removed_key_block'],initial_score=float(maps['initial'][qb,k]),
                      learned_score=float(maps[learned_policy][qb,k])))
            repeated=evaluate(model,prompt_ids,label_ids,torch,c)
            drift=abs(repeated['label_loss']-baseline['label_loss']);tol=max(a.loss_tolerance,5*drift)
            metrics=ranking_summary(trials,[int(x) for x in a.top_sizes.split(',')],tol)
            metrics.update(effective_tolerance=tol,baseline_repeat_drift=drift,
                           full_candidate_count=len(c.plan['candidate_key_blocks']),exhaustive=not a.max_candidates or len(candidates)==len(c.plan['candidate_key_blocks']))
        elif kind==4:
            sets=compare_sets(base,learned,qb,maps['initial'][qb],maps[learned_policy][qb],a.all_pairs)
            for r,k in sets['pairs']:
                trial(row_swap(base,qb,r,k),dict(removed=r,added=k,initial_removed_score=float(maps['initial'][qb,r]),
                                              initial_added_score=float(maps['initial'][qb,k]),
                                              learned_removed_score=float(maps[learned_policy][qb,r]),
                                              learned_added_score=float(maps[learned_policy][qb,k]),trial_type='single_swap'))
            whole=trial(learned,dict(policy='learned_row',trial_type='whole_row'))
            repeated=evaluate(model,prompt_ids,label_ids,torch,c)
            drift=abs(repeated['label_loss']-baseline['label_loss']);tol=max(a.loss_tolerance,5*drift)
            singles=[r for r in trials if r['trial_type']=='single_swap']
            metrics=dict(changed_blocks=len(sets['added']),selection=sets,pairing='cartesian' if a.all_pairs else 'score-ranked one-to-one',
                         whole_row_utility=baseline['label_loss']-whole['label_loss'],
                         mean_swap_utility=float(np.mean([r['utility'] for r in singles])) if singles else None,
                         swap_improvement_fraction=float(np.mean([r['utility']>tol for r in singles])) if singles else None,
                         effective_tolerance=tol,baseline_repeat_drift=drift)
        else:
            losses={'initial':baseline['label_loss']};repeat_drift={}
            for policy in policies(a):
                cached=checkpoint.get(dict(policy=policy))
                if cached is not None:
                    losses[policy]=cached['label_loss'];repeat_drift[policy]=cached['repeat_drift']
                    trials.append(cached)
                    print(case_dir.name,'reusing policy',policy,flush=True)
                    continue
                if policy!='initial':
                    c.record_policy=policy;c.recording=True;c.frozen_masks={}
                    result=evaluate(model,prompt_ids,label_ids,torch,c)
                    c.recording=False;losses[policy]=result['label_loss']
                repeated=evaluate(model,prompt_ids,label_ids,torch,c)
                repeat_drift[policy]=abs(repeated['label_loss']-losses[policy])
                trials.append(dict(policy=policy,label_loss=losses[policy],baseline_loss=baseline['label_loss'],
                                   utility=baseline['label_loss']-losses[policy],repeat_drift=repeat_drift[policy]))
                checkpoint.save(trials[-1])
            metrics=dict(losses=losses,utilities={k:baseline['label_loss']-v for k,v in losses.items()},
                         baseline_repeat_drift=repeat_drift,effective_tolerance=max(a.loss_tolerance,5*max(repeat_drift.values())),
                         positive_1x1_control='unit positive gain; untrained exact Top-k invariance control',
                         fixed_kernel='vertical union main diagonal, overlap counted once; untrained')
        csv_file(case_dir/'trials.csv',trials)
        (case_dir/'trials.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in trials),encoding='utf-8')
        meta.update(status='complete',baseline=baseline,metrics=metrics)
        write_json(case_dir/'experiment.json',meta)
        return meta
    except BaseException as e:
        meta.update(status='failed',error=f'{type(e).__name__}: {e}');write_json(case_dir/'experiment.json',meta);raise
    finally:
        adapter._run_prefill=original
        for attn,forward,fast in saved:
            attn.forward=forward
            if fast is None: delattr(attn,'fastprefillconfig')
            else: attn.fastprefillconfig=fast


def main(kind):
    a=parser(kind).parse_args()
    if not a.learned_policy_name.isidentifier() or a.learned_policy_name in POLICIES[:3]:
        raise ValueError('learned-policy-name must be an identifier distinct from the control policies')
    if not 0<a.ratio<1 or a.stride<=0 or 128%a.stride or a.max_candidates<0 or a.loss_tolerance<0:
        raise ValueError('Invalid ratio/stride/candidate cap/tolerance')
    samples=[int(x) for x in a.sample_indices.split(',')]
    layers=[int(x) for x in a.layers.split(',')];heads=[int(x) for x in a.heads.split(',')]
    queries=a.query_blocks.split(',')
    if kind==5 and (len(layers)!=1 or len(heads)!=1 or len(queries)!=1):
        raise ValueError('Experiment5 runs full policies once per input; specify one diagnostic layer/head/query location')
    if min(samples+layers+heads)<0 or any(x!='last' and int(x)<0 for x in queries): raise ValueError('Negative case index')
    if any(int(x)<=0 for x in a.top_sizes.split(',')): raise ValueError('top-sizes must be positive')
    a.selector='initial';a.query_block='last';a.layer=layers[0];a.head=heads[0]
    if a.dry_run:
        print(json.dumps(dict(experiment=kind,arguments=vars(a),cases=len(samples)*(1 if kind==5 else len(layers)*len(heads)*len(queries)),
              dependencies=['torch+CUDA','transformers==4.51.x','triton','block_sparse_attn'],
              note='No results computed. Experiment5 runs each full policy once per input with sparse background.'),indent=2));return
    out=Path(a.output)
    if not Path(a.data).is_file() or not Path(a.conv_weights).is_file(): raise FileNotFoundError('Missing input data or convolution checkpoint')
    prepare_run(out,kind,a)
    if not a.no_plots:
        from plot_analysis import plot
    env=None
    cases=[]; records=[]
    for sample in samples:
        # Full-policy ablation is one independent input, not duplicated over layer/head knobs.
        locations=[(layers[0],heads[0],queries[0])] if kind==5 else [(l,h,q) for l in layers for h in heads for q in queries]
        for layer,head,query in locations:
            a.sample_index=sample;a.layer=layer;a.head=head;a.query_block=query
            name=f's{sample}_l{layer}_h{head}_q{query}'
            meta_path=out/name/'experiment.json'
            existing=json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.is_file() else None
            if existing is not None and existing.get('status')=='complete':
                case=existing
                print('Reusing completed case',name,flush=True)
            else:
                if env is None: env=setup(a)
                case=run_case(kind,a,env,out/name)
            cases.append(case)
            record=dict(input_id=case['input_id'],case=name,sample_index=sample,layer=layer,head=head,query=query)
            m=case['metrics']
            if kind==3:
                record.update(initial_spearman=m['initial_spearman'],learned_spearman=m['learned_spearman'],
                              spearman_gain=m['learned_spearman']-m['initial_spearman'] if m['learned_spearman'] is not None and m['initial_spearman'] is not None else None)
            elif kind==4: record.update({k:m[k] for k in ['whole_row_utility','mean_swap_utility','swap_improvement_fraction']})
            else: record.update({k+'_utility':v for k,v in m['utilities'].items()})
            records.append(record);csv_file(out/'cases.csv',records)
    keys=[k for k in records[0] if k not in {'input_id','case','sample_index','layer','head','query'}]
    summary={'experiment':kind,'case_count':len(cases),'aggregates':{k:input_cluster_summary(records,k,a.seed) for k in keys},
             'inference_scope':'conditional block utility for experiments3/4; full policy NLL for experiment5',
             'independence':'Inputs are bootstrap units; candidate blocks/layers/heads are not independent replicates',
             'data_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
             'weight_sha256':cases[0]['weight_sha256'],'model_type':cases[0]['model_type'],
             'model_path':a.model,'weight_path':cases[0]['weight_path'],
             'learned_policy':a.learned_policy_name,'topk_ratio':a.ratio,'status':'complete'}
    write_json(out/'summary.json',summary)
    if not a.no_plots:
        from plot_analysis import plot
        plot(out,kind)
    print(json.dumps(summary,indent=2))
