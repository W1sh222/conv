"""Full-policy kernel-size NLL sweep: fixed V+diagonal kernels and an optional learned Llama kernel."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments3'))
import experiment_common as common
from analysis_core import fixed_vertical_diagonal_kernel, validate_checkpoint_layout
from resume_state import atomic_json, file_hash
from sweep_core import TASKS, SIZES, summarize

MODEL_LOAD_BUG_HASHES={
    '50cb9ff5230113acabeac8b1284788a671369624bb0eeb029e935ca77ed03b05',
    '4862bdc2cf33d48da917a0bd69841e987bf4b2b2f0ace7e4a7baa1d4b99e4b52',
    'b38593ab8a2eaee23ae46b6e4ee6b86c044070e8bb002d7c3d158cd48eb812e3',
    '0eb62e3cfe4ebfdff43cd9b8894b416ed31d962a4345146b3266933522a089c5',
    '8ddf799b45b8828766924a899bc2a79ef994d77ce17d19518e6fde6de3df2d52',
    'e91bc0957e8dd6e0b73ee73e81b6cd1302860178972cddcce2f140ecb1771090',
}


def reference_sampled_scores(q,k,torch,block_size=128,stride=8,norm=1.0,chunk_size=0,causal=True):
    """FP32 reference for inverse-antidiagonal sampling, not full attention.

    Bound workspace to one head and 128 sampled queries. Keep the deployed
    estimator's padding, sampled-index causality, row normalization and sums.
    """
    from conv_sparse_ops import choose_conv_prefill_chunk_size
    unit=stride*128
    chunk_size=chunk_size or choose_conv_prefill_chunk_size(max(q.shape[-2],k.shape[-2]))
    chunk_size=math.ceil(max(chunk_size,unit)/unit)*unit
    qlen,klen=q.shape[-2],k.shape[-2]
    qp=math.ceil(qlen/chunk_size)*chunk_size;kp=math.ceil(klen/chunk_size)*chunk_size
    q=torch.nn.functional.pad(q,(0,0,0,qp-qlen))
    k=torch.nn.functional.pad(k,(0,0,0,kp-klen))
    rb=block_size//stride;qs=qp//stride;ks=kp//stride
    real_sampled_q=math.ceil(qlen/stride)
    out=torch.zeros((q.shape[0],q.shape[1],qp//block_size,kp//block_size),device=q.device,dtype=q.dtype)
    key_indices=torch.arange(ks,device=q.device)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            for start in range(0,qs,128):
                end=min(start+128,qs)
                logits=torch.zeros((end-start,ks),device=q.device,dtype=torch.float32)
                for i in range(stride):
                    qq=q[b,h,start*stride+stride-1-i:end*stride:stride].float()
                    kk=k[b,h,i::stride].float()
                    logits.addmm_(qq,kk.transpose(0,1))
                    del qq,kk
                logits.mul_(1.0/(math.sqrt(q.shape[-1])*stride*norm))
                rows=torch.arange(start,end,device=q.device)
                if causal:logits.masked_fill_(key_indices[None,:]>rows[:,None]+(ks-qs),float('-inf'))
                probs=torch.softmax(logits,dim=-1)
                probs[rows>=real_sampled_q]=0
                summed=probs.reshape((end-start)//rb,rb,ks//rb,rb).sum(dim=(1,3))
                out[b,h,start//rb:end//rb]=summed.to(out.dtype)
                del logits,probs,summed
    return out,dict(q_real_blocks=math.ceil(qlen/block_size),k_real_blocks=math.ceil(klen/block_size),
                    q_full_blocks=qp//block_size,k_full_blocks=kp//block_size,num_blocks_per_chunk=chunk_size//block_size)


def cpu_ratio_mask(scores,ratio,offset=0,causal=True):
    """Select real causal blocks without CUDA topk/scatter; stable key-index ties."""
    values=np.asarray(scores,dtype=np.float32)
    if values.ndim!=4 or not 0<ratio<=1:raise ValueError('Invalid block scores/ratio')
    values=np.nan_to_num(values,nan=-1e4,posinf=1e4,neginf=-1e4)
    mask=np.zeros(values.shape,dtype=bool)
    qb,kb=values.shape[-2:]
    for row in range(qb):
        visible=min(kb,max(0,row+int(offset)+1)) if causal else kb
        if not visible:continue
        count=min(visible,max(1,int(np.ceil(np.float32(visible)*np.float32(ratio)))))
        chosen=np.argsort(-values[...,row,:visible],axis=-1,kind='stable')[...,:count]
        np.put_along_axis(mask[...,row,:visible],chosen,True,axis=-1)
    return mask


@contextmanager
def single_process_load_environment(environ=None):
    """Prevent HF 4.51 from promoting device_map=auto to TP in scheduler jobs.

    This runner is one process, with optional Accelerate device-map sharding.
    Clear launcher-only variables temporarily, preserving GPU visibility.
    """
    env=os.environ if environ is None else environ
    if int(env.get('RANK','0'))!=0 or (env.get('LOCAL_RANK') is not None and int(env.get('WORLD_SIZE','1'))>1):
        raise ValueError('This diagnostic runner is single-process: launch with python/bash, not multi-rank torchrun')
    keys=('WORLD_SIZE','RANK','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK',
          'ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT')
    saved={key:env.pop(key) for key in keys if key in env}
    if saved:print('Single-process model loading: temporarily ignoring launcher variables:',', '.join(saved),flush=True)
    try:yield
    finally:env.update(saved)


def ensure_run_manifest(out,identity,resume):
    manifest=out/'run_manifest.json'
    if out.exists() and any(out.iterdir()):
        if not resume or not manifest.is_file():
            raise ValueError('Use --resume with identical configuration/data/weights, or a new output directory')
        prior=json.loads(manifest.read_text())
        if prior==identity:return
        source_key=str(Path(__file__).relative_to(ROOT))
        old_hash=prior.get('source_sha256',{}).get(source_key)
        upgraded=json.loads(json.dumps(prior))
        if source_key in upgraded.get('source_sha256',{}):
            upgraded['source_sha256'][source_key]=identity['source_sha256'][source_key]
        if 'mask_backend' in identity.get('arguments',{}):
            upgraded['arguments'].setdefault('mask_backend','cpu')
        # The reported failure occurred before the first model was loaded. Only
        # migrate that exact engine version if no policy result has been saved.
        if (old_hash not in MODEL_LOAD_BUG_HASHES or upgraded!=identity
            or any(out.glob('*/*/*.json')) or (out/'summary.json').exists()):
            raise ValueError('Resume configuration/source differs; use a new output directory')
        backup=out/'run_manifest.before_model_load_fix.json'
        if not backup.exists():atomic_json(backup,prior)
        atomic_json(manifest,identity)
        print('Updated failed run manifest for model-loading/block-selection repair; no saved losses mixed',flush=True)
    else:
        out.mkdir(parents=True,exist_ok=True);atomic_json(manifest,identity)


def parser():
    p=common.parser(5)
    p.description=__doc__
    p.set_defaults(model='/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Llama-3.1-8B-Instruct',
                   conv_weights=None,learned_policy_name='learned_checkpoint',no_plots=True)
    p.add_argument('--kernel-sizes',default=','.join(map(str,SIZES)))
    p.add_argument('--tasks',default=','.join(TASKS))
    p.add_argument('--samples-per-task',type=int,default=2)
    p.add_argument('--fixed-only',action='store_true',help='Run six fixed kernels without a learned checkpoint')
    p.add_argument('--seq-length',type=int,default=131072)
    p.add_argument('--mask-backend',choices=['cpu','cuda'],default='cpu',
                   help='CPU is the stable diagnostic selector; CUDA uses the repository Top-k routine')
    return p


def setup(a):
    import torch, transformers
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    if not transformers.__version__.startswith('4.51.'): raise RuntimeError('Use the Transformers 4.51 evaluation environment')
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ft_scripts/sparse_ruler'))
    from xattn.src import Conv as conv,load_transformers_451 as adapter
    from conv_sparse_ops import kernels_conv_block_scores_infer_full,apply_conv_energy
    config=AutoConfig.from_pretrained(a.model)
    if config.model_type!='llama': raise ValueError('This launcher evaluates Llama; do not load a Qwen checkpoint')
    if a.rope_factor is not None:
        config.rope_scaling={'rope_type':'yarn','factor':a.rope_factor,'original_max_position_embeddings':a.rope_original_length}
        config.max_position_embeddings=math.ceil(a.rope_factor*a.rope_original_length)
    if a.max_position_embeddings:config.max_position_embeddings=a.max_position_embeddings
    if a.seq_length>config.max_position_embeddings:raise ValueError('Model context too short')
    weight=None
    if not a.fixed_only:
        weight=torch.load(a.conv_weights,map_location='cpu',weights_only=True).float()
        if weight.ndim==5 and weight.shape[2]==1:weight=weight[:,:,0]
        validate_checkpoint_layout('llama',config.num_hidden_layers,config.num_attention_heads,weight.shape)
        if not torch.isfinite(weight).all():raise ValueError('Nonfinite learned weights')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(a.model,use_fast=True)
    dm=a.device_map if a.device_map in {'auto','balanced','balanced_low_0','sequential'} else {'':a.device_map}
    with single_process_load_environment():
        model=AutoModelForCausalLM.from_pretrained(a.model,config=config,torch_dtype=getattr(torch,a.dtype),
                                                device_map=dm,attn_implementation='sdpa',tp_plan=None).eval()
    model.requires_grad_(False)
    return model,tokenizer,config,torch,adapter,conv,kernels_conv_block_scores_infer_full,weight,apply_conv_energy


class KernelController(common.old.MaskController):
    def __init__(self,a,length,torch,adapter,conv,score_fn,weight,apply,policy):
        self.current_layer=None;self.raw=score_fn;self.weight=weight;self.apply=apply;self.policy=policy
        self.reference_layers=[]
        super().__init__(a,length,torch,adapter,conv,self.estimate)

    def estimate(self,*args,**kwargs):
        try:
            scores,meta=self.raw(*args,**kwargs)
            if scores.is_cuda:self.torch.cuda.synchronize(scores.device)
        except Exception as exc:
            raise RuntimeError(f'Score-estimator failure before selection: layer={self.current_layer}, policy={self.policy}') from exc
        if not self.torch.isfinite(scores).all():
            q,k=args[:2]
            diagnostic=dict(layer=self.current_layer,policy=self.policy,stage='raw_score_estimator',
                score_shape=list(scores.shape),nonfinite_scores=int((~self.torch.isfinite(scores)).sum().item()),
                q_nonfinite=int((~self.torch.isfinite(q)).sum().item()),
                k_nonfinite=int((~self.torch.isfinite(k)).sum().item()),meta=meta)
            atomic_json(Path(self.args.output)/'nonfinite_diagnostic.json',diagnostic)
            if diagnostic['q_nonfinite'] or diagnostic['k_nonfinite']:
                raise ValueError(f'Nonfinite Q/K before scoring: {diagnostic}; inspect upstream attention/model activations')
            print(f'Nonfinite fused scores with finite Q/K: layer={self.current_layer}, policy={self.policy}; recomputing sampled scores in FP32',flush=True)
            del scores
            scores,meta=reference_sampled_scores(q,k,self.torch,**kwargs)
            if not self.torch.isfinite(scores).all():raise ValueError(f'FP32 reference estimator also nonfinite: {diagnostic}')
            self.reference_layers.append(self.current_layer)
        if self.policy.startswith('fixed_'):
            size=int(self.policy.split('_')[1].split('x')[0])
            if size==1:energy=scores
            else:
                kernel=self.torch.as_tensor(fixed_vertical_diagonal_kernel(size),device=scores.device)
                kernel=kernel[None].expand(scores.shape[1],-1,-1).contiguous()
                energy=self.apply(scores,kernel)
        else:energy=self.apply(scores,self.weight[self.current_layer].to(scores.device))
        try:
            if energy.is_cuda:self.torch.cuda.synchronize(energy.device)
        except Exception as exc:
            raise RuntimeError(f'Convolution failure before selection: layer={self.current_layer}, policy={self.policy}') from exc
        real=energy[:,:,:self.nblocks,:self.nblocks]
        valid=self.torch.ones(self.nblocks,self.nblocks,device=scores.device,dtype=self.torch.bool).tril()
        if not self.torch.isfinite(real[...,valid]).all():
            raise ValueError(f'Nonfinite refined causal scores: layer={self.current_layer}, policy={self.policy}; raw scores were finite')
        return energy,meta

    def select_mask(self,scores,topk_ratio,offset,causal=True):
        # Refinement already ran on the full padded map. Crop only now, so
        # padding/future blocks cannot enter the real block-selection support.
        real=scores[:,:,:self.nblocks,:self.nblocks]
        if self.args.mask_backend=='cpu':
            try:
                mask=cpu_ratio_mask(real.detach().float().cpu().numpy(),topk_ratio,offset,causal)
                return self.torch.from_numpy(mask).to(device=scores.device)
            except Exception as exc:
                raise RuntimeError(f'CPU block selection failed: layer={self.current_layer}, policy={self.policy}, shape={tuple(real.shape)}') from exc
        return self.original_selector(real,topk_ratio,offset,causal)

    def prefill(self,attn,*args,**kwargs):
        self.current_layer=attn.layer_idx
        out=super().prefill(attn,*args,**kwargs)
        if not self.torch.isfinite(out).all():
            diagnostic=dict(layer=self.current_layer,policy=self.policy,stage='block_sparse_attention_output',
                            nonfinite_output=int((~self.torch.isfinite(out)).sum().item()))
            atomic_json(Path(self.args.output)/'nonfinite_diagnostic.json',diagnostic)
            raise ValueError(f'Nonfinite sparse attention output: {diagnostic}; scores/mask already passed validation')
        return out


def evaluate_policy(a,env,item,policy):
    model,tokenizer,config,torch,adapter,conv,score_fn,weight,apply=env
    _,_,prompt_ids,label_ids=common.old.prepare_tokens(tokenizer,item,a)
    if len(prompt_ids)+len(label_ids)>min(a.seq_length,config.max_position_embeddings):
        raise ValueError('Prompt+answer exceeds context; never truncate')
    c=KernelController(a,len(prompt_ids),torch,adapter,conv,score_fn,weight,apply,policy)
    fast=adapter.BaseFastPrefillConfig(metric='conv',stride=a.stride,block_topk_ratio=a.ratio,print_detail=False)
    original=adapter._run_prefill;saved=[]
    original_selector=conv._topk_ratio_mask_from_scores;c.original_selector=original_selector
    try:
        adapter._run_prefill=c.prefill
        conv._topk_ratio_mask_from_scores=c.select_mask
        for layer in model.model.layers:
            attn=layer.self_attn;saved.append((attn,attn.forward,getattr(attn,'fastprefillconfig',None)))
            attn.fastprefillconfig=fast;attn.forward=adapter.forward_eval_451.__get__(attn,type(attn))
        result=common.old.label_nll(model,prompt_ids,label_ids,torch)
        c.recording=False
        repeated=common.old.label_nll(model,prompt_ids,label_ids,torch)
        return dict(policy=policy,**result,repeat_drift=abs(result['label_loss']-repeated['label_loss']),
                    prompt_tokens=len(prompt_ids),label_tokens=len(label_ids),
                    mask_backend=a.mask_backend,
                    score_estimator='fused sampling with FP32 sampled reference on nonfinite scores',
                    reference_estimator_layers=c.reference_layers,
                    tie_rule='lower key-block index first' if a.mask_backend=='cpu' else 'repository torch.topk tie behavior',
                    scope='All layers/heads; each policy selects on its own activations; prompt-only sparse prefill')
    finally:
        adapter._run_prefill=original
        conv._topk_ratio_mask_from_scores=original_selector
        for attn,forward,fast in saved:
            attn.forward=forward
            if fast is None:delattr(attn,'fastprefillconfig')
            else:attn.fastprefillconfig=fast


def main():
    a=parser().parse_args();sizes=[int(x) for x in a.kernel_sizes.split(',')];tasks=a.tasks.split(',')
    if not sizes or sizes[0]!=1 or len(set(sizes))!=len(sizes) or any(s not in SIZES for s in sizes):raise ValueError('Sizes must start with 1, be unique and drawn from 1,3,5,7,9,11')
    if len(set(tasks))!=len(tasks) or not tasks or any(t not in TASKS for t in tasks):raise ValueError('Invalid tasks')
    if a.samples_per_task<1 or not 0<a.ratio<1 or a.stride<=0 or 128%a.stride:raise ValueError('Invalid sample count/budget/stride')
    if a.background!='sparse' or a.chat_template or a.max_label_tokens or a.max_prompt_tokens:
        raise ValueError('Use full sparse policies and prepared templates without token truncation')
    if a.fixed_only and a.conv_weights:raise ValueError('Choose fixed-only OR a learned checkpoint')
    if not a.fixed_only and not a.conv_weights:raise ValueError('Supply a matching Llama --conv-weights, or use --fixed-only; Qwen step9750 cannot be applied to Llama')
    if a.learned_policy_name.startswith('fixed_'):raise ValueError('Learned policy name must differ from fixed kernels')
    a.layer=16;a.head=8;a.query_block='last';a.selector='initial'
    policies=[f'fixed_{s}x{s}' for s in sizes]+([] if a.fixed_only else [a.learned_policy_name])
    if a.dry_run:
        print(json.dumps(dict(arguments=vars(a),policies=policies,inputs=len(tasks)*a.samples_per_task,
                             policy_evaluations=len(tasks)*a.samples_per_task*len(policies),
                             forwards_per_evaluation='one fresh-policy NLL plus one frozen-mask repeat check'),indent=2));return
    rows=[json.loads(x) for x in Path(a.data).read_text(encoding='utf-8').splitlines() if x.strip()]
    inputs=[]
    for task in tasks:
        group=[r for r in rows if r['task']==task]
        if len(group)<a.samples_per_task:raise ValueError('Insufficient samples: '+task)
        inputs.extend(group[:a.samples_per_task])
    identities=[(r['task'],r['sample_index']) for r in inputs]
    if len(identities)!=len(set(identities)):raise ValueError('Duplicate task/sample identities')
    out=Path(a.output)
    identity={'arguments':{k:v for k,v in vars(a).items() if k not in ('resume','dry_run','output','no_plots')},
              'data_sha256':file_hash(a.data),'weight_sha256':file_hash(a.conv_weights) if a.conv_weights else None,
              'source_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in
                  [Path(__file__),Path(__file__).with_name('sweep_core.py'),ROOT/'experiments3/analysis_core.py',
                   ROOT/'experiments3/experiment_common.py',ROOT/'experiments/block_label_swap/run_experiment.py',
                   ROOT/'ft_scripts/sparse_ruler/conv_sparse_ops.py']},
              'model_metadata':{n:file_hash(Path(a.model)/n) for n in ('config.json','tokenizer_config.json','tokenizer.json') if (Path(a.model)/n).is_file()}}
    ensure_run_manifest(out,identity,a.resume)
    env=None;results=[]
    for item in inputs:
        case=out/item['task']/f"s{item['sample_index']}";case.mkdir(parents=True,exist_ok=True)
        for policy in policies:
            path=case/(policy+'.json')
            if path.is_file():
                result=json.loads(path.read_text())
                if (result.get('status')!='complete' or result.get('task')!=item['task'] or
                    result.get('sample_index')!=item['sample_index'] or result.get('policy')!=policy or
                    not math.isfinite(result['label_loss']) or
                    abs(float(np.mean(result['token_losses']))-result['label_loss'])>1e-12):
                    raise ValueError('Invalid saved input-policy result: '+str(path))
                print('Reusing',item['task'],item['sample_index'],policy,flush=True)
            else:
                if env is None:env=setup(a)
                print('Evaluating',item['task'],item['sample_index'],policy,flush=True)
                result=evaluate_policy(a,env,item,policy)
                result.update(task=item['task'],sample_index=item['sample_index'],status='complete')
                atomic_json(path,result)
            results.append(result);common.csv_file(out/'individual_losses.csv',results)
    summary=summarize(results,tasks,policies);summary.update(data_sha256=identity['data_sha256'],weight_sha256=identity['weight_sha256'])
    atomic_json(out/'summary.json',summary)
    common.csv_file(out/'task_mean_losses.csv',[dict(task=t,**summary['task_mean_nll'][t]) for t in tasks]+
                    [dict(task='Macro average',**summary['macro_mean_nll'])])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
