"""Full-policy kernel-size NLL sweep: fixed V+diagonal kernels and an optional learned Llama kernel."""
import argparse
import json
import math
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
    model=AutoModelForCausalLM.from_pretrained(a.model,config=config,torch_dtype=getattr(torch,a.dtype),
                                            device_map=dm,attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    return model,tokenizer,config,torch,adapter,conv,kernels_conv_block_scores_infer_full,weight,apply_conv_energy


class KernelController(common.old.MaskController):
    def __init__(self,a,length,torch,adapter,conv,score_fn,weight,apply,policy):
        self.current_layer=None;self.raw=score_fn;self.weight=weight;self.apply=apply;self.policy=policy
        super().__init__(a,length,torch,adapter,conv,self.estimate)

    def estimate(self,*args,**kwargs):
        scores,meta=self.raw(*args,**kwargs)
        if self.policy.startswith('fixed_'):
            size=int(self.policy.split('_')[1].split('x')[0])
            if size==1:energy=scores
            else:
                kernel=self.torch.as_tensor(fixed_vertical_diagonal_kernel(size),device=scores.device)
                kernel=kernel[None].expand(scores.shape[1],-1,-1).contiguous()
                energy=self.apply(scores,kernel)
        else:energy=self.apply(scores,self.weight[self.current_layer].to(scores.device))
        real=energy[:,:,:self.nblocks,:self.nblocks]
        valid=self.torch.ones(self.nblocks,self.nblocks,device=scores.device,dtype=self.torch.bool).tril()
        if not self.torch.isfinite(real[...,valid]).all():raise ValueError('Nonfinite causal scores')
        return energy,meta

    def prefill(self,attn,*args,**kwargs):
        self.current_layer=attn.layer_idx
        return super().prefill(attn,*args,**kwargs)


def evaluate_policy(a,env,item,policy):
    model,tokenizer,config,torch,adapter,conv,score_fn,weight,apply=env
    _,_,prompt_ids,label_ids=common.old.prepare_tokens(tokenizer,item,a)
    if len(prompt_ids)+len(label_ids)>min(a.seq_length,config.max_position_embeddings):
        raise ValueError('Prompt+answer exceeds context; never truncate')
    c=KernelController(a,len(prompt_ids),torch,adapter,conv,score_fn,weight,apply,policy)
    fast=adapter.BaseFastPrefillConfig(metric='conv',stride=a.stride,block_topk_ratio=a.ratio,print_detail=False)
    original=adapter._run_prefill;saved=[]
    try:
        adapter._run_prefill=c.prefill
        for layer in model.model.layers:
            attn=layer.self_attn;saved.append((attn,attn.forward,getattr(attn,'fastprefillconfig',None)))
            attn.fastprefillconfig=fast;attn.forward=adapter.forward_eval_451.__get__(attn,type(attn))
        result=common.old.label_nll(model,prompt_ids,label_ids,torch)
        c.recording=False
        repeated=common.old.label_nll(model,prompt_ids,label_ids,torch)
        return dict(policy=policy,**result,repeat_drift=abs(result['label_loss']-repeated['label_loss']),
                    prompt_tokens=len(prompt_ids),label_tokens=len(label_ids),
                    scope='All layers/heads; each policy selects on its own activations; prompt-only sparse prefill')
    finally:
        adapter._run_prefill=original
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
    manifest=out/'run_manifest.json'
    if out.exists() and any(out.iterdir()):
        if not a.resume or not manifest.is_file() or json.loads(manifest.read_text())!=identity:
            raise ValueError('Use --resume with identical configuration/data/weights, or a new output directory')
    else:out.mkdir(parents=True,exist_ok=True);atomic_json(manifest,identity)
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
