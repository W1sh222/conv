"""Prepare small, model-templated RULER task sets, without loading LLM weights."""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments3'))
from resume_state import atomic_json, file_hash
from sweep_core import TASKS, convert


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--tasks',default=','.join(TASKS))
    p.add_argument('--num-samples',type=int,default=2)
    p.add_argument('--seq-length',type=int,default=131072)
    p.add_argument('--seed',type=int,default=20261008)
    p.add_argument('--raw-data-dir',type=Path,help='Explicit existing RULER directory with TASK/validation.jsonl; must use Llama template')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();tasks=a.tasks.split(',')
    if not tasks or len(set(tasks))!=len(tasks) or any(t not in TASKS for t in tasks): raise ValueError('Invalid task list')
    if a.num_samples<1 or a.seq_length<4096: raise ValueError('Invalid sample count/context')
    if a.dry_run:
        print(json.dumps(vars(a),default=str,indent=2));return
    from transformers import AutoTokenizer
    import yaml
    config=json.loads((Path(a.model)/'config.json').read_text())
    if config['model_type']!='llama' or a.seq_length>config['max_position_embeddings']:
        raise ValueError('This sweep requires a Llama model with sufficient context')
    identity={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ('dry_run','output')}
    identity['model_config_sha256']=file_hash(Path(a.model)/'config.json')
    identity['converter_sha256']=file_hash(Path(__file__).with_name('sweep_core.py'))
    ruler=ROOT/'eval/RULER/scripts'
    identity['generator_sources']={str(p.relative_to(ROOT)):file_hash(p) for p in
        [Path(__file__),ruler/'synthetic.yaml',ruler/'data/template.py',ruler/'data/synthetic/constants.py',
         ruler/'data/synthetic/qa.py',ruler/'data/synthetic/niah.py',ruler/'data/synthetic/freq_words_extraction.py']}
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=a.output/'preparation_manifest.json'
    if manifest.is_file():
        if json.loads(manifest.read_text())!=identity: raise ValueError('Data preparation configuration differs; use a new data directory')
    elif any(a.output.iterdir()): raise ValueError('Nonempty data directory without provenance')
    else: atomic_json(manifest,identity)
    target=a.output/'observation.jsonl';ready=a.output/'data_ready.json'
    if ready.is_file():
        if file_hash(target)!=json.loads(ready.read_text())['data_sha256']: raise ValueError('Prepared data changed')
        print('Reusing prepared inputs:',target);return
    custom=yaml.safe_load((ruler/'synthetic.yaml').read_text())
    base=runpy.run_path(str(ruler/'data/synthetic/constants.py'))['TASKS']
    template=runpy.run_path(str(ruler/'data/template.py'))['Templates']['meta-llama3']
    tokenizer=AutoTokenizer.from_pretrained(a.model,use_fast=True)
    converted=[]
    for task in tasks:
        raw_root=a.raw_data_dir or a.output/'raw'
        raw=raw_root/task/'validation.jsonl'
        task_ready=a.output/(task+'_ready.json')
        if a.raw_data_dir is None and not task_ready.is_file():
            # Preserve an incomplete generator output before retrying this task.
            if raw.is_file():
                backup=raw.with_name('validation.incomplete.'+file_hash(raw)+'.jsonl')
                if not backup.exists(): raw.rename(backup)
            spec={**base[custom[task]['task']],**custom[task]}
            command=[sys.executable,'-u',str(ruler/'data/synthetic'/f"{spec['task']}.py"),
                     '--save_dir',str(raw_root),'--save_name',task,'--subset','validation',
                     '--tokenizer_path',a.model,'--tokenizer_type','hf',
                     '--max_seq_length',str(a.seq_length),'--tokens_to_generate',str(spec['tokens_to_generate']),
                     '--num_samples',str(a.num_samples),'--random_seed',str(a.seed),
                     '--template',template.format(task_template=spec['template'])+spec.get('answer_prefix','')]
            for key,value in spec['args'].items(): command+=['--'+key,str(value)]
            if spec['task']=='qa': command+=['--pre_samples','0']
            print('Preparing',task,flush=True);subprocess.run(command,check=True,cwd=ruler/'data')
        if not raw.is_file(): raise FileNotFoundError(raw)
        if task_ready.is_file() and json.loads(task_ready.read_text())['raw_sha256']!=file_hash(raw):
            raise ValueError('Raw task data changed: '+task)
        rows=[json.loads(x) for x in raw.read_text(encoding='utf-8').splitlines() if x.strip()]
        if len(rows)<a.num_samples: raise ValueError('Insufficient inputs for '+task)
        for index,row in enumerate(rows[:a.num_samples]):
            item=convert(row,task);item['sample_index']=index
            n=len(tokenizer.encode(item['prompt'],add_special_tokens=True))
            m=len(tokenizer.encode(item['label'],add_special_tokens=False))
            if not n or not m or n+m>a.seq_length: raise ValueError('Prompt+label outside context; no truncation: '+task)
            item.update(prompt_tokens=n,label_tokens=m)
            converted.append(item)
        atomic_json(task_ready,{'raw_sha256':file_hash(raw)})
    temp=target.with_suffix('.jsonl.tmp')
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in converted),encoding='utf-8')
    temp.replace(target)
    atomic_json(ready,{'data_sha256':file_hash(target),'input_count':len(converted),'tasks':tasks})
    print('Prepared',len(converted),'inputs:',target)


if __name__=='__main__':main()
