"""Prepare small, model-templated RULER task sets, without loading LLM weights."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments3'))
from resume_state import atomic_json, file_hash
from sweep_core import TASKS, convert

PATH_BUG_SOURCE_HASHES = {
    '22bc014304b1aaa8680000054a007be76546346d5cc2d5d4d49fb3d929890bd2',
    '8b02f76cd2cf0164a577a7bd9a6334e148d696d56f03c98764f2099721e223b1',
}


def ensure_manifest(output, identity, source_key):
    """Allow only the known path-only repair on unfinished preparation runs."""
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/'preparation_manifest.json'
    backup=output/'preparation_manifest.before_path_fix.json'
    if manifest.is_file():
        previous=json.loads(manifest.read_text(encoding='utf-8'))
        if previous==identity:
            return backup.is_file()
        old_hash=previous.get('generator_sources',{}).get(source_key)
        migrated=json.loads(json.dumps(previous))
        if source_key in migrated.get('generator_sources',{}):
            migrated['generator_sources'][source_key]=identity['generator_sources'][source_key]
        if (old_hash not in PATH_BUG_SOURCE_HASHES or migrated!=identity
            or (output/'data_ready.json').is_file()):
            raise ValueError('Data preparation configuration differs; use a new data directory')
        if not backup.is_file(): atomic_json(backup,previous)
        atomic_json(manifest,identity)
        print('Updated unfinished preparation manifest for output-path repair',flush=True)
        return True
    if any(output.iterdir()): raise ValueError('Nonempty data directory without provenance')
    atomic_json(manifest,identity)
    return False


def recover_misplaced_raw(source, destination):
    """Copy the known misplaced output, preserving the original file."""
    if not source.is_file(): return False
    if destination.is_file():
        if file_hash(source)!=file_hash(destination):
            raise ValueError('Misplaced and destination task data differ; refusing to overwrite')
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        temp=destination.with_name(destination.name+'.recover.tmp')
        shutil.copy2(source,temp);temp.replace(destination)
    print('Recovered generated task data:',source,'->',destination,flush=True)
    return True


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
    # The generators run with a different cwd. Never pass their output/model paths
    # relative to this process's cwd. Keep the original spelling for provenance.
    requested_output=a.output
    a.output=a.output.resolve()
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
    legacy_allowed=ensure_manifest(a.output,identity,str(Path(__file__).relative_to(ROOT)))
    if requested_output.is_absolute():
        try: legacy_output=requested_output.relative_to(ROOT)
        except ValueError: legacy_output=None
    else: legacy_output=requested_output
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
        raw_root=(a.raw_data_dir or a.output/'raw').resolve()
        raw=raw_root/task/'validation.jsonl'
        task_ready=a.output/(task+'_ready.json')
        recovered=False
        if legacy_allowed and a.raw_data_dir is None and legacy_output is not None and not task_ready.is_file():
            misplaced=(ruler/'data'/legacy_output/'raw'/task/'validation.jsonl').resolve()
            if misplaced!=raw: recovered=recover_misplaced_raw(misplaced,raw)
        if a.raw_data_dir is None and not task_ready.is_file() and not recovered:
            # Preserve an incomplete generator output before retrying this task.
            if raw.is_file():
                backup=raw.with_name('validation.incomplete.'+file_hash(raw)+'.jsonl')
                if not backup.exists(): raw.rename(backup)
            spec={**base[custom[task]['task']],**custom[task]}
            command=[sys.executable,'-u',str(ruler/'data/synthetic'/f"{spec['task']}.py"),
                     '--save_dir',str(raw_root),'--save_name',task,'--subset','validation',
                     '--tokenizer_path',str(Path(a.model).resolve()),'--tokenizer_type','hf',
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
