"""Merge disjoint completed task sweeps without altering their raw results."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments3'))
import experiment_common as common
from resume_state import atomic_json,file_hash
from sweep_core import ALL_TASKS
from run_sweep import paired_summary


def merge(directories,output):
    summaries=[];manifests=[];tasks=[];inputs=[];rows=[];failures=[]
    for directory in directories:
        summary=json.loads((directory/'summary.json').read_text())
        manifest=json.loads((directory/'run_manifest.json').read_text())
        if summary['status'] not in ('complete','complete_with_skips'):
            raise ValueError('Sweep is not finished: '+str(directory))
        if set(tasks)&set(summary['tasks']):raise ValueError('Overlapping task sets; refusing to count inputs twice')
        if summaries and summary['policies']!=summaries[0]['policies']:raise ValueError('Different kernel policies')
        if manifests:
            excluded={'tasks','data'}
            args=lambda m:{k:v for k,v in m['arguments'].items() if k not in excluded}
            if args(manifest)!=args(manifests[0]):raise ValueError('Different evaluation configurations')
            if manifest['model_metadata']!=manifests[0]['model_metadata'] or manifest['weight_sha256']!=manifests[0]['weight_sha256']:
                raise ValueError('Different model/checkpoint provenance')
            for key,value in manifests[0]['source_sha256'].items():
                # These two files were extended only to accept extra task keys
                # and serialize QA1 alternative answers. Numerical code is unchanged.
                if key.replace('\\','/').endswith(('kernel_task_sweep/run_sweep.py','kernel_task_sweep/sweep_core.py')):continue
                if manifest['source_sha256'].get(key)!=value:raise ValueError('Shared numerical implementation differs: '+key)
        failures.extend(summary.get('failures',[]))
        excluded_ids={(x['task'],x['sample_index']) for x in summary.get('excluded_samples',[])}
        for task in summary['tasks']:
            for case in sorted((directory/task).glob('s*')):
                if not case.is_dir():continue
                index=int(case.name[1:]);inputs.append(dict(task=task,sample_index=index))
                if (task,index) in excluded_ids:continue
                for policy in summary['policies']:
                    result=json.loads((case/(policy+'.json')).read_text())
                    if (result.get('status')!='complete' or result.get('task')!=task or result.get('sample_index')!=index or
                        result.get('policy')!=policy or not math.isfinite(result['label_loss']) or
                        abs(sum(result['token_losses'])/len(result['token_losses'])-result['label_loss'])>1e-12):
                        raise ValueError('Invalid saved policy result: '+str(case))
                    rows.append(result)
        if len([i for i in inputs if i['task'] in summary['tasks']])!=summary['attempted_samples']:
            raise ValueError('Input count disagrees with summary')
        tasks.extend(summary['tasks']);summaries.append(summary);manifests.append(manifest)
    if set(tasks)!=set(ALL_TASKS):raise ValueError('Merged sources must cover all 13 RULER tasks')
    tasks=list(ALL_TASKS)
    if output.exists() and any(output.iterdir()):raise ValueError('Choose a new empty merged output directory')
    output.mkdir(parents=True,exist_ok=True)
    summary=paired_summary(rows,inputs,tasks,summaries[0]['policies'],failures)
    summary['source_runs']=[dict(directory=str(d),summary_sha256=file_hash(d/'summary.json'),
                                manifest_sha256=file_hash(d/'run_manifest.json')) for d in directories]
    summary['source_note']='Original four-task sweep plus newly added tasks; common inference parameters checked; no raw results modified'
    atomic_json(output/'summary.json',summary)
    common.csv_file(output/'individual_losses.csv',rows)
    common.csv_file(output/'failed_samples.csv',failures)
    if summary['macro_mean_nll'] is not None:
        common.csv_file(output/'task_mean_losses.csv',
            [dict(task=t,**summary['task_mean_nll'][t]) for t in tasks]+
            [dict(task='Macro average',**summary['macro_mean_nll'])])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'experiments5/kernel_task_results/llama128k_all13tasks_merged')
    a=p.parse_args();merge(a.inputs,a.output)
