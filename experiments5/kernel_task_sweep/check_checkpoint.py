"""Check model/checkpoint layout on CPU before generating expensive task inputs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments3'))
from analysis_core import validate_checkpoint_layout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--conv-weights',required=True)
    a=p.parse_args()
    config=json.loads((Path(a.model)/'config.json').read_text())
    if config['model_type']!='llama':raise ValueError('This sweep requires a Llama backbone')
    import torch
    weight=torch.load(a.conv_weights,map_location='cpu',weights_only=True)
    if weight.ndim==5 and weight.shape[2]==1:weight=weight[:,:,0]
    validate_checkpoint_layout('llama',config['num_hidden_layers'],config['num_attention_heads'],weight.shape)
    if not torch.isfinite(weight).all():raise ValueError('Nonfinite learned checkpoint')
    print('Matching Llama checkpoint:',a.conv_weights,'shape=',tuple(weight.shape),flush=True)


if __name__=='__main__':main()
