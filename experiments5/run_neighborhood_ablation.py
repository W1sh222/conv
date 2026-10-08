"""Experiment5: full-policy NLL of initial, positive 1x1, fixed and learned kernels."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments3'))
from experiment_common import main
if __name__=='__main__': main(5)
