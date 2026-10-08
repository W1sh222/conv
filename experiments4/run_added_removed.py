"""Experiment4: blocks added by learned selection versus removed blocks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments3'))
from experiment_common import main
if __name__=='__main__': main(4)
