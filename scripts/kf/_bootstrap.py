import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'navsim'))
os.environ.setdefault('NAVSIM_DEVKIT_ROOT',str(ROOT/'navsim'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')

os.environ['SIMWAM_KF_OFFLINE']='1'
