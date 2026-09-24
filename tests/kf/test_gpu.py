import os
import subprocess
import sys
import pytest
import torch

@pytest.mark.gpu
@pytest.mark.integration
def test_real_gpu_regression(tmp_path):
    if not torch.cuda.is_available() or not os.environ.get('SIMWAM_IL_CHECKPOINT'):
        pytest.skip('CUDA and real C0/NAVSIM resources are unavailable')
    subprocess.run([sys.executable,'scripts/kf/gpu_checks.py','--output',str(tmp_path/'gpu.json')],check=True)
