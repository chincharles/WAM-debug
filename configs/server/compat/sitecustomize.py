"""Restore the exact legacy np.int alias needed by the untouched NAVSIM v1.1.

Only active in the PPU launcher's isolated Python 3.12 / NumPy 1.26 environment.
No metric implementation or numerical operation is replaced.
"""
import os
if os.environ.get('SIMWAM_RUNTIME') == 'ppu':
    try:
        import numpy as np
    except ModuleNotFoundError:
        pass  # The installer has not installed NumPy into the new venv yet.
    else:
        if 'int' not in vars(np):
            np.int = int
