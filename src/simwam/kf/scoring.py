"""Unit and infrastructure invariants shared by real scoring and CPU tests."""
import numpy as np

def validate_score_inputs(poses,tokens,allowed,cache_paths):
    if poses.shape!=(len(tokens),8,3) or not np.isfinite(poses).all():
        raise ValueError('Model numerical failure: trajectories must be finite [M,8,3]')
    for token in tokens:
        if token not in allowed:raise ValueError(f'Token outside selected manifest: {token}')
        if token not in cache_paths:raise FileNotFoundError(f'Metric cache missing: {token}')

def reward_to_report(reward):
    if not np.isfinite(reward) or not 0<=reward<=1:raise ValueError('Reward must be finite in [0,1]')
    return 100*reward
