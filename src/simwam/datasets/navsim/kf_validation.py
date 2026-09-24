"""Manifest-restricted current-image dataset and strict official NAVSIM v1 reward."""
from dataclasses import asdict
import numpy as np
import torch
from .navsim_dataset import NavSimVideoDataset
from .pdm_reward import NavSimPDMReward
from simwam.kf.contracts import read_manifest

class KFNavSimDataset(NavSimVideoDataset):
    def __init__(self, *, manifest_path, **kwargs):
        self.manifest = read_manifest(manifest_path)
        self.allowed = {r['scene_token']:r['log_id'] for r in self.manifest}
        super().__init__(**kwargs)
        if set(self.tokens) != set(self.allowed):
            raise ValueError(f'Manifest/scene filter mismatch: missing={sorted(set(self.allowed)-set(self.tokens))[:5]}')
        self.tokens = [r['scene_token'] for r in self.manifest]
        self._frame_indices = [self.scene_filter.num_history_frames-1]
        self._image_is_pad = torch.zeros(1,dtype=torch.bool)

    def _apply_split_log_filter(self, scene_filter, split_config_path, split_logs_key):
        # Manifest is the sole allowed scene set; validate against source in prepare/preflight.
        scene_filter.tokens = list(self.allowed)
        scene_filter.log_names = sorted(set(self.allowed.values()))
        return scene_filter

    def _build_sensor_config(self,camera_layout):
        config = super()._build_sensor_config(camera_layout)
        config.cam_f0 = [self.scene_filter.num_history_frames-1]
        return config

    def __getitem__(self,idx):
        sample = super().__getitem__(idx)
        sample['log_id']=self.allowed[sample['token']]
        return sample

class StrictPDMReward(NavSimPDMReward):
    def __init__(self, *, allowed_tokens, official=False, **kwargs):
        super().__init__(**kwargs)
        self.allowed_tokens=set(allowed_tokens)
        if official:
            from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer,PDMScorerConfig
            self.scorer=PDMScorer(self.simulator.proposal_sampling,PDMScorerConfig())
        for token in self.allowed_tokens:
            if not self.available(token): raise FileNotFoundError(f'Metric cache missing: {token}')
        self.last_results=[]

    def score_batch(self,abs_poses,tokens,metric_cache_map=None,device=None):
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import pdm_score
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        poses=abs_poses.detach().float().cpu().numpy() if isinstance(abs_poses,torch.Tensor) else np.asarray(abs_poses)
        from simwam.kf.scoring import validate_score_inputs
        validate_score_inputs(poses,tokens,self.allowed_tokens,self.metric_cache_loader.metric_cache_paths)
        self.last_results=[]
        for pose,token in zip(poses,tokens):
            if token not in self.allowed_tokens: raise ValueError(f'Token outside selected manifest: {token}')
            try:
                cache=self.metric_cache_loader.get_from_token(token) if metric_cache_map is None else metric_cache_map[token]
                result=asdict(pdm_score(metric_cache=cache,
                    model_trajectory=Trajectory(pose.astype(np.float32),TrajectorySampling(num_poses=8,interval_length=.5)),
                    future_sampling=self.simulator.proposal_sampling,simulator=self.simulator,scorer=self.scorer))
                if not all(np.isfinite(v) for v in result.values()) or not 0 <= result['score'] <= 1:
                    raise ValueError('Invalid official score values')
            except Exception as exc:
                raise RuntimeError(f'Infrastructure/scoring failure for scene {token}: {exc}') from exc
            self.last_results.append({key:float(value) for key,value in result.items()})
        return torch.tensor([r['score'] for r in self.last_results],device=device or 'cpu',dtype=torch.float32)
