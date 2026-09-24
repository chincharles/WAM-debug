"""Frozen video-only sampler. Never consumes expert actions or true future images."""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class FuturePacket:
    latents: torch.Tensor
    frame_times_s: torch.Tensor
    latent_positions: torch.Tensor
    scene_tokens: list[str]
    candidate_ids: list[int]
    seeds: list[int]
    generator_hash: str
    num_future_frames: int

    def __post_init__(self):
        if self.num_future_frames != 8 or self.latents.ndim != 5 or self.latents.requires_grad:
            raise ValueError('FuturePacket requires detached five-dimensional F8 latents')
        if not all(len(x) == self.latents.shape[0] for x in (self.scene_tokens,self.candidate_ids,self.seeds)):
            raise ValueError('Future metadata must align with candidate batch')

def latent_positions(z, temporal_factor, patch_size=2):
    _, _, t, h, w = z.shape
    return torch.stack(torch.meshgrid(
        torch.arange(1,t+1,device=z.device).float()*temporal_factor*.5,
        (torch.arange(h//patch_size,device=z.device).float()+.5)*patch_size,
        (torch.arange(w//patch_size,device=z.device).float()+.5)*patch_size,
        indexing='ij'), dim=-1).reshape(-1,3)

@torch.no_grad()
def sample_video_latents(model, first, context, mask, seed, num_steps, initial_latents=None):
    model.video_expert.eval()
    model.video_expert.requires_grad_(False)
    factor = model.vae.temporal_downsample_factor
    if 8 % factor:
        raise ValueError('Loaded VAE temporal compression is incompatible with F8')
    shape = (1, model.vae.model.z_dim, 8//factor+1, first.shape[-2], first.shape[-1])
    z = (torch.randn(shape, generator=torch.Generator().manual_seed(seed)) if initial_latents is None
         else initial_latents.clone()).to(device=first.device, dtype=first.dtype)
    if tuple(z.shape) != shape:
        raise ValueError('Initial video latent shape mismatch')
    z[:,:,0:1] = first
    sched = model.infer_video_scheduler
    ts, ds = sched.build_inference_schedule(num_inference_steps=num_steps, device=z.device, dtype=z.dtype)
    for timestep, delta in zip(ts, ds):
        pre = model.video_expert.pre_dit(x=z, timestep=timestep.reshape(1), context=context,
            context_mask=mask, action=None,
            fuse_vae_embedding_in_latents=model.video_expert.fuse_vae_embedding_in_latents)
        n = pre['tokens'].shape[1]
        attn = model._build_mot_attention_mask(video_seq_len=n, action_seq_len=8,
            video_tokens_per_frame=int(pre['meta']['tokens_per_frame']), device=z.device)[:n,:n]
        tokens = model.mot.forward_video_only(video_tokens=pre['tokens'], video_freqs=pre['freqs'],
            video_t_mod=pre['t_mod'], video_context_payload={'context':pre['context'],'mask':pre['context_mask']},
            video_attention_mask=attn)
        z = sched.step(model.video_expert.post_dit(tokens,pre), delta,z)
        z[:,:,0:1] = first
    return z.detach()

@torch.no_grad()
def generate_future_latents(model, observed_condition, *, num_future_frames,
                            scene_tokens, candidate_ids, seeds):
    if num_future_frames == 0:
        return None
    if num_future_frames != 8:
        raise ValueError('P0 supports F=0/8 only')
    if not len(scene_tokens) == len(candidate_ids) == len(seeds) == observed_condition['context'].shape[0]:
        raise ValueError('Candidate metadata alignment error')
    # Serial microbatch=1 bounds peak video activations independently of K.
    result = []
    for i, seed in enumerate(seeds):
        result.append(sample_video_latents(model, observed_condition['first_frame_latents'][i:i+1],
            observed_condition['context'][i:i+1], observed_condition['context_mask'][i:i+1],
            seed, model.kf['generation']['num_inference_steps'])[:,:,1:])
    z = torch.cat(result).detach()
    return FuturePacket(z, torch.arange(1,9,device=z.device)*.5,
        latent_positions(z,model.vae.temporal_downsample_factor), list(scene_tokens),
        list(candidate_ids),list(seeds),model.kf_metadata['il_hash'],8)
