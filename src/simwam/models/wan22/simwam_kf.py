"""KF model: upstream action flow with optional explicit future residuals."""
import torch
from torch import nn
from .simwam_grpo import SimWAMGRPO
from .future_adapter import FutureAdapter
from .future_generator import generate_future_latents
from simwam.kf.contracts import UPSTREAM_SHA, SCHEMA, stable_seed

class SimWAMKF(SimWAMGRPO):
    def setup_kf(self, kf):
        self.kf = kf
        if hasattr(self.vae.model,"requires_grad_"):
            self.vae.model.requires_grad_(False).eval()
        a = kf['adapter']
        layers = a['layer_indices']
        if len(set(layers)) != len(layers) or any(i < 0 or i >= len(self.action_expert.blocks) for i in layers):
            raise ValueError('FutureAdapter layer indices outside action expert')
        if not a['enabled']:
            raise ValueError('P0 requires registered adapters even for F0')
        self.action_expert.future_adapters = nn.ModuleDict({str(i):FutureAdapter(
            self.action_expert.hidden_dim, self.vae.model.z_dim,
            inner_dim=a['inner_dim'],num_heads=a['num_heads'],gate_init=a['gate_init'],dropout=a['dropout'])
            for i in layers}).to(device=self.device,dtype=self.torch_dtype)
        self.kf_metadata = {'schema_version':SCHEMA,'upstream_sha':UPSTREAM_SHA,'supported_F':[0,8],
                            'kf':kf,'il_hash':'unloaded','normalization':'upstream_fixed_odo_absolute'}

    def expand_condition(self, cond, group_size):
        out = super().expand_condition(cond,group_size)
        if 'first_frame_latents' in cond:
            out['first_frame_latents'] = cond['first_frame_latents'].repeat_interleave(group_size,0)
        return out

    def with_future(self, cond, tokens, k, base_seed, rollout, future_frames):
        out = self.expand_condition(cond,k)
        if future_frames == 0:
            out['future_packet'] = None
            return out
        seen = {}; seeds=[]; expanded=[]; ids=[]
        for token in tokens:
            occurrence = seen.get(token,0); seen[token]=occurrence+1
            for candidate in range(k):
                expanded.append(token); ids.append(candidate)
                seeds.append(stable_seed(base_seed,'future',rollout,token,occurrence,candidate))
        out['future_packet'] = generate_future_latents(self,out,num_future_frames=future_frames,
            scene_tokens=expanded,candidate_ids=ids,seeds=seeds)
        return out

    @torch.no_grad()
    def infer_action(self, prompt, input_image, action_horizon, proprio=None, context=None,
                     context_mask=None, negative_prompt=None, text_cfg_scale=1., num_inference_steps=10,
                     sigma_shift=None, seed=0, rand_device='cpu', tiled=False, future_frames=0,
                     scene_token='inference', future_packet=None):
        if future_frames not in (0,8) or action_horizon != 8:
            raise ValueError('KF inference requires F0/F8 and horizon=8')
        if seed is None or rand_device != 'cpu':
            raise ValueError('KF requires explicit seed and CPU random stream')
        self.eval()
        if prompt is not None:
            if context is not None: raise ValueError('prompt/context are mutually exclusive')
            context, context_mask = self.encode_prompt(prompt)
        if context is None or context_mask is None: raise ValueError('Missing current text context')
        if input_image.ndim == 3: input_image=input_image[None]
        if context.ndim == 2: context=context[None]
        if context_mask.ndim == 1: context_mask=context_mask[None]
        if proprio is not None and proprio.ndim == 1: proprio=proprio[None]
        if input_image.shape[0] != 1: raise ValueError('Deployment requires one observation')
        cond = self.build_action_condition(input_image,8,context,context_mask,proprio,tiled)
        if future_packet is not None:
            if future_frames != 8: raise ValueError('Packet requires F8')
            cond['future_packet']=future_packet
        elif future_frames:
            cond = self.with_future(cond,[scene_token],1,seed,0,future_frames)
        x = torch.randn((1,8,3),generator=torch.Generator().manual_seed(seed)).to(self.device,self.torch_dtype)
        ts, ds = self.infer_action_scheduler.build_inference_schedule(num_inference_steps=num_inference_steps,
            device=self.device,dtype=self.torch_dtype,shift_override=sigma_shift)
        for t,d in zip(ts,ds):
            x = self.infer_action_scheduler.step(self.action_velocity(x,t.reshape(1),cond),d,x)
        return {'action':x[0].detach().float().cpu()}

    def load_checkpoint(self,path,optimizer=None):
        if getattr(self,'lora_enabled',False):
            raise ValueError('Load C1 before configure_grpo; resume requires complete training state')
        payload = torch.load(path,map_location='cpu',weights_only=False)
        metadata = payload.get('kf_metadata')
        if metadata is not None:
            if metadata['schema_version'] != SCHEMA or metadata['upstream_sha'] != UPSTREAM_SHA:
                raise ValueError('Incompatible KF checkpoint metadata')
            if metadata['kf']['adapter'] != self.kf['adapter']:
                raise ValueError('Adapter configuration mismatch')
        incompatible = self.mot.load_state_dict(payload['mot'],strict=False)
        missing = incompatible.missing_keys
        if metadata is None:
            missing = [k for k in missing if '.future_adapters.' not in k]
        if missing or incompatible.unexpected_keys:
            raise ValueError(f'Checkpoint mismatch: missing={missing}, unexpected={incompatible.unexpected_keys}')
        if self.proprio_encoder is not None:
            self.proprio_encoder.load_state_dict(payload['proprio_encoder'],strict=True)
        if metadata is not None: self.kf_metadata.update(metadata)
        return payload

    def save_checkpoint(self,path,optimizer=None,step=None):
        from .lora import merged_state_dict
        if optimizer is not None:
            raise ValueError("Deployment export is not resume; use save_training_state")
        payload={'mot':merged_state_dict(self.mot),'step':step,'kf_metadata':self.kf_metadata,'lora_merged':True}
        if self.proprio_encoder is not None: payload['proprio_encoder']=self.proprio_encoder.state_dict()
        torch.save(payload,path)
