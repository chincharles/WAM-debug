"""Build the one fixed-prompt cache used by P0, with the upstream encoder semantics."""
import _bootstrap
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

def main():
    p=argparse.ArgumentParser(description='Create P0 fixed-prompt umT5 cache from local weights; never downloads')
    p.add_argument('--weights',required=True,help='Local converted models_t5_umt5-xxl-enc-bf16.safetensors')
    p.add_argument('--tokenizer',required=True,help='Local google/umt5-xxl tokenizer directory')
    p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.dry_run:print(vars(a));return
    import torch
    from simwam.datasets.navsim.navsim_dataset import NavSimVideoDataset
    from simwam.models.wan22.helpers.loader import _load_registered_model
    from simwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
    from simwam.models.wan22.simwam import SimWAM
    if not Path(a.weights).is_file() or not Path(a.tokenizer).is_dir():raise FileNotFoundError('Prepare local umT5 weights and tokenizer first')
    prompt=NavSimVideoDataset.build_prompt_fixed(torch.empty(0,3),torch.empty(0),0.,0.,use_dynamic_prompt=False)
    output=Path(a.output_dir)/f'{hashlib.sha256(prompt.encode()).hexdigest()}.t5_len256.wan22ti2v5b.pt'
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
    encoder=_load_registered_model(a.weights,'wan_video_text_encoder',torch.bfloat16,a.device).eval().requires_grad_(False)
    tokenizer=HuggingfaceTokenizer(name=a.tokenizer,seq_len=256,clean='whitespace')
    context,mask=SimWAM.encode_prompt(SimpleNamespace(text_encoder=encoder,tokenizer=tokenizer,device=torch.device(a.device)),prompt)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'context':context[0].cpu(),'mask':mask[0].cpu(),'prompt':prompt},output)
    print(output)
if __name__=='__main__':main()
