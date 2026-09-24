"""Explicit future cross-attention, owned by each action expert including reference."""
import torch
from torch import nn

class FutureAdapter(nn.Module):
    def __init__(self, hidden_dim, latent_channels, inner_dim=256, num_heads=8,
                 gate_init=0., dropout=0., patch_size=2):
        super().__init__()
        if inner_dim % num_heads:
            raise ValueError('Adapter inner_dim must be divisible by num_heads')
        self.patch_size = patch_size
        self.latent_projection = nn.Linear(latent_channels * patch_size**2, inner_dim)
        self.position_projection = nn.Linear(3, inner_dim)
        self.modality = nn.Parameter(torch.randn(inner_dim) * .02)
        self.query_projection = nn.Linear(hidden_dim, inner_dim)
        self.attention = nn.MultiheadAttention(inner_dim, num_heads, dropout=dropout, batch_first=True)
        self.output_projection = nn.Linear(inner_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, hidden, packet):
        if packet is None:
            return hidden
        z = packet.latents
        b, c, t, h, w = z.shape
        p = self.patch_size
        if h % p or w % p or b != hidden.shape[0]:
            raise ValueError('Future patch grid/batch mismatch')
        z = z.reshape(b, c, t, h//p, p, w//p, p).permute(0,2,3,5,1,4,6).reshape(b,-1,c*p*p)
        positions = packet.latent_positions.to(device=z.device, dtype=z.dtype)
        if positions.shape != (t*(h//p)*(w//p), 3):
            raise ValueError('Future latent position metadata mismatch')
        tokens = self.latent_projection(z) + self.position_projection(positions)[None] + self.modality
        query = self.query_projection(hidden)
        output = self.attention(query, tokens, tokens, need_weights=False)[0]
        return hidden + self.alpha.tanh() * self.output_projection(output)

def apply_future_adapter(expert, layer_index, hidden, packet):
    if packet is None:
        return hidden
    adapters = expert.future_adapters
    key = str(layer_index)
    return adapters[key](hidden, packet) if key in adapters else hidden
