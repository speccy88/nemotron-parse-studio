"""Exact geometry specialization of NVIDIA's frozen vision encoder.

The learned positional field and content/padded dimensions are constant for a
profile. Evaluate the *unchanged* pinned helper once in float32 outside the export
graph, and register its result as a buffer. This is partial evaluation, not an
approximation: antialiasing, border sampling, partial patches and the one-patch
adaptive pooling path are all preserved. Exported profiles must never be applied
to different padded/content dimensions (validated by the runtime manifest).
"""
import importlib
import torch
from torch import nn

class GeometrySpecializedVision(nn.Module):
    def __init__(self, model, image_size, content_size):
        super().__init__()
        original = model.encoder.model_encoder.radio_model.model
        pg = original.patch_generator
        self.image_size = tuple(map(int, image_size))
        self.content_size = tuple(map(int, content_size))
        count = (self.image_size[0] // 16) * (self.image_size[1] // 16)
        helper = importlib.import_module(type(original).__module__)._normalized_2d_position_pages
        with torch.no_grad():
            positions = helper(
                torch, patches=torch.zeros(1, count, 1152, dtype=torch.float32),
                pos_embed=pg.pos_embed.detach().float(), source_rows=128,
                source_cols=128, image_sizes=[self.image_size], patch_size=16,
                content_sizes=[self.content_size], content_normalized=True,
            )
        self.register_buffer('positions', positions.contiguous())
        self.embedder = pg.embedder
        self.register_buffer('class_tokens', pg.cls_token.token.detach().unsqueeze(0))
        self.blocks = original.blocks

    def forward(self, patches):
        patches = self.embedder(patches).unsqueeze(0) + self.positions
        hidden = torch.cat((self.class_tokens, patches), dim=1)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden
