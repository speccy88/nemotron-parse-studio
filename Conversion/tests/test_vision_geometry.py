"""Regression tests for partial patches, white padding and one-patch axes."""
import importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Conversion'))
from vision_geometry import GeometrySpecializedVision
spec=importlib.util.spec_from_file_location('pinned_geometry',ROOT/'Models/hf_prefix_geometry.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
_normalized_2d_position_pages=module._normalized_2d_position_pages
class FakeVision(nn.Module):
    def __init__(self):
        super().__init__();self.patch_generator=SimpleNamespace(embedder=nn.Linear(768,1152,bias=False),pos_embed=nn.Parameter(torch.randn(1,16384,1152)),cls_token=SimpleNamespace(token=nn.Parameter(torch.randn(10,1152))))
        self.blocks=nn.ModuleList([])
    def forward(self,x,size,content):
        pg=self.patch_generator
        positioned=_normalized_2d_position_pages(torch,patches=pg.embedder(x)[None],pos_embed=pg.pos_embed,source_rows=128,source_cols=128,image_sizes=[size],patch_size=16,content_sizes=[content],content_normalized=True)
        return torch.cat((pg.cls_token.token[None],positioned),1)
def test_geometry_specialization_preserves_reference_and_no_unsupported_ops():
    torch.manual_seed(41);torch.set_num_threads(4)
    original=FakeVision().eval().requires_grad_(False)
    model=SimpleNamespace(encoder=SimpleNamespace(model_encoder=SimpleNamespace(radio_model=SimpleNamespace(model=original))))
    # Includes one-patch adaptive pooling path, odd sizes/partial patches, padding
    # and ordinary landscape/portrait. Random patches ensure no sample is baked in.
    for content in [(1,1),(16,31),(31,16),(17,63),(253,509),(800,640),(321,961)]:
        size=((content[0]+15)//16*16,(content[1]+63)//64*64)
        wrapper=GeometrySpecializedVision(model,size,content).eval()
        patches=torch.randn(size[0]//16*(size[1]//16),768)
        with torch.no_grad():torch.testing.assert_close(wrapper(patches),original(patches,size,content),rtol=0,atol=0)
        graph=torch.export.export(wrapper,(patches,))
        assert not any('grid_sampler' in str(n.target) or 'upsample_bilinear2d_aa' in str(n.target) for n in graph.graph.nodes)
