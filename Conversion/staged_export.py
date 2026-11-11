#!/usr/bin/env python3
"""Incremental, fixed-geometry Nemotron export experiment (NOT a native backend).

Requires pinned model downloaded into --model. First proves PyTorch partition
parity; optional --aimodel invokes the documented Core AI PyTorch Extensions.
No .aimodel output is labelled validated without separate runtime comparison.
"""
from __future__ import annotations
import argparse,json,platform,time,traceback,hashlib
from vision_geometry import GeometrySpecializedVision
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

class Vision(nn.Module):
    def __init__(self,model,size,content):
        super().__init__();self.vision=model.encoder.model_encoder.radio_model.model
        self.size=size;self.content=content
    def forward(self,patches): return self.vision(patches,self.size,self.content)

class Adapter(nn.Module):
    def __init__(self,model,size):
        super().__init__();self.neck=model.encoder;self.height,self.width=size
    def forward(self,hidden):
        n=self.neck
        # Shipped checkpoint summary_idxs is [0,1]; use it, rather than assuming.
        summary=hidden[:,n.model_encoder.radio_model.summary_idxs].flatten(1)
        features=n.layer_norm1(n.conv1(hidden[:,10:].transpose(1,2)).transpose(1,2))
        features=features.transpose(1,2).reshape(1,1024,self.height//16,self.width//16)
        features=n.layer_norm2(n.conv2(features).flatten(2).transpose(1,2))
        summary=n.layer_norm3(n.sum_proj(summary)).unsqueeze(1)
        return torch.cat((features,summary),dim=1)

class Decoder(nn.Module):
    """Functional explicit cache: [layers,batch,kv_heads,past_tokens,head_dim].

    A static-shape experiment, not mutable Core AI cache integration. Input
    embeddings are vision+prompt for prefill and token embedding for each step.
    """
    def __init__(self,model):
        super().__init__();self.decoder=model.decoder;self.lm_head=model.lm_head
    def forward(self,embeddings,past_keys,past_values,key_mask):
        h=self.decoder.layernorm_embedding(embeddings);keys=[];values=[]
        b,l,_=h.shape
        for i,layer in enumerate(self.decoder.layers):
            a=layer.self_attn;n=layer.self_attn_layer_norm(h)
            q=a.q_proj(n).view(b,l,a.num_heads,a.head_dim).transpose(1,2)
            k=a.k_proj(n).view(b,l,a.num_kv_heads,a.head_dim).transpose(1,2)
            v=a.v_proj(n).view(b,l,a.num_kv_heads,a.head_dim).transpose(1,2)
            k=torch.cat((past_keys[i],k),dim=-2);v=torch.cat((past_values[i],v),dim=-2)
            keys.append(k);values.append(v)
            past=k.shape[-2]-l
            allowed=torch.arange(k.shape[-2],device=h.device)[None,:] <= (torch.arange(l,device=h.device)[:,None]+past)
            allowed=allowed[None,None]&key_mask[:,None,None,:]
            repeat=a.num_heads//a.num_kv_heads
            out=F.scaled_dot_product_attention(q,k.repeat_interleave(repeat,dim=1),v.repeat_interleave(repeat,dim=1),attn_mask=allowed,dropout_p=0.)
            h=h+a.out_proj(out.transpose(1,2).reshape(b,l,-1))
            h=h+layer.fc2(F.gelu(layer.fc1(layer.final_layer_norm(h))))
        logits=self.lm_head(self.decoder.layer_norm(h[:,-1:]))
        return logits,torch.stack(keys),torch.stack(values)

def diff(a,b):
    d=(a.float()-b.float()).abs()
    return {'max_absolute_error':d.max().item(),'mean_absolute_error':d.mean().item(),'finite':bool(torch.isfinite(a).all()),'shape':list(a.shape)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path('Models'))
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('Conversion/evidence/partitions'))
    p.add_argument('--specialize-geometry',action='store_true',help='Exactly precompute frozen positional interpolation for declared page geometry')
    p.add_argument('--runtime',action='store_true',help='Execute exported assets through macOS Core AI and compare with PyTorch (USE_OS_COREAI=1 required)')
    p.add_argument('--aimodel',action='store_true',help='Attempt Core AI conversion after PyTorch parity, requires coreai-torch')
    p.add_argument('--stages', nargs='+', choices=['vision','adapter','decoder_prefill','decoder_step'], default=['vision','adapter','decoder_prefill','decoder_step'])
    p.add_argument('--torch-export',action='store_true',help='Also execute each torch.export graph and compare')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report={'revision':'a9a1febb1b9bf497fc67bd1d23fa33e170b249db','macos':platform.mac_ver()[0],'core_ai_end_to_end_validated':False,'stages':{},'limitations':['Fixed single-page geometry','Explicit functional cache, not Core AI mutable cache','No full Swift tokenizer/generation integration']}
    try:
        from PIL import Image
        from transformers import AutoModel,AutoProcessor
        model=AutoModel.from_pretrained(args.model.resolve(),trust_remote_code=True,dtype=torch.float32,local_files_only=True).eval().requires_grad_(False)
        processor=AutoProcessor.from_pretrained(args.model.resolve(),trust_remote_code=True,local_files_only=True)
        x=processor(images=[Image.open(args.image).convert('RGB')],text='</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>',return_tensors='pt',add_special_tokens=False)
        size=tuple(x['image_sizes'][0].tolist());content=tuple(x['content_sizes'][0].tolist())
        original_vision=Vision(model,size,content).eval()
        vision=(GeometrySpecializedVision(model,size,content) if args.specialize_geometry else original_vision).eval();adapter=Adapter(model,size).eval();decoder=Decoder(model).eval()
        with torch.inference_mode():
            hidden=vision(x['pixel_values']);emb=adapter(hidden)
            if args.specialize_geometry:
                report['stages']['specialized_vision_parity']=diff(hidden,original_vision(x['pixel_values']))
            report['geometry']={'image_size_hw':list(size),'content_size_hw':list(content),'patch_count':int(x['pixel_values'].shape[0]),'source_image_sha256':hashlib.sha256(args.image.read_bytes()).hexdigest()}
            official_emb,_=model.encoder(x['pixel_values'],x['image_sizes'],x['content_sizes'])
            report['stages']['vision_adapter_parity']=diff(emb,official_emb)
            embeddings=torch.cat((emb,model.decoder.embed_tokens(x['input_ids'])),dim=1)
            empty=torch.empty(10,1,8,0,64);mask=torch.ones(1,embeddings.shape[1],dtype=torch.bool)
            logits,k,v=decoder(embeddings,empty,empty,mask)
            ref=model(**x)
            report['stages']['decoder_prefill_parity']=diff(logits,ref.logits[:,-1:])
            report['stages']['prefill_keys_parity']=diff(k,torch.stack([layer.keys for layer in ref.past_key_values.layers]))
            report['stages']['prefill_values_parity']=diff(v,torch.stack([layer.values for layer in ref.past_key_values.layers]))
            token=logits[:,-1].argmax(-1).unsqueeze(-1)
            nxt=model.decoder.embed_tokens(token);next_mask=torch.ones(1,k.shape[-2]+1,dtype=torch.bool)
            step,k2,v2=decoder(nxt,k,v,next_mask)
            ref2=model(input_ids=token,past_key_values=ref.past_key_values)
            report['stages']['decoder_cache_step_parity']=diff(step,ref2.logits[:,-1:])
            report['stages']['step_keys_parity']=diff(k2,torch.stack([layer.keys for layer in ref2.past_key_values.layers]))
            report['stages']['step_values_parity']=diff(v2,torch.stack([layer.values for layer in ref2.past_key_values.layers]))
            examples=[('vision',vision,(x['pixel_values'],)),('adapter',adapter,(hidden,)),('decoder_prefill',decoder,(embeddings,empty,empty,mask)),('decoder_step',decoder,(nxt,k,v,next_mask))]
        for name,module,inputs in examples:
            if name not in args.stages: continue
            print(name,flush=True)
            if args.torch_export or args.aimodel:
                inputs=tuple(t.clone() for t in inputs)
                exported=torch.export.export(module,inputs)
                if name == 'vision' and args.specialize_geometry:
                    forbidden=[str(n.target) for n in exported.graph.nodes if 'upsample_bilinear2d_aa' in str(n.target) or 'grid_sampler' in str(n.target)]
                    assert not forbidden, forbidden
                    report['stages']['unsupported_vision_ops_remaining']=forbidden
                with torch.inference_mode():
                    expected=module(*inputs);actual=exported.module()(*inputs)
                first=lambda val: val[0] if isinstance(val,tuple) else val
                report['stages'][name+'_torch_export']=diff(first(actual),first(expected))
                if args.aimodel:
                    from coreai_torch import TorchConverter,get_decomp_table
                    from coreai.runtime import AIModelAssetMetadata
                    exported=exported.run_decompositions(get_decomp_table())
                    input_names={'vision':['patches'],'adapter':['hidden'],'decoder_prefill':['embeddings','past_keys','past_values','key_mask'],'decoder_step':['embeddings','past_keys','past_values','key_mask']}[name]
                    output_names=['output'] if name in ('vision','adapter') else ['logits','keys','values']
                    converter=TorchConverter(mode=TorchConverter.Mode.RELEASE).add_exported_program(exported_program=exported,input_names=input_names,output_names=output_names)
                    program=converter.to_coreai()
                    meta=AIModelAssetMetadata();meta.author='NVIDIA Corporation';meta.license='OpenMDW-1.1';meta.model_description='Nemotron Parse experimental '+name
                    program.save_asset(args.output/(name+'.aimodel'),meta)
                    report['stages'][name+'_core_ai']='Exported; NOT executed or numerically validated in Core AI'
                    if args.runtime:
                        import os,asyncio,numpy as np
                        if os.environ.get('USE_OS_COREAI') != '1': raise RuntimeError('USE_OS_COREAI=1 required; portable runtime is not native validation')
                        from coreai.runtime import AIModel,NDArray,SpecializationOptions
                        async def validate():
                            started=time.perf_counter()
                            runtime=await AIModel.load(args.output/(name+'.aimodel'),SpecializationOptions.cpu_only())
                            fn=runtime.load_function('main')
                            result=await fn({key:NDArray(t.numpy()) for key,t in zip(input_names,inputs)})
                            expected_outputs=expected if isinstance(expected,tuple) else (expected,)
                            measured={key:diff(torch.from_numpy(result[key].numpy().copy()),target) for key,target in zip(output_names,expected_outputs)}
                            measured['runtime_seconds']=time.perf_counter()-started
                            measured['runtime']='macOS Core AI CPU, USE_OS_COREAI=1'
                            return measured
                        report['stages'][name+'_native_runtime']=asyncio.run(validate())
                        report['stages'][name+'_core_ai']='Exported and executed through macOS Core AI; see measured errors'

        report['pytorch_partition_parity_passed']=all(s.get('max_absolute_error',0)<1e-4 for s in report['stages'].values() if isinstance(s,dict))
    except Exception as e:
        report['error']=repr(e);report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    if 'error' in report or not report.get('pytorch_partition_parity_passed',False): raise SystemExit(1)

if __name__=='__main__': main()
