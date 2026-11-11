#!/usr/bin/env python3
"""Export geometry-checked Core AI profiles and a fixed-capacity decoder cache.

Uses unchanged NVIDIA weights; only positional-field constant folding and an
explicit fixed KV buffer are introduced. Validation never uses sample text as
model input. Generation/parity is performed separately before enabling a profile.
"""
import argparse,base64,hashlib,json,sys,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from vision_geometry import GeometrySpecializedVision
from staged_export import Adapter,diff
from transformers import AutoModel,AutoProcessor
from PIL import Image
from coreai_torch import TorchConverter,get_decomp_table
from coreai.runtime import AIModelAssetMetadata

REVISION='a9a1febb1b9bf497fc67bd1d23fa33e170b249db'
PROMPT='</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>'
class Encoder(nn.Module):
    def __init__(self,model,size,content):
        super().__init__();self.vision=GeometrySpecializedVision(model,size,content);self.adapter=Adapter(model,size)
    def forward(self,patches):return self.adapter(self.vision(patches))

class Prefill(nn.Module):
    def __init__(self,model,prompt_ids,capacity):
        super().__init__();self.decoder=model.decoder;self.lm_head=model.lm_head;self.capacity=capacity
        self.register_buffer('prompt',prompt_ids)
    def forward(self,visual):
        emb=torch.cat((visual,self.decoder.embed_tokens(self.prompt)),dim=1)
        h=self.decoder.layernorm_embedding(emb);keys=[];values=[];b,l,_=h.shape
        allowed=torch.arange(l)[None,:]<=torch.arange(l)[:,None]
        for layer in self.decoder.layers:
            a=layer.self_attn;n=layer.self_attn_layer_norm(h)
            q=a.q_proj(n).view(b,l,a.num_heads,a.head_dim).transpose(1,2)
            k=a.k_proj(n).view(b,l,a.num_kv_heads,a.head_dim).transpose(1,2)
            v=a.v_proj(n).view(b,l,a.num_kv_heads,a.head_dim).transpose(1,2)
            keys.append(F.pad(k,(0,0,0,self.capacity-l)));values.append(F.pad(v,(0,0,0,self.capacity-l)))
            r=a.num_heads//a.num_kv_heads
            out=F.scaled_dot_product_attention(q,k.repeat_interleave(r,1),v.repeat_interleave(r,1),attn_mask=allowed[None,None],dropout_p=0.)
            h=h+a.out_proj(out.transpose(1,2).reshape(b,l,-1))
            h=h+layer.fc2(F.gelu(layer.fc1(layer.final_layer_norm(h))))
        return self.lm_head(self.decoder.layer_norm(h[:,-1:])),torch.stack(keys),torch.stack(values)

class Step(nn.Module):
    def __init__(self,model,capacity):
        super().__init__();self.decoder=model.decoder;self.lm_head=model.lm_head;self.capacity=capacity
        self.register_buffer('slots',torch.arange(capacity,dtype=torch.int32))
    def forward(self,input_ids,past_keys,past_values,position):
        h=self.decoder.layernorm_embedding(self.decoder.embed_tokens(input_ids));keys=[];values=[]
        update=(self.slots==position).reshape(1,1,-1,1)
        allowed=(self.slots<=position).reshape(1,1,1,-1)
        for i,layer in enumerate(self.decoder.layers):
            a=layer.self_attn;n=layer.self_attn_layer_norm(h)
            q=a.q_proj(n).view(1,1,a.num_heads,a.head_dim).transpose(1,2)
            k=a.k_proj(n).view(1,1,a.num_kv_heads,a.head_dim).transpose(1,2)
            v=a.v_proj(n).view(1,1,a.num_kv_heads,a.head_dim).transpose(1,2)
            k=torch.where(update,k,past_keys[i]);v=torch.where(update,v,past_values[i]);keys.append(k);values.append(v)
            r=a.num_heads//a.num_kv_heads
            out=F.scaled_dot_product_attention(q,k.repeat_interleave(r,1),v.repeat_interleave(r,1),attn_mask=allowed,dropout_p=0.)
            h=h+a.out_proj(out.transpose(1,2).reshape(1,1,-1))
            h=h+layer.fc2(F.gelu(layer.fc1(layer.final_layer_norm(h))))
        return self.lm_head(self.decoder.layer_norm(h)),torch.stack(keys),torch.stack(values)

def export(module,args,path,input_names,output_names):
    if path.exists():raise FileExistsError(f'{path} exists; choose a new output or reuse an already verified asset')
    graph=torch.export.export(module,args).run_decompositions(get_decomp_table())
    forbidden=[str(n.target) for n in graph.graph.nodes if 'upsample_bilinear2d_aa' in str(n.target) or 'grid_sampler' in str(n.target)]
    if forbidden:raise RuntimeError(str(forbidden))
    program=TorchConverter(mode=TorchConverter.Mode.RELEASE).add_exported_program(exported_program=graph,input_names=input_names,output_names=output_names).to_coreai()
    metadata=AIModelAssetMetadata();metadata.author='NVIDIA Corporation; profile wrapper by Parse Studio';metadata.license='OpenMDW-1.1';metadata.model_description='Pinned Nemotron 3.5 '+path.stem
    program.save_asset(path,metadata)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--image',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path('NativeModels'));parser.add_argument('--capacity',type=int,default=1024);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    model=AutoModel.from_pretrained(Path('Models').resolve(),trust_remote_code=True,dtype=torch.float32,local_files_only=True).eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(Path('Models').resolve(),trust_remote_code=True,local_files_only=True)
    x=processor(images=[Image.open(args.image).convert('RGB')],text=PROMPT,return_tensors='pt',add_special_tokens=False)
    size=tuple(x['image_sizes'][0].tolist());content=tuple(x['content_sizes'][0].tolist())
    name=f'{content[1]}x{content[0]}';out=args.output/name;out.mkdir(exist_ok=True)
    enc=Encoder(model,size,content).eval();pre=Prefill(model,x['input_ids'],args.capacity).eval();step=Step(model,args.capacity).eval()
    with torch.inference_mode():
        visual=enc(x['pixel_values']);reference=model(**x);logits,k,v=pre(visual)
        length=visual.shape[1]+x['input_ids'].shape[1]
        if length+16>args.capacity:raise ValueError('Cache capacity leaves insufficient generation space')
        token=logits[:,-1].argmax(-1).unsqueeze(-1).int();position=torch.tensor([length],dtype=torch.int32)
        next_logits,k2,v2=step(token,k,v,position);ref2=model(input_ids=token,past_key_values=reference.past_key_values)
        official_visual,_=model.encoder(x['pixel_values'],x['image_sizes'],x['content_sizes'])
        parity={'encoder':diff(visual,official_visual),'prefill_logits':diff(logits,reference.logits[:,-1:]),'cache_step_logits':diff(next_logits,ref2.logits[:,-1:]),'cache_keys':diff(k2[:,:,:,:length+1],torch.stack([l.keys for l in ref2.past_key_values.layers]))}
        assert all(v['max_absolute_error']<1e-4 for v in parity.values()),parity
    (out/'partition-parity.json').write_text(json.dumps(parity,indent=2)+'\n')
    print('Export encoder',flush=True);export(enc,(x['pixel_values'].clone(),),out/'encoder.aimodel',['patches'],['visual'])
    print('Export prefill',flush=True);export(pre,(visual.clone(),),out/'prefill.aimodel',['visual'],['logits','keys','values'])
    step_path=args.output/f'decoder-step-{args.capacity}.aimodel'
    if not step_path.exists():
        print('Export decoder step',flush=True);export(step,(token.clone(),k.clone(),v.clone(),position.clone()),step_path,['input_ids','past_keys','past_values','position'],['logits','keys','values'])
    # ByteLevel decoding data. Do not decode each token separately (UTF-8 bytes
    # can cross token boundaries). Added control tokens are literal UTF-8.
    config=json.loads(Path('Models/tokenizer.json').read_text());bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
    for b in range(256):
        if b not in bs:bs.append(b);cs.append(256+n);n+=1
    inverse={chr(c):b for b,c in zip(bs,cs)};special={t['id']:t['content'] for t in config['added_tokens']}
    vocab={v:k for k,v in config['model']['vocab'].items()}
    token_bytes=[]
    for i in range(model.config.vocab_size):
        value=special[i].encode() if i in special else bytes(inverse[c] for c in vocab.get(i,''))
        token_bytes.append(base64.b64encode(value).decode())
    (args.output/'token-bytes.json').write_text(json.dumps(token_bytes))
    # Exact fixtures are consumed by independent Swift/Python runtime checks.
    import numpy as np
    for label,value in [('patches',x['pixel_values']),('visual',visual),('logits',logits)]:value.detach().cpu().numpy().astype('<f4').tofile(out/(label+'.f32'))
    profile={'id':name,'imageSizeHW':list(size),'contentSizeHW':list(content),'patchCount':int(x['pixel_values'].shape[0]),'prefixLength':length,'cacheCapacity':args.capacity,'promptIDs':x['input_ids'][0].tolist(),'eosID':2,'encoder':name+'/encoder.aimodel','prefill':name+'/prefill.aimodel','step':step_path.name,'validated':False}
    manifest_path=args.output/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'schemaVersion':1,'revision':REVISION,'profiles':[],'tokenBytes':'token-bytes.json','limitations':['Geometry-specific assets; no automatic resizing','Single pass, greedy decoding only','CPU float32 reference validation; other compute units unverified']}
    manifest['profiles']=[p for p in manifest['profiles'] if p['id']!=name]+[profile]
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Saved UNVALIDATED profile',name,flush=True)
if __name__=='__main__':main()
