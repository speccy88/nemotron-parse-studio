#!/usr/bin/env python3
"""Run the pinned NVIDIA model on a local RGB image; no network inference.

Synthetic validation output may be stored in evidence/. Never point --output at
an evidence directory with confidential input unless you intend to retain it.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, time
from pathlib import Path

REVISION = 'a9a1febb1b9bf497fc67bd1d23fa33e170b249db'
PROMPT = '</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path('Models'))
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='mps')
    p.add_argument('--max-new-tokens',type=int,default=1024)
    args=p.parse_args()
    import torch, transformers
    from PIL import Image
    from transformers import AutoModel,AutoProcessor,GenerationConfig
    args.output.mkdir(parents=True,exist_ok=True)
    report={'revision':REVISION,'model_directory':str(args.model.resolve()),'input':args.image.name,
            'input_sha256':hashlib.sha256(args.image.read_bytes()).hexdigest(),
            'torch':torch.__version__,'transformers':transformers.__version__,
            'device':args.device,'dtype':'float32','macos':platform.mac_ver()[0],
            'core_ai':False,'prompt':PROMPT,'max_new_tokens':args.max_new_tokens}
    start=time.perf_counter()
    try:
        print('Loading pinned reference model...',flush=True)
        model=AutoModel.from_pretrained(args.model.resolve(),trust_remote_code=True,dtype=torch.float32,local_files_only=True).to(args.device).eval()
        processor=AutoProcessor.from_pretrained(args.model.resolve(),trust_remote_code=True,local_files_only=True)
        report['model_load_seconds']=time.perf_counter()-start
        image=Image.open(args.image).convert('RGB')
        inputs=processor(images=[image],text=PROMPT,return_tensors='pt',add_special_tokens=False).to(args.device)
        report['image_size']=[image.width,image.height]
        report['input_shapes']={k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')}
        gen=GenerationConfig.from_pretrained(args.model.resolve(),local_files_only=True)
        gen.max_new_tokens=args.max_new_tokens
        gen.do_sample=False
        gen.num_beams=1
        report['generation_settings']={'do_sample':False,'num_beams':1,'repetition_penalty':gen.repetition_penalty,'max_new_tokens':gen.max_new_tokens,'seed':0}
        torch.manual_seed(0)
        print('Running real '+args.device+' inference...',flush=True)
        started=time.perf_counter()
        with torch.inference_mode(): out=model.generate(**inputs,generation_config=gen)
        if args.device=='mps': torch.mps.synchronize()
        report['inference_seconds']=time.perf_counter()-started
        raw=processor.batch_decode(out,skip_special_tokens=False)[0]
        (args.output/'raw.txt').write_text(raw)
        report['output_token_count']=out.shape[-1]
        report['raw_sha256']=hashlib.sha256(raw.encode()).hexdigest()
        report['completed']=True
        report['at_token_limit']=(out.shape[-1]-inputs['input_ids'].shape[-1])>=args.max_new_tokens
        print(raw,flush=True)
    except Exception as e:
        report['completed']=False;report['error']=repr(e)
        raise
    finally:
        report['total_seconds']=time.perf_counter()-start
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
