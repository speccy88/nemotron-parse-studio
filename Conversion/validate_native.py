#!/usr/bin/env python3
"""Promote a Core AI profile only after native Swift and pinned PyTorch agree.

Numerical gates: input patches bit-exact; visual relative L2 <1e-3; prefill logits
absolute error <1e-3. Additionally require byte-exact generated serialization,
which checks decoder/cache/tokenizer/coordinates/tables jointly. Tests are small
functional fixtures, not a claim of corpus-wide model accuracy.
"""
import base64,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Backend'))
from worker_client import WorkerClient
REV='a9a1febb1b9bf497fc67bd1d23fa33e170b249db'
def stats(a,b):
    delta=np.abs(a-b)
    return {'max_absolute_error':float(delta.max()),'mean_absolute_error':float(delta.mean()),'relative_l2':float(np.linalg.norm(a-b)/max(1e-12,np.linalg.norm(b)))}
def main():
    manifest_path=ROOT/'NativeModels/manifest.json';manifest=json.loads(manifest_path.read_text())
    reports=[]
    with WorkerClient() as worker:
        for profile,page in [('512x256','simple'),('640x800','mixed')]:
            run=ROOT/f'Conversion/evidence/native-swift-{page}'
            native=json.loads((run/'prediction.json').read_text())
            config=next(p for p in manifest['profiles'] if p['id']==profile)
            image=ROOT/f'Conversion/evidence/pages/{page}.png'
            response=worker.request('parse',{'request_id':'native-reference-'+page,'image_base64':base64.b64encode(image.read_bytes()).decode(),'width':config['contentSizeHW'][1],'height':config['contentSizeHW'][0], 'settings':{'backend':'transformers','model_directory':str(ROOT/'Models'),'revision':REV,'device':'cpu','mode':'single','temperature':0,'repetition_penalty':1.05,'max_new_tokens':256},'options':{}},timeout=300)
            (run/'reference.json').write_text(json.dumps(response,ensure_ascii=False,indent=2)+'\n')
            values={name:stats(np.fromfile(run/(name+'.f32'),dtype='<f4'),np.fromfile(ROOT/'NativeModels'/profile/(name+'.f32'),dtype='<f4')) for name in ['patches','visual','logits']}
            passed=values['patches']['max_absolute_error']==0 and values['visual']['relative_l2']<1e-3 and values['logits']['max_absolute_error']<1e-3 and native['raw']==response['raw']
            report={'profile':profile,'passed':passed,'native_runtime':'Swift Core AI, CPU float32, macOS27','native_raw_equals_reference':native['raw']==response['raw'],'numerical_comparison':values,'native_metrics':native['metrics'],'reference_metrics':response['metrics'],'raw_sha256':hashlib.sha256(native['raw'].encode()).hexdigest(),'revision':REV,'reference_transport':'private stdin/stdout pipes','image_sha256':hashlib.sha256(image.read_bytes()).hexdigest()}
            (run/'validation.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
            config['validated']=passed
            print(json.dumps(report,indent=2),flush=True)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'Conversion/evidence/native-validation.json').write_text(json.dumps(reports,indent=2)+'\n')
    if not all(r['passed'] for r in reports): raise SystemExit(1)
if __name__=='__main__':main()
