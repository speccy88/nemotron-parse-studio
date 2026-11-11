#!/usr/bin/env python3
"""Real persistent-worker single/multipass, reformat, cancellation and retry test."""
import argparse,base64,json,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'Backend'))
from worker_client import WorkerClient,WorkerError

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['all','single','multipass','cancel'],default='all');args=parser.parse_args()
    out=ROOT/'Conversion/evidence/stdio';out.mkdir(parents=True,exist_ok=True)
    settings={'backend':'transformers','model_directory':str(ROOT/'Models'),'revision':'a9a1febb1b9bf497fc67bd1d23fa33e170b249db','device':'mps','temperature':0,'max_new_tokens':2048,'max_crops':8}
    body={'image_base64':base64.b64encode((ROOT/'Conversion/evidence/pages/mixed.png').read_bytes()).decode(),'width':640,'height':800,'settings':settings,'options':{}}
    reports=[]
    with WorkerClient() as worker:
        pid=worker.request('health')['pid']
        for mode in (['single','double','triple'] if args.scope=='all' else ['single'] if args.scope=='single' else ['double','triple'] if args.scope=='multipass' else []):
            body['settings']['mode']=mode;body['request_id']=str(uuid.uuid4())
            result=worker.request('parse',body)
            (out/(mode+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            assert any(b.get('table') for b in result['blocks']) and 'Apples' in result['raw']
            reprocessed=worker.request('postprocess',{'raw':result['mergedRaw'],'originalRaw':result['raw'],'width':640,'height':800,'options':{'split_chart_value_percent':True,'relocate_chart_captions':True}})
            assert reprocessed['raw']==result['raw']
            reports.append({'mode':mode,'passed':True,'metrics':result['metrics'],'raw_unchanged_after_formatting':True})
            print(mode,'passed',flush=True)
        if args.scope in ('all','multipass'):
            body['settings'].update(mode='single',max_new_tokens=16);body['request_id']=str(uuid.uuid4())
            try:worker.request('parse',body);raise AssertionError('Token limit accepted')
            except WorkerError as error:
                assert error.payload.get('raw') or error.payload.get('stages')
                (out/'token-limit.json').write_text(json.dumps(error.payload,indent=2)+'\n')
                reports.append({'token_limit_retains_partial_output':True})
        if args.scope in ('all','cancel'):
            body['settings'].update(mode='triple',max_new_tokens=2048);body['request_id']=str(uuid.uuid4())
            pending=worker.submit('parse',body)
            # Wait for actual job registration, not an arbitrary model-load delay.
            for _ in range(100):
                if worker.request('health')['activeRequests']:break
                time.sleep(.02)
            start=time.monotonic();assert worker.request('cancel',{'request_id':body['request_id']})['cancelled']
            try:pending.result(60);raise AssertionError('Cancelled parse accepted')
            except WorkerError as error:
                (out/'cancelled.json').write_text(json.dumps(error.payload,indent=2)+'\n')
            delay=time.monotonic()-start
            body.update(request_id=str(uuid.uuid4()),image_base64=base64.b64encode((ROOT/'Conversion/evidence/pages/simple.png').read_bytes()).decode(),width=512,height=256);body['settings']['mode']='single'
            retry=worker.request('parse',body);assert 'Hello Nemotron.' in retry['raw']
            reports.append({'cancelled_then_retry_passed':True,'cancellation_seconds':delay})
        assert worker.request('health')['pid']==pid
        sockets=subprocess.run(['/usr/sbin/lsof','-a','-p',str(pid),'-i','-Fn'],capture_output=True,text=True)
        assert not sockets.stdout
        reports.append({'persistent_pid':pid,'worker_socket_count':0,'transport':'private stdin/stdout pipes'})
    (out/(args.scope+'-report.json')).write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
