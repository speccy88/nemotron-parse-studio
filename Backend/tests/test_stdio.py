import json,os,subprocess,sys,time
from pathlib import Path
from worker_client import WorkerClient,WorkerError
import pytest

ROOT=Path(__file__).resolve().parents[2]
def test_persistent_private_stdio_and_no_sockets():
    with WorkerClient() as worker:
        health=worker.request('health')
        assert health['transport']=='stdio' and health['pid']==worker.process.pid
        futures=[worker.submit('postprocess',{'raw':f'<x_0><y_0>Page {i}<x_1><y_1><class_Text>','width':100,'height':100}) for i in range(6)]
        for i,f in enumerate(futures):assert f.result(10)['blocks'][0]['text']==f'Page {i}'
        assert worker.request('health')['pid']==health['pid']
        sockets=subprocess.run(['/usr/sbin/lsof','-a','-p',str(worker.process.pid),'-i','-Fn'],capture_output=True,text=True)
        assert not sockets.stdout
        with pytest.raises(WorkerError,match='Unknown worker method'):worker.request('invalid')
        assert worker.request('health')['status']=='ok'
    assert worker.process.returncode==0

def test_malformed_frame_recovers_and_stdout_is_only_protocol():
    process=subprocess.Popen([sys.executable,str(ROOT/'Backend/studio_worker.py'),'--stdio'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    stdout,stderr=process.communicate('not json\n'+json.dumps({'id':'recovery','method':'health','params':{}})+'\n',timeout=10)
    replies=[json.loads(line) for line in stdout.splitlines()]
    assert len(replies)==2 and 'error' in replies[0]
    assert replies[1]['id']=='recovery' and replies[1]['result']['transport']=='stdio'
    assert process.returncode==0

def test_listening_port_arguments_removed():
    result=subprocess.run([sys.executable,str(ROOT/'Backend/studio_worker.py'),'--host','127.0.0.1','--port','8765'],capture_output=True,text=True,timeout=10)
    assert result.returncode!=0

def test_cancel_before_job_dispatch_is_not_lost():
    import studio_worker as service
    engine=service.Engine()
    assert engine.cancel('early')['cancelled']
    with pytest.raises(service.Cancelled,match='before inference'):
        engine.parse({'request_id':'early'})
    assert not engine.jobs and not engine.cancelled_before_start
