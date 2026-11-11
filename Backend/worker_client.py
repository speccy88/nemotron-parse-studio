"""Private-pipe client for integration tests and local automation (no sockets)."""
from concurrent.futures import Future
import json,subprocess,sys,threading,uuid
from pathlib import Path
class WorkerError(RuntimeError):
    def __init__(self,payload):self.payload=payload;super().__init__(payload.get('message','Worker failed'))
class WorkerClient:
    def __init__(self,python=sys.executable):
        self.process=subprocess.Popen([python,str(Path(__file__).with_name('studio_worker.py')),'--stdio'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
        self.pending={};self.lock=threading.Lock();self.writer=threading.Lock()
        self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start()
    def _read(self):
        try:
            for line in self.process.stdout:
                value=json.loads(line)
                with self.lock:future=self.pending.pop(value['id'],None)
                if future:
                    if 'error' in value:future.set_exception(WorkerError(value['error']))
                    else:future.set_result(value['result'])
        finally:
            with self.lock:
                for f in self.pending.values():f.set_exception(RuntimeError('Worker exited'))
                self.pending.clear()
    def submit(self,method,params=None):
        identity=str(uuid.uuid4());future=Future()
        with self.lock:self.pending[identity]=future
        with self.writer:self.process.stdin.write(json.dumps({'id':identity,'method':method,'params':params or {}})+'\n');self.process.stdin.flush()
        return future
    def request(self,method,params=None,timeout=300):return self.submit(method,params).result(timeout)
    def close(self):
        self.process.stdin.close()
        try:self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=5)
        self.reader.join(timeout=2)
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
