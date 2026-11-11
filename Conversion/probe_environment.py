#!/usr/bin/env python3
"""Emit reproducible environment checks without downloading or running weights."""
import importlib.util,json,platform,subprocess,sys
from pathlib import Path

def command(argv):
    try:
        r=subprocess.run(argv,capture_output=True,text=True,timeout=15)
        return {'returncode':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()}
    except Exception as e: return {'error':repr(e)}

report={
 'macos':platform.mac_ver()[0], 'machine':platform.machine(), 'python':sys.version,
 'xcode':command(['xcodebuild','-version']),
 'coreai_build':command(['xcrun','--find','coreai-build']),
 'core_ai_minimum_macos':'27.0',
 'core_ai_runtime_supported_os':tuple(int(x) for x in (platform.mac_ver()[0] or '0').split('.')) >= (27,0),
 'core_ai_end_to_end_validated':False,
 'coreai_torch_installed':importlib.util.find_spec('coreai_torch') is not None,
 'sources':['https://developer.apple.com/documentation/coreai.md','https://github.com/apple/coreai-models','https://huggingface.co/nvidia/Nemotron-3.5-Parse/tree/a9a1febb1b9bf497fc67bd1d23fa33e170b249db']}
try:
 import torch
 report['torch']=torch.__version__;report['mps_available']=torch.backends.mps.is_available();report['cuda_available']=torch.cuda.is_available()
except ImportError: pass
out=Path(sys.argv[1]) if len(sys.argv)>1 else Path('Conversion/evidence/environment.json')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
