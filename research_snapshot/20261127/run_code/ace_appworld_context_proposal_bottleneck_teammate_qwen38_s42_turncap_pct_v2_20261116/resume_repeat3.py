import json,os,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
old_pid=json.loads((r/'controller.json').read_text())['pid']
while Path(f'/proc/{old_pid}').exists():
    time.sleep(5)
(r/'STOP').unlink(missing_ok=True)
command=json.loads((r/'launch_command.json').read_text())['command']
with (r/'controller.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(command,cwd=json.loads((r/'protocol.json').read_text())['source_worktree'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1'))
(r/'repeat3_resume.json').write_text(json.dumps({'pid':p.pid,'previous_pid':old_pid,'stage3_repeats':3,'stage3_jobs':1500},indent=2))
