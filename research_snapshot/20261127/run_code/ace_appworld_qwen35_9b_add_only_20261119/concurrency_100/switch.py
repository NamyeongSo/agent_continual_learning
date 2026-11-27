from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

from filelock import FileLock
from scripts.ace import schedule_qwen35_9b_add_only as schedule

base = schedule.base
root = Path(sys.argv[1]).resolve()
archive = root / "concurrency_100"
reflection = root / "thinking/reflection"
prior = int(sys.argv[2])
limit = 100
status = archive / "status.json"

def write_status(state, **fields):
    base.write(status, {"state": state, "updated_at": base.now(), "pid": os.getpid(), **fields})

def switch():
    write_status("draining", prior_scheduler_pid=prior, requested_concurrency=limit)
    deadline = time.monotonic() + 7200
    while schedule.pid_alive(prior):
        if time.monotonic() >= deadline:
            raise RuntimeError("Old scheduler did not exit while draining")
        time.sleep(5)
    with FileLock(str(root / "scheduler.lock"), timeout=0), FileLock(str(reflection / "controller.lock"), timeout=0):
        assert (reflection / "STOP").exists()
        assert base.read(reflection / "status.json")["state"] in {"stopped", "completed"}
        counts = {str(i): len(list((reflection / f"stage{i}").glob("*/result.json"))) for i in (1,2,3)}
        for mode in ("thinking", "nonthinking"):
            for phase in ("reflection", "accumulation"):
                target = root / mode / phase
                before = archive / "before" / mode / phase
                before.mkdir(parents=True, exist_ok=True)
                schedule.study.verify_sources(target)
                for name in ("protocol.json", "source_hashes.json"):
                    shutil.copy2(target / name, before / name)
                protocol = base.read(target / "protocol.json")
                if phase == "reflection":
                    relative = "scripts/ace/run_context_proposal_bottleneck.py"
                    frozen = target / "source_snapshot" / relative
                    shutil.copy2(frozen, before / Path(relative).name)
                    shutil.copy2(schedule.study.ROOT / relative, frozen)
                    hashes = base.read(target / "source_hashes.json")
                    hashes[relative] = base.digest(frozen.read_bytes())
                    base.write(target / "source_hashes.json", hashes)
                    for stage in protocol["stages"]:
                        stage["max_workers"] = limit
                else:
                    protocol["evaluation_workers"] = protocol["global_concurrency"] = limit
                    source = Path(protocol["source_reflection_run"]) / "protocol.json"
                    protocol["source_protocol_sha256"] = base.digest(source.read_bytes())
                base.write(target / "protocol.json", protocol)
                schedule.study.verify_sources(target)
        path = root / "plan.json"
        shutil.copy2(path, archive / "previous_plan.json")
        base.write(path, {**base.read(path), "evaluation_concurrency": limit, "reflection_concurrency": limit})
        (reflection / "STOP").unlink()
    command = [sys.executable, "-u", "-m", "scripts.ace.schedule_qwen35_9b_add_only", "--output", str(root)]
    with (root / "scheduler.log").open("a") as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    base.write(root / "scheduler_process.json", {"pid": child.pid, "command": command, "started_at": base.now()})
    write_status("restarted", scheduler_pid=child.pid, preserved_stage_results=counts, concurrency=limit)
    note = archive / "journal_note.txt"
    note.write_text(f"사용자 요청으로 실행 동시성을 50에서 100으로 바꿨습니다. 진행 중인 rollout을 마친 뒤 실행기를 재시작했습니다. "
                    f"보존된 Reflection 결과: {counts}. 두 모드의 Reflection과 모든 평가에 동시성 100을 적용했습니다. "
                    f"이전 프로토콜과 소스 해시는 {archive}에 보존했습니다. 스케줄러 PID {child.pid}.\n")
    helper = Path.home() / ".codex/skills/experiment-journal/scripts/journal.py"
    subprocess.run([sys.executable, str(helper), "log", base.read(root / "journal_record.json")["id"],
                    "--file", str(note)], check=False)

try:
    switch()
except Exception as exc:
    write_status("error", error=str(exc))
    raise
