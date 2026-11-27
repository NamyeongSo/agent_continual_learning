# SPDX-License-Identifier: Apache-2.0
"""Keep explicit hardware resumptions isolated from the historical GPU1 default."""

import io
import json
from pathlib import Path

import pytest

from scripts.ace import run_context_proposal_bottleneck as study


@pytest.mark.parametrize("device", [0, 1])
def test_configure_environment_uses_pinned_device(monkeypatch, device):
    monkeypatch.setattr(study.os, "environ", {})
    study.configure_environment({"base_url": "http://127.0.0.1:8006/v1", "gpu_physical_id": device})
    assert study.os.environ["CUDA_VISIBLE_DEVICES"] == str(device)


def test_unmodified_cohorts_keep_gpu1():
    assert study.physical_gpu({}) == 1


@pytest.mark.parametrize("device", [-1, True, "0", "0,1", None])
def test_invalid_device_is_rejected(device):
    with pytest.raises(ValueError, match="gpu_physical_id"):
        study.physical_gpu({"gpu_physical_id": device})


@pytest.mark.parametrize("pinned,served,valid", [(0, "0", True), (0, "1", False), (1, "0", False)])
def test_endpoint_still_rejects_the_wrong_physical_gpu(tmp_path, monkeypatch, pinned, served, valid):
    process = tmp_path / "proc" / "123"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"vllm\0serve\0test-model\0--port\08006\0")
    (process / "environ").write_bytes(f"CUDA_VISIBLE_DEVICES={served}\0".encode())
    monkeypatch.setattr(study, "Path", lambda path: tmp_path / "proc" if path == "/proc" else Path(path))
    monkeypatch.setattr(study.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(
        json.dumps({"data": [{"id": "test-model"}]}).encode()))
    monkeypatch.setattr(study.subprocess, "check_output", lambda *args, **kwargs: "0, gpu0, H200\n1, gpu1, H200\n")
    protocol = {"base_url": "http://127.0.0.1:8006/v1", "model": "test-model", "gpu_physical_id": pinned}
    if valid:
        assert study.check_endpoint(protocol)["server"]["cuda_visible_devices"] == served
    else:
        with pytest.raises(ValueError, match=f"physical GPU {pinned}"):
            study.check_endpoint(protocol)
