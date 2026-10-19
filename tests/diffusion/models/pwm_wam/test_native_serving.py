# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HTTP ownership, error recovery and original client response contract."""

import threading

import pytest
from fastapi.testclient import TestClient

from vllm_omni.diffusion.models.pwm_wam.serving import create_app

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeService:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.info = {"model_name": "action128"}
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def close(self):
        self.closed = True

    def predict(self, request):
        if request.get("invalid"):
            raise ValueError("Invalid action mask")
        if request.get("block"):
            self.entered.set()
            assert self.release.wait(5)
        return {"args": {"name": "test"}, "outputs": [{"content": {"action": [[0.0] * 128]}, "files": []}]}


def test_error_recovery_and_original_envelope(tmp_path):
    service = FakeService(tmp_path)
    with TestClient(create_app(service)) as client:
        assert client.get("/healthz").json() == {"status": "ready"}
        assert client.get("/info").json()["models"] == ["action128"]
        assert client.post("/generate", json={"model": "missing"}).status_code == 404
        assert client.post("/generate", json={"invalid": True}).status_code == 422
        response = client.post("/generate", json={})
        assert response.status_code == 200
        assert len(response.json()["outputs"][0]["content"]["action"][0]) == 128
    assert service.closed


def test_concurrent_request_rejected_without_releasing_owner(tmp_path):
    service = FakeService(tmp_path)
    with TestClient(create_app(service)) as client:
        results = []
        thread = threading.Thread(target=lambda: results.append(client.post("/generate", json={"block": True})))
        thread.start()
        try:
            assert service.entered.wait(5)
            response = client.post("/generate", json={})
            assert response.status_code == 503
            assert response.headers["Retry-After"] == "1"
        finally:
            service.release.set()
            thread.join(5)
        assert results[0].status_code == 200
        assert client.post("/generate", json={}).status_code == 200


def test_output_files_cannot_escape_root(tmp_path):
    service = FakeService(tmp_path)
    (tmp_path / "clip.mp4").write_bytes(b"test-video")
    with TestClient(create_app(service)) as client:
        assert client.get("/outputs/clip.mp4").content == b"test-video"
        assert client.get("/outputs/%2e%2e/config.json").status_code == 404
