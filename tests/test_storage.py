"""Storage-level robustness tests."""
import shutil
import tempfile

import torch

from disk_offload import DiskTensorStore


def test_write_retries_past_transient_permission_error(monkeypatch):
    """On Windows, os.replace() can transiently fail with PermissionError if
    a background prefetch thread still has the destination file open for
    reading. Writes must retry instead of crashing training."""
    store = DiskTensorStore(tempfile.mkdtemp(prefix="offload_storage_test_"))
    try:
        calls = {"count": 0}
        real_replace = __import__("os").replace

        def flaky_replace(src, dst):
            calls["count"] += 1
            if calls["count"] < 3:
                raise PermissionError("simulated transient lock")
            real_replace(src, dst)

        monkeypatch.setattr("os.replace", flaky_replace)
        monkeypatch.setattr("disk_offload.storage.time.sleep", lambda _: None)

        store.save("w", torch.ones(3))
        assert calls["count"] == 3
        assert torch.allclose(store.load("w"), torch.ones(3))
    finally:
        shutil.rmtree(store.root, ignore_errors=True)
