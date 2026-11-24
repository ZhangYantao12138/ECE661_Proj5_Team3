import io
import pytest
from scripts.local_gpu import commands
from src import data


def test_corrupt_download_preserves_existing_archive(tmp_path, monkeypatch):
    archive = tmp_path/'cifar-10-python.tar.gz'
    archive.write_bytes(b'old-corrupt-copy')
    calls = []
    def download(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(b'truncated-response')
    monkeypatch.setattr(data.urllib.request, 'urlopen', download)
    with pytest.raises(RuntimeError, match='3 attempts'):
        data.prepare(tmp_path)
    assert len(calls) == 3
    assert archive.read_bytes() == b'old-corrupt-copy'
    assert not (tmp_path/'cifar-10-batches-py').exists()


def test_final_freezes_before_test_access():
    steps = commands('final')
    freeze = next(i for i, c in enumerate(steps) if 'freeze' in c)
    assert all(i > freeze for i, c in enumerate(steps) if '--final' in c)
    assert len(commands('attack')) == 6
    assert all('--include-strength' in c for c in commands('attack'))
    assert 'scripts.run_training' in commands('train')[0]
