import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from satquery.official_parts import PartsReader, SelectedTIFFCheckpoints, extract_selected_tar


def parts(tmp_path, data):
    for i, offset in enumerate(range(0, len(data), 31)):
        block = data[offset:offset+31]
        (tmp_path/f'{i:05d}.part').write_bytes(block)
        (tmp_path/f'{i:05d}.sha256').write_text(hashlib.sha256(block).hexdigest())
    (tmp_path/'VERIFIED.json').write_text(json.dumps({'size':len(data),'md5':hashlib.md5(data).hexdigest()}))


def test_cross_part_reads_and_corruption(tmp_path):
    data=bytes(range(255))*3
    parts(tmp_path,data)
    with PartsReader(tmp_path) as source:
        assert source.read(43)==data[:43]
        assert source.read()==data[43:]
    (tmp_path/'00001.part').write_bytes(b'x'*31)
    with pytest.raises(ValueError,match='checksum'), PartsReader(tmp_path) as source:
        source.read()


def test_selective_tar_uses_known_destination_not_archive_path(tmp_path):
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as tar:
        for name in ['../../wanted.tif','unwanted.tif']:
            info=tarfile.TarInfo(name);info.size=3
            tar.addfile(info,io.BytesIO(b'abc'))
    out=tmp_path/'out'
    extract_selected_tar(io.BytesIO(buf.getvalue()),{'wanted.tif':'sensor/wanted.tif'},out)
    assert (out/'sensor/wanted.tif').read_bytes()==b'abc'
    assert len(list(out.rglob('*.tif')))==1


def test_missing_band_rejected(tmp_path):
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w'):pass
    with pytest.raises(ValueError,match='Missing'):
        extract_selected_tar(io.BytesIO(buf.getvalue()),{'missing.tif':'missing.tif'},tmp_path/'out')


def test_drive_part_reads_are_bounded(tmp_path, monkeypatch):
    data = b'a' * (2 * 1024 * 1024 + 17)
    (tmp_path / '00000.part').write_bytes(data)
    (tmp_path / '00000.sha256').write_text(hashlib.sha256(data).hexdigest())
    (tmp_path / 'VERIFIED.json').write_text(json.dumps({
        'size': len(data), 'md5': hashlib.md5(data).hexdigest(),
    }))
    original_open = Path.open
    requests = []

    class BoundedFile:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def read(self, size=-1):
            requests.append(size)
            assert 0 < size <= 1024 * 1024, 'Unbounded Drive read'
            return self.file.read(size)

    def open_checked(path, *args, **kwargs):
        file = original_open(path, *args, **kwargs)
        return BoundedFile(file) if path.suffix == '.part' else file

    monkeypatch.setattr(Path, 'open', open_checked)
    with PartsReader(tmp_path) as source:
        assert source.read() == data
        assert source.finished
    assert len(requests) >= 3


def test_selected_tiffs_restore_from_verified_batches(tmp_path):
    output = tmp_path / 'raw'
    wanted = {
        'a.tif': 'sensor/a.tif',
        'b.tif': 'sensor/b.tif',
        'c.tif': 'sensor/c.tif',
    }
    for name, relative in wanted.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    cache = SelectedTIFFCheckpoints(tmp_path / 'checkpoints', output, wanted,
                                    identity='selection-one', batch_size=2)
    cache.record({'a.tif', 'b.tif'})
    cache.record({'a.tif', 'b.tif', 'c.tif'}, final=True)
    assert len(list((tmp_path / 'checkpoints').glob('*.zip'))) == 2

    for path in output.rglob('*.tif'):
        path.unlink()
    restored = SelectedTIFFCheckpoints(
        tmp_path / 'checkpoints', output, wanted,
        identity='selection-one', batch_size=2,
    ).restore()
    assert restored == set(wanted)
    assert (output / 'sensor/a.tif').read_bytes() == b'a.tif'


def test_selected_tiff_checkpoint_identity_mismatch_is_rejected(tmp_path):
    output = tmp_path / 'raw'
    wanted = {'a.tif': 'sensor/a.tif'}
    SelectedTIFFCheckpoints(tmp_path / 'checkpoints', output, wanted,
                            identity='selection-one')
    with pytest.raises(ValueError, match='identity'):
        SelectedTIFFCheckpoints(tmp_path / 'checkpoints', output, wanted,
                                identity='selection-two')
