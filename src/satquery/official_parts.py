"""Read verified split official archives without joining them on local disk."""

import hashlib
import io
import json
import tarfile
import tempfile
import zipfile
from pathlib import Path


class PartsReader(io.RawIOBase):
    """Sequential stream; verify each downloaded part before exposing its bytes."""

    def __init__(self, folder, progress=None):
        super().__init__()
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / 'VERIFIED.json').read_text())
        self.paths = sorted(self.folder.glob('*.part'))
        if not self.paths or [p.name for p in self.paths] != [
            f'{i:05d}.part' for i in range(len(self.paths))
        ]:
            raise ValueError('Missing or unordered archive parts')
        if sum(p.stat().st_size for p in self.paths) != self.manifest['size']:
            raise ValueError('Archive size mismatch')
        self.index = 0
        self.buffer = io.BytesIO()
        self.digest = hashlib.md5()
        self.finished = False
        self.progress = progress

    def readable(self):
        return True

    def readinto(self, target):
        data = self.read(len(target))
        target[:len(data)] = data
        return len(data)

    def read(self, size=-1):
        if size == 0:
            return b''
        result = []
        remaining = size
        while remaining != 0:
            block = self.buffer.read(remaining)
            if block:
                result.append(block)
                if remaining > 0:
                    remaining -= len(block)
                continue
            if self.index == len(self.paths):
                if not self.finished:
                    if self.digest.hexdigest() != self.manifest['md5']:
                        raise ValueError('Official archive checksum mismatch')
                    self.finished = True
                break
            path = self.paths[self.index]
            expected = path.with_suffix('.sha256').read_text().strip()
            # Large single reads can disconnect Colab's Drive filesystem.
            # Verify bounded reads in a local spool before exposing any bytes.
            spool = tempfile.TemporaryFile()  # noqa: SIM115 — owned by this reader until consumed
            part_digest = hashlib.sha256()
            try:
                with path.open('rb') as source:
                    while block := source.read(1024 * 1024):
                        spool.write(block)
                        part_digest.update(block)
                        self.digest.update(block)
                if part_digest.hexdigest() != expected:
                    raise ValueError(f'Part checksum mismatch: {path.name}')
                spool.seek(0)
            except BaseException:
                spool.close()
                raise
            self.buffer.close()
            self.buffer = spool
            self.index += 1
            if self.progress:
                self.progress(f'{self.folder.name}: verified part {self.index}/{len(self.paths)}')
        return b''.join(result)

    def close(self):
        self.buffer.close()
        super().close()


class SelectedTIFFCheckpoints:
    """Persist verified batches of selected TIFFs and restore them after restart."""

    def __init__(self, folder, output, wanted, *, identity, batch_size=1000):
        self.folder = Path(folder)
        self.output = Path(output).resolve()
        self.wanted = dict(wanted)
        self.batch_size = batch_size
        self.saved = set()
        self.folder.mkdir(parents=True, exist_ok=True)
        expected = {
            'identity': identity,
            'batch_size': batch_size,
            'wanted_sha256': hashlib.sha256(json.dumps(
                sorted(self.wanted.items()), separators=(',', ':'),
            ).encode()).hexdigest(),
        }
        manifest = self.folder / 'MANIFEST.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != expected:
                raise ValueError('Selected TIFF checkpoint identity mismatch')
        else:
            if any(self.folder.iterdir()):
                raise ValueError('Selected TIFF checkpoints lack an identity manifest')
            pending = manifest.with_suffix('.partial')
            pending.write_text(json.dumps(expected, sort_keys=True))
            pending.replace(manifest)

    def restore(self):
        for receipt in sorted(self.folder.glob('batch-*.json')):
            info = json.loads(receipt.read_text())
            archive = receipt.with_suffix('.zip')
            if not archive.exists() or self._sha256(archive) != info['sha256']:
                raise ValueError(f'Checkpoint checksum mismatch: {archive.name}')
            names = set(info['names'])
            if names & self.saved or not names <= set(self.wanted):
                raise ValueError(f'Invalid checkpoint contents: {archive.name}')
            expected_paths = {self.wanted[name] for name in names}
            with zipfile.ZipFile(archive) as package:
                if set(package.namelist()) != expected_paths:
                    raise ValueError(f'Checkpoint paths mismatch: {archive.name}')
                for relative in sorted(expected_paths):
                    target = (self.output / relative).resolve()
                    if self.output not in target.parents:
                        raise ValueError('Unsafe checkpoint destination')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with package.open(relative) as source, target.open('wb') as dest:
                        while block := source.read(1024 * 1024):
                            dest.write(block)
            self.saved.update(names)
        return set(self.saved)

    def record(self, seen, *, final=False):
        available = sorted(set(seen) - self.saved)
        while len(available) >= self.batch_size or (final and available):
            names = available[:self.batch_size]
            index = len(list(self.folder.glob('batch-*.json')))
            archive = self.folder / f'batch-{index:05d}.zip'
            pending = archive.with_suffix('.partial')
            with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as package:
                for name in names:
                    path = self.output / self.wanted[name]
                    if not path.is_file():
                        raise ValueError(f'Missing TIFF for checkpoint: {name}')
                    package.write(path, self.wanted[name])
            pending.replace(archive)
            receipt = archive.with_suffix('.json')
            receipt_pending = receipt.with_suffix('.partial')
            receipt_pending.write_text(json.dumps({
                'sha256': self._sha256(archive), 'names': names,
            }))
            receipt_pending.replace(receipt)
            self.saved.update(names)
            available = available[self.batch_size:]
        return set(self.saved)

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()


def extract_selected_tar(stream, wanted, output, progress=None, on_selected=None):
    """Copy selected regular TIFFs to trusted paths, never archive-provided paths."""
    output = Path(output).resolve()
    seen = set()
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            name = Path(member.name).name
            if name not in wanted:
                continue
            if name in seen or not member.isfile() or member.size > 2*1024*1024:
                raise ValueError(f'Invalid or duplicate TIFF: {name}')
            target = (output / wanted[name]).resolve()
            if output not in target.parents:
                raise ValueError('Unsafe destination')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as dest:
                while block := source.read(1024*1024):
                    dest.write(block)
            seen.add(name)
            if on_selected:
                on_selected(seen)
            if progress and len(seen) % 1000 == 0:
                progress(f'Extracted {len(seen)}/{len(wanted)} selected TIFFs')
    if seen != set(wanted):
        raise ValueError(f'Missing {len(set(wanted)-seen)} selected TIFFs')
    return seen
