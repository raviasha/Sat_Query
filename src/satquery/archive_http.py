"""Sequential official archive stream with bounded HTTP ranges and final checksum."""

import hashlib
import io
import json
from pathlib import Path

from .remote_lmdb import HTTPRanges


class HTTPArchiveReader(io.RawIOBase):
    """Network transport fallback; callers must drain to EOF to verify the MD5."""

    def __init__(self, url, size, md5, *, chunk_size=8 * 1024 * 1024,
                 fetch=None, progress=None, cache_folder=None):
        super().__init__()
        if size <= 0 or chunk_size <= 0:
            raise ValueError('Archive and chunk sizes must be positive')
        self.size = size
        self.expected = md5
        self.chunk_size = chunk_size
        self.fetch = fetch or HTTPRanges(url, size).read
        self.progress = progress
        self.url = url
        self.cache_folder = Path(cache_folder) if cache_folder is not None else None
        if self.cache_folder is not None:
            self._prepare_cache()
        self.position = 0
        self.buffer = io.BytesIO()
        self.digest = hashlib.md5()
        self.finished = False

    def _prepare_cache(self):
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        identity = {
            'url': self.url,
            'size': self.size,
            'md5': self.expected,
            'chunk_size': self.chunk_size,
        }
        manifest = self.cache_folder / 'MANIFEST.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError('Archive cache identity mismatch')
            return
        if any(self.cache_folder.iterdir()):
            raise ValueError('Archive cache has chunks but no identity manifest')
        pending = manifest.with_suffix('.partial')
        pending.write_text(json.dumps(identity, sort_keys=True))
        pending.replace(manifest)

    def _load_or_fetch(self, start, length):
        if self.cache_folder is None:
            return self.fetch(start, length)
        index = start // self.chunk_size
        part = self.cache_folder / f'{index:05d}.part'
        receipt = part.with_suffix('.sha256')
        if part.exists() and receipt.exists():
            block = part.read_bytes()
            if len(block) != length or hashlib.sha256(block).hexdigest() != receipt.read_text().strip():
                raise ValueError(f'Archive cache checksum mismatch: {part.name}')
            return block
        block = self.fetch(start, length)
        if len(block) != length:
            raise OSError('Incomplete archive range')
        digest = hashlib.sha256(block).hexdigest()
        pending_part = part.with_suffix('.partial')
        pending_receipt = receipt.with_suffix('.partial')
        pending_part.write_bytes(block)
        pending_part.replace(part)
        pending_receipt.write_text(digest)
        pending_receipt.replace(receipt)
        return block

    def readable(self):
        return True

    def readinto(self, target):
        value = self.read(len(target))
        target[:len(value)] = value
        return len(value)

    def read(self, size=-1):
        if self.closed:
            raise ValueError('Read from closed archive')
        chunks = []
        remaining = size
        while remaining != 0:
            block = self.buffer.read(remaining)
            if block:
                chunks.append(block)
                if remaining > 0:
                    remaining -= len(block)
                continue
            if self.position == self.size:
                if self.digest.hexdigest() != self.expected:
                    raise ValueError('Official archive checksum mismatch')
                self.finished = True
                break
            length = min(self.chunk_size, self.size - self.position)
            block = self._load_or_fetch(self.position, length)
            if len(block) != length:
                raise OSError('Incomplete archive range')
            self.digest.update(block)
            self.position += length
            self.buffer.close()
            self.buffer = io.BytesIO(block)
            if self.progress and (self.position % (256 * 1024 * 1024) == 0
                                  or self.position == self.size):
                self.progress(f'Official HTTP stream: {self.position}/{self.size} bytes')
        return b''.join(chunks)

    def close(self):
        self.buffer.close()
        super().close()
