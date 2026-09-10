"""Read selected keys from a pinned, little-endian 64-bit LMDB using byte ranges.

Layout follows LMDB 0.9.35 mdb.c. This is deliberately not a general LMDB driver:
only 4096/16384-byte pages, plain byte keys, branch/leaf pages and overflow values.
"""

import bisect
import struct
import time
from functools import lru_cache
from urllib.request import Request, urlopen


class HTTPRanges:
    def __init__(self, url, size, *, attempts=5):
        self.url, self.size, self.attempts = url, size, attempts

    def read(self, start, length):
        if start < 0 or length < 1 or start + length > self.size:
            raise ValueError("Range outside source file")
        expected = f"bytes {start}-{start + length - 1}/{self.size}"
        for attempt in range(self.attempts):
            try:
                req = Request(
                    self.url,
                    headers={
                        "Range": f"bytes={start}-{start + length - 1}",
                        "User-Agent": "SatQuery-subset/1.0",
                        "Accept-Encoding": "identity",
                    },
                )
                with urlopen(req, timeout=45) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not return the exact requested byte range")
                    data = response.read(length + 1)
                if len(data) != length:
                    raise OSError("Incomplete range response")
                return data
            except (OSError, ValueError):
                if attempt == self.attempts - 1:
                    raise
                time.sleep(min(2**attempt, 16))


class LMDBReader:
    def __init__(self, read, size):
        self.read, self.size = read, size
        header = self.read(0, 4096)
        if len(header) != 4096:
            raise ValueError("Incomplete LMDB header")
        self.page_size = struct.unpack_from("<I", header, 40)[0]
        if self.page_size not in (4096, 16384):
            raise ValueError("Unsupported LMDB page size")
        raw = self.read(0, 2 * self.page_size)
        if len(raw) != 2 * self.page_size:
            raise ValueError("Incomplete LMDB metadata")
        versions = []
        for offset in [0, self.page_size]:
            page = raw[offset : offset + self.page_size]
            if (
                struct.unpack_from("<II", page, 16) != (0xBEEFC0DE, 1)
                or struct.unpack_from("<H", page, 10)[0] != 8
                or struct.unpack_from("<I", page, 40)[0] != self.page_size
                or struct.unpack_from("<H", page, 92)[0] != 0
            ):
                raise ValueError("Unsupported LMDB layout or key flags")
            versions.append(
                (struct.unpack_from("<Q", page, 144)[0], struct.unpack_from("<Q", page, 128)[0])
            )
        self.transaction, self.root = max(versions)
        self.page = lru_cache(maxsize=4096)(self._page)

    def _page(self, number):
        if number < 2 or (number + 1) * self.page_size > self.size:
            raise ValueError("Invalid LMDB page number")
        value = self.read(number * self.page_size, self.page_size)
        if len(value) != self.page_size or struct.unpack_from("<Q", value)[0] != number:
            raise ValueError("Invalid LMDB page header")
        return value

    def get(self, key):
        key = key.encode() if isinstance(key, str) else key
        number = self.root
        for _ in range(20):
            page = self.page(number)
            flags, lower, upper = struct.unpack_from("<HHH", page, 10)
            if flags not in (1, 2) or not (16 <= lower <= upper <= self.page_size) or lower % 2:
                raise ValueError("Unsupported or malformed LMDB page")
            nodes = []
            for i in range((lower - 16) // 2):
                offset = struct.unpack_from("<H", page, 16 + 2 * i)[0]
                if not upper <= offset <= self.page_size - 8:
                    raise ValueError("Invalid LMDB node offset")
                lo, hi, node_flags, keysize = struct.unpack_from("<HHHH", page, offset)
                end = offset + 8 + keysize
                if end > self.page_size:
                    raise ValueError("Invalid LMDB key length")
                nodes.append((page[offset + 8 : end], lo, hi, node_flags, end))
            keys = [node[0] for node in nodes]
            if keys != sorted(keys):
                raise ValueError("Unordered LMDB keys")
            index = bisect.bisect_right(keys, key) - 1
            if index < 0:
                raise KeyError(key)
            found, lo, hi, node_flags, end = nodes[index]
            if flags == 1:
                number = lo + (hi << 16) + (node_flags << 32)
                continue
            if found != key:
                raise KeyError(key)
            length = lo + (hi << 16)
            if length > 2 * 1024 * 1024:
                raise ValueError("Record exceeds supported size")
            if node_flags == 0:
                if end + length > self.page_size:
                    raise ValueError("Invalid inline record size")
                return page[end : end + length]
            if node_flags != 1 or end + 8 > self.page_size:
                raise ValueError("Unsupported LMDB node flags")
            overflow = struct.unpack_from("<Q", page, end)[0]
            header = self.page(overflow)
            if struct.unpack_from("<H", header, 10)[0] != 4:
                raise ValueError("Expected overflow page")
            pages = struct.unpack_from("<I", header, 12)[0]
            if (
                length > pages * self.page_size - 16
                or (overflow + pages) * self.page_size > self.size
            ):
                raise ValueError("Invalid overflow record extent")
            value = self.read(overflow * self.page_size + 16, length)
            if len(value) != length:
                raise ValueError("Incomplete LMDB record")
            return value
        raise ValueError("LMDB tree depth exceeded")
