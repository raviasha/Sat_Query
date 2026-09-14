import hashlib

import pytest

from satquery.archive_http import HTTPArchiveReader


def test_stream_reads_bounded_ranges_and_verifies_digest():
    data = b'abcdefghijklm'
    requests = []

    def fetch(start, length):
        requests.append((start, length))
        return data[start:start + length]

    with HTTPArchiveReader('unused', len(data), hashlib.md5(data).hexdigest(),
                           chunk_size=4, fetch=fetch) as stream:
        assert stream.read(3) == data[:3]
        target = bytearray(5)
        assert stream.readinto(target) == 5
        assert bytes(target) == data[3:8]
        assert stream.read() == data[8:]
        assert stream.finished
        assert stream.read() == b''
    assert requests == [(0, 4), (4, 4), (8, 4), (12, 1)]


def test_corrupt_archive_is_rejected():
    with HTTPArchiveReader('unused', 3, hashlib.md5(b'abc').hexdigest(),
                           fetch=lambda start, length: b'bad') as stream:
        with pytest.raises(ValueError, match='checksum'):
            stream.read()
        assert not stream.finished


def test_short_range_is_rejected_without_advancing():
    with HTTPArchiveReader('unused', 3, hashlib.md5(b'abc').hexdigest(),
                           fetch=lambda start, length: b'a') as stream:
        with pytest.raises(OSError, match='Incomplete'):
            stream.read()
        assert stream.position == 0


def test_cached_ranges_survive_interruption_and_avoid_repeat_downloads(tmp_path):
    data = b'abcdefghijklm'
    first_requests = []

    def first_fetch(start, length):
        first_requests.append((start, length))
        return data[start:start + length]

    with HTTPArchiveReader(
        'source-a', len(data), hashlib.md5(data).hexdigest(), chunk_size=4,
        fetch=first_fetch, cache_folder=tmp_path,
    ) as stream:
        assert stream.read(5) == data[:5]
    assert first_requests == [(0, 4), (4, 4)]

    resumed_requests = []

    def resumed_fetch(start, length):
        resumed_requests.append((start, length))
        return data[start:start + length]

    with HTTPArchiveReader(
        'source-a', len(data), hashlib.md5(data).hexdigest(), chunk_size=4,
        fetch=resumed_fetch, cache_folder=tmp_path,
    ) as stream:
        assert stream.read() == data
        assert stream.finished
    assert resumed_requests == [(8, 4), (12, 1)]


def test_cache_identity_mismatch_is_rejected(tmp_path):
    data = b'abcdefgh'
    with HTTPArchiveReader(
        'source-a', len(data), hashlib.md5(data).hexdigest(), chunk_size=4,
        fetch=lambda start, length: data[start:start + length],
        cache_folder=tmp_path,
    ) as stream:
        assert stream.read(1) == b'a'

    with pytest.raises(ValueError, match='identity'):
        HTTPArchiveReader(
            'source-b', len(data), hashlib.md5(data).hexdigest(), chunk_size=4,
            fetch=lambda start, length: data[start:start + length],
            cache_folder=tmp_path,
        )
