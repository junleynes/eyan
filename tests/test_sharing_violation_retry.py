"""
Tests for fetch_network_file()'s retry on SharingViolation -- a real,
observed production error: NtStatus 0xC0000043, "the process cannot access
the file because it is being used by another process". This share's own
name included "raysync" (a file-sync tool), which commonly holds a brief
lock on a file while actively writing/syncing it -- typically for a few
seconds, not indefinitely. This status code specifically means the file
was genuinely found (unlike a not-found or permissions error), so a short
retry is a real fix for the common case, not a blind hope.
"""
import unittest.mock as mock
import io

import pytest

from smbprotocol.exceptions import SharingViolation

import pipeline


def _fake_env(open_file_side_effect):
    """Common patches every test here needs -- an SMB layer standing in for
    a real share, with sleep mocked out so tests run instantly rather than
    waiting through real retry delays."""
    return (
        mock.patch('pipeline.smbclient.open_file', side_effect=open_file_side_effect),
        mock.patch('pipeline.smbclient.register_session'),
        mock.patch('pipeline._network_share_root', return_value='\\\\server\\share'),
        mock.patch('pipeline.time.sleep'),
    )


def test_succeeds_after_a_transient_sharing_violation(tmp_path):
    call_count = [0]
    def flaky(path, mode=None):
        call_count[0] += 1
        if call_count[0] == 1:
            raise SharingViolation(mock.MagicMock())
        return io.BytesIO(b'fake file content')

    patches = _fake_env(flaky)
    with patches[0], patches[1], patches[2], patches[3], \
         mock.patch('pipeline.app.config', {'UPLOAD_FOLDER': str(tmp_path)}):
        result = pipeline.fetch_network_file('test.wav', category='music')
    assert call_count[0] == 2
    assert (tmp_path / result).exists()


def test_gives_up_after_repeated_sharing_violations_with_a_clear_message(tmp_path):
    call_count = [0]
    def always_locked(path, mode=None):
        call_count[0] += 1
        raise SharingViolation(mock.MagicMock())

    patches = _fake_env(always_locked)
    with patches[0], patches[1], patches[2], patches[3] as mock_sleep, \
         mock.patch('pipeline.app.config', {'UPLOAD_FOLDER': str(tmp_path)}):
        with pytest.raises(ValueError, match='currently in use'):
            pipeline.fetch_network_file('test.wav', category='music')
    assert call_count[0] == 4  # 1 initial + 3 retries
    assert mock_sleep.call_count == 3


def test_a_broken_exception_message_does_not_crash_error_reporting(tmp_path):
    # The exact failure mode discovered while building this: some
    # smbprotocol exceptions build their own message lazily from the raw
    # SMB response header, which can itself fail to parse (a MagicMock
    # stand-in for a header is exactly this case). The final error must
    # still be raised cleanly, not replaced by a crash while formatting it.
    def always_locked(path, mode=None):
        raise SharingViolation(mock.MagicMock())

    patches = _fake_env(always_locked)
    with patches[0], patches[1], patches[2], patches[3], \
         mock.patch('pipeline.app.config', {'UPLOAD_FOLDER': str(tmp_path)}):
        with pytest.raises(ValueError, match='currently in use'):
            pipeline.fetch_network_file('test.wav', category='music')


def test_unrelated_exceptions_are_not_retried_at_all(tmp_path):
    call_count = [0]
    def not_found(path, mode=None):
        call_count[0] += 1
        raise Exception('STATUS_OBJECT_NAME_NOT_FOUND')

    patches = _fake_env(not_found)
    with patches[0], patches[1], patches[2], patches[3] as mock_sleep, \
         mock.patch('pipeline.app.config', {'UPLOAD_FOLDER': str(tmp_path)}):
        with pytest.raises(Exception, match='STATUS_OBJECT_NAME_NOT_FOUND'):
            pipeline.fetch_network_file('missing.wav', category='music')
    assert call_count[0] == 1  # no retry
    assert mock_sleep.call_count == 0


def test_partial_local_file_is_cleaned_up_between_retries(tmp_path):
    # A partial copy from a failed attempt must not linger and get appended
    # to (or confused with) a later successful attempt's own fresh copy.
    call_count = [0]
    def flaky_partial_write(path, mode=None):
        call_count[0] += 1
        if call_count[0] == 1:
            raise SharingViolation(mock.MagicMock())
        return io.BytesIO(b'the real, complete file')

    patches = _fake_env(flaky_partial_write)
    with patches[0], patches[1], patches[2], patches[3], \
         mock.patch('pipeline.app.config', {'UPLOAD_FOLDER': str(tmp_path)}):
        result = pipeline.fetch_network_file('test.wav', category='music')
    content = (tmp_path / result).read_bytes()
    assert content == b'the real, complete file'


def test_safe_exception_text_falls_back_when_str_raises():
    class BrokenException(Exception):
        def __str__(self):
            raise RuntimeError('cannot format this')

    result = pipeline._safe_exception_text(BrokenException())
    assert result == 'BrokenException'


def test_safe_exception_text_normal_case():
    assert pipeline._safe_exception_text(ValueError('a normal message')) == 'a normal message'
