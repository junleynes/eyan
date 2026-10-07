"""
A transcription that FAILED must not be reported as a file with no speech.

Real, reported case: Speech to Text answered "No speech was transcribed.
Check that the whisper service ... is reachable ... and that the file
actually contains audio", and Vertical Shorts quietly fell back to picking
moments on picture alone -- for an episode full of dialogue, on a server
whose whisper service was up. transcribe_video() returned ([], []) for every
kind of trouble (service refused, HTTP 500, timed out, sent `segments: null`,
audio could not be extracted) AND for genuine silence, with the actual
reason printed only to the server console, so nobody at a browser could
tell which it was or what to fix.

Pinned here:
  * every way of coming back empty carries its own reason, and whether it
    was a failure or a real "nothing to hear";
  * replies real servers send -- `segments` missing or null when only word
    timings were computed -- are transcripts, not errors;
  * Speech to Text shows the reason; Vertical Shorts refuses to pick moments
    after a failure (it promises story-checked moments) but still falls
    back, saying why, for a source that truly has no dialogue.
"""
import io
import shutil
import subprocess
import unittest.mock as mock

import pytest
import requests

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main
import pipeline

pytestmark = pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg not available')

WORDS = [{'word': ' Kumusta', 'start': 0.5, 'end': 1.0}, {'word': ' ka.', 'start': 1.0, 'end': 1.4}]
SEGS = [{'id': 0, 'start': 0.5, 'end': 1.4, 'text': ' Kumusta ka.'}]


def _clip(path, audio=True):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=red:s=160x90:d=2:r=25']
    if audio:
        cmd += ['-f', 'lavfi', '-i', 'sine=frequency=330:duration=2', '-c:a', 'aac', '-shortest']
    subprocess.run(cmd + ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True, timeout=60)
    return str(path)


def _reply(payload=None, status=200, text=''):
    r = mock.Mock()
    r.status_code, r.text = status, text
    r.json.return_value = payload
    r.raise_for_status.side_effect = (requests.exceptions.HTTPError(f'{status} Server Error', response=r)
                                      if status >= 400 else None)
    return r


@pytest.fixture
def src(tmp_path):
    return _clip(tmp_path / 'talk.mp4')


def test_reply_parsing_copes_with_what_servers_leave_out():
    full = pipeline._whisper_reply({'segments': SEGS, 'words': WORDS})
    assert full == ([{'start': 0.5, 'end': 1.0, 'word': 'Kumusta'}, {'start': 1.0, 'end': 1.4, 'word': 'ka.'}],
                    [{'start': 0.5, 'end': 1.4, 'text': 'Kumusta ka.'}])
    # Only word timings computed: `segments` null, or not there at all. Lines are built from the words.
    for reply in ({'segments': None, 'words': WORDS}, {'words': WORDS}):
        words, segs = pipeline._whisper_reply(reply)
        assert len(words) == 2 and [s['text'] for s in segs] == ['Kumusta ka.']
    assert pipeline._whisper_reply({'segments': SEGS}) == ([], full[1])
    assert pipeline._whisper_reply({'text': '', 'segments': [], 'words': None}) == ([], [])
    # One malformed row costs that row, not the transcript.
    words, segs = pipeline._whisper_reply({'segments': SEGS + [{'text': 'no times'}], 'words': WORDS + [None]})
    assert len(words) == 2 and len(segs) == 1
    with pytest.raises(ValueError):
        pipeline._whisper_reply(['not', 'an', 'object'])


def test_a_transcript_comes_back_with_a_clean_outcome_and_asks_for_both_granularities(src):
    with mock.patch.object(pipeline.requests, 'post', return_value=_reply({'segments': SEGS, 'words': WORDS})) as post:
        words, segs, outcome = pipeline.transcribe_video_detailed(src)
    assert len(words) == 2 and len(segs) == 1 and outcome == {'ok': True, 'reason': None}
    kw = post.call_args.kwargs
    assert kw['data']['timestamp_granularities[]'] == ['word', 'segment']
    assert kw['timeout'] == pipeline.WHISPER_TIMEOUT >= 1800
    # The two-value form the promo pipeline uses is unchanged.
    with mock.patch.object(pipeline.requests, 'post', return_value=_reply({'segments': SEGS, 'words': WORDS})):
        assert pipeline.transcribe_video(src) == (words, segs)


@pytest.mark.parametrize('answer, ok, says', [
    (_reply({'text': '', 'segments': [], 'words': []}), True, 'found no speech in the audio'),
    (_reply(None, 500, '{"detail":"Model large-v2 is not loaded"}'), False,
     'HTTP 500 for model "' + pipeline.WHISPER_MODEL + '"): {"detail":"Model large-v2 is not loaded"}'),
    (_reply(None, 404, ''), False, 'HTTP 404'),
    (requests.exceptions.ConnectionError('refused'), False, 'could not connect to the speech-to-text service'),
    (requests.exceptions.ReadTimeout('read timed out'), False, 'did not answer within'),
    (_reply(['unexpected']), False, 'could not be read'),
])
def test_every_way_of_coming_back_empty_says_which_it_was(src, answer, ok, says):
    side = {'side_effect': answer} if isinstance(answer, Exception) else {'return_value': answer}
    with mock.patch.object(pipeline.requests, 'post', **side):
        words, segs, outcome = pipeline.transcribe_video_detailed(src)
        assert pipeline.transcribe_video(src) == ([], [])
    assert (words, segs) == ([], []) and outcome['ok'] is ok and says in outcome['reason'], outcome


def test_a_file_with_no_audio_track_is_nothing_to_hear_not_a_failure(tmp_path):
    silent = _clip(tmp_path / 'mute.mp4', audio=False)
    post = mock.Mock(side_effect=AssertionError('nothing to send'))
    with mock.patch.object(pipeline.requests, 'post', post):
        assert pipeline.transcribe_video_detailed(silent) == ([], [], {'ok': True, 'reason': 'the file has no audio track'})


def _client():
    client = main.app.test_client()
    with client.session_transaction() as s:
        s.update(authed=True, user_id=1, username='admin', role='admin', csrf_token='t')
    return client, {'X-CSRF-Token': 't'}


def test_speech_to_text_tab_shows_the_reason(src, monkeypatch):
    monkeypatch.setattr(pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)
    client, headers = _client()

    def post_file():
        with open(src, 'rb') as f:
            return client.post('/api/stt/transcribe', data={'stt_file': (io.BytesIO(f.read()), 'talk.mp4')},
                               headers=headers, content_type='multipart/form-data')

    with mock.patch.object(pipeline.requests, 'post', return_value=_reply(None, 500, 'CUDA out of memory')):
        r = post_file()
    assert r.status_code == 502
    err = r.get_json()['error']
    assert err.startswith('Transcription failed: ') and 'HTTP 500' in err and 'CUDA out of memory' in err
    assert 'actually contains audio' not in err, 'the old message blamed the file for a server error'

    with mock.patch.object(pipeline.requests, 'post', return_value=_reply({'segments': [], 'words': []})):
        r = post_file()
    assert r.status_code == 422 and r.get_json()['error'] == \
        'Nothing to transcribe: the speech-to-text service found no speech in the audio.'

    with mock.patch.object(pipeline.requests, 'post', return_value=_reply({'segments': None, 'words': WORDS})):
        r = post_file()
    body = r.get_json()
    assert r.status_code == 200 and body['ok'] and body['text'] == 'Kumusta ka.' and body['words'] == 2
