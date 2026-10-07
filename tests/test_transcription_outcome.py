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


# ---- which of the file's audio is transcribed ----
#
# A broadcast master is not a stereo MP4. The stand-in "speech" below is a
# 300 Hz tone and the stand-in "music" a 900 Hz one, and the stand-in
# service answers by which of them it is sent -- so these tests are about
# WHAT WAS SENT, which is the whole of the problem: a programme that is all
# dialogue came back "no speech in the audio" because the audio sent was a
# silent track, or music and effects, or two channels cancelling each other.

import io as _io
import wave

import numpy as np

import shorts_core as sc

SPEECH, MUSIC, SILENT = 'sine=f=300:d=6', 'sine=f=900:d=6', 'anullsrc=r=48000:cl=mono:d=6'
_VIDEO = ['-f', 'lavfi', '-i', 'color=c=gray:s=160x90:d=6:r=25']
_ENC = ['-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'pcm_s16le', '-ar', '48000', '-shortest']


def _ff(*args):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', *args], check=True, timeout=120)


def _mono_tracks(path, kinds):
    """A .mov with one mono audio track per letter: s(peech), m(usic), 0 (silent)."""
    args = list(_VIDEO)
    for k in kinds:
        args += ['-f', 'lavfi', '-i', {'s': SPEECH, 'm': MUSIC, '0': SILENT}[k]]
    for i in range(len(kinds)):
        args += ['-map', f'{i + 1}:a']
    _ff(*args, '-map', '0:v', *_ENC, str(path))
    return str(path)


def _tone(raw_wav):
    """(level in dB, dominant frequency) of a WAV file's bytes."""
    w = wave.open(_io.BytesIO(raw_wav))
    x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(float) / 32768.0
    level = 20 * np.log10(max(1e-9, float(np.sqrt(np.mean(x ** 2)))))
    part = x[:w.getframerate() * 2]
    freq = float(np.argmax(np.abs(np.fft.rfft(part))) * w.getframerate() / len(part)) if level > -90 else 0.0
    return level, freq


class _Listener:
    """A speech-to-text service that hears speech in a 300 Hz tone and in nothing else."""

    def __init__(self):
        self.sent = []

    def __call__(self, url, files=None, data=None, timeout=None):
        level, freq = _tone(files['file'][1].read())
        self.sent.append((round(level), round(freq, -1)))
        heard = level > -40 and abs(freq - 300) < 20
        return _reply({'segments': SEGS if heard else [], 'words': WORDS if heard else []})


def _transcribe(path):
    ear = _Listener()
    with mock.patch.object(pipeline.requests, 'post', side_effect=ear):
        words, segs, outcome = pipeline.transcribe_video_detailed(path)
    return bool(segs), outcome, ear.sent


def test_dialogue_on_tracks_3_and_4_of_eight_is_found_and_the_silent_tracks_are_never_sent(tmp_path):
    src = _mono_tracks(tmp_path / 'master.mov', '00ss0000')
    heard, outcome, sent = _transcribe(src)
    assert heard and outcome == {'ok': True, 'reason': None, 'audio': 'tracks 3+4', 'take': [[2, 0], [3, 0]]}
    assert len(sent) == 1 and sent[0][1] == 300 and sent[0][0] > -30, 'one request, and it was the dialogue'
    # What used to be sent -- the first track -- is silence.
    old = tmp_path / 'old.wav'
    _ff('-i', src, '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(old))
    assert _tone(old.read_bytes())[0] < -80


def test_music_and_effects_first_is_tried_found_wanting_and_the_full_mix_is_next(tmp_path):
    heard, outcome, sent = _transcribe(_mono_tracks(tmp_path / 'me.mov', 'mmss'))
    assert heard and outcome['audio'] == 'tracks 3+4' and outcome['take'] == [[2, 0], [3, 0]]
    assert [f for _, f in sent] == [900, 300]


def test_a_stereo_pair_out_of_phase_is_sent_as_one_side_not_summed_to_nothing(tmp_path):
    src = str(tmp_path / 'phase.mov')
    _ff(*_VIDEO, '-f', 'lavfi', '-i', SPEECH, '-filter_complex',
        '[1:a]asplit[a][b];[b]volume=-1[c];[a][c]amerge=inputs=2[s]', '-map', '0:v', '-map', '[s]', *_ENC, src)
    old = tmp_path / 'old.wav'
    _ff('-i', src, '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(old))
    assert _tone(old.read_bytes())[0] < -80, 'folded to mono, this file is silence'
    heard, outcome, sent = _transcribe(src)
    assert heard and outcome == {'ok': True, 'reason': None} and sent[0][0] > -30
    # An ordinary stereo file is sent as both sides together, and reported as plainly as ever.
    plain = str(tmp_path / 'plain.mp4')
    _ff(*_VIDEO, '-f', 'lavfi', '-i', SPEECH, '-ac', '2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
        '-shortest', plain)
    assert _transcribe(plain)[:2] == (True, {'ok': True, 'reason': None})


def test_a_surround_mix_has_its_centre_channel_tried_first(tmp_path):
    src = str(tmp_path / 'five_one.mp4')
    _ff(*_VIDEO, '-f', 'lavfi', '-i', SPEECH, '-f', 'lavfi', '-i', MUSIC, '-f', 'lavfi', '-i', SILENT,
        '-filter_complex', '[2:a][2:a][1:a][3:a][2:a][2:a]amerge=inputs=6,aformat=channel_layouts=5.1[a]',
        '-map', '0:v', '-map', '[a]', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', src)
    heard, outcome, sent = _transcribe(src)
    assert heard and sent == [(sent[0][0], 300)] and outcome['audio'] == 'channel 3'
    assert 'take' not in outcome, 'the shorts fold a surround mix down whole; they do not play its centre alone'


def test_nothing_found_says_what_was_tried_and_silence_is_not_sent_at_all(tmp_path):
    heard, outcome, sent = _transcribe(_mono_tracks(tmp_path / 'music.mov', 'mm0m'))
    assert not heard and outcome['ok'] and len(sent) == 1, 'the pair; track 4 only repeats it and is not sent again'
    assert outcome['reason'].startswith('the speech-to-text service found no speech in the audio; tried tracks 1+2 at ')
    assert outcome['reason'].endswith('the file has 4 audio tracks') and 'track 4' not in outcome['reason']
    post = mock.Mock(side_effect=AssertionError('nothing to send'))
    with mock.patch.object(pipeline.requests, 'post', post):
        assert pipeline.transcribe_video_detailed(_mono_tracks(tmp_path / 'quiet.mov', '000')) == (
            [], [], {'ok': True, 'reason': 'every audio channel in the file is silent (below -55 dB; 3 audio tracks)'})
    # A service that fails is still a failure, on whichever attempt.
    with mock.patch.object(pipeline.requests, 'post', return_value=_reply(None, 500, 'CUDA out of memory')):
        _, _, outcome = pipeline.transcribe_video_detailed(_mono_tracks(tmp_path / 'm2.mov', '0s'))
    assert outcome['ok'] is False and 'HTTP 500' in outcome['reason']


def test_dialogue_returned_as_text_without_timings_is_a_failure_to_report_not_silence(src):
    with pytest.raises(ValueError, match='no timings'):
        pipeline._whisper_reply({'text': 'Kumusta ka. Mabuti naman.'})
    with mock.patch.object(pipeline.requests, 'post', return_value=_reply({'text': 'Kumusta ka.', 'segments': None})):
        words, segs, outcome = pipeline.transcribe_video_detailed(src)
    assert (words, segs) == ([], []) and outcome['ok'] is False and 'no timings' in outcome['reason']


def _samples(*channels, seconds=2, rate=8000):
    t = np.arange(seconds * rate) / rate
    made = {'s': 0.2 * np.sin(2 * np.pi * 300 * t), 'm': 0.2 * np.sin(2 * np.pi * 900 * t), '0': np.zeros_like(t),
            'q': 0.0005 * np.sin(2 * np.pi * 300 * t)}
    for k in range(1, 7):               # n1..n6: six different noises, nothing in common with one another
        made[f'n{k}'] = 0.1 * np.random.default_rng(k).standard_normal(len(t))
    cols = []
    for c in channels:
        cols.append(-made[c[1:]] if c.startswith('-') else made[c])
    return np.stack(cols, axis=1).astype(np.float32)


def _mono(n):
    return [{'channels': 1, 'layout': ''} for _ in range(n)]


def test_choosing_what_to_send():
    takes, levels = pipeline.stt_takes(_mono(4), _samples('0', '0', 's', 's'))
    assert takes == [[2, 3]] and levels[0] < -100 and -20 < levels[2] < -14
    assert pipeline.stt_takes(_mono(4), _samples('m', 'm', 's', 's'))[0] == [[0, 1], [2, 3]]
    assert pipeline.stt_takes(_mono(2), _samples('s', '-s'))[0] == [[0]], 'out of phase: one side, never the sum'
    assert pipeline.stt_takes(_mono(3), _samples('s', 'n1', 'm'))[0] == [[0], [1], [2]], 'unrelated: each on its own'
    assert pipeline.stt_takes(_mono(4), _samples('s', 's', 's', 's'))[0] == [[0, 1]], 'copies are not tried again'
    assert pipeline.stt_takes(_mono(2), _samples('q', '0'))[0] == [], 'below the floor is silence'
    assert pipeline.stt_takes(_mono(3), _samples('0', '0', '0')) == ([], [-120.0] * 3)
    # At most a handful of attempts, in file order.
    many = pipeline.stt_takes(_mono(6), _samples('n1', 'n2', 'n3', 'n4', 'n5', 'n6'))[0]
    assert many == [[0], [1], [2], [3]] and len(many) == pipeline.STT_MAX_TAKES
    # Surround that says it is surround: the centre first, then the rest as usual.
    five_one = [{'channels': 6, 'layout': '5.1(side)'}]
    assert pipeline.stt_takes(five_one, _samples('m', 'm', 's', '0', '0', '0'))[0] == [[2], [0, 1]]
    assert pipeline.stt_takes(five_one, _samples('s', 's', '0', '0', '0', '0'))[0] == [[0, 1]], 'an empty centre is skipped'
    # Eight unlabelled channels are eight tracks, not 7.1.
    assert pipeline.stt_takes([{'channels': 8, 'layout': ''}], _samples('m', 'm', 's', 's', '0', '0', '0', '0'))[0] == [
        [0, 1], [2, 3]]

    assert pipeline.describe_take(_mono(8), [2, 3]) == 'tracks 3+4'
    assert pipeline.describe_take(_mono(8), [4]) == 'track 5'
    assert pipeline.describe_take(_mono(1), [0]) == 'the audio track'
    assert pipeline.describe_take([{'channels': 2, 'layout': 'stereo'}], [0, 1]) == 'channels 1+2'
    assert pipeline.describe_take([{'channels': 8, 'layout': ''}], [4, 5]) == 'channels 5+6'
    two = [{'channels': 2, 'layout': 'stereo'}, {'channels': 6, 'layout': '5.1'}]
    assert pipeline.describe_take(two, [4]) == 'track 2 channel 3' and pipeline.describe_take(two, [0, 1]) == 'track 1 channels 1+2'


def test_the_shorts_take_their_sound_from_where_the_dialogue_was_found(tmp_path):
    assert sc.take_to_stereo([[2, 0]]) == '[0:a:2]pan=stereo|c0=c0|c1=c0'
    assert sc.take_to_stereo([[0, 4], [0, 5]]) == '[0:a:0]pan=stereo|c0=c4|c1=c5'
    assert sc.take_to_stereo([[2, 0], [3, 0]]) == ('[0:a:2]pan=mono|c0=c0[tl];[0:a:3]pan=mono|c0=c0[tr];'
                                                    '[tl][tr]amerge=inputs=2,pan=stereo|c0=c0|c1=c1')
    src = _mono_tracks(tmp_path / 'master.mov', '00ss')
    info = sc.probe_source('ffprobe', src)
    segs = [{'a': 0, 'b': 49, 'layout': 'fit', 'x': None, 'keys': None}]

    def sound_of(out, **extra):
        ok, err = sc.render_short('ffmpeg', src, str(out), 25, 50, dict(info, **extra), segs, work_dir=str(tmp_path),
                                  crf=30, preset='ultrafast')
        assert ok, err
        wav = tmp_path / (out.name + '.wav')
        _ff('-i', str(out), '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(wav))
        return _tone(wav.read_bytes())
    level, freq = sound_of(tmp_path / 'with.mp4', audio_take=[[2, 0], [3, 0]])
    assert level > -30 and abs(freq - 300) < 20, 'the dialogue'
    assert sound_of(tmp_path / 'without.mp4')[0] < -60, 'what the first track alone gives: nothing'
    cmd = sc.build_render_cmd('ffmpeg', src, 'o.mp4', 0, 50, dict(info, audio_take=[[2, 0], [3, 0]]), segs)
    assert '[aout]' in cmd and '-af' not in cmd and 'loudnorm' in cmd[cmd.index('-filter_complex') + 1]
    plain = sc.build_render_cmd('ffmpeg', src, 'o.mp4', 0, 50, info, segs)
    assert '-af' in plain and '[aout]' not in plain, 'a file with ordinary audio is rendered exactly as before'
