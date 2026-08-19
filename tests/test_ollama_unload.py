"""
Tests for unload_ollama_model() -- the shared-GPU fix that explicitly frees
Ollama's vision model from VRAM right after a scoring pass finishes, rather
than waiting out its keep-alive window while faster-whisper (often sharing
the same GPU) tries to load its own model into what's left.
"""
import unittest.mock as mock

import pipeline


def test_sends_keep_alive_zero_with_no_prompt():
    with mock.patch('pipeline.requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        pipeline.unload_ollama_model('qwen3-vl:8b')
    url, kwargs = mock_post.call_args
    assert url[0] == f'{pipeline.OLLAMA_URL}/api/generate'
    # keep_alive=0 with no prompt/images is Ollama's documented "unload,
    # don't run inference" signal -- confirming the payload is exactly that
    # and nothing more (no prompt that would trigger real generation).
    assert mock_post.call_args.kwargs['json'] == {'model': 'qwen3-vl:8b', 'keep_alive': 0}


def test_no_model_name_does_not_call_ollama():
    with mock.patch('pipeline.requests.post') as mock_post:
        pipeline.unload_ollama_model(None)
        pipeline.unload_ollama_model('')
    assert not mock_post.called


def test_connection_failure_does_not_raise():
    # Best-effort by design: Ollama's own keep-alive timeout is still the
    # fallback if this doesn't get through, so a failure here must not take
    # down the render job that's calling it.
    with mock.patch('pipeline.requests.post', side_effect=Exception('connection refused')):
        pipeline.unload_ollama_model('qwen3-vl:8b')  # must not raise


def test_toggle_is_in_the_self_describing_production_spec():
    assert 'unload_vision_after_scoring' in pipeline.PRODUCTION_DEFAULT_SPEC
    assert pipeline.PRODUCTION_DEFAULT_SPEC['unload_vision_after_scoring']['type'] == 'bool'


def test_toggle_defaults_to_true(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'PRODUCTION_DEFAULTS_FILE', str(tmp_path / 'prod.json'))
    assert pipeline.load_production_defaults()['unload_vision_after_scoring'] is True


def test_toggle_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'PRODUCTION_DEFAULTS_FILE', str(tmp_path / 'prod.json'))
    pipeline.save_production_defaults({'unload_vision_after_scoring': False})
    assert pipeline.load_production_defaults()['unload_vision_after_scoring'] is False
