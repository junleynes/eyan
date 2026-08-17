"""
Tests for load_production_defaults/save_production_defaults -- the settings
backing Config > Production. These are deployment-wide values (broadcast
loudness, house mix, scene-detection tuning) that every future render reads,
so a save path that silently accepts a malformed value would corrupt the
delivery spec for everyone, not just the person who made the mistake.
"""
import os
import pipeline


def test_defaults_load_without_a_saved_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'PRODUCTION_DEFAULTS_FILE', str(tmp_path / 'nonexistent.json'))
    values = pipeline.load_production_defaults()
    assert values['target_loudness'] == -14.0
    assert values['detector'] == 'content'
    assert values['broadcast_stereo'] is False


def test_save_then_load_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'PRODUCTION_DEFAULTS_FILE', str(tmp_path / 'prod.json'))
    pipeline.save_production_defaults({'target_loudness': -23.0, 'broadcast_stereo': True, 'detector': 'adaptive'})
    values = pipeline.load_production_defaults()
    assert values['target_loudness'] == -23.0
    assert values['broadcast_stereo'] is True
    assert values['detector'] == 'adaptive'
    # Untouched fields keep their built-in default
    assert values['true_peak'] == -1.5


def test_invalid_values_are_rejected_not_written_through(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'PRODUCTION_DEFAULTS_FILE', str(tmp_path / 'prod.json'))
    pipeline.save_production_defaults({'target_loudness': -20.0})
    # A bad follow-up save shouldn't corrupt the previously-good value
    pipeline.save_production_defaults({'target_loudness': 'not-a-number', 'detector': 'nonsense-value'})
    values = pipeline.load_production_defaults()
    assert values['target_loudness'] == -20.0  # unchanged, bad value ignored
    assert values['detector'] == 'content'      # unchanged, invalid choice ignored


def test_choice_field_only_accepts_listed_values(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'PRODUCTION_DEFAULTS_FILE', str(tmp_path / 'prod.json'))
    pipeline.save_production_defaults({'detector': 'adaptive'})
    assert pipeline.load_production_defaults()['detector'] == 'adaptive'
    pipeline.save_production_defaults({'detector': 'made-up-detector'})
    assert pipeline.load_production_defaults()['detector'] == 'adaptive'  # unchanged


def test_text_field_blank_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'PRODUCTION_DEFAULTS_FILE', str(tmp_path / 'prod.json'))
    pipeline.save_production_defaults({'vision_model': 'llava:13b'})
    assert pipeline.load_production_defaults()['vision_model'] == 'llava:13b'
    # A blank vision model would fail every scoring call with a confusing
    # error rather than an obvious "you cleared this" one -- so it falls
    # back to the built-in rather than saving empty.
    pipeline.save_production_defaults({'vision_model': ''})
    assert pipeline.load_production_defaults()['vision_model'] == 'qwen3-vl:8b'
