"""
Tests for resolve_delivery_format() -- validates a requested delivery-format
key against EXPORT_FORMATS, degrading to the MP4 master rather than raising
or passing an invalid key through to the render/export pipeline.
"""
import pipeline


def test_valid_format_passes_through_unchanged():
    for fmt in pipeline.EXPORT_FORMATS:
        assert pipeline.resolve_delivery_format(fmt) == fmt


def test_unknown_format_falls_back_to_mp4_master():
    assert pipeline.resolve_delivery_format('not_a_real_format') == 'mp4_high'


def test_empty_string_falls_back_to_mp4_master():
    assert pipeline.resolve_delivery_format('') == 'mp4_high'


def test_none_falls_back_to_mp4_master():
    # A stale/older client (or a form field that genuinely wasn't submitted)
    # could send None rather than a missing key -- must not raise.
    assert pipeline.resolve_delivery_format(None) == 'mp4_high'


def test_every_export_format_has_a_real_extension_and_label():
    # Cheap sanity check on the format table itself: every entry
    # resolve_delivery_format can return must actually be usable downstream.
    for key, spec in pipeline.EXPORT_FORMATS.items():
        assert spec.get('ext'), f'{key} has no ext'
        assert spec.get('label'), f'{key} has no label'
