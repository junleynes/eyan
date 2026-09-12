"""
Tests for a real, user-reported problem: a two-column PDF rundown (video
timecode on the left, audio/SOT script text on the right) fed through the
old, position-blind pypdf extract_text() had no way to tell which column
any given piece of text came from -- so a time-of-day mention buried in
the right column's own prose (e.g. "airs at 8:50PM on GMA Prime") could
get flattened into the extracted text in a form that looked exactly like
a real left-column video timecode, and be misread as a scene-selection
cue that was never actually there.

_extract_pdf_video_column_only() detects a genuine two-column layout via
each word's own real x-position (not the page's content-stream order,
which is what a naive flatten actually reads) and, only when a clear
column gutter is found, builds text from the left column's words alone.
An ordinary single-column script -- the common case -- never triggers
this at all and reads exactly as it always did, via pypdf, completely
unaffected.
"""
import io
import unittest.mock as mock

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main
import pipeline
from werkzeug.datastructures import FileStorage


def _build_two_column_pdf(rows):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica", 10)
    width, height = letter
    y = height - 100
    for left, right in rows:
        c.drawString(50, y, left)
        c.drawString(250, y, right)
        y -= 40
    c.save()
    return buf.getvalue()


def _build_single_column_pdf(lines):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica", 10)
    width, height = letter
    y = height - 100
    for line in lines:
        c.drawString(60, y, line)
        y -= 20
    c.save()
    return buf.getvalue()


def test_two_column_pdf_extracts_left_column_only():
    raw = _build_two_column_pdf([
        ("00:03:33:00 M1 -37 HABOL TACKLE", "Reporter walks into the venue and begins narration."),
        ("00:00:35:00 M2 -12 BUNOT", "This segment airs at 8:50PM on GMA Prime tonight."),
    ])
    text, used = pipeline._extract_pdf_video_column_only(raw)
    assert used is True
    assert "HABOL TACKLE" in text
    assert "BUNOT" in text
    # The exact bug: right-column prose, including its own time-of-day
    # mention, must never appear in the extracted (video-column) text at all.
    assert "8:50PM" not in text
    assert "Reporter" not in text
    assert "GMA Prime" not in text


def test_single_column_pdf_is_not_treated_as_two_column():
    # No real column gutter anywhere -- must fall back cleanly (None, False)
    # so extract_script_text uses the ordinary, unaffected pypdf path.
    raw = _build_single_column_pdf([
        "00:03:33:00 M1 -37 HABOL TACKLE",
        "This is an ordinary single-column script with normal prose.",
        "It happens to mention 8:50PM on GMA Prime, and that's fine here",
        "since there's no real column layout for it to be confused with.",
    ])
    text, used = pipeline._extract_pdf_video_column_only(raw)
    assert used is False
    assert text is None


def test_full_extract_script_text_entry_point_uses_column_extraction_for_two_column_pdf():
    raw = _build_two_column_pdf([
        ("00:03:33:00 M1 -37 HABOL TACKLE", "Reporter walks into the venue and begins narration."),
        ("00:00:35:00 M2 -12 BUNOT", "This segment airs at 8:50PM on GMA Prime tonight."),
    ])
    fs = FileStorage(stream=io.BytesIO(raw), filename="rundown.pdf")
    text, err = pipeline.extract_script_text(fs)
    assert err is None
    assert "HABOL TACKLE" in text
    assert "8:50PM" not in text


def test_full_extract_script_text_entry_point_falls_back_for_single_column_pdf():
    raw = _build_single_column_pdf([
        "00:03:33:00 M1 -37 HABOL TACKLE",
        "Ordinary paragraph text follows, mentioning 8:50PM on GMA Prime.",
    ])
    fs = FileStorage(stream=io.BytesIO(raw), filename="script.pdf")
    text, err = pipeline.extract_script_text(fs)
    assert err is None
    # Single-column fallback keeps everything, unaffected by this feature --
    # this text was never at risk of column confusion in the first place.
    assert "HABOL TACKLE" in text
    assert "8:50PM" in text


def test_extracted_two_column_text_produces_clean_cues_with_no_spurious_entries():
    raw = _build_two_column_pdf([
        ("00:03:33:00 M1 -37 HABOL TACKLE", "Reporter walks into the venue and begins narration."),
        ("00:03:38:00 M1 -43", "Continued interview footage with the athlete."),
        ("00:00:35:00 M2 -12 BUNOT", "This segment airs at 8:50PM on GMA Prime tonight."),
        ("00:00:43:00 M2 -20", "Second telecast will be on GTV at 10:30PM."),
    ])
    text, used = pipeline._extract_pdf_video_column_only(raw)
    assert used is True
    cues = pipeline.parse_script_cues(text, segment_offsets={1: 0.0, 2: 100.0})
    assert len(cues) == 4
    descs = [c['desc'] for c in cues]
    assert not any('8:50' in d or 'GMA Prime' in d or 'GTV' in d for d in descs)


def test_two_column_detection_returns_false_gracefully_without_pdfplumber():
    raw = _build_two_column_pdf([
        ("00:03:33:00 M1 -37 HABOL TACKLE", "Reporter walks into the venue."),
    ])
    with mock.patch.dict('sys.modules', {'pdfplumber': None}):
        text, used = pipeline._extract_pdf_video_column_only(raw)
    assert used is False
    assert text is None


def test_varying_line_lengths_on_a_single_column_page_do_not_false_positive():
    # Regression guard for a real bug caught during development: an
    # earlier version of the column-gap detector pooled every word's x0
    # across the WHOLE page into one list, rather than checking within
    # each row -- a short first line followed by a much longer second
    # line left a coincidental gap in that pooled, page-wide list that
    # looked exactly like a column gutter, even though there's no real
    # column structure here at all, and truncated the extracted text mid
    # word. Detection must be per-row (a real gutter recurs at the same
    # x-position across several rows), not global.
    raw = _build_single_column_pdf([
        "00:03:33:00 M1 -37 HABOL TACKLE",
        "Ordinary paragraph text follows, mentioning 8:50PM on GMA Prime.",
    ])
    text, used = pipeline._extract_pdf_video_column_only(raw)
    assert used is False
    assert text is None
