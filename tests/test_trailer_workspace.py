"""The Episodic Plug page as a three-step workspace over the existing form, review grid and result.

The page's behaviour is checked in a browser during development; here, what must stay true of the markup the
workspace relies on: its steps, its ready check, its sticky actions, and that the form it wraps still has
every control the generate and render code reads.
"""
import re
import unittest.mock as mock

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main


def _page():
    client = main.app.test_client()
    with client.session_transaction() as sess:
        sess.update(authed=True, user_id=1, username='admin', role='admin', csrf_token='t')
    return client.get('/').get_data(as_text=True)


def test_the_episodic_page_has_three_steps_over_the_existing_screens():
    html = _page()
    for step in ('setup', 'review', 'result'):
        assert f"data-trstep={step}" in html and f"trScreen('{step}')" in html
    assert html.index('id=tr-steps') < html.index('<div class="work">') < html.index('id=tr-preview-area') < html.index('id=tr-progress-area')
    for ident in ('tr-ready', 'job-summary-text', 'tr-step-count', 'tr-preview-area', 'tr-area', 'tr-progress-area',
                  'trailer-preview-btn', 'trailer-submit-btn', 'tr-render-btn', 'tr-close'):
        assert len(re.findall(rf'id=["\']?{ident}(?=[\s>"\'])', html)) == 1, ident


def test_the_screens_are_chosen_by_a_class_not_by_moving_the_form():
    html = _page()
    for rule in ('#p-trailer.tr-s-review .work-in', '#p-trailer.tr-s-setup #tr-preview-area', '#p-trailer.tr-s-result #tr-preview-area'):
        assert rule in html, rule
    # The form, its submit path and the settings the ready check reads are all still there.
    for piece in ("action=/api/trailer/generate", "id=template-select", "id=trailer-length-select", "id=genre-select",
                  "id=delivery-format-select", "id=priority-prompt", "id=negative-prompt", "function submitTrailer(", "function trReady(",
                  "function trUseLast(", "if(typeof trScreen === 'function') trScreen('setup')"):
        assert piece in html, piece


def test_the_essentials_come_before_the_long_steering_block():
    html = _page()
    assert html.index('id=template-select') < html.index('class="promo-steer card"')
    assert html.index('id=trailer-length-select') < html.index('class="promo-steer card"')
    assert html.index('class="promo-steer card"') < html.index('id=generator-settings')


def test_cleaner_edits_settings_are_on_by_default_and_posted_with_the_form():
    html = _page()
    assert 'id="promo-clean"' in html
    for name, default in (('cut_clean', 'on'), ('card_fit', 'fit'), ('vo_smart', 'on')):
        m = re.search(rf'<select name={name} [^>]*>(.*?)</select>', html, re.S)
        assert m, name
        assert re.search(rf'<option value={default} selected>', m.group(1)), name
    assert html.index('class="promo-essentials"') < html.index('id="promo-clean"') < html.index('class="promo-steer card"')


def test_the_review_has_a_place_for_the_edit_report_and_the_ready_panel_a_time_budget():
    html = _page()
    assert html.count('id=tr-edit-report') == 1
    for piece in ('function renderEditReport(', 'renderEditReport(d)', 'function trCardSecs(', "'Time budget'"):
        assert piece in html, piece
