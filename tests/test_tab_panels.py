"""
Every top-level tab's panel must be a sibling of the others, never inside one.

Real, reported bug: the Music Generation panel was missing its closing
</div>, so every panel after it in the page (Text to SFX, Text to speech,
Speech to text, Scene detection, AI chat, API, Docs, Jobs, Config) was
parsed as a CHILD of the music panel. A panel is only shown while it is the
active one, and a child of a hidden panel is hidden whatever its own state,
so those tabs highlighted when clicked and showed an empty page. Nothing
errors when this happens -- not the server, not the browser console -- which
is why it needs a test: one unbalanced tag anywhere in a 10,000-line
template silently takes out every tab below it.
"""
import unittest.mock as mock
from html.parser import HTMLParser

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main


class _Panels(HTMLParser):
    """Records, for each <div class="... panel ..."> with an id, the id of
    the panel it is nested inside (None for a top-level one)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []          # one entry per open <div>: its panel id, or None
        self.parents = {}

    def handle_starttag(self, tag, attrs):
        if tag != 'div':
            return
        a = dict(attrs)
        is_panel = 'panel' in (a.get('class') or '').split() and a.get('id')
        if is_panel:
            self.parents[a['id']] = next((p for p in reversed(self.stack) if p), None)
        self.stack.append(a['id'] if is_panel else None)

    def handle_endtag(self, tag):
        if tag == 'div' and self.stack:
            self.stack.pop()


def nested_panels(html):
    p = _Panels()
    p.feed(html)
    return {child: parent for child, parent in p.parents.items() if parent}


def test_the_checker_itself_catches_a_missing_close():
    good = '<div id=a class="panel active"><div>x</div></div><div id=b class=panel></div>'
    bad = '<div id=a class="panel active"><div>x</div><div id=b class=panel></div>'
    assert nested_panels(good) == {} and nested_panels(bad) == {'b': 'a'}
    # A sub-panel is a different class and is meant to live inside its panel.
    assert nested_panels('<div id=a class=panel><div id=a1 class="sub-panel"></div></div>') == {}


def test_no_tab_panel_is_nested_inside_another():
    client = main.app.test_client()
    with client.session_transaction() as s:
        s.update(authed=True, user_id=1, username='admin', role='admin', csrf_token='t')
    r = client.get('/')
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    p = _Panels()
    p.feed(html)
    assert len(p.parents) >= 12, 'the page really has its tab panels'
    assert {c: par for c, par in p.parents.items() if par} == {}, \
        'a panel inside another panel is hidden whenever that one is: look for an unclosed <div> in the outer panel'
