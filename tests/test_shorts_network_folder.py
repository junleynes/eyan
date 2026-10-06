"""
The Vertical Shorts tab's own network folder (Config > Network >
"Vertical Shorts video (HIRES)").

Shorts are cut from whole episodes, which usually live somewhere other than
the material promos are built from, so the tab browses a category of its
own instead of sharing the promo generator's HIRES folder. What has to hold:

  * it is a real category everywhere a category is accepted (config, browse,
    search, fetch, favourites), limited to video files;
  * a server set up before the category existed keeps working: while its
    path is blank it reads the HIRES folder, login included;
  * once it has a path, that path and ITS login are used, and nothing about
    HIRES changes;
  * the maximum source video size applies to it, as it does to HIRES.

No real SMB server here, so smbclient is mocked; what is under test is
which folder and which login each call resolves to.
"""
import json
import unittest.mock as mock

import pytest

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main
import library_db
import pipeline

HIRES = {'path': r'\\10.0.1.130\media\HIRES', 'username': 'promo', 'password': 'p1'}
SHORTS = {'path': '//10.0.1.140/archive/Episodes', 'username': 'shorts', 'password': 'p2'}


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """A network_folders.json of this test's own, starting with HIRES only --
    i.e. a server configured before the Vertical Shorts folder existed."""
    f = tmp_path / 'network_folders.json'
    f.write_text(json.dumps({'hires': HIRES}))
    monkeypatch.setattr(library_db, 'NETWORK_FOLDERS_FILE', str(f))
    return f


def _admin():
    client = main.app.test_client()
    with client.session_transaction() as s:
        s.update(authed=True, user_id=1, username='admin', role='admin', csrf_token='t')
    return client, {'X-CSRF-Token': 't'}


def test_it_is_a_video_category_of_its_own():
    assert 'shorts' in library_db.NETWORK_CATEGORY_KEYS
    cats = pipeline._network_categories()
    assert cats['shorts']['exts'] == cats['hires']['exts'] and cats['shorts']['fallback'] == 'hires'
    assert set(cats) == set(library_db.NETWORK_CATEGORY_KEYS), 'every stored category is browsable and vice versa'
    assert pipeline._network_category('shorts') is not pipeline._network_category('hires')


def test_blank_reads_the_hires_folder_with_the_hires_login(folders):
    assert pipeline._network_share_root('shorts') == HIRES['path']
    with mock.patch.object(pipeline.smbclient, 'register_session') as reg:
        pipeline._network_session('shorts')
    reg.assert_called_once_with('10.0.1.130', username='promo', password='p1', connection_timeout=10)
    # A login typed in without a path is not a folder of its own yet.
    library_db.save_network_folder('shorts', {'username': 'someone', 'password': 'x'})
    assert pipeline._network_folder_row('shorts') == HIRES


def test_its_own_path_and_login_once_set_and_hires_is_untouched(folders):
    assert library_db.save_network_folder('shorts', SHORTS) == (True, None)
    assert pipeline._network_share_root('shorts') == r'\\10.0.1.140\archive\Episodes'
    assert pipeline._network_share_root('hires') == HIRES['path']
    with mock.patch.object(pipeline.smbclient, 'register_session') as reg:
        pipeline._network_session('shorts')
    reg.assert_called_once_with('10.0.1.140', username='shorts', password='p2', connection_timeout=10)
    # Clearing the path goes back to HIRES.
    library_db.save_network_folder('shorts', {'path': ''})
    assert pipeline._network_share_root('shorts') == HIRES['path']


def test_nothing_configured_at_all_says_which_folder_to_set(tmp_path, monkeypatch):
    monkeypatch.setattr(library_db, 'NETWORK_FOLDERS_FILE', str(tmp_path / 'none.json'))
    with pytest.raises(ValueError, match='Vertical Shorts video'):
        pipeline.list_network_files('shorts')
    # Other categories never borrow a folder.
    monkeypatch.setattr(library_db, 'NETWORK_FOLDERS_FILE', str(tmp_path / 'h.json'))
    (tmp_path / 'h.json').write_text(json.dumps({'hires': HIRES}))
    for cat in ('tcard', 'endcard', 'music', 'vo', 'sfx', 'script'):
        assert pipeline._network_share_root(cat) == ''


def test_browsing_lists_video_from_the_resolved_folder(folders):
    library_db.save_network_folder('shorts', SHORTS)
    seen = []

    def scandir(path):
        seen.append(path)
        ep = mock.Mock(); ep.name = 'episode_101.mov'
        ep.is_dir.return_value, ep.is_file.return_value = False, True
        ep.stat.return_value = mock.Mock(st_size=10, st_mtime=1.0)
        wav = mock.Mock(); wav.name = 'bed.wav'
        wav.is_dir.return_value, wav.is_file.return_value = False, True
        return [ep, wav]
    with mock.patch.object(pipeline.smbclient, 'register_session'), \
            mock.patch.object(pipeline.smbclient, 'scandir', side_effect=scandir):
        full, sub, dirs, files = pipeline.list_network_files('shorts', 'Season 1')
    assert seen == [r'\\10.0.1.140\archive\Episodes\Season 1'] and full == seen[0]
    assert [f['name'] for f in files] == ['episode_101.mov'], 'video only'


def test_the_maximum_video_size_applies_to_it(folders, monkeypatch):
    checked = []
    monkeypatch.setattr(pipeline, 'check_video_size', lambda size, label=None: (checked.append(label), (False, 'too big'))[1])
    with mock.patch.object(pipeline.smbclient, 'register_session'), \
            mock.patch.object(pipeline.smbclient, 'stat', return_value=mock.Mock(st_size=10 ** 12)):
        with pytest.raises(ValueError, match='too big'):
            pipeline.fetch_network_file('episode_101.mp4', 'shorts')
    assert checked == ['episode_101.mp4']


def test_config_api_shows_saves_and_validates_it(folders):
    client, headers = _admin()
    cats = client.get('/api/network/shares').get_json()['categories']
    assert cats['shorts'] == {'path': '', 'username': '', 'has_password': False, 'fallback': 'hires'}
    assert cats['hires']['fallback'] is None and cats['hires']['path'] == HIRES['path']
    r = client.post('/api/network/shares', json=dict(SHORTS, category='shorts'), headers=headers)
    assert r.status_code == 200 and r.get_json()['ok']
    cats = client.get('/api/network/shares').get_json()['categories']
    assert cats['shorts']['path'] == SHORTS['path'] and cats['shorts']['has_password'] is True
    assert 'password' not in cats['shorts'], 'a saved password is never sent back'
    assert json.loads(folders.read_text())['hires'] == HIRES


def test_the_shorts_tab_browses_its_own_category():
    client, _ = _admin()
    html = client.get('/').get_data(as_text=True)
    assert "openNetworkBrowser('shorts_file','shorts'," in html
    assert "openNetworkBrowser('shorts_file','hires'," not in html
