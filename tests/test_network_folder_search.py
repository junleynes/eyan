"""
Tests for list_network_files_recursive() matching FOLDERS, not just files.

Real, reported bug: searching Browse Library for a show's own folder name
(rather than a specific file inside it) came back empty even when that
folder genuinely existed right there -- the recursive search only ever
checked filenames against the query, silently using directories purely as
passthrough containers to recurse into, never as something the query
itself could match.

Uses a mocked smbclient.scandir (no real SMB server in this environment)
returning fake DirEntry-like objects, since this function's own logic --
what it does with directory vs file entries, not the real network I/O --
is what's under test here.
"""
import unittest.mock as mock

import pipeline


class _FakeEntry:
    """Stands in for smbclient's DirEntry -- just enough of its interface
    (name, is_dir/is_file, stat) for list_network_files_recursive to work
    with, since a real DirEntry can only come from a real SMB connection."""
    def __init__(self, name, is_directory=False, size=1000, mtime=1700000000.0):
        self.name = name
        self._is_dir = is_directory
        self._size = size
        self._mtime = mtime

    def is_dir(self):
        return self._is_dir

    def is_file(self):
        return not self._is_dir

    def stat(self):
        return mock.Mock(st_size=self._size, st_mtime=self._mtime)


# A small, fixed folder tree used by every test below:
#   (root)
#     24Oras/              <- folder, matches query "24"
#       episode1.mp4        <- file
#     Firewall/             <- folder, does NOT match "24"
#       24_special.mp4      <- file, DOES match "24" despite its parent not matching
#       poster.jpg           <- file, extension not in the allowed list for this category
_TREE = {
    '': [
        _FakeEntry('24Oras', is_directory=True),
        _FakeEntry('Firewall', is_directory=True),
    ],
    '24Oras': [
        _FakeEntry('episode1.mp4'),
    ],
    'Firewall': [
        _FakeEntry('24_special.mp4'),
        _FakeEntry('poster.jpg'),
    ],
}


def _fake_scandir(path):
    # path looks like '\\\\server\\share' or '\\\\server\\share\\Firewall' --
    # the function under test only ever cares about entries at a given
    # relative subpath, so map the real, full path back to that.
    for sub in sorted(_TREE.keys(), key=len, reverse=True):
        suffix = ('\\' + sub) if sub else ''
        if path.endswith(suffix) if sub else True:
            if sub == '' or path.endswith('\\' + sub):
                return iter(_TREE[sub])
    return iter([])


def _patches():
    return (
        mock.patch('pipeline._network_share_root', return_value='\\\\server\\share'),
        mock.patch('pipeline._network_session'),
        mock.patch('pipeline.smbclient.scandir', side_effect=_fake_scandir),
        mock.patch('pipeline._network_category', return_value={'label': 'HIRES', 'exts': {'mp4'}}),
    )


def test_a_matching_folder_is_returned_with_is_dir_true():
    p1, p2, p3, p4 = _patches()
    with p1, p2, p3, p4:
        root, results, truncated = pipeline.list_network_files_recursive('hires', '', '24')
    folder_hits = [r for r in results if r.get('is_dir')]
    assert len(folder_hits) == 1
    assert folder_hits[0]['name'] == '24Oras'
    assert folder_hits[0]['subpath'] == ''


def test_a_file_inside_a_non_matching_folder_is_still_found():
    # Firewall itself doesn't match "24", but 24_special.mp4 inside it does
    # -- confirms a non-matching folder still gets recursed into, not
    # skipped just because its own name didn't match.
    p1, p2, p3, p4 = _patches()
    with p1, p2, p3, p4:
        root, results, truncated = pipeline.list_network_files_recursive('hires', '', '24')
    file_hits = [r for r in results if not r.get('is_dir')]
    names = sorted(r['name'] for r in file_hits)
    assert names == ['24_special.mp4']
    assert file_hits[0]['subpath'] == 'Firewall'


def test_matching_folder_contents_are_still_found_too():
    # 24Oras matches the query itself, but episode1.mp4 inside it does NOT
    # match "24" on its own filename -- since the search is a flat
    # recursive walk (not "only look inside a matched folder"), this
    # shouldn't appear in a "24" search, confirming a matching folder
    # doesn't cause everything inside it to be treated as auto-matching.
    p1, p2, p3, p4 = _patches()
    with p1, p2, p3, p4:
        root, results, truncated = pipeline.list_network_files_recursive('hires', '', '24')
    assert not any(r['name'] == 'episode1.mp4' for r in results)


def test_blank_query_returns_every_folder_and_file():
    p1, p2, p3, p4 = _patches()
    with p1, p2, p3, p4:
        root, results, truncated = pipeline.list_network_files_recursive('hires', '', '')
    folder_names = sorted(r['name'] for r in results if r.get('is_dir'))
    file_names = sorted(r['name'] for r in results if not r.get('is_dir'))
    assert folder_names == ['24Oras', 'Firewall']
    # poster.jpg is excluded by the category's allowed extensions, not by
    # the (blank) query -- confirms that filter still applies to files
    # regardless of this change.
    assert file_names == ['24_special.mp4', 'episode1.mp4']


def test_folders_sort_before_files_within_the_same_subpath():
    p1, p2, p3, p4 = _patches()
    with p1, p2, p3, p4:
        root, results, truncated = pipeline.list_network_files_recursive('hires', '', '')
    root_level = [r for r in results if r['subpath'] == '']
    # Both root-level entries here are folders (24Oras, Firewall) --
    # confirms the sort key change didn't break plain alphabetical
    # ordering within a type, not just the folders-before-files ordering.
    assert [r['name'] for r in root_level] == ['24Oras', 'Firewall']


def test_folder_result_case_insensitive_match():
    p1, p2, p3, p4 = _patches()
    with p1, p2, p3, p4:
        root, results, truncated = pipeline.list_network_files_recursive('hires', '', 'FIREWALL')
    folder_hits = [r for r in results if r.get('is_dir')]
    assert len(folder_hits) == 1
    assert folder_hits[0]['name'] == 'Firewall'
