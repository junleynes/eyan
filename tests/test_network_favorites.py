"""
Tests for network_favorites_list/network_favorite_add/network_favorite_remove
-- the per-user bookmarked-folder feature in the network browser. Built and
manually verified with real browser sessions when it shipped, but that
verification doesn't persist; these make it permanent regression coverage.
"""
import os
import sys
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with mock.patch('requests.post'), mock.patch('requests.get'):
    import library_db


def _use_temp_lib_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / 'test_library.db')
    monkeypatch.setattr(library_db, 'LIBRARY_DB_PATH', db_path)
    library_db.library_db_init()


def test_add_then_list(tmp_path, monkeypatch):
    _use_temp_lib_db(tmp_path, monkeypatch)
    library_db.network_favorite_add(1, 'hires', '24Oras\\2026\\August', 'August 24 Oras')
    items = library_db.network_favorites_list(1, 'hires')
    assert len(items) == 1
    assert items[0]['label'] == 'August 24 Oras'
    assert items[0]['path'] == '24Oras\\2026\\August'
    assert items[0]['category'] == 'hires'


def test_readding_the_same_location_is_a_harmless_no_op(tmp_path, monkeypatch):
    _use_temp_lib_db(tmp_path, monkeypatch)
    library_db.network_favorite_add(1, 'hires', '24Oras\\2026\\August', 'First label')
    library_db.network_favorite_add(1, 'hires', '24Oras\\2026\\August', 'Second label')
    items = library_db.network_favorites_list(1, 'hires')
    # UNIQUE(user_id, category, path) + INSERT OR IGNORE -- no duplicate row,
    # and the original label wins rather than silently being overwritten.
    assert len(items) == 1
    assert items[0]['label'] == 'First label'


def test_different_paths_in_the_same_category_are_both_kept(tmp_path, monkeypatch):
    _use_temp_lib_db(tmp_path, monkeypatch)
    library_db.network_favorite_add(1, 'hires', 'ShowA\\2026', 'Show A')
    library_db.network_favorite_add(1, 'hires', 'ShowB\\2026', 'Show B')
    assert len(library_db.network_favorites_list(1, 'hires')) == 2


def test_category_filter_only_returns_that_category(tmp_path, monkeypatch):
    _use_temp_lib_db(tmp_path, monkeypatch)
    library_db.network_favorite_add(1, 'hires', 'ShowA', 'Show A')
    library_db.network_favorite_add(1, 'music', 'Beds\\Action', 'Action beds')
    assert len(library_db.network_favorites_list(1, 'hires')) == 1
    assert len(library_db.network_favorites_list(1, 'music')) == 1
    assert len(library_db.network_favorites_list(1)) == 2  # no category = everything


def test_the_category_root_can_be_favorited(tmp_path, monkeypatch):
    # path='' represents "the category's own root", not a missing value --
    # confirm an empty-string path round-trips correctly rather than being
    # treated as falsy/skipped anywhere along the way.
    _use_temp_lib_db(tmp_path, monkeypatch)
    library_db.network_favorite_add(1, 'hires', '', '(category root)')
    items = library_db.network_favorites_list(1, 'hires')
    assert len(items) == 1
    assert items[0]['path'] == ''


def test_favorites_are_isolated_per_user(tmp_path, monkeypatch):
    _use_temp_lib_db(tmp_path, monkeypatch)
    library_db.network_favorite_add(1, 'hires', 'ShowA', 'User 1 favorite')
    library_db.network_favorite_add(2, 'hires', 'ShowA', 'User 2 favorite')
    user1 = library_db.network_favorites_list(1, 'hires')
    user2 = library_db.network_favorites_list(2, 'hires')
    assert len(user1) == 1 and len(user2) == 1
    assert user1[0]['label'] == 'User 1 favorite'
    assert user2[0]['label'] == 'User 2 favorite'
    # Same (category, path) for both users is fine -- the uniqueness
    # constraint is per-user, not global.
    assert user1[0]['id'] != user2[0]['id']


def test_removing_your_own_favorite_works(tmp_path, monkeypatch):
    _use_temp_lib_db(tmp_path, monkeypatch)
    library_db.network_favorite_add(1, 'hires', 'ShowA', 'Show A')
    fav_id = library_db.network_favorites_list(1, 'hires')[0]['id']
    removed = library_db.network_favorite_remove(1, fav_id)
    assert removed is True
    assert library_db.network_favorites_list(1, 'hires') == []


def test_cannot_remove_another_users_favorite(tmp_path, monkeypatch):
    # The important security property: ownership is enforced in the DELETE
    # itself (WHERE id=? AND user_id=?), not just left to the UI to not
    # expose the button.
    _use_temp_lib_db(tmp_path, monkeypatch)
    library_db.network_favorite_add(2, 'hires', 'ShowA', 'User 2 favorite')
    fav_id = library_db.network_favorites_list(2, 'hires')[0]['id']
    removed = library_db.network_favorite_remove(1, fav_id)
    assert removed is False
    assert len(library_db.network_favorites_list(2, 'hires')) == 1


def test_removing_a_nonexistent_favorite_returns_false_not_an_error(tmp_path, monkeypatch):
    _use_temp_lib_db(tmp_path, monkeypatch)
    assert library_db.network_favorite_remove(1, 999999) is False
