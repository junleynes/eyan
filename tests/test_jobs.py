"""
Tests for job_new/job_set/job_get/job_cancel/job_list_all/job_set_orig_name --
the SQLite-backed job tracking that replaced an in-memory dict. The whole
point of this layer is durability across a restart, so that's the behavior
most worth protecting with a real test, alongside the basic CRUD lifecycle.
"""
import re
import sqlite3
import time
import unittest.mock as mock

import pytest

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main
import pipeline


def _use_temp_jobs_db(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'JOBS_DB_PATH', str(tmp_path / 'test_jobs.db'))
    pipeline.jobs_db_init()


def test_basic_lifecycle(tmp_path, monkeypatch):
    _use_temp_jobs_db(tmp_path, monkeypatch)
    jid = pipeline.job_new(user_id=1, username='admin')
    j = pipeline.job_get(jid)
    assert j['percent'] == 0
    assert j['step'] == 'Queued'
    assert j['done'] is False

    pipeline.job_set(jid, percent=50, step='Rendering')
    j = pipeline.job_get(jid)
    assert j['percent'] == 50
    assert j['step'] == 'Rendering'
    assert j['done'] is False  # untouched fields stay untouched


def test_result_round_trips_as_a_dict_not_a_json_string(tmp_path, monkeypatch):
    _use_temp_jobs_db(tmp_path, monkeypatch)
    jid = pipeline.job_new()
    pipeline.job_set(jid, done=True, result={'trailer_url': '/uploads/x.mp4', 'scenes': [1, 2, 3]})
    j = pipeline.job_get(jid)
    assert isinstance(j['result'], dict)
    assert j['result']['trailer_url'] == '/uploads/x.mp4'
    assert j['result']['scenes'] == [1, 2, 3]


def test_job_set_orig_name(tmp_path, monkeypatch):
    _use_temp_jobs_db(tmp_path, monkeypatch)
    jid = pipeline.job_new()
    pipeline.job_set_orig_name(jid, 'episode42.mp4')
    assert pipeline.job_get(jid)['orig_name'] == 'episode42.mp4'


def test_job_get_on_unknown_id_returns_none(tmp_path, monkeypatch):
    _use_temp_jobs_db(tmp_path, monkeypatch)
    assert pipeline.job_get('does-not-exist') is None


def test_job_set_on_unknown_id_is_a_no_op_not_an_error(tmp_path, monkeypatch):
    _use_temp_jobs_db(tmp_path, monkeypatch)
    pipeline.job_set('does-not-exist', percent=50)  # must not raise


def test_error_marks_the_job_done(tmp_path, monkeypatch):
    _use_temp_jobs_db(tmp_path, monkeypatch)
    jid = pipeline.job_new()
    pipeline.job_set(jid, error='Something went wrong')
    j = pipeline.job_get(jid)
    assert j['done'] is True
    assert j['status'] == 'error'
    assert j['error'] == 'Something went wrong'


def test_job_list_all_returns_every_job(tmp_path, monkeypatch):
    _use_temp_jobs_db(tmp_path, monkeypatch)
    jid1 = pipeline.job_new(user_id=1, username='alice')
    jid2 = pipeline.job_new(user_id=2, username='bob')
    all_jobs = pipeline.job_list_all()
    assert jid1 in all_jobs and jid2 in all_jobs
    assert all_jobs[jid1]['username'] == 'alice'
    assert all_jobs[jid2]['username'] == 'bob'


# --------------------------------------------------------------------------
# Whose job it is
# --------------------------------------------------------------------------
# Real, reported: Vertical Shorts jobs turned up in the episodic plug tab's
# "Your jobs". Every feature's jobs share this table and nothing in a row
# said which feature it belonged to, so that list -- "this account's jobs"
# -- was all of them.

def test_a_job_says_which_feature_it_belongs_to(tmp_path, monkeypatch):
    _use_temp_jobs_db(tmp_path, monkeypatch)
    plug = pipeline.job_new(user_id=1, username='ana')
    short = pipeline.job_new(user_id=1, username='ana', kind='shorts')
    sched = pipeline.job_new(user_id=1, username='ana', kind='schedule')
    assert [pipeline.job_get(j)['kind'] for j in (plug, short, sched)] == ['promo', 'shorts', 'schedule']
    assert {j['kind'] for j in pipeline.job_list_all().values()} == {'promo', 'shorts', 'schedule'}
    with pytest.raises(ValueError):
        pipeline.job_new(user_id=1, kind='trailer')
    assert len(pipeline.job_list_all()) == 3, 'and a job of no known kind is not created'


def test_a_jobs_table_from_before_kinds_is_filed_by_its_labels(tmp_path, monkeypatch):
    """On the server this is deployed to, the table already exists and holds
    the last hour's jobs. Left unfiled they would all still count as
    episodic plugs, and the fix would look like it had not worked."""
    path = str(tmp_path / 'old_jobs.db')
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE jobs (id TEXT PRIMARY KEY, percent INTEGER NOT NULL DEFAULT 0, step TEXT, '
                 'done INTEGER NOT NULL DEFAULT 0, error TEXT, status TEXT, result_json TEXT, '
                 'cancel_requested INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL, user_id INTEGER, '
                 'username TEXT, orig_name TEXT)')
    rows = [('a', 'EP101.mov'), ('b', 'EP101.mov (vertical shorts: analysis)'),
            ('c', 'EP101.mov (vertical shorts: 20 to render)'), ('d', 'EP101.mov (vertical shorts: analysis + render)'),
            ('e', 'Week 42.psd (schedule plug, 15s)'), ('f', None)]
    conn.executemany("INSERT INTO jobs (id, done, created, user_id, orig_name) VALUES (?, 1, ?, 7, ?)",
                     [(jid, time.time(), name) for jid, name in rows])
    conn.commit()
    conn.close()
    monkeypatch.setattr(pipeline, 'JOBS_DB_PATH', path)
    pipeline.jobs_db_init()
    kinds = {jid: j['kind'] for jid, j in pipeline.job_list_all().items()}
    assert kinds == {'a': 'promo', 'b': 'shorts', 'c': 'shorts', 'd': 'shorts', 'e': 'schedule', 'f': 'promo'}
    # Filed once, on the upgrade: a later start leaves what jobs say they are alone.
    plug = pipeline.job_new(user_id=7, kind='promo')
    pipeline.job_set_orig_name(plug, 'My Show (schedule plug, special).mov')
    pipeline.jobs_db_init()
    assert pipeline.job_get(plug)['kind'] == 'promo' and pipeline.job_get('e')['kind'] == 'schedule'
    assert pipeline.job_get(pipeline.job_new(user_id=7, kind='shorts'))['kind'] == 'shorts'


def _session(user_id=7, role='user', username='ana'):
    client = main.app.test_client()
    with client.session_transaction() as s:
        s.update(authed=True, user_id=user_id, username=username, role=role, csrf_token='t')
    return client


def test_the_episodic_plug_tab_lists_only_episodic_plug_jobs(tmp_path, monkeypatch):
    _use_temp_jobs_db(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, 'JOB_QUEUE', [])
    names = {}
    for kind, label, done in (('promo', 'EP101.mov', False), ('promo', 'EP100.mov', True),
                              ('shorts', 'EP101.mov (vertical shorts: analysis)', False),
                              ('shorts', 'EP099.mov (vertical shorts: 8 to render)', True),
                              ('schedule', 'Week 42.psd (schedule plug, 15s)', False)):
        jid = pipeline.job_new(user_id=7, username='ana', kind=kind)
        pipeline.job_set_orig_name(jid, label)
        pipeline.job_set(jid, percent=40, step='Working', status='running')
        if done:
            pipeline.job_set(jid, percent=100, done=True, status='done')
        names[label] = jid
    queued = pipeline.job_new(user_id=7, username='ana', kind='shorts')
    pipeline.job_set_orig_name(queued, 'EP102.mov (vertical shorts: analysis)')
    pipeline.JOB_QUEUE.append(queued)
    other = pipeline.job_new(user_id=8, username='ben')
    pipeline.job_set_orig_name(other, 'Somebody else.mov')

    ana = _session(role='admin')                          # an admin still sees only their own here
    d = ana.get('/api/monitor').get_json()
    assert [j['orig_name'] for j in d['active']] == ['EP101.mov']
    assert [j['orig_name'] for j in d['finished']] == ['EP100.mov'] and d['queued'] == []
    assert {j['kind'] for j in d['active'] + d['finished']} == {'promo'}
    # What the tab picks up after a reload to carry on showing: never another feature's job.
    assert d['active'][0]['job_id'] == names['EP101.mov']

    every = ana.get('/api/monitor?kind=all').get_json()
    assert len(every['active']) == 3 and len(every['finished']) == 2
    assert [j['orig_name'] for j in every['queued']] == ['EP102.mov (vertical shorts: analysis)']
    assert all(j['user_id'] == 7 for k in ('active', 'queued', 'finished') for j in every[k])
    only = ana.get('/api/monitor?kind=shorts').get_json()
    assert [len(only[k]) for k in ('active', 'queued', 'finished')] == [1, 1, 1]
    assert {j['kind'] for k in ('active', 'queued', 'finished') for j in only[k]} == {'shorts'}
    assert [j['orig_name'] for j in ana.get('/api/monitor?kind=schedule').get_json()['active']] == \
        ['Week 42.psd (schedule plug, 15s)']
    assert ana.get('/api/monitor?kind=everything').status_code == 400

    # The admin view is every job of every kind and account, each saying what it is.
    jobs = ana.get('/api/admin/jobs').get_json()
    assert len(jobs['active']) == 4 and len(jobs['queued']) == 1 and len(jobs['finished']) == 2
    assert sorted(j['kind'] for j in jobs['active']) == ['promo', 'promo', 'schedule', 'shorts']
    assert jobs['queued'][0]['kind'] == 'shorts'
    assert _session(user_id=8, username='ben').get('/api/admin/jobs').status_code == 403


def test_the_page_asks_for_the_right_jobs_in_each_place():
    html = _session(role='admin').get('/').get_data(as_text=True)
    # The episodic tab's list and its resume-after-reload: episodic jobs only (the default).
    assert html.count("fetch('/api/monitor')") == 2
    # The Dashboard's count: everything this account has running.
    assert html.count("fetch('/api/monitor?kind=all')") == 1
    # The other tabs no longer refresh a list their jobs are not in.
    for mark in ('SH.jobId = start.job_id', 'SP.jobId = start.job_id'):
        at = [m.end() for m in re.finditer(re.escape(mark), html)]
        assert at and all('refreshMonitor' not in html[a:a + 80] for a in at), mark
    # File names are shown as text in that list, not as markup.
    monitor = html[html.index('function renderActive(items)'):html.index('window.refreshMonitor = refreshMonitor')]
    assert "(j.orig_name || 'Untitled')" in monitor
    assert monitor.count("escapeHtmlLite(j.orig_name || 'Untitled')") == monitor.count("(j.orig_name || 'Untitled')") == 3


class TestCancellation:
    def test_cancel_a_queued_job_marks_it_cancelled(self, tmp_path, monkeypatch):
        _use_temp_jobs_db(tmp_path, monkeypatch)
        jid = pipeline.job_new()
        pipeline.JOB_QUEUE.append(jid)
        try:
            ok = pipeline.job_cancel(jid)
            assert ok is True
            j = pipeline.job_get(jid)
            assert j['done'] is True
            assert j['error'] == 'Cancelled'
            assert jid not in pipeline.JOB_QUEUE
        finally:
            if jid in pipeline.JOB_QUEUE:
                pipeline.JOB_QUEUE.remove(jid)

    def test_cancelling_a_running_job_raises_on_its_next_progress_update(self, tmp_path, monkeypatch):
        _use_temp_jobs_db(tmp_path, monkeypatch)
        jid = pipeline.job_new()  # not in JOB_QUEUE -- simulates an already-running job
        pipeline.job_cancel(jid)
        raised = False
        try:
            pipeline.job_set(jid, percent=10, step='still going')
        except pipeline.JobCancelled:
            raised = True
        assert raised

    def test_cannot_cancel_an_already_finished_job(self, tmp_path, monkeypatch):
        _use_temp_jobs_db(tmp_path, monkeypatch)
        jid = pipeline.job_new()
        pipeline.job_set(jid, done=True, status='success')
        ok = pipeline.job_cancel(jid)
        assert ok is False


class TestRestartInterruption:
    """The actual point of moving this off an in-memory dict: a job that was
    running when the process stopped should be truthfully marked as
    interrupted on the next startup, not silently lost (the in-memory
    version) and not left looking like it's still running forever."""

    def test_unfinished_job_marked_interrupted_after_reinit(self, tmp_path, monkeypatch):
        _use_temp_jobs_db(tmp_path, monkeypatch)
        jid = pipeline.job_new()
        pipeline.job_set(jid, percent=45, step='Rendering scene 3/8')

        # Simulate a real process restart: re-run exactly what happens at
        # actual server startup, against the SAME db file (not a fresh one).
        pipeline.jobs_db_init()

        j = pipeline.job_get(jid)
        assert j['done'] is True
        assert j['status'] == 'error'
        assert 'restart' in j['error'].lower()

    def test_completed_job_is_untouched_by_reinit(self, tmp_path, monkeypatch):
        _use_temp_jobs_db(tmp_path, monkeypatch)
        jid = pipeline.job_new()
        pipeline.job_set(jid, done=True, status='success', result={'trailer_url': '/uploads/done.mp4'})
        pipeline.jobs_db_init()
        j = pipeline.job_get(jid)
        assert j['status'] == 'success'
        assert j['result']['trailer_url'] == '/uploads/done.mp4'

    def test_queued_job_also_marked_interrupted(self, tmp_path, monkeypatch):
        # A job that never even started rendering is just as much "not
        # actually happening anymore" after a restart as one that was
        # halfway through -- both are done=0 before the restart.
        _use_temp_jobs_db(tmp_path, monkeypatch)
        jid = pipeline.job_new()  # freshly created, done=0, status='queued'
        pipeline.jobs_db_init()
        j = pipeline.job_get(jid)
        assert j['done'] is True
        assert j['status'] == 'error'
