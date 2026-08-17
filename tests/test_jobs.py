"""
Tests for job_new/job_set/job_get/job_cancel/job_list_all/job_set_orig_name --
the SQLite-backed job tracking that replaced an in-memory dict. The whole
point of this layer is durability across a restart, so that's the behavior
most worth protecting with a real test, alongside the basic CRUD lifecycle.
"""
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
