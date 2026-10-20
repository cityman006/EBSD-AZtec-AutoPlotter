"""Windows control-file sharing races retain atomic old/new JSON records."""
import json
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from ebsd_autoplot.storage import atomic_json


def windows_permission(code):
    error=PermissionError(13,'simulated Windows sharing/access violation')
    error.winerror=code
    return error


@pytest.mark.parametrize('winerror',[5,32,33])
def test_transient_windows_share_retries_keep_old_record_until_atomic_commit(tmp_path,winerror):
    path=tmp_path/'control.json';path.write_text('{"cancel": false,"old": true}',encoding='utf-8')
    previous=path.read_bytes();new={'cancel':True,'name':'中文'}
    import ebsd_autoplot.storage as storage
    real_replace=storage.os.replace;calls=[]
    def contested_replace(source,target):
        calls.append((source,target))
        assert path.read_bytes()==previous
        assert json.loads(Path(source).read_text(encoding='utf-8'))==new
        if len(calls)<=2:raise windows_permission(winerror)
        return real_replace(source,target)
    with patch('ebsd_autoplot.storage.os.replace',side_effect=contested_replace),patch('ebsd_autoplot.storage.time.sleep') as sleep:
        atomic_json(path,new)
    assert len(calls)==3 and sleep.call_count==2
    assert json.loads(path.read_text(encoding='utf-8'))==new
    assert not list(tmp_path.glob('.j*.tmp'))


@pytest.mark.parametrize('error',[OSError(22,'invalid replacement'),windows_permission(87),PermissionError(13,'POSIX/non-Windows access denied')])
def test_nonretryable_error_is_immediate_and_removes_only_owned_temp(tmp_path,error):
    path=tmp_path/'control.json';path.write_text('{"old": true}',encoding='utf-8');before=path.read_bytes()
    unrelated=tmp_path/'.j_unrelated.tmp';unrelated.write_text('user data')
    with patch('ebsd_autoplot.storage.os.replace',side_effect=error) as replace,patch('ebsd_autoplot.storage.time.sleep') as sleep:
        with pytest.raises(type(error)):atomic_json(path,{'new':True})
    assert replace.call_count==1 and not sleep.called
    assert path.read_bytes()==before and unrelated.read_text()=='user data'
    assert list(tmp_path.glob('.j*.tmp'))==[unrelated]


def test_permanent_share_violation_has_bounded_timeout_and_preserves_old_record(tmp_path):
    path=tmp_path/'control.json';path.write_text('{"old": true}',encoding='utf-8');before=path.read_bytes()
    clock=[100.];attempts=[];sleeps=[]
    def replace(source,target):
        attempts.append(clock[0]);assert path.read_bytes()==before
        assert Path(source).exists();raise windows_permission(5)
    def sleep(delay):
        sleeps.append(delay);clock[0]+=delay
    with patch('ebsd_autoplot.storage.os.replace',side_effect=replace),patch('ebsd_autoplot.storage.time.monotonic',side_effect=lambda:clock[0]),patch('ebsd_autoplot.storage.time.sleep',side_effect=sleep):
        with pytest.raises(PermissionError):atomic_json(path,{'new':True})
    assert len(attempts)>1 and 1.99<=sum(sleeps)<=2.001
    assert max(sleeps)<=.1 and path.read_bytes()==before
    assert not list(tmp_path.glob('.j*.tmp'))


def test_serialization_failure_preserves_old_file_and_cleans_owned_partial(tmp_path):
    path=tmp_path/'control.json';path.write_text('{"old": true}',encoding='utf-8');before=path.read_bytes()
    with pytest.raises(TypeError):atomic_json(path,{'unsupported':object()})
    assert path.read_bytes()==before and not list(tmp_path.glob('.j*.tmp'))
