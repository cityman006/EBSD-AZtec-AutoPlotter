import json
import shutil
import pytest
from ebsd_autoplot.batch import inventory, create_batch, execute_batch, request_cancel
from test_pipeline import write_synth


def test_twenty_independent_files_and_twenty_shards(tmp_path):
    raw=tmp_path/'independent';raw.mkdir()
    write_synth(tmp_path/'base',h=20,w=4)
    for i in range(20):shutil.copyfile(tmp_path/'base/sample.h5oina',raw/f'sample_{i}.h5')
    records=inventory([raw]);assert len(records)==20
    assert len({r['scan_id'] for r in records})==20
    import h5py
    folder=tmp_path/'montage';folder.mkdir()
    parts=[]
    for i in range(20):
        name=f'scan01_rows{i:06d}-{i:06d}_EBSD.h5'
        with h5py.File(tmp_path/'base/sample.h5oina','r') as src,h5py.File(folder/name,'w') as dst:
            src.copy('1',dst)
            for key in list(dst['1/EBSD/Data']):
                data=dst['1/EBSD/Data'][key][i*4:(i+1)*4];del dst['1/EBSD/Data'][key];dst['1/EBSD/Data'][key]=data
        parts.append({'file':name,'row_start':i,'row_end_exclusive':i+1})
    (folder/'manifest.json').write_text(json.dumps({'source_name':'original.h5oina','scans':[{'scan_path':'/1/EBSD','parts':parts}]}))
    grouped=inventory([folder,*folder.glob('*.h5')]);assert len(grouped)==1
    assert grouped[0]['parts']==20 and grouped[0]['ny']==20


def test_failure_continues_snapshot_recovery_and_unique_runs(tmp_path):
    write_synth(tmp_path/'a',h=4,w=4);write_synth(tmp_path/'b',h=4,w=4)
    paths=[tmp_path/'a/sample.h5oina',tmp_path/'b/sample.h5oina']
    options=['--analysis','v1','--no-ipf','--no-render']
    run=create_batch(paths,tmp_path/'out',options)
    (paths[0]).write_bytes(b'changed')
    result=execute_batch(run,progress=lambda _:None)
    assert [r['status'] for r in result['records']]==['failed','success']
    assert (run/'batch_failures.csv').is_file()
    # Resume verifies retained successes instead of overwriting their run.
    original=result['records'][1]['run_path']
    resumed=execute_batch(run,progress=lambda _:None)
    assert resumed['records'][1]['run_path']==original
    assert create_batch([paths[1]],tmp_path/'out',options)!=run


def test_cancelled_batch_has_no_false_success(tmp_path):
    write_synth(tmp_path/'a')
    run=create_batch([tmp_path/'a/sample.h5oina'],tmp_path/'out',['--no-ipf','--no-render'])
    request_cancel(run); result=execute_batch(run,progress=lambda _:None)
    assert result['status']=='cancelled' and result['records'][0]['status']=='queued'
    with pytest.raises(ValueError):create_batch([tmp_path/'a'],tmp_path/'out',['--output-root','elsewhere'])
