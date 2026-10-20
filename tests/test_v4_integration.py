import csv
import json
from ebsd_autoplot.batch import create_batch, execute_batch
from test_pipeline import write_synth


def test_quantitative_batch_end_to_end(tmp_path):
    write_synth(tmp_path/'raw',h=12,w=16)
    run=create_batch([tmp_path/'raw/sample.h5oina'],tmp_path/'out',
        ['--analysis','v4_quantitative','--min-grain-pixels','1','--no-ipf','--no-render'])
    result=execute_batch(run,progress=lambda _:None)
    assert result['status']=='success',result
    mp=json.loads((__import__('pathlib').Path(result['records'][0]['run_path'])/'analysis_manifest.json').read_text())
    assert 'v4_quantitative' in mp['checkpoints'] and mp['status']=='success'
    with (run/'Summary.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==2 and rows[0]['source_name']=='sample.h5oina'
    assert 'unavailable' in rows[1]['RX_status']
