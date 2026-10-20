import base64
import io
from pathlib import Path
import xml.etree.ElementTree as ET
import h5py
from PIL import Image
from ebsd_autoplot.compute import prepare
from ebsd_autoplot.io import H5Source
from ebsd_autoplot.render import maps
from test_pipeline import write_synth


def test_export_embeds_native_pixels_and_vector_labels(tmp_path):
    write_synth(tmp_path/'raw',h=17,w=25)
    cache=prepare(H5Source(tmp_path/'raw/sample.h5oina'),tmp_path/'cache',do_ipf=False,progress=lambda _:None)
    out=tmp_path/'render'
    maps(cache,out,formats=('png','tiff','pdf','svg'),style={'maps':['Phase','BC'],'layout':(2,3)},progress=lambda _:None)
    assert Image.open(out/'native_maps/BC.tiff').size==(25,17)
    assert Image.open(out/'single_maps/BC.png').size==(25,17)
    svg=ET.parse(out/'single_maps/BC.svg');images=svg.findall('.//{http://www.w3.org/2000/svg}image')
    assert len(images)==1
    payload=images[0].attrib['{http://www.w3.org/1999/xlink}href'].split(',',1)[1]
    assert Image.open(io.BytesIO(base64.b64decode(payload))).size==(25,17)
    assert svg.findall('.//{http://www.w3.org/2000/svg}text')
    assert (out/'single_maps/BC.pdf').read_bytes().startswith(b'%PDF-')
    assert not (out/'native_maps/KAM.png').exists()
