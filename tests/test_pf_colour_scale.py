import numpy as np
from ebsd_autoplot import texture_v3


def test_sharp_pf_scale_covers_the_actual_peak(tmp_path,monkeypatch):
    seen=[];original=texture_v3.Normalize
    def normalize(vmin,vmax):
        seen.append((vmin,vmax));return original(vmin,vmax)
    monkeypatch.setattr(texture_v3,'Normalize',normalize)
    poles=np.repeat([[.9949409123839288,.10046183785216795,0]],50,axis=0)
    mud=texture_v3._plot_pf(tmp_path/'rim',poles,title='Synthetic sharp rim',formats=('svg',))
    assert seen[-1][0]==0 and seen[-1][1]>=np.nanmax(mud)
