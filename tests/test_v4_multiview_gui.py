import os,json
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import h5py,numpy as np,pytest
pytest.importorskip('PySide6')
from PySide6 import QtWidgets as W,QtCore as C
from gui_qt import Studio


def cache_file(path):
    h,w=800,1000
    raw=np.zeros((7,h,w),np.float32);raw[3]=1;raw[4]=7;raw[5]=1.;raw[6]=128
    with h5py.File(path,'w') as f:
        f['features']=raw;f['indexed']=np.ones((h,w),np.uint8);f['kam_valid']=np.ones((h,w),np.uint8)
        for axis in 'XYZ':f['ipf_'+axis]=np.full((h,w,3),[45,90,180],np.uint8)
        f.attrs['metadata_json']=json.dumps({'x_step_um':2.,'y_step_um':3.,'phase_specs':{'7':{'name':'alpha','rgb':[10,20,30],'space_group':194}}})


def test_real_map_views_linked_native_camera_roi_and_language(tmp_path):
    app=W.QApplication.instance() or W.QApplication([])
    cache=tmp_path/'features.h5';cache_file(cache)
    win=Studio();win.preferences=tmp_path/'preferences.json';win.resize(1600,1100);win.show()
    win.nav.setCurrentRow(2)
    win.workspace.set_linked(True);win.workspace.set_layout(4);app.processEvents()
    assert all(win.workspace.load_cache(cache))
    assert [v.field for v in win.workspace.views]==['BC','Phase','KAM','IPF_Z']
    first=win.workspace.views[0];first.apply_navigation({'center':(500.,400.),'scale':3.})
    state=first.navigation_state();first.navigation_changed.emit(state)
    for view in win.workspace.views:
        assert view.navigation_state()['scale']==pytest.approx(3.)
        assert view.navigation_state()['center']==pytest.approx(state['center'],abs=.5)
        assert view.transform().m22()/view.transform().m11()==pytest.approx(1.5)
    first.cursor_moved.emit((42,35));first.roi_selected.emit((40,30,120,100))
    assert win.roi.text()=='40 30 120 100'
    assert all(len(v.cursor_items)==2 and v.roi_item.rect().width()==120 for v in win.workspace.views)
    win.workspace.set_field('BC',index=2)
    assert [v.field for v in win.workspace.views]==['BC','Phase','BC','IPF_Z']
    assert win.workspace.views[2].roi_item is not None
    win.toggle_language()
    assert win.workspace.views[2].navigation_state()['scale']==pytest.approx(3.)
    assert win.roi.text()=='40 30 120 100'
    win.workspace.set_linked(False)
    before=win.workspace.views[1].navigation_state()
    first.apply_navigation({'center':(620,520),'scale':4});first.navigation_changed.emit(first.navigation_state())
    assert win.workspace.views[1].navigation_state()==before
    win.workspace.set_layout(1);win.workspace.set_layout(4);app.processEvents()
    assert win.workspace.views[1].navigation_state()['scale']==pytest.approx(3.)
    # Latest V4 adds tile workers; wait until their signal receivers are alive.
    for view in win.workspace.views:
        if hasattr(view,'generation'):view.generation+=1;view.tile_timer.stop()
    C.QThreadPool.globalInstance().waitForDone(10000);app.processEvents()
    win.close();app.processEvents()


def test_cache_loaded_before_show_fits_active_panel_after_layout(tmp_path):
    from PySide6 import QtTest
    app=W.QApplication.instance() or W.QApplication([])
    cache=tmp_path/'features.h5';cache_file(cache)
    win=Studio();win.preferences=tmp_path/'preferences.json';win.resize(1600,1100)
    win.nav.setCurrentRow(2);win.workspace.set_layout(4);win.workspace.load_cache(cache);win.show()
    QtTest.QTest.qWait(180);app.processEvents()
    view=win.viewer
    ideal=min((view.viewport().width()-10)/view.width_px,(view.viewport().height()-10)/(view.height_px*view.aspect_ratio))
    try:assert view.transform().m11()==pytest.approx(ideal,rel=.05)
    finally:
        for v in win.workspace.views:
            if hasattr(v,'generation'):v.generation+=1;v.tile_timer.stop()
        C.QThreadPool.globalInstance().waitForDone(10000);app.processEvents();win.close();app.processEvents()
