"""Exercise linked routing with real Qt graphics widgets and injected MapView hooks."""
import os
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore as C, QtGui as G, QtWidgets as W

from ebsd_autoplot.linked_views import LinkedMapWorkspace


class Viewer(W.QGraphicsView):
    navigation_changed=C.Signal(object); cursor_moved=C.Signal(object)
    roi_selected=C.Signal(tuple); coordinate=C.Signal(str)
    def __init__(self):
        super().__init__(); self.setScene(W.QGraphicsScene(self))
        self.cache=None; self.field="BC"; self.width_px=self.height_px=0
        self.meta={}; self._state={"center": (50.,40.), "scale":1.}
        self.cursor=None; self.roi=None; self.load_calls=[]; self.navigation_applies=0
    def load_cache(self,path,field="BC"):
        self.load_calls.append((str(path),field))
        if field == "IPF_Z" and "missing" in Path(path).name: raise ValueError("IPF_Z not present in selected source")
        self.cache=path; self.field=field; self.meta={"x_step_um":2.,"y_step_um":3.}
        self.width_px,self.height_px=100,80
        self.scene().setSceneRect(0,0,100,80)
        self.apply_navigation({"center":(50.,40.),"scale":1.})
    def clear_cache(self):
        self.cache=None; self.width_px=self.height_px=0; self.scene().clear()
        self.cursor=self.roi=None
    def navigation_state(self): return dict(self._state)
    def apply_navigation(self,state):
        self._state=dict(state); self.navigation_applies+=1
        self.setTransform(G.QTransform.fromScale(state["scale"],state["scale"]*1.5))
        self.centerOn(*state["center"])
        # A noisy viewer hook must still not cause routing recursion.
        self.navigation_changed.emit(dict(state))
    def set_linked_cursor(self,value): self.cursor=value
    def set_linked_roi(self,value): self.roi=value
    def move_camera(self,center,scale):
        self.apply_navigation({"center":tuple(center),"scale":scale})


@pytest.fixture(scope="module")
def app():
    return W.QApplication.instance() or W.QApplication([])


@pytest.fixture
def workspace(app,tmp_path):
    first=tmp_path/"source.h5"; first.write_bytes(b"unchanged cache source")
    widget=LinkedMapWorkspace(Viewer,language=lambda:"en")
    widget.resize(900,600); widget.show(); app.processEvents()
    assert widget.load_cache(first) == [True]*4
    yield widget,first
    widget.close(); widget.deleteLater(); app.processEvents()


def test_layout_persistence_and_active_view(workspace,app):
    widget,_=workspace
    original=tuple(widget.views)
    assert widget.view_count==1
    widget.set_layout(2); app.processEvents()
    assert widget.frames[0].isVisible() and widget.frames[1].isVisible() and not widget.frames[2].isVisible()
    widget.set_layout(4); widget.set_active_view(3)
    assert widget.active_view is widget.views[3]
    widget.set_layout(1)
    assert widget.active_index==0 and tuple(widget.views)==original
    widget.set_layout(4)
    assert all(view.cache is not None for view in widget.views)
    with pytest.raises(ValueError): widget.set_layout(3)
    widget.retranslate("zh")
    assert widget.layout_label.text()=="视图数量"
    phase_index=widget.map_selectors[0].findData("Phase")
    assert widget.map_selectors[0].itemText(phase_index)=="相位"
    widget.set_field("Phase",index=0)
    assert widget.views[0].field=="Phase" and widget.map_selectors[0].currentData()=="Phase"


def test_native_navigation_cursor_roi_link_and_link_off(workspace):
    widget,_=workspace; widget.set_layout(4)
    widget.views[0].move_camera((23.,31.),2.5)
    for view in widget.views:
        assert view.navigation_state()=={"center":(23.,31.),"scale":2.5}
        assert view.transform().m11()==2.5 and view.transform().m22()==3.75
        assert view.navigation_applies<10  # recursive echoes blocked
    widget.views[2].cursor_moved.emit((11.,17.))
    assert all(view.cursor==(11.,17.) for view in widget.views)
    rois=[]; widget.roi_selected.connect(rois.append)
    widget.views[1].roi_selected.emit((7,9,12,15))
    assert all(view.roi==(7,9,12,15) for view in widget.views)
    assert rois==[(7,9,12,15)] and widget.active_index==1
    widget.set_linked(False)
    assert all(view.cursor is None for view in widget.views)
    widget.views[0].move_camera((40.,42.),4.)
    assert widget.views[1].navigation_state()["scale"]==2.5
    widget.views[0].cursor_moved.emit((3.,4.))
    assert widget.views[0].cursor==(3.,4.) and widget.views[1].cursor is None


def test_field_switch_preserves_only_target_camera_and_map(workspace):
    widget,_=workspace; widget.set_layout(4)
    widget.views[2].move_camera((13.,19.),3.)
    widget.views[2].cursor_moved.emit((6.,8.))
    widget.views[2].roi_selected.emit((3,4,15,20))
    previous=[view.field for view in widget.views]
    calls=[len(view.load_calls) for view in widget.views]
    widget.set_field("IPF_X",index=2)
    assert widget.views[2].field=="IPF_X"
    assert widget.views[2].navigation_state()=={"center":(13.,19.),"scale":3.}
    assert [view.field for view in widget.views[:2]]==previous[:2]
    assert len(widget.views[0].load_calls)==calls[0]
    assert "Δx 2" in widget.status_labels[2].text()
    assert all(view.roi==(3,4,15,20) for view in widget.views)
    assert all(view.cursor==(6.,8.) for view in widget.views)
    widget.set_layout(1); widget.set_layout(4)
    assert all(view.navigation_state()=={"center":(13.,19.),"scale":3.} for view in widget.views)
    assert all(view.roi==(3,4,15,20) for view in widget.views)


def test_different_cache_and_changed_source_never_link_stale_maps(workspace,tmp_path):
    widget,first=workspace; widget.set_layout(4)
    second=tmp_path/"other.h5"; second.write_bytes(b"different source")
    assert widget.load_view_cache(1,second,"Phase")
    before=widget.views[1].navigation_state()
    widget.views[0].move_camera((25.,37.),7.)
    assert widget.views[1].navigation_state()==before
    widget.views[0].cursor_moved.emit((5.,6.))
    assert widget.views[1].cursor is None
    first.write_bytes(b"changed source cache requires reload")
    widget.views[0].cursor_moved.emit((9.,10.))
    assert widget.views[0].cache is None and widget.views[2].cache is None
    assert widget.views[0].cursor is None and widget.views[0].roi is None
    assert "reload" in widget.status_labels[0].text()
    assert widget.views[1].cache==second


def test_missing_map_clears_prior_picture_and_never_substitutes(workspace,tmp_path):
    widget,_=workspace; widget.set_layout(4)
    missing=tmp_path/"missing_ipf.h5"; missing.write_bytes(b"source without IPF")
    errors=[]; widget.load_error.connect(lambda index,error: errors.append((index,error)))
    assert widget.load_view_cache(0,missing,"IPF_Z") is False
    assert widget.views[0].cache is None and widget.views[0].width_px==0
    assert widget.map_selectors[0].currentText()=="IPF_Z"
    assert "Unavailable" in widget.status_labels[0].text()
    assert errors==[(0,"IPF_Z not present in selected source")]
    assert widget.views[1].cache is not None
