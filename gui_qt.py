"""Qt Materials Intelligence Studio. Run with the optional requirements-gui.txt."""
from __future__ import annotations
import json
import sys
import os
from pathlib import Path
import h5py
import numpy as np
try:
    from PySide6 import QtCore as C, QtGui as G, QtWidgets as W
except ImportError:
    raise SystemExit('PySide6 is optional. Install requirements-gui.txt or use gui.py (Tk).')
from ebsd_autoplot.storage import atomic_json
from ebsd_autoplot.batch import create_batch, inventory, request_cancel

ROOT=Path(__file__).resolve().parent
NAV=[('Dashboard','总览'),('Data Manager','数据管理'),('EBSD Maps','EBSD 地图'),
     ('Crystallography','晶体学'),('Grain & Recrystallization','晶粒与再结晶'),
     ('Batch Processing','批量处理'),('Slicing & Dataset','切片与数据集'),
     ('Multimodal EDS','多模态 EDS'),('GNN / Mechanics / Diffusion','AI 研究'),
     ('Publication Studio','论文出图'),('Results History','结果历史'),('Settings','设置')]


class ImportWorker(C.QThread):
    result=C.Signal(object); error=C.Signal(str)
    def __init__(self, paths):
        super().__init__(); self.paths=tuple(paths)
    def run(self):
        try: self.result.emit(inventory(self.paths))
        except Exception as exc: self.error.emit(str(exc))


class MapView(W.QGraphicsView):
    coordinate=C.Signal(str); roi_selected=C.Signal(tuple)
    def __init__(self):
        super().__init__(); self.setScene(W.QGraphicsScene(self))
        self.setDragMode(W.QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True); self.setTransformationAnchor(W.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.cache=None; self.field='BC'; self.width_px=0; self.height_px=0; self.start=None
        self.roi_item=None
    def load_cache(self, path, field='BC'):
        self.cache=Path(path); self.field=field
        with h5py.File(self.cache,'r') as f:
            self.height_px,self.width_px=f['indexed'].shape
            self.meta=json.loads(f.attrs['metadata_json'])
            step=max(1,int(np.ceil(max(self.width_px,self.height_px)/1400)))
            if field.startswith('IPF'):
                key='ipf_'+field[-1]
                if key not in f: raise ValueError('Selected cache has no calibrated orientation IPF data')
                arr=f[key][::step,::step,:]
            elif field=='Phase':
                ph=f['features'][4,::step,::step].astype(int); arr=np.zeros((*ph.shape,3),np.uint8)
                for key,spec in self.meta['phase_specs'].items(): arr[ph==int(key)]=spec['rgb']
            else:
                values=f['features'][5 if field=='KAM' else 6,::step,::step]
                valid=f['kam_valid' if field=='KAM' else 'indexed'][::step,::step]>0
                from matplotlib import colormaps
                arr=(colormaps['viridis' if field=='KAM' else 'gray'](np.clip(values/(5 if field=='KAM' else 255),0,1))[...,:3]*255).astype('u1')
                arr[~valid]=0
        self._image(arr, self.width_px, self.height_px)
    def load_image(self, path):
        self.cache=None
        reader=G.QImageReader(str(path)); size=reader.size()
        self.width_px=size.width(); self.height_px=size.height()
        reader.setScaledSize(size.scaled(1600,1600,C.Qt.AspectRatioMode.KeepAspectRatio))
        img=reader.read()
        if img.isNull(): raise ValueError(reader.errorString())
        self._set_pixmap(G.QPixmap.fromImage(img),size.width(),size.height())
    def _image(self, arr, width, height):
        arr=np.ascontiguousarray(arr,dtype='u1')
        img=G.QImage(arr.data,arr.shape[1],arr.shape[0],arr.strides[0],G.QImage.Format.Format_RGB888).copy()
        self._set_pixmap(G.QPixmap.fromImage(img),width,height)
    def _set_pixmap(self,pix,width,height):
        self.scene().clear(); self.roi_item=None
        item=self.scene().addPixmap(pix); item.setTransform(G.QTransform.fromScale(width/pix.width(),height/pix.height()))
        self.scene().setSceneRect(0,0,width,height); self.fitInView(self.sceneRect(),C.Qt.AspectRatioMode.KeepAspectRatio)
    def wheelEvent(self,event):
        self.scale(1.2 if event.angleDelta().y()>0 else 1/1.2,1.2 if event.angleDelta().y()>0 else 1/1.2)
    def mousePressEvent(self,event):
        if event.modifiers() & C.Qt.KeyboardModifier.ControlModifier:
            self.start=self.mapToScene(event.position().toPoint()); event.accept(); return
        super().mousePressEvent(event)
    def mouseReleaseEvent(self,event):
        if self.start is not None:
            stop=self.mapToScene(event.position().toPoint()); rect=C.QRectF(self.start,stop).normalized().intersected(self.sceneRect())
            self.start=None
            x,y=max(0,int(rect.x())),max(0,int(rect.y()))
            w,h=min(self.width_px-x,int(rect.width())),min(self.height_px-y,int(rect.height()))
            if w>0 and h>0:
                if self.roi_item is not None: self.scene().removeItem(self.roi_item)
                pen=G.QPen(G.QColor('#f59e0b')); pen.setCosmetic(True); pen.setWidth(2)
                self.roi_item=self.scene().addRect(x,y,w,h,pen); self.roi_selected.emit((x,y,w,h))
            event.accept(); return
        super().mouseReleaseEvent(event)
    def mouseMoveEvent(self,event):
        pos=self.mapToScene(event.position().toPoint()); x,y=int(pos.x()),int(pos.y())
        if 0<=x<self.width_px and 0<=y<self.height_px:
            msg=f'X={x} Y={y}'
            if self.cache:
                with h5py.File(self.cache,'r') as f:
                    vals=f['features'][:,y,x]
                    msg+=f' | {x*self.meta["x_step_um"]:.3f}, {y*self.meta["y_step_um"]:.3f} µm | Phase {int(vals[4])} | BC {vals[6]:.1f}'
                    msg+=f' | KAM {vals[5]:.3f}°' if f['kam_valid'][y,x] else ' | KAM unavailable'
            self.coordinate.emit(msg)
        super().mouseMoveEvent(event)


class Studio(W.QMainWindow):
    def __init__(self):
        super().__init__()
        # Offscreen Windows Qt has no platform font database. Load local system
        # fonts explicitly; no font files are redistributed with this project.
        if not G.QFontDatabase.families() and os.name=='nt':
            fonts=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
            for name in ('segoeui.ttf','msyh.ttc'):
                if (fonts/name).is_file():G.QFontDatabase.addApplicationFont(str(fonts/name))
        self.setWindowTitle('EBSD-AZtec AutoPlotter V4 — Materials Intelligence Studio')
        self.resize(1460,920); self.setAcceptDrops(True)
        self.preferences=ROOT/'config/studio_preferences.json'
        try: self.saved=json.loads(self.preferences.read_text(encoding='utf-8'))
        except (OSError,ValueError): self.saved={}
        self.paths=[]; self.records=[]; self.process=None; self.batch_run=None; self.worker=None
        self.active_batch_run=None; self.active_control=None
        self.lang=self.saved.get('language','zh'); self.dark=self.saved.get('dark',True)
        self._build(); self.set_theme(self.dark); self.translate()
        self.timer=C.QTimer(self); self.timer.timeout.connect(self.refresh_queue); self.timer.start(600)
    def _build(self):
        toolbar=self.addToolBar('Workspace'); toolbar.setMovable(False)
        self.add_files=W.QPushButton(); self.add_files.clicked.connect(self.choose_files); toolbar.addWidget(self.add_files)
        self.add_folder=W.QPushButton(); self.add_folder.clicked.connect(self.choose_folder); toolbar.addWidget(self.add_folder)
        self.open_cache=W.QPushButton('Open cache / 打开缓存'); self.open_cache.clicked.connect(self.choose_cache); toolbar.addWidget(self.open_cache)
        self.map_select=W.QComboBox(); self.map_select.addItems(['BC','Phase','KAM','IPF_X','IPF_Y','IPF_Z']); self.map_select.currentTextChanged.connect(self.change_map); toolbar.addWidget(self.map_select)
        self.info=W.QLabel('Import H5 / H5OINA — AZtec color not calibrated'); toolbar.addWidget(self.info)
        self.nav=W.QListWidget(); self.nav.currentRowChanged.connect(self.navigate)
        left=W.QDockWidget('Workspace / 工作区',self); left.setWidget(self.nav); self.addDockWidget(C.Qt.DockWidgetArea.LeftDockWidgetArea,left)
        self.viewer=MapView(); self.viewer.coordinate.connect(self.statusBar().showMessage); self.viewer.roi_selected.connect(self.set_roi)
        self.setCentralWidget(self.viewer)
        self.params=W.QWidget(); form=W.QFormLayout(self.params)
        self.help=W.QLabel(); self.help.setWordWrap(True); form.addRow(self.help)
        self.mode=W.QComboBox(); self.mode.addItems(['v1','v1v2','v3_texture','v3_schmid','v3_variants','v3_prior_beta','v4_quantitative'])
        form.addRow('Analysis / 分析',self.mode)
        self.roi=W.QLineEdit(); self.roi.setPlaceholderText('X Y WIDTH HEIGHT (blank = full scan)'); form.addRow('ROI pixels',self.roi)
        self.axis=W.QLineEdit('0 0 1'); form.addRow('Load axis X Y Z',self.axis)
        self.block=W.QSpinBox(); self.block.setRange(1,1024); self.block.setValue(32); form.addRow('Block rows',self.block)
        self.budget=W.QSpinBox(); self.budget.setRange(16,65536); self.budget.setValue(2048); self.budget.setSuffix(' MB'); form.addRow('Memory budget',self.budget)
        self.weight=W.QComboBox(); self.weight.addItems(['area','grain']); form.addRow('ODF weighting',self.weight)
        self.font=W.QComboBox()
        from matplotlib.font_manager import fontManager
        fonts=sorted(set(x.name for x in fontManager.ttflist)); self.font.addItems(fonts)
        self.font.setCurrentText('Arial' if 'Arial' in fonts else 'DejaVu Sans'); form.addRow('Export font',self.font)
        self.formats={}
        fmtrow=W.QWidget(); fl=W.QHBoxLayout(fmtrow); fl.setContentsMargins(0,0,0,0)
        for fmt in ('png','tiff','pdf','svg'):
            cb=W.QCheckBox(fmt.upper()); cb.setChecked(fmt=='png'); fl.addWidget(cb); self.formats[fmt]=cb
        form.addRow('Formats',fmtrow)
        self.map_checks={}
        maps_widget=W.QWidget(); ml=W.QGridLayout(maps_widget)
        for i,name in enumerate(['IPF_X','IPF_Y','IPF_Z','Phase','KAM','BC','MAD','PQ']):
            cb=W.QCheckBox(name);cb.setChecked(True);self.map_checks[name]=cb;ml.addWidget(cb,i//3,i%3)
        form.addRow('Maps',maps_widget)
        self.layout_select=W.QComboBox();self.layout_select.addItems(['2 × 4','2 × 3','3 × 3']);form.addRow('Panel layout',self.layout_select)
        self.out=W.QLineEdit(self.saved.get('output_root',str(ROOT/'outputs'))); form.addRow('Output root',self.out)
        pick=W.QPushButton('Choose output / 选择输出'); pick.clicked.connect(self.choose_output); form.addRow(pick)
        self.run_button=W.QPushButton(); self.run_button.clicked.connect(self.start_batch); form.addRow(self.run_button)
        self.cancel_button=W.QPushButton(); self.cancel_button.clicked.connect(self.cancel); form.addRow(self.cancel_button)
        self.resume_button=W.QPushButton('Resume batch / 恢复批次'); self.resume_button.clicked.connect(self.resume_batch); form.addRow(self.resume_button)
        self.render_button=W.QPushButton('Render Only / 缓存重绘'); self.render_button.clicked.connect(self.render_only); form.addRow(self.render_button)
        self.quant_button=W.QPushButton('RX / Euler / Grain Size (cache)');self.quant_button.clicked.connect(self.quantitative);form.addRow(self.quant_button)
        self.rx_config=W.QLineEdit('{}');self.rx_config.setPlaceholderText('Phase ID → thresholds JSON');form.addRow('RX phase rules',self.rx_config)
        self.history_button=W.QPushButton('Refresh history / 刷新历史'); self.history_button.clicked.connect(self.history); form.addRow(self.history_button)
        theme=W.QPushButton('Light / Dark'); theme.clicked.connect(lambda:self.set_theme(not self.dark)); form.addRow(theme)
        language=W.QPushButton('中文 / English'); language.clicked.connect(self.toggle_language); form.addRow(language)
        self.panel=W.QDockWidget('Parameters / 参数',self)
        scroll=W.QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(self.params);scroll.setMinimumWidth(310);scroll.setMaximumWidth(470)
        self.panel.setWidget(scroll); self.addDockWidget(C.Qt.DockWidgetArea.RightDockWidgetArea,self.panel)
        bottom=W.QDockWidget('Queue & Logs / 队列与日志',self); self.bottomtabs=W.QTabWidget()
        self.table=W.QTableWidget(0,8); self.table.setHorizontalHeaderLabels(['Source','Scan','Nx × Ny','Step µm','Phases','Shards','Est. cache MB','State'])
        self.table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.ResizeToContents)
        self.bottomtabs.addTab(self.table,'Data / Queue'); self.log=W.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(10000); self.bottomtabs.addTab(self.log,'Logs')
        self.history_list=W.QListWidget(); self.history_list.itemDoubleClicked.connect(self.open_history); self.bottomtabs.addTab(self.history_list,'History')
        bottom.setWidget(self.bottomtabs); self.addDockWidget(C.Qt.DockWidgetArea.BottomDockWidgetArea,bottom)
    def translate(self):
        self.nav.clear(); self.nav.addItems([x[1 if self.lang=='zh' else 0] for x in NAV]); self.nav.setCurrentRow(0)
        self.add_files.setText('导入文件' if self.lang=='zh' else 'Import files')
        self.add_folder.setText('导入文件夹' if self.lang=='zh' else 'Import folder')
        self.run_button.setText('运行批量分析' if self.lang=='zh' else 'Run batch')
        self.cancel_button.setText('取消任务' if self.lang=='zh' else 'Cancel task')
    def toggle_language(self):
        self.lang='en' if self.lang=='zh' else 'zh'; self.translate(); self.save_settings()
    def set_theme(self,dark):
        self.dark=dark
        bg,fg,field=('#18212f','#e5eaf2','#243246') if dark else ('#f3f5f8','#18212f','#ffffff')
        self.setStyleSheet(f'QWidget {{background:{bg};color:{fg};font-family:Segoe UI;font-size:12px;}} QLineEdit,QComboBox,QListWidget,QPlainTextEdit,QTableWidget {{background:{field};padding:5px;}} QPushButton {{background:{field};border:1px solid #61738b;border-radius:5px;padding:7px;}} QPushButton:hover {{border:1px solid #48a8df;}} QDockWidget::title {{padding:6px;}}')
        self.save_settings()
    def save_settings(self):
        atomic_json(self.preferences,{'language':self.lang,'dark':self.dark,'output_root':self.out.text()})
    def navigate(self,row):
        descriptions={0:'Import scans → choose analysis → run → inspect immutable results. Ctrl-drag ROI; wheel zoom; drag pan.',
            1:'Independent H5 files remain separate. Montage row shards are grouped using validated provenance.',
            2:'Standard IPF uses cached symmetry-aware orientation colors. Euler RGB is a separate representation.',
            3:'Select V3 texture, Schmid, variants or prior-beta. External MTEX validation pending.',
            4:'Quantitative proxy tools are enabled in M2; no metallurgical RX certification.',
            5:'One analysis child at a time; failures do not stop subsequent samples. Frozen batch snapshot.',
            6:'Numeric slicing tools are enabled in M3; geometric patches are not independent specimens.',
            7:'EDS window intensity is not wt.% or at.%. EDS availability is checked separately.',
            8:'Research tools require QC and explicit labels. Supervised mechanics unavailable without labels.',
            9:'Choose font and formats; Render Only uses the selected cache without scientific recomputation.',
            10:'Double-click an archived analysis to preview its cache. Existing results are never overwritten.',
            11:'Optional dependencies: PySide6 for Qt; torch only for AI. Tk fallback remains gui.py.'}
        self.help.setText(descriptions.get(row,''))
        if row==3:self.mode.setCurrentText('v3_texture')
        if row==4:self.mode.setCurrentText('v4_quantitative')
        if row==10:self.history()
    def error(self,text):
        self.log.appendPlainText('ERROR: '+str(text)); W.QMessageBox.warning(self,'EBSD Studio',str(text))
    def choose_files(self):
        paths,_=W.QFileDialog.getOpenFileNames(self,'EBSD inputs','','EBSD (*.h5 *.h5oina)')
        if paths:self.import_paths(paths)
    def choose_folder(self):
        path=W.QFileDialog.getExistingDirectory(self,'EBSD folder / montage')
        if path:self.import_paths([path])
    def choose_output(self):
        path=W.QFileDialog.getExistingDirectory(self,'Output root')
        if path:self.out.setText(path);self.save_settings()
    def import_paths(self,paths):
        if self.worker and self.worker.isRunning():return
        self.paths=list(dict.fromkeys(self.paths+list(paths))); self.add_files.setEnabled(False)
        self.worker=ImportWorker(self.paths); self.worker.result.connect(self.imported); self.worker.error.connect(self.error)
        self.worker.finished.connect(lambda:self.add_files.setEnabled(True)); self.worker.start()
    def imported(self,records):
        self.records=records; self.populate(records)
        self.info.setText(f'{len(records)} scans | {sum(r["nx"]*r["ny"] for r in records):,} measured grid points')
    def populate(self,records):
        self.table.setRowCount(len(records))
        for i,r in enumerate(records):
            values=[r['source_name'],r['scan_id'],f'{r["nx"]} × {r["ny"]}',f'{r["dx_um"]:g} × {r["dy_um"]:g}',
                    ', '.join(p['name'] for p in r['phases'].values()),r['parts'],round(r['estimated_cache_bytes']/1024**2,1),r['status']]
            for j,v in enumerate(values):self.table.setItem(i,j,W.QTableWidgetItem(str(v)))
    def options(self):
        formats=[k for k,v in self.formats.items() if v.isChecked()]
        if not formats:raise ValueError('Select at least one export format')
        maps=[k for k,v in self.map_checks.items() if v.isChecked()]
        layout=self.layout_select.currentText().split(' × ')
        if not maps or int(layout[0])*int(layout[1])<len(maps):raise ValueError('Select maps and a layout with enough cells')
        args=['--analysis',self.mode.currentText(),'--batch',str(self.block.value()),'--font',self.font.currentText(),
              '--v3-weighting',self.weight.currentText(),'--v3-load-axis',*self.axis.text().split(),'--formats',*formats,
              '--maps',*maps,'--layout',*layout]
        if self.mode.currentText()=='v4_quantitative':args+=['--rx-threshold-json',json.dumps(json.loads(self.rx_config.text()))]
        if self.roi.text().strip():
            vals=[int(x) for x in self.roi.text().split()]
            if len(vals)!=4 or min(vals[:2])<0 or min(vals[2:])<1:raise ValueError('ROI requires X Y WIDTH HEIGHT within scan')
            for r in self.records:
                if vals[0]+vals[2]>r['nx'] or vals[1]+vals[3]>r['ny']:raise ValueError('ROI exceeds a selected scan')
            args+=['--roi',*map(str,vals)]
        return args
    def start_batch(self):
        if self.process and self.process.state()!=C.QProcess.ProcessState.NotRunning:return
        try:
            if not self.paths:raise ValueError('Import one or more scans first')
            run=create_batch(self.paths,self.out.text(),self.options(),memory_budget_mb=self.budget.value())
            self.launch(['-u','-m','ebsd_autoplot.batch',str(run)],batch_run=run)
            self.batch_run=run
        except Exception as exc:self.error(exc)
    def busy(self):
        return self.process is not None and self.process.state()!=C.QProcess.ProcessState.NotRunning
    def launch(self,args,*,batch_run=None,control=None):
        if self.process and self.process.state()!=C.QProcess.ProcessState.NotRunning:raise ValueError('A task is already running')
        self.active_batch_run=batch_run;self.active_control=control
        self.process=C.QProcess(self); self.process.setWorkingDirectory(str(ROOT))
        self.process.setProcessChannelMode(C.QProcess.ProcessChannelMode.MergedChannels)
        env=C.QProcessEnvironment.systemEnvironment(); env.insert('PYTHONUTF8','1'); self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self.read_log)
        self.process.finished.connect(self.finished); self.process.errorOccurred.connect(self.process_error)
        self.process.start(sys.executable,args); self.run_button.setEnabled(False)
    def process_error(self,error):
        self.log.appendPlainText('Process error: '+str(error))
        if error==C.QProcess.ProcessError.FailedToStart:
            self.active_batch_run=None;self.active_control=None;self.run_button.setEnabled(True)
            self.log.appendPlainText('FAILED: process could not start')
    def read_log(self):self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace').rstrip())
    def finished(self,code,status):
        self.read_log(); self.run_button.setEnabled(True)
        self.active_batch_run=None;self.active_control=None
        self.log.appendPlainText('SUCCESS' if code==0 else f'FAILED / CANCELLED: exit {code}; inspect manifest and log')
        self.refresh_queue();self.history()
    def cancel(self):
        if not self.busy():return
        if self.active_batch_run:
            request_cancel(self.active_batch_run);self.log.appendPlainText('Cancellation requested; preserving checkpoints')
        if self.active_control:atomic_json(self.active_control,{'cancel':True})
    def resume_batch(self):
        if self.busy():self.error('A task is already running');return
        path,_=W.QFileDialog.getOpenFileName(self,'Select batch_manifest.json',self.out.text(),'JSON (*.json)')
        if path:
            try:
                run=Path(path).parent
                # A new explicit Resume clears only the batch cancellation request.
                (run/'cancel.json').unlink(missing_ok=True)
                self.launch(['-u','-m','ebsd_autoplot.batch',str(run)],batch_run=run)
                self.batch_run=run
            except Exception as exc:self.error(exc)
    def refresh_queue(self):
        if self.batch_run:
            try:self.populate(json.loads((self.batch_run/'batch_manifest.json').read_text(encoding='utf-8'))['records'])
            except (OSError,ValueError):pass
    def choose_cache(self):
        path,_=W.QFileDialog.getOpenFileName(self,'Select features.h5',self.out.text(),'Cache (*.h5)')
        if path:
            try:self.viewer.load_cache(path,self.map_select.currentText())
            except Exception as exc:self.error(exc)
    def change_map(self,field):
        if self.viewer.cache:
            try:self.viewer.load_cache(self.viewer.cache,field)
            except Exception as exc:self.error(exc)
    def set_roi(self,roi):self.roi.setText(' '.join(map(str,roi)))
    def render_only(self):
        if self.busy():self.error('A task is already running');return
        if not self.viewer.cache:self.error('Open an archived features.h5 first');return
        try:
            import uuid
            control=Path(self.out.text())/'.job_control'/('qt_render_'+uuid.uuid4().hex+'.json')
            atomic_json(control,{'cancel':False})
            self.launch(['-u','-m','ebsd_autoplot','--cache',str(self.viewer.cache),'--font',self.font.currentText(),
                         '--formats',*[k for k,v in self.formats.items() if v.isChecked()],
                         '--maps',*[k for k,v in self.map_checks.items() if v.isChecked()],
                         '--layout',*self.layout_select.currentText().split(' × '),'--control-file',str(control)],control=control)
        except Exception as exc:self.error(exc)
    def cached_action(self,action,config):
        if self.process and self.process.state()!=C.QProcess.ProcessState.NotRunning:
            raise ValueError('A task is already running')
        if not self.viewer.cache:self.error('Open an archived features.h5 first');return
        import uuid
        self.control=Path(self.out.text())/'.job_control'/('qt_action_'+uuid.uuid4().hex+'.json')
        atomic_json(self.control,{'cancel':False});self.batch_run=None
        self.launch(['-u','-m','ebsd_autoplot.studio_actions',action,'--cache',str(self.viewer.cache),
                     '--output-root',self.out.text(),'--config',json.dumps(config),'--control-file',str(self.control)])
    def quantitative(self):
        try:
            formats=[k for k,v in self.formats.items() if v.isChecked()]
            self.cached_action('quantitative',{'formats':formats,'font':self.font.currentText(),'thresholds':json.loads(self.rx_config.text())})
        except Exception as exc:self.error(exc)
    def history(self):
        self.history_list.clear()
        for path in sorted(Path(self.out.text()).glob('**/analysis_manifest.json'),reverse=True):
            try:
                m=json.loads(path.read_text(encoding='utf-8')); item=W.QListWidgetItem(f'{m["status"]} | {m["source_name"]} | {path.parent.name}')
                item.setData(C.Qt.ItemDataRole.UserRole,str(path));self.history_list.addItem(item)
            except (OSError,ValueError,KeyError):continue
    def open_history(self,item):
        path=Path(item.data(C.Qt.ItemDataRole.UserRole)).parent/'analysis_cache/features.h5'
        try:self.viewer.load_cache(path,self.map_select.currentText())
        except Exception as exc:self.error(exc)
    def dragEnterEvent(self,event):
        if event.mimeData().hasUrls():event.acceptProposedAction()
    def dropEvent(self,event):self.import_paths([u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()])
    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():event.ignore();self.log.appendPlainText('Wait for input validation');return
        if self.process and self.process.state()!=C.QProcess.ProcessState.NotRunning:
            self.cancel();event.ignore();self.log.appendPlainText('Wait for safe task cancellation, then close');return
        self.save_settings();event.accept()


def main():
    app=W.QApplication.instance() or W.QApplication(sys.argv)
    win=Studio();win.show()
    if '--smoke' in sys.argv:C.QTimer.singleShot(1200,win.close);C.QTimer.singleShot(1500,app.quit)
    return app.exec()

if __name__=='__main__':raise SystemExit(main())
