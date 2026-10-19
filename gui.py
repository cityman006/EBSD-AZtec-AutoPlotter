"""Windows GUI: Tkinter by default, optional tkinterdnd2 for drag & drop."""
from pathlib import Path
import json,os
import subprocess,sys,threading,queue
try:
    from tkinterdnd2 import TkinterDnD,DND_FILES
except ImportError:
    TkinterDnD=None
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox
from ebsd_autoplot.jobs import JobManager
from ebsd_autoplot.storage import atomic_json, long_path, canonical_path
from ebsd_autoplot.archive import locate_cache
from ebsd_autoplot.pipeline import recover_interrupted

class App:
    def __init__(self):
        self.dnd_enabled=False
        if TkinterDnD:
            try:
                self.root=TkinterDnD.Tk()
                self.dnd_enabled=True
            except (RuntimeError, tk.TclError) as exc:
                print('Drag and drop unavailable; use Select file:', exc, flush=True)
                self.root=tk.Tk()
        else:
            self.root=tk.Tk()
        self.root.title('EBSD-AZtec Auto Plotter | V1+V2')
        self.root.geometry('920x760')
        self.preferences=Path(__file__).resolve().parent/'config'/'ui_preferences.json'
        try: saved=json.loads(self.preferences.read_text(encoding='utf-8'))
        except (OSError,ValueError): saved={}
        self.infile=tk.StringVar();self.outfile=tk.StringVar(value=saved.get('output_root',''))
        self.an=tk.StringVar(value='v1');self.roi=tk.StringVar(value='0 0 800 800')
        self.threads=tk.StringVar(value='32');self.font=tk.StringVar(value='Arial');self.formats=tk.StringVar(value='png pdf svg')
        self.msg_queue=queue.Queue();self.manager=JobManager(Path(__file__).resolve().parent,self.msg_queue)
        self.active_job=tk.StringVar();self.history_items=[]
        self.dataset_var=tk.StringVar(value='Dataset directory: choose input and output root')
        self.target_var=tk.StringVar(value='This run target: created when analysis/render starts')
        ttk.Label(self.root,text='EBSD-AZtec Auto Plotter',font=('Segoe UI',19,'bold')).pack(anchor='w',padx=20,pady=(18,8))
        ttk.Label(self.root,text='H5OINA / H5 montage files | Symmetry-aware IPF/KAM | V2 ROI grain analysis').pack(anchor='w',padx=20)
        main=ttk.Frame(self.root,padding=20);main.pack(fill='both',expand=True)
        ttk.Label(main,text='Input: raw .h5oina / .h5 or folder of 20 split H5 files').pack(anchor='w')
        r=ttk.Frame(main);r.pack(fill='x',pady=(3,12)); ttk.Entry(r,textvariable=self.infile).pack(side='left',fill='x',expand=True)
        ttk.Button(r,text='Select file',command=self.choose_file).pack(side='left',padx=3)
        ttk.Button(r,text='Select folder',command=self.choose_folder).pack(side='left')
        if self.dnd_enabled:
            r.drop_target_register(DND_FILES)
            r.dnd_bind('<<Drop>>', lambda e:self.infile.set(self.root.tk.splitlist(e.data)[0]))
        else:
            ttk.Label(main,text='Drag and drop unavailable; Select file works. Run REPAIR_WINDOWS.cmd to repair.').pack(anchor='w')
        ttk.Label(main,text='Total output root (one dataset folder per input)').pack(anchor='w')
        r=ttk.Frame(main);r.pack(fill='x',pady=(3,12));ttk.Entry(r,textvariable=self.outfile).pack(side='left',fill='x',expand=True)
        ttk.Button(r,text='Browse',command=self.choose_out).pack(side='left')
        ttk.Label(main,textvariable=self.dataset_var,wraplength=850).pack(anchor='w')
        ttk.Label(main,textvariable=self.target_var,wraplength=850).pack(anchor='w',pady=(2,8))
        self.infile.trace_add('write',lambda *_:self.update_paths())
        self.outfile.trace_add('write',lambda *_:self.update_paths())
        ttk.Label(main,text='Processing mode').pack(anchor='w')
        combo=ttk.Combobox(main,textvariable=self.an,values=['inspect','v1','v1v2','v2'],state='readonly')
        combo.pack(anchor='w',pady=6)
        ttk.Label(main,text='V2 ROI (X Y WIDTH HEIGHT); for huge scans, do V1 first, then V2 ROI').pack(anchor='w')
        ttk.Entry(main,textvariable=self.roi,width=35).pack(anchor='w',pady=(3,10))
        ttk.Label(main,text='Processing block rows (lower value uses less memory)').pack(anchor='w')
        ttk.Entry(main,textvariable=self.threads,width=10).pack(anchor='w',pady=5)
        opts=ttk.Frame(main);opts.pack(anchor='w',pady=5)
        ttk.Label(opts,text='Font').pack(side='left')
        installed_fonts=sorted(set(tkfont.families(self.root)),key=str.casefold)
        preferred_fonts=['Arial','Times New Roman','Calibri','Segoe UI','Cambria','Verdana','Tahoma','Courier New']
        font_values=[name for name in preferred_fonts if name.casefold() in {f.casefold() for f in installed_fonts}]
        font_values.extend(name for name in installed_fonts if name.casefold() not in {f.casefold() for f in font_values})
        if self.font.get() not in font_values:font_values.insert(0,self.font.get())
        self.font_combo=ttk.Combobox(opts,textvariable=self.font,values=font_values,state='readonly',width=22)
        self.font_combo.pack(side='left',padx=5)
        ttk.Label(opts,text='Formats').pack(side='left',padx=(8,0))
        self.format_flags={name:tk.BooleanVar(value=name in {'png','pdf','svg'}) for name in ('png','tiff','pdf','svg')}
        self.format_popup=None
        self.formats_button=ttk.Button(opts,text='PNG, PDF, SVG  ▾',command=self._show_formats_menu)
        self.formats_button.pack(side='left',padx=5)
        r=ttk.Frame(main);r.pack(anchor='w',pady=8)
        self.startbtn=ttk.Button(r,text='Run analysis / Generate maps',command=self.start);self.startbtn.pack(side='left')
        ttk.Button(r,text='Stop',command=self.stop).pack(side='left',padx=10)
        ttk.Button(r,text='Force stop selected job',command=lambda:self.stop(force=True)).pack(side='left')
        ttk.Button(r,text='Open sample results',command=self.open_dataset).pack(side='left',padx=4)
        ttk.Button(r,text='View history',command=self.view_history).pack(side='left',padx=4)
        ttk.Button(r,text='Preview old output',command=self.preview_old_output).pack(side='left',padx=4)
        self.job_combo=ttk.Combobox(main,textvariable=self.active_job,state='readonly',width=65)
        self.job_combo.pack(anchor='w',pady=4)
        ttk.Button(main,text='Recover interrupted records',command=self.recover_records).pack(anchor='w')
        ttk.Label(main,text='Log').pack(anchor='w')
        self.log=tk.Text(main,height=12,wrap='word');self.log.pack(fill='both',expand=True)
        self.root.after(150,self.poll)
    def _sync_formats(self):
        selected=[name for name,var in self.format_flags.items() if var.get()]
        self.formats.set(' '.join(selected))
        label=', '.join(name.upper() for name in selected) if selected else 'None'
        self.formats_button.configure(text=label+'  ▾')
    def _show_formats_menu(self):
        if self.format_popup is not None and self.format_popup.winfo_exists():
            self.format_popup.destroy();self.format_popup=None;return
        popup=tk.Toplevel(self.root);self.format_popup=popup
        popup.overrideredirect(True);popup.transient(self.root)
        panel=ttk.Frame(popup,padding=8,relief='solid',borderwidth=1);panel.pack(fill='both',expand=True)
        for name,var in self.format_flags.items():
            ttk.Checkbutton(panel,text=name.upper(),variable=var,command=self._sync_formats).pack(anchor='w')
        ttk.Separator(panel,orient='horizontal').pack(fill='x',pady=5)
        actions=ttk.Frame(panel);actions.pack(fill='x')
        ttk.Button(actions,text='Select all',command=self._select_all_formats).pack(side='left')
        ttk.Button(actions,text='Clear all',command=self._clear_all_formats).pack(side='left',padx=(6,0))
        popup.bind('<Escape>',lambda _event:self._close_formats_popup())
        popup.bind('<FocusOut>',lambda _event:popup.after_idle(self._close_formats_popup))
        popup.update_idletasks()
        x=self.formats_button.winfo_rootx()
        y=self.formats_button.winfo_rooty()+self.formats_button.winfo_height()
        popup.geometry(f'+{x}+{y}');popup.deiconify();popup.lift();popup.focus_force()
    def _close_formats_popup(self):
        popup=self.format_popup
        if popup is None or not popup.winfo_exists():return
        focus=self.root.focus_get()
        if focus is None or str(focus.winfo_toplevel())!=str(popup):
            popup.destroy();self.format_popup=None
    def _select_all_formats(self):
        for flag in self.format_flags.values():flag.set(True)
        self._sync_formats()
    def _clear_all_formats(self):
        for flag in self.format_flags.values():flag.set(False)
        self._sync_formats()
    def choose_file(self):
        f=filedialog.askopenfilename(filetypes=[('Oxford H5OINA/HDF5','*.h5oina *.h5 *.hdf5'),('All files','*.*')])
        if f:self.infile.set(f)
    def choose_folder(self):
        f=filedialog.askdirectory()
        if f:self.infile.set(f)
    def choose_out(self):
        f=filedialog.askdirectory()
        if f:self.outfile.set(f)
    def update_paths(self):
        if not self.outfile.get().strip() or not self.infile.get().strip():
            self.dataset_var.set('Dataset directory: choose input and output root');return
        matches=self._matching_datasets()
        if len(matches)>1:
            self.dataset_var.set('Dataset directories: '+' | '.join(str(p) for p in matches));return
        existing=matches[0] if matches else None
        if existing:self.dataset_var.set('Dataset directory: '+str(existing))
        else:
            source=Path(self.infile.get()).expanduser()
            stem=source.stem if source.is_file() else source.name+'_montage'
            self.dataset_var.set('Dataset directory (new): '+str(Path(self.outfile.get()).expanduser()/stem))
    def start(self):
        formats=self.formats.get().split()
        if not formats:
            messagebox.showerror('No formats selected','Select at least one output format: PNG, TIFF, PDF, or SVG.');return
        if not self.outfile.get().strip():
            messagebox.showinfo('Choose output root','Choose the total output directory once; this setting will be remembered.');self.choose_out()
            if not self.outfile.get().strip():return
        if self.an.get()=='v2':
            messagebox.showinfo('Select analysis cache','Open View history, select an analysis record and choose Run V2 from selected cache.');return
        if not long_path(self.infile.get()).exists():messagebox.showerror('Input missing','Choose a valid input file or folder');return
        try:
            self.preferences.parent.mkdir(parents=True,exist_ok=True)
            atomic_json(self.preferences,{'output_root':self.outfile.get()})
        except OSError as exc:
            messagebox.showerror('Cannot save output preference',str(exc));return
        argv=[sys.executable,'-u','-m','ebsd_autoplot',self.infile.get(),'--output-root',self.outfile.get(),
              '--analysis',self.an.get(),'--batch',self.threads.get(),'--font',self.font.get(), '--formats',*formats]
        if self.an.get() in ('v2','v1v2'):
            try:
                roi=self.roi.get().split()
                if len(roi)!=4:raise ValueError()
                [int(v) for v in roi]
            except ValueError:
                messagebox.showerror('Invalid ROI','Enter X Y WIDTH HEIGHT as integers');return
            argv+=['--roi',*roi]
        self._submit(argv)
    def _datasets(self):
        if not self.outfile.get().strip():return []
        root=long_path(self.outfile.get())
        if not root.is_dir():return []
        return [p.parent for p in root.rglob('dataset_manifest.json')]
    def _matching_datasets(self):
        try: src=str(canonical_path(self.infile.get()))
        except OSError:return []
        matches=[]
        for folder in self._datasets():
            try: data=json.loads((folder/'dataset_manifest.json').read_text(encoding='utf-8'))
            except (OSError,ValueError):continue
            candidates=[data.get('input_path',''),*[x.get('path','') for x in data.get('source',{}).get('source_files',[])]]
            if os.path.normcase(src) in [os.path.normcase(x) for x in candidates]:matches.append(folder)
        return matches
    def _current_dataset(self):
        matches=self._matching_datasets()
        return matches[0] if len(matches)==1 else None
    def open_dataset(self):
        if len(self._matching_datasets())>1:self.view_history();return
        folder=self._current_dataset()
        if folder is None:messagebox.showinfo('No archived dataset','Run an analysis first for this input.');return
        os.startfile(str(folder))
    def view_history(self):
        win=tk.Toplevel(self.root);win.title('Historical analyses and renders');win.geometry('900x480')
        listing=tk.Listbox(win,selectmode='browse');listing.pack(fill='both',expand=True,padx=10,pady=10)
        listing.history_items=[]
        for dataset in self._datasets():
            for kind,folder,manifest_name in [('analysis','analysis_runs','analysis_manifest.json'),('render','renders','render_manifest.json')]:
                for mp in sorted(dataset.rglob(manifest_name)):
                    run=mp.parent
                    if run.parent.name!=folder:continue
                    try:m=json.loads(mp.read_text(encoding='utf-8'))
                    except (OSError,ValueError):continue
                    desc=f"{dataset.name} | {kind} | {m.get('started_at',m.get('completed_at',''))} | {m.get('status')} | id={m.get('analysis_run_id',m.get('render_id',run.name))} | font={m.get('actual_font','')} | formats={','.join(m.get('formats',[]))}"
                    listing.insert('end',desc);listing.history_items.append((kind,run,m))
        actions=ttk.Frame(win);actions.pack(fill='x',padx=10,pady=(0,10))
        ttk.Button(actions,text='Open selected record',command=lambda:self._open_history(listing)).pack(side='left')
        ttk.Button(actions,text='Reformat selected analysis',command=lambda:self._reformat_history(listing,win)).pack(side='left',padx=8)
        ttk.Button(actions,text='Run V2 from selected cache',command=lambda:self._run_v2_history(listing,win)).pack(side='left')
        ttk.Button(actions,text='Resume selected task',command=lambda:self._resume_history(listing,win)).pack(side='left',padx=5)
    def preview_old_output(self):
        folder=filedialog.askdirectory(title='Select an old output folder to preview')
        if not folder:return
        from ebsd_autoplot.archive import preview_legacy,import_legacy
        try:preview=preview_legacy(folder)
        except Exception as exc:messagebox.showerror('Preview failed',str(exc));return
        win=tk.Toplevel(self.root);win.title('Legacy output migration preview');win.geometry('900x520')
        text=tk.Text(win,wrap='word');text.pack(fill='both',expand=True,padx=10,pady=10)
        text.insert('end',f"Confirmed source folders: {len(preview['confirmed'])}\nUnknown source files: {len(preview['unassigned_files'])}\n\n")
        for item in preview['confirmed']:
            text.insert('end',f"{item['status']} | source={item['source_path'] or 'unknown'} | folder={item['directory']} | files={item['file_count']}\n")
        if preview['unassigned_files']:
            text.insert('end','\nUnassigned (will be kept under _legacy_unassigned if copied):\n')
            for name in preview['unassigned_files'][:80]:text.insert('end',name+'\n')
        text.configure(state='disabled')
        buttons=ttk.Frame(win);buttons.pack(fill='x',padx=10,pady=(0,10))
        ttk.Button(buttons,text='Copy confirmed results',command=lambda:self._import_legacy(import_legacy,folder,True,False,win)).pack(side='left')
        ttk.Button(buttons,text='Copy unknown as unassigned',command=lambda:self._import_legacy(import_legacy,folder,False,True,win)).pack(side='left',padx=8)
    def _import_legacy(self,importer,folder,confirmed,unassigned,win):
        if not self.outfile.get().strip():
            messagebox.showinfo('Choose output root','Choose the new total output root first.');self.choose_out()
            if not self.outfile.get().strip():return
        label='provenance-confirmed results' if confirmed else 'unknown-source files'
        if not messagebox.askyesno('Copy legacy results',f'Copy {label} into the selected output root? The old folder will remain unchanged.'):return
        try:copied=importer(folder,self.outfile.get(),include_confirmed=confirmed,include_unassigned=unassigned)
        except Exception as exc:messagebox.showerror('Import failed',str(exc));return
        messagebox.showinfo('Import complete','Copied to:\n'+'\n'.join(map(str,copied)))
        win.destroy()
    def _selected_history(self,listing):
        sel=listing.curselection()
        return listing.history_items[sel[0]] if sel else None
    def _open_history(self,listing):
        item=self._selected_history(listing)
        if item:os.startfile(str(item[1]))
    def _cache_for_record(self,run,manifest):
        candidate=run/'analysis_cache'/'features.h5'
        if candidate.is_file():return candidate
        rel=manifest.get('cache_path')
        if rel:
            dataset=next((p for p in run.parents if (p/'dataset_manifest.json').is_file()),run.parent.parent)
            candidate=(dataset/rel).resolve()
            if dataset.resolve() in candidate.parents and candidate.is_file():return candidate
        return None
    def _reformat_history(self,listing,win):
        item=self._selected_history(listing)
        if not item:return
        kind,run,manifest=item
        if kind!='analysis':messagebox.showinfo('Choose an analysis','Select an analysis record to redraw its cached maps.');return
        if manifest.get('status') not in ('success','partial','cancelled'):messagebox.showerror('Analysis incomplete','Select a record with a verified V1 checkpoint.');return
        cache=self._cache_for_record(run,manifest)
        if cache is None:messagebox.showerror('Cache missing','The selected analysis record has no complete features.h5 cache.');return
        formats=self.formats.get().split()
        if not formats:messagebox.showerror('No formats selected','Select at least one output format: PNG, TIFF, PDF, or SVG.');return
        argv=[sys.executable,'-u','-m','ebsd_autoplot','--cache',str(cache),'--font',self.font.get(),
              '--formats',*formats]
        win.destroy();self.log.insert('end',f'Render Only from {cache}\n')
        self._submit(argv)
    def _run_v2_history(self,listing,win):
        item=self._selected_history(listing)
        if not item:return
        kind,run,manifest=item
        if kind!='analysis':messagebox.showinfo('Choose an analysis','Select a successful analysis cache first.');return
        if manifest.get('status') not in ('success','partial','cancelled'):messagebox.showerror('Analysis incomplete','Select a record with a verified V1 checkpoint.');return
        cache=self._cache_for_record(run,manifest)
        if cache is None:messagebox.showerror('Cache missing','Selected analysis has no complete features.h5 cache.');return
        try:
            roi=self.roi.get().split()
            if len(roi)!=4:raise ValueError()
            [int(v) for v in roi]
        except ValueError:
            messagebox.showerror('Invalid ROI','Enter X Y WIDTH HEIGHT as integers');return
        argv=[sys.executable,'-u','-m','ebsd_autoplot','--analysis','v2','--analysis-cache',str(cache),'--roi',*roi]
        win.destroy();self.log.insert('end',f'V2 analysis from {cache}\n')
        self._submit(argv)
    def _resume_history(self,listing,win):
        item=self._selected_history(listing)
        if not item:return
        kind,run,manifest=item
        if manifest.get('status')=='success':messagebox.showinfo('Task complete','Choose an interrupted, cancelled or failed record.');return
        win.destroy();self._submit([sys.executable,'-u','-m','ebsd_autoplot','--resume',str(run)])
    def recover_records(self):
        if not self.outfile.get().strip():return
        try:records=recover_interrupted(self.outfile.get())
        except Exception as exc:messagebox.showerror('Recovery failed',str(exc));return
        messagebox.showinfo('Recovery complete',f'Marked {len(records)} abandoned records. Choose one in history and use Resume selected task.')
    def _submit(self,argv):
        argv=tuple(argv);root=self.outfile.get();cache='';input_snapshot=self.infile.get()
        style_snapshot=(self.font.get(),self.formats.get(),self.roi.get())
        for flag in ('--cache','--analysis-cache','--resume'):
            if flag not in argv:continue
            selected=Path(argv[argv.index(flag)+1]).resolve()
            if flag=='--resume':
                dataset=next(p for p in selected.parents if (p/'dataset_manifest.json').is_file())
                mp=selected if selected.is_file() else selected/('render_manifest.json' if selected.parent.name=='renders' else 'analysis_manifest.json')
                saved=json.loads(mp.read_text(encoding='utf-8'));input_snapshot=saved.get('input_path','')
                config=saved.get('options',{});style_snapshot=(config.get('font',''),tuple(config.get('formats',[])),tuple(config.get('roi') or []))
                if saved.get('cache_path'):cache=str(dataset/saved['cache_path'])
            else:
                cache=str(selected);dataset,_,_=locate_cache(selected);input_snapshot=''
            root=str(dataset.parent);break
        snapshot=self.manager.submit(argv,root,input_path=input_snapshot,cache_path=cache,style=style_snapshot)
        self.active_job.set(snapshot.job_id)
        self.job_combo.configure(values=tuple(self.manager.jobs))
        self.target_var.set('Started '+snapshot.job_id+' | output root: '+str(snapshot.output_root))
        return snapshot
    def stop(self,force=False):
        job_id=self.active_job.get()
        if job_id in self.manager.jobs:
            self.manager.cancel(job_id,force=force)
            self.log.insert('end',f'[{job_id}] '+('Force stop' if force else 'Cancellation requested; waiting for a safe checkpoint')+'\n')
    def poll(self):
        while not self.msg_queue.empty():
            job_id,kind,line=self.msg_queue.get_nowait()
            if kind=='done':line='Task '+line+'\n'
            if kind=='line' and line.startswith('JOB_EVENT ') and self.active_job.get()==job_id:
                try:
                    event=json.loads(line[len('JOB_EVENT '):])
                    self.target_var.set(job_id+' | '+event['kind']+': '+event['path'])
                except (ValueError,KeyError):pass
            self.log.insert('end',f'[{job_id}] '+line);self.log.see('end')
        self.root.after(150,self.poll)
    def run(self):self.root.mainloop()
if __name__=='__main__':
    import argparse,json,os,ctypes
    parser=argparse.ArgumentParser()
    parser.add_argument('--ready-file',type=Path)
    parser.add_argument('--smoke-exit-ms',type=int,default=0)
    args=parser.parse_args()
    app=App()
    def ready():
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        buttons=[w for w in descendants(app.root) if isinstance(w,ttk.Button)]
        chooser=[w for w in buttons if w.cget('text')=='Select file']
        hwnd=app.root.winfo_id()
        visible=bool(ctypes.windll.user32.IsWindowVisible(hwnd)) if os.name=='nt' else bool(app.root.winfo_viewable())
        result={'pid':os.getpid(),'python':sys.executable,'title':app.root.title(),
                'hwnd':hwnd,'mapped':bool(app.root.winfo_ismapped()),'visible':visible,
                'event_loop':True,'file_selection_command':bool(chooser and chooser[0].cget('command')),
                'drag_and_drop':app.dnd_enabled}
        print('GUI_READY '+json.dumps(result),flush=True)
        if args.ready_file:
            atomic_json(args.ready_file,result)
        if args.smoke_exit_ms:
            app.root.after(args.smoke_exit_ms,app.root.destroy)
    app.root.after(300,ready)
    app.run()
