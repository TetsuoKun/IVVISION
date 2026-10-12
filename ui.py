"""Follow-cam cropper.

Load the raw video + the boxes sheet produced by track_and_export.py, pick an id,
and export a phone-shaped crop that follows that person's box midpoint.

Run:  python follow_cam_ui.py
Needs: opencv-python, pandas, numpy, openpyxl (for .xlsx). ffmpeg on PATH for audio.
"""
import base64
import os
import queue
import shutil
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import pandas as pd

PRESETS = {
    "Phone portrait 1080x1920 (9:16)": (1080, 1920),
    "Phone portrait 720x1280 (9:16)": (720, 1280),
    "Square 1080x1080 (1:1)": (1080, 1080),
    "Landscape 1920x1080 (16:9)": (1920, 1080),
}
PREVIEW_W, PREVIEW_H = 560, 420


# ---------------- Processing (no UI code below until the App class) ----------------
def load_boxes(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name="boxes")


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30,
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "n": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def track_path(df, id_col, track_id, n_frames, smooth_frames):
    """(n_frames, 2) array of the crop centre for every frame.

    Frames where the id is missing hold its last known position (and frames
    before it first appears use its first position). A centred moving average
    removes box jitter without making the camera lag behind the person.
    """
    sub = df[df[id_col] == track_id]
    if sub.empty:
        raise ValueError(f"No rows for {id_col} = {track_id}")
    per_frame = sub.groupby("frame")[["mid_x", "mid_y"]].mean()
    per_frame = per_frame.reindex(range(n_frames)).ffill().bfill()
    if smooth_frames > 1:
        per_frame = per_frame.rolling(smooth_frames, center=True, min_periods=1).mean()
    return per_frame.to_numpy()


def crop_size(src_w, src_h, out_w, out_h, zoom):
    """Largest window with the output's aspect ratio that fits the source, shrunk by zoom."""
    aspect = out_w / out_h
    if src_w / src_h > aspect:
        ch, cw = src_h, src_h * aspect
    else:
        cw, ch = src_w, src_w / aspect
    return max(2, int(round(cw / zoom))), max(2, int(round(ch / zoom)))


def crop_origin(cx, cy, cw, ch, src_w, src_h):
    """Top-left of a cw x ch window centred on (cx, cy), clamped inside the frame."""
    x0 = min(max(int(round(cx - cw / 2)), 0), src_w - cw)
    y0 = min(max(int(round(cy - ch / 2)), 0), src_h - ch)
    return x0, y0


def crop_frame(frame, x0, y0, cw, ch, out_w, out_h):
    crop = frame[y0:y0 + ch, x0:x0 + cw]
    interp = cv2.INTER_AREA if cw > out_w else cv2.INTER_CUBIC
    return cv2.resize(crop, (out_w, out_h), interpolation=interp)


def export_video(video_path, centres, cw, ch, out_w, out_h, out_path, on_progress, cancel):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    silent_path = os.path.splitext(out_path)[0] + "_noaudio.mp4"
    writer = cv2.VideoWriter(silent_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    i = 0
    while not cancel.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        cx, cy = centres[min(i, len(centres) - 1)]
        x0, y0 = crop_origin(cx, cy, cw, ch, src_w, src_h)
        writer.write(crop_frame(frame, x0, y0, cw, ch, out_w, out_h))
        i += 1
        if i % 10 == 0:
            on_progress(i)
    cap.release()
    writer.release()

    if cancel.is_set():
        os.remove(silent_path)
        return "Cancelled."

    if shutil.which("ffmpeg") is None:
        os.replace(silent_path, out_path)
        return f"Saved {out_path} (ffmpeg not found, so no audio)."

    on_progress(i, "Adding audio with ffmpeg...")
    proc = subprocess.run([
        "ffmpeg", "-y", "-i", silent_path, "-i", video_path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        out_path,
    ], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + proc.stderr[-1500:])
    os.remove(silent_path)
    return f"Saved {out_path}"


def to_photo(img_bgr, max_w, max_h):
    h, w = img_bgr.shape[:2]
    s = min(max_w / w, max_h / h)
    small = cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", small)
    return tk.PhotoImage(data=base64.b64encode(png.tobytes()))


# ---------------- UI ----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Follow-cam cropper")
        self.df = None
        self.info = None
        self.preview_cap = None
        self.centres = None
        self.cancel = threading.Event()
        self.events = queue.Queue()
        self._preview_job = None

        self.video_var = tk.StringVar()
        self.boxes_var = tk.StringVar()
        self.id_col_var = tk.StringVar(value="id")
        self.track_var = tk.StringVar()
        self.preset_var = tk.StringVar(value=next(iter(PRESETS)))
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.smooth_var = tk.IntVar(value=15)
        self.frame_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="Pick a video and a boxes sheet.")

        self._build()
        self.after(100, self._poll_events)

    def _build(self):
        pad = {"padx": 6, "pady": 4}
        top = ttk.Frame(self)
        top.pack(fill="x", **pad)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Raw video").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.video_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(top, text="Browse...", command=self.pick_video).grid(row=0, column=2)

        ttk.Label(top, text="Boxes sheet").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.boxes_var).grid(row=1, column=1, sticky="ew")
        ttk.Button(top, text="Browse...", command=self.pick_boxes).grid(row=1, column=2)

        opts = ttk.Frame(self)
        opts.pack(fill="x", **pad)

        ttk.Label(opts, text="Use").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(opts, text="Corrected id", variable=self.id_col_var, value="id",
                        command=self.refresh_ids).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(opts, text="Raw tracker id", variable=self.id_col_var, value="raw_id",
                        command=self.refresh_ids).grid(row=0, column=2, sticky="w")

        ttk.Label(opts, text="Follow id").grid(row=1, column=0, sticky="w")
        self.track_box = ttk.Combobox(opts, textvariable=self.track_var, state="readonly", width=24)
        self.track_box.grid(row=1, column=1, columnspan=2, sticky="w")
        self.track_box.bind("<<ComboboxSelected>>", lambda e: self.rebuild())

        ttk.Label(opts, text="Output size").grid(row=2, column=0, sticky="w")
        preset_box = ttk.Combobox(opts, textvariable=self.preset_var, values=list(PRESETS),
                                  state="readonly", width=32)
        preset_box.grid(row=2, column=1, columnspan=2, sticky="w")
        preset_box.bind("<<ComboboxSelected>>", lambda e: self.schedule_preview())

        ttk.Label(opts, text="Zoom").grid(row=3, column=0, sticky="w")
        ttk.Scale(opts, from_=1.0, to=3.0, variable=self.zoom_var, length=220,
                  command=lambda v: self.schedule_preview()).grid(row=3, column=1, columnspan=2, sticky="w")
        self.zoom_lbl = ttk.Label(opts, width=6)
        self.zoom_lbl.grid(row=3, column=3, sticky="w")

        ttk.Label(opts, text="Smoothing (frames)").grid(row=4, column=0, sticky="w")
        ttk.Scale(opts, from_=1, to=61, variable=self.smooth_var, length=220,
                  command=lambda v: self.schedule_preview(rebuild=True)).grid(row=4, column=1, columnspan=2, sticky="w")
        self.smooth_lbl = ttk.Label(opts, width=6)
        self.smooth_lbl.grid(row=4, column=3, sticky="w")

        prev = ttk.LabelFrame(self, text="Preview")
        prev.pack(fill="both", expand=True, **pad)
        imgs = ttk.Frame(prev)
        imgs.pack()
        self.full_lbl = ttk.Label(imgs)
        self.full_lbl.pack(side="left", padx=4)
        self.crop_lbl = ttk.Label(imgs)
        self.crop_lbl.pack(side="left", padx=4)
        self.frame_scale = ttk.Scale(prev, from_=0, to=0, variable=self.frame_var,
                                     command=lambda v: self.schedule_preview())
        self.frame_scale.pack(fill="x", padx=6, pady=4)

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", **pad)
        self.export_btn = ttk.Button(bottom, text="Export...", command=self.start_export)
        self.export_btn.pack(side="left")
        self.cancel_btn = ttk.Button(bottom, text="Cancel", command=self.cancel.set, state="disabled")
        self.cancel_btn.pack(side="left", padx=4)
        self.progress = ttk.Progressbar(bottom, length=300)
        self.progress.pack(side="left", padx=8, fill="x", expand=True)
        ttk.Label(self, textvariable=self.status_var).pack(fill="x", padx=8, pady=(0, 6))

    # ----- loading -----
    def pick_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
        if not path:
            return
        self.video_var.set(path)
        try:
            self.info = video_info(path)
        except ValueError as e:
            messagebox.showerror("Video", str(e))
            return
        if self.preview_cap is not None:
            self.preview_cap.release()
        self.preview_cap = cv2.VideoCapture(path)
        self.rebuild()

    def pick_boxes(self):
        path = filedialog.askopenfilename(filetypes=[("Boxes sheet", "*.xlsx *.csv"), ("All", "*.*")])
        if not path:
            return
        self.boxes_var.set(path)
        try:
            self.df = load_boxes(path)
        except Exception as e:
            messagebox.showerror("Boxes sheet", f"Could not read {path}:\n{e}")
            return
        self.refresh_ids()

    def refresh_ids(self):
        if self.df is None:
            return
        col = self.id_col_var.get()
        counts = self.df[col].dropna().astype(int).value_counts().sort_index()
        values = [f"{tid}   ({n} frames)" for tid, n in counts.items()]
        self.track_box["values"] = values
        if values:
            self.track_box.current(0)
        self.rebuild()

    def n_frames(self):
        n = self.info["n"]
        if self.df is not None and len(self.df):
            n = max(n, int(self.df["frame"].max()) + 1)
        return n

    def rebuild(self):
        """Recompute the smoothed centre path for the chosen id, then refresh the preview."""
        self.centres = None
        if self.info is None or self.df is None or not self.track_var.get():
            return
        track_id = int(self.track_var.get().split()[0])
        try:
            self.centres = track_path(self.df, self.id_col_var.get(), track_id,
                                      self.n_frames(), int(self.smooth_var.get()))
        except ValueError as e:
            self.status_var.set(str(e))
            return
        self.frame_scale.configure(to=max(0, self.info["n"] - 1))
        self.status_var.set(f"Following {self.id_col_var.get()} {track_id}.")
        self.schedule_preview()

    # ----- preview -----
    def crop_geometry(self):
        out_w, out_h = PRESETS[self.preset_var.get()]
        cw, ch = crop_size(self.info["w"], self.info["h"], out_w, out_h, self.zoom_var.get())
        return out_w, out_h, cw, ch

    def schedule_preview(self, rebuild=False):
        # Debounce so dragging a slider doesn't seek the video dozens of times a second
        self.zoom_lbl.configure(text=f"{self.zoom_var.get():.2f}x")
        self.smooth_lbl.configure(text=str(int(self.smooth_var.get())))
        if self._preview_job:
            self.after_cancel(self._preview_job)
        self._preview_job = self.after(120, self.rebuild if rebuild else self.draw_preview)

    def draw_preview(self):
        self._preview_job = None
        if self.centres is None or self.preview_cap is None:
            return
        i = int(self.frame_var.get())
        self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = self.preview_cap.read()
        if not ok:
            return
        out_w, out_h, cw, ch = self.crop_geometry()
        cx, cy = self.centres[min(i, len(self.centres) - 1)]
        x0, y0 = crop_origin(cx, cy, cw, ch, self.info["w"], self.info["h"])
        cropped = crop_frame(frame, x0, y0, cw, ch, out_w, out_h)

        full = frame.copy()
        dim = full.copy()
        dim[:] = (dim * 0.35).astype(np.uint8)
        dim[y0:y0 + ch, x0:x0 + cw] = full[y0:y0 + ch, x0:x0 + cw]
        cv2.rectangle(dim, (x0, y0), (x0 + cw, y0 + ch), (0, 255, 255), 4)
        cv2.circle(dim, (int(cx), int(cy)), 8, (0, 255, 255), -1)

        self._full_photo = to_photo(dim, PREVIEW_W, PREVIEW_H)
        self._crop_photo = to_photo(cropped, PREVIEW_W // 2, PREVIEW_H)
        self.full_lbl.configure(image=self._full_photo)
        self.crop_lbl.configure(image=self._crop_photo)

    # ----- export -----
    def start_export(self):
        if self.centres is None:
            messagebox.showinfo("Export", "Load a video and a boxes sheet and pick an id first.")
            return
        track_id = self.track_var.get().split()[0]
        default = f"{os.path.splitext(os.path.basename(self.video_var.get()))[0]}_id{track_id}.mp4"
        out_path = filedialog.asksaveasfilename(defaultextension=".mp4", initialfile=default,
                                                filetypes=[("MP4", "*.mp4")])
        if not out_path:
            return
        out_w, out_h, cw, ch = self.crop_geometry()
        self.cancel.clear()
        self.export_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.configure(maximum=max(1, self.info["n"]), value=0)

        def on_progress(i, msg=None):
            self.events.put(("progress", i, msg))

        def work():
            try:
                msg = export_video(self.video_var.get(), self.centres.copy(), cw, ch, out_w, out_h,
                                   out_path, on_progress, self.cancel)
                self.events.put(("done", msg, None))
            except Exception as e:
                self.events.put(("error", str(e), None))

        threading.Thread(target=work, daemon=True).start()

    def _poll_events(self):
        # Worker thread talks to Tk only through this queue (Tk isn't thread-safe)
        try:
            while True:
                kind, a, b = self.events.get_nowait()
                if kind == "progress":
                    self.progress.configure(value=a)
                    self.status_var.set(b or f"Exporting frame {a} / {self.info['n']}...")
                else:
                    self.export_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    self.status_var.set(a if kind == "done" else "Export failed.")
                    if kind == "error":
                        messagebox.showerror("Export failed", a)
        except queue.Empty:
            pass
        self.after(100, self._poll_events)


if __name__ == "__main__":
    App().mainloop()
