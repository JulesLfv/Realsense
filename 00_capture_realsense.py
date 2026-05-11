#!/usr/bin/env python3
"""
Acquisition clinique Intel RealSense D435i — 4 vues par patient.

Usage:
    python 00_capture_realsense.py [patient_id] [--timer 5]

Sans patient_id, une seule fenêtre permet de voir les acquisitions existantes,
choisir ou créer un patient, puis capturer les 4 vues caméra.

Contrôles (mode manuel) :
    ESPACE   — capturer la vue courante
    T        — basculer mode retardateur on/off
    Q        — quitter (abandon)

En mode retardateur (--timer ou touche T) :
    La capture part automatiquement après N secondes.
"""

import argparse
import shutil
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

import numpy as np

# ── dépendances optionnelles (vérification précoce) ─────────────────────────
try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 non installé. Lancer : pip install pyrealsense2")

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d non installé. Lancer : pip install open3d")

try:
    import cv2
except ImportError:
    sys.exit("opencv-python non installé. Lancer : pip install opencv-python")

# ── constantes ───────────────────────────────────────────────────────────────
VIEWS = ["Face", "Profil gauche", "Dos", "Profil droit"]
VIEW_KEYS = ["front", "left", "back", "right"]
VIEW_CHOICES = ["Face", "Profil gauche", "Dos", "Profil droit"]

# Résolution depth/color
DEPTH_W, DEPTH_H, DEPTH_FPS = 848, 480, 30
COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 4.0
# ── filtres depth ────────────────────────────────────────────────────────────
def _build_filters() -> list:
    """Construit la chaîne de filtres post-traitement (ordre obligatoire)."""
    spa  = rs.spatial_filter()
    spa.set_option(rs.option.filter_magnitude, 2)
    spa.set_option(rs.option.filter_smooth_alpha, 0.5)
    spa.set_option(rs.option.filter_smooth_delta, 20)
    tmp  = rs.temporal_filter()
    hole = rs.hole_filling_filter()
    return [spa, tmp, hole]


def _apply_filters(depth_frame, filters: list):
    for f in filters:
        depth_frame = f.process(depth_frame)
    return depth_frame


# ── acquisition d'un nuage ───────────────────────────────────────────────────
def _capture_frame(pipeline, align, filters: list, depth_scale: float):
    """
    Retourne (color_image RGB, pcd).
    Le depth est aligné sur la couleur, donc chaque point depth correspond au
    pixel couleur de même index.
    """
    for _ in range(30):
        frames = pipeline.wait_for_frames(timeout_ms=300)
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        depth_frame = _apply_filters(depth_frame, filters)
        color_image = np.asanyarray(color_frame.get_data())

        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        colors = color_image.reshape(-1, 3) / 255.0

        if len(verts) != len(colors):
            raise RuntimeError(
                f"Dimensions depth/couleur incoherentes : {len(verts)} points, "
                f"{len(colors)} pixels couleur."
            )

        mask = (verts[:, 2] > MIN_DEPTH_M) & (verts[:, 2] < MAX_DEPTH_M)
        verts = verts[mask]
        colors = colors[mask]
        if len(verts) == 0:
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print(f"[PLY] Capture : {len(verts):,} points")
        return color_image, pcd

    raise RuntimeError("Impossible d'obtenir des frames valides depuis la caméra.")


# ── affichage overlay ─────────────────────────────────────────────────────────
def _draw_overlay(img: np.ndarray, view_idx: int, n_done: int, timer_active: bool,
                  countdown: float | None = None) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    # Barre supérieure semi-transparente
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    # Vue courante
    view_label = VIEWS[view_idx] if view_idx < len(VIEWS) else "Terminé"
    cv2.putText(out, f"Vue {view_idx + 1}/4 : {view_label}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Instructions
    mode_txt = "mode : RETARDATEUR (T=off)" if timer_active else "mode : MANUEL  (T=retardateur)"
    cv2.putText(out, f"ESPACE=capturer   {mode_txt}   Q=quitter",
                (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Compte à rebours
    if countdown is not None and countdown > 0:
        txt = f"{countdown:.0f}"
        scale = 5.0
        thickness = 8
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.putText(out, txt,
                    (w // 2 - tw // 2, h // 2 + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 60, 255), thickness)

    # Points captures
    for i in range(4):
        color = (0, 220, 0) if i < n_done else (80, 80, 80)
        cv2.circle(out, (w - 40 - (3 - i) * 28, 45), 9, color, -1)

    return out


def _format_size(size_bytes: int) -> str:
    units = ["o", "Ko", "Mo", "Go"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f} {unit}" if unit == "o" else f"{size:.1f} {unit}"
        size /= 1024


def _list_patient_dirs(output_root: Path) -> list[Path]:
    if not output_root.exists():
        return []
    return sorted(
        (p for p in output_root.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=lambda p: p.name.lower(),
    )


def _start_realsense_pipeline():
    """Démarre la RealSense et retourne (pipeline, profile, align, filters)."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.rgb8, COLOR_FPS)
    profile = pipeline.start(config)
    return pipeline, profile, rs.align(rs.stream.color), _build_filters()


def _rgb_to_photoimage(img: np.ndarray, max_width: int = 880) -> tk.PhotoImage:
    """Convertit une image RGB en PhotoImage Tk sans dépendance Pillow."""
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(img.shape[0] * scale)))
    img = np.ascontiguousarray(img)
    height, width = img.shape[:2]
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return tk.PhotoImage(data=header + img.tobytes(), format="PPM")


def _populate_files_tree(files_tree: ttk.Treeview, patient_dir: Path | None):
    files_tree.delete(*files_tree.get_children())
    if patient_dir is None:
        return
    if not patient_dir.exists():
        files_tree.insert("", tk.END, text="Dossier introuvable", values=("", ""))
        return
    entries = sorted(patient_dir.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    if not entries:
        files_tree.insert("", tk.END, text="(dossier vide)", values=("", ""))
        return
    for path in entries:
        if path.is_dir():
            files_tree.insert("", tk.END, iid=str(path), text=f"{path.name}/", values=("dossier", ""))
        else:
            files_tree.insert(
                "", tk.END, iid=str(path), text=path.name,
                values=("fichier", _format_size(path.stat().st_size)),
            )


class AcquisitionApp:
    def __init__(self, output_root: Path, timer_sec: int):
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.timer_sec = timer_sec if timer_sec > 0 else 5

        self.patients: list[Path] = []
        self.patient_id: str | None = None
        self.out_dir: Path | None = None
        self.pipeline = None
        self.align = None
        self.filters = None
        self.depth_scale: float | None = None
        self.captured_plys: list[Path] = []
        self.view_idx = 0
        self.timer_active = timer_sec > 0
        self.timer_start: float | None = None
        self.running = False
        self.starting = False
        self.capturing = False
        self.last_photo = None

        self.root = tk.Tk()
        self.root.title("Acquisition RealSense - Patients et capture")
        self.root.geometry("1180x720")
        self.root.minsize(960, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._build_ui()
        self.refresh_patients()
        self.root.after(100, self.start_camera)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        left.rowconfigure(2, weight=1)
        left.rowconfigure(5, weight=2)

        ttk.Label(left, text="ID patient").grid(row=0, column=0, sticky="w")
        self.patient_entry = ttk.Entry(left, width=28)
        self.patient_entry.grid(row=1, column=0, sticky="ew", pady=(3, 8))
        self.patient_entry.bind("<Return>", lambda _event: self.create_or_select_patient())

        ttk.Label(left, text="Dossiers patients").grid(row=2, column=0, sticky="nw")
        self.patients_list = tk.Listbox(left, width=32, exportselection=False)
        self.patients_list.grid(row=3, column=0, sticky="nsew", pady=(3, 8))
        self.patients_list.bind("<<ListboxSelect>>", self.on_patient_selected)

        ttk.Label(left, text="Fichiers du dossier").grid(row=4, column=0, sticky="w")
        self.files_tree = ttk.Treeview(
            left, columns=("type", "size"), show="tree headings",
            height=12, selectmode="extended",
        )
        self.files_tree.heading("#0", text="Nom")
        self.files_tree.heading("type", text="Type")
        self.files_tree.heading("size", text="Taille")
        self.files_tree.column("#0", width=190, stretch=True)
        self.files_tree.column("type", width=70, anchor=tk.CENTER, stretch=False)
        self.files_tree.column("size", width=80, anchor=tk.E, stretch=False)
        self.files_tree.grid(row=5, column=0, sticky="nsew", pady=(3, 8))

        left_buttons = ttk.Frame(left)
        left_buttons.grid(row=6, column=0, sticky="ew")
        ttk.Button(left_buttons, text="Creer/choisir", command=self.create_or_select_patient).pack(side=tk.LEFT)
        ttk.Button(left_buttons, text="Actualiser", command=self.refresh_patients).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(left_buttons, text="Supprimer", command=self.delete_selected).pack(side=tk.LEFT, padx=(8, 0))

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.preview = ttk.Label(right, anchor=tk.CENTER, text="Demarrage camera...")
        self.preview.grid(row=0, column=0, sticky="nsew")

        controls = ttk.Frame(right)
        controls.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        controls.columnconfigure(6, weight=1)

        self.start_btn = ttk.Button(
            controls, text="Nouvelle acquisition",
            command=self.start_capture, state=tk.DISABLED,
        )
        self.start_btn.grid(row=0, column=0)
        self.capture_btn = ttk.Button(controls, text="Capturer", command=self.request_capture, state=tk.DISABLED)
        self.capture_btn.grid(row=0, column=1, padx=(8, 0))
        self.timer_btn = ttk.Button(controls, text="Retardateur", command=self.toggle_timer, state=tk.DISABLED)
        self.timer_btn.grid(row=0, column=2, padx=(8, 0))
        ttk.Label(controls, text="sec").grid(row=0, column=3, padx=(8, 0))
        self.timer_spin = ttk.Spinbox(
            controls, from_=1, to=30, width=4,
            command=self.update_timer_seconds,
        )
        self.timer_spin.set(str(self.timer_sec))
        self.timer_spin.grid(row=0, column=4, padx=(4, 0))
        self.retake_view = tk.StringVar(value=VIEW_CHOICES[0])
        self.retake_combo = ttk.Combobox(
            controls, textvariable=self.retake_view, values=VIEW_CHOICES,
            state="readonly", width=15,
        )
        self.retake_combo.grid(row=0, column=5, padx=(18, 0))
        self.retake_btn = ttk.Button(
            controls, text="Refaire cette vue",
            command=self.retake_selected_view, state=tk.DISABLED,
        )
        self.retake_btn.grid(row=0, column=6, padx=(8, 0))
        ttk.Button(controls, text="Quitter", command=self.close).grid(row=0, column=7, padx=(8, 0))

        self.status_var = tk.StringVar(value=f"Dossier racine : {self.output_root}")
        ttk.Label(right, textvariable=self.status_var).grid(row=2, column=0, sticky="w", pady=(8, 0))

    def refresh_patients(self):
        self.patients = _list_patient_dirs(self.output_root)
        self.patients_list.delete(0, tk.END)
        for patient_dir in self.patients:
            n_files = sum(1 for p in patient_dir.iterdir() if p.is_file())
            self.patients_list.insert(tk.END, f"{patient_dir.name}  ({n_files} fichier(s))")
        self.status_var.set(f"{len(self.patients)} dossier(s) dans {self.output_root}")

    def current_patient_dir(self) -> Path | None:
        selection = self.patients_list.curselection()
        if not selection:
            return None
        idx = selection[0]
        return self.patients[idx] if 0 <= idx < len(self.patients) else None

    def on_patient_selected(self, _event=None):
        patient_dir = self.current_patient_dir()
        if patient_dir is None:
            return
        self.patient_entry.delete(0, tk.END)
        self.patient_entry.insert(0, patient_dir.name)
        _populate_files_tree(self.files_tree, patient_dir)

    def create_or_select_patient(self):
        patient_id = self.patient_entry.get().strip()
        if not patient_id:
            messagebox.showwarning("ID patient manquant", "Entre ou sélectionne un ID patient.")
            return
        self.patient_id = patient_id
        self.out_dir = self.output_root / patient_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.refresh_patients()
        for idx, path in enumerate(self.patients):
            if path.name == patient_id:
                self.patients_list.selection_clear(0, tk.END)
                self.patients_list.selection_set(idx)
                self.patients_list.see(idx)
                break
        _populate_files_tree(self.files_tree, self.out_dir)
        self.status_var.set(f"Patient selectionne : {patient_id}")

    def selected_file_paths(self) -> list[Path]:
        paths = []
        for item_id in self.files_tree.selection():
            path = Path(item_id)
            if path.exists():
                paths.append(path)
        return paths

    def delete_selected(self):
        if self.starting or self.capturing:
            messagebox.showwarning("Suppression impossible", "Attends la fin de la capture en cours.")
            return

        targets = self.selected_file_paths()
        if not targets:
            patient_dir = self.current_patient_dir()
            targets = [patient_dir] if patient_dir is not None else []
        if not targets:
            messagebox.showwarning("Rien a supprimer", "Selectionne un fichier ou un dossier patient.")
            return

        if len(targets) == 1:
            target = targets[0]
            kind = "dossier" if target.is_dir() else "fichier"
            message = f"Supprimer ce {kind} ?\n\n{target}"
        else:
            kind = "element"
            message = "Supprimer ces elements ?\n\n" + "\n".join(str(p) for p in targets)
        if not messagebox.askyesno(
            "Confirmer la suppression",
            message,
        ):
            return

        for target in targets:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            if self.out_dir == target or (self.out_dir and not self.out_dir.exists()):
                self.patient_id = None
                self.out_dir = None
                self.patient_entry.delete(0, tk.END)

        self.refresh_patients()
        if self.out_dir and self.out_dir.exists():
            _populate_files_tree(self.files_tree, self.out_dir)
        else:
            _populate_files_tree(self.files_tree, None)
        if len(targets) == 1:
            self.status_var.set(f"{kind.capitalize()} supprime.")
        else:
            self.status_var.set(f"{len(targets)} elements supprimes.")

    def start_camera(self):
        if self.starting or self.running:
            return
        self.starting = True
        self.start_btn.config(state=tk.DISABLED)
        self.status_var.set("Demarrage camera...")
        self.root.update_idletasks()

        try:
            self.pipeline, profile, self.align, self.filters = _start_realsense_pipeline()
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"Depth scale : {self.depth_scale:.6f} m/unit")
            self.status_var.set("Chauffe du capteur...")
            self.root.update_idletasks()
            for _ in range(30):
                self.pipeline.wait_for_frames(timeout_ms=5000)
        except Exception as e:
            messagebox.showerror("Erreur camera", str(e))
            self.stop_pipeline()
            self.starting = False
            self.start_btn.config(state=tk.NORMAL)
            return

        self.starting = False
        self.running = True
        self.start_btn.config(state=tk.NORMAL)
        self.timer_btn.config(state=tk.NORMAL)
        self.retake_btn.config(state=tk.NORMAL)
        self.status_var.set("Camera prete. Choisis un patient puis lance une acquisition.")
        self.update_preview()

    def start_capture(self):
        if self.starting:
            return
        if not self.running:
            self.start_camera()
        if not self.running:
            return

        self.create_or_select_patient()
        if not self.patient_id or not self.out_dir:
            return

        self.captured_plys = []
        self.view_idx = 0
        self.timer_start = None
        self.capture_btn.config(state=tk.NORMAL)
        self.timer_btn.config(state=tk.NORMAL)
        self.retake_btn.config(state=tk.NORMAL)
        self.status_var.set("Camera prete. Capture la vue face.")

    def update_preview(self):
        if not self.running or self.pipeline is None:
            return

        trigger = False
        countdown = None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            aligned = self.align.process(frames)
            color_f = aligned.get_color_frame()
            if color_f:
                preview = np.asanyarray(color_f.get_data())

                if self.timer_active and self.timer_start is not None:
                    remaining = self.timer_sec - (time.monotonic() - self.timer_start)
                    if remaining <= 0:
                        trigger = True
                        self.timer_start = None
                    else:
                        countdown = remaining

                frame = _draw_overlay(preview, self.view_idx, len(self.captured_plys),
                                      self.timer_active, countdown)
                self.last_photo = _rgb_to_photoimage(frame)
                self.preview.config(image=self.last_photo, text="")
        except Exception as e:
            self.status_var.set(f"Erreur preview : {e}")

        if trigger:
            self.capture_current_view()

        if self.running:
            self.root.after(15, self.update_preview)

    def request_capture(self):
        if self.timer_active:
            self.update_timer_seconds()
            self.timer_start = time.monotonic()
            self.status_var.set(f"Retardateur {self.timer_sec}s...")
        else:
            self.capture_current_view()

    def update_timer_seconds(self):
        try:
            self.timer_sec = max(1, int(self.timer_spin.get()))
        except ValueError:
            self.timer_sec = 5
            self.timer_spin.set(str(self.timer_sec))

    def toggle_timer(self):
        self.update_timer_seconds()
        self.timer_active = not self.timer_active
        self.timer_start = None
        if self.timer_active:
            self.status_var.set(f"Retardateur active ({self.timer_sec}s).")
        else:
            self.status_var.set("Retardateur desactive.")

    def capture_current_view(self):
        if not self.running or not self.out_dir or self.pipeline is None:
            return
        if self.view_idx >= len(VIEW_KEYS):
            return

        ok = self.capture_view(self.view_idx)
        if not ok:
            return

        self.view_idx += 1
        if self.view_idx < len(VIEW_KEYS):
            self.status_var.set(f"OK. Prochaine vue : {VIEWS[self.view_idx]}")
            self.timer_start = None
        else:
            self.finish_capture()

    def capture_view(self, view_idx: int) -> bool:
        if not self.running or not self.out_dir or self.pipeline is None or self.depth_scale is None:
            return False

        view_name = VIEW_KEYS[view_idx]
        self.status_var.set(f"Capture {view_name}...")
        self.root.update_idletasks()
        self.capturing = True
        try:
            color_img, pcd = _capture_frame(
                self.pipeline, self.align, self.filters, self.depth_scale,
            )
        except RuntimeError as e:
            self.capturing = False
            self.status_var.set(f"Erreur capture : {e}")
            return False
        self.capturing = False

        jpg_path = self.out_dir / f"{view_name}.jpg"
        ply_path = self.out_dir / f"{view_name}.ply"
        cv2.imwrite(str(jpg_path), cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
        o3d.io.write_point_cloud(str(ply_path), pcd)
        if ply_path not in self.captured_plys:
            self.captured_plys.append(ply_path)
        _populate_files_tree(self.files_tree, self.out_dir)
        return True

    def retake_selected_view(self):
        if not self.running:
            messagebox.showwarning("Camera arretee", "Demarre la camera avant de refaire une vue.")
            return
        try:
            view_idx = VIEW_CHOICES.index(self.retake_view.get())
        except ValueError:
            view_idx = 0
        if self.capture_view(view_idx):
            self.status_var.set(f"Vue refaite : {VIEW_CHOICES[view_idx]}")

    def finish_capture(self):
        self.capture_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.NORMAL)

        if self.out_dir:
            self.status_var.set(f"Acquisition terminee. Camera toujours active : {self.out_dir}")
            _populate_files_tree(self.files_tree, self.out_dir)
            self.refresh_patients()

    def stop_pipeline(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None
        self.depth_scale = None

    def close(self):
        self.running = False
        self.capturing = False
        self.stop_pipeline()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def run_gui(output_root: Path, timer_sec: int):
    AcquisitionApp(output_root, timer_sec).run()


# ── boucle principale ────────────────────────────────────────────────────────
def run(patient_id: str, output_root: Path, timer_sec: int):
    out_dir = output_root / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dossier patient : {out_dir}")

    pipeline, profile, align, filters = _start_realsense_pipeline()

    # Récupérer les intrinsèques (pour référence future)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()
    print(f"Depth scale : {depth_scale:.6f} m/unit")

    # Chauffe — laisser le capteur se stabiliser (auto-exposition, IR)
    print("Chauffe du capteur…", end=" ", flush=True)
    for _ in range(30):
        pipeline.wait_for_frames(timeout_ms=5000)
    print("OK")

    captured_plys = []
    view_idx      = 0
    timer_active  = timer_sec > 0
    timer_start   = None    # moment où le décompte a commencé
    window        = "RealSense D435i — Acquisition"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    print("\nContrôles : ESPACE=capturer  T=retardateur  Q=quitter\n")

    try:
        while view_idx < 4:
            # Lire la frame live
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            color_f = aligned.get_color_frame()
            if not color_f:
                continue
            preview = np.asanyarray(color_f.get_data())

            # Calcul countdown si retardateur actif
            countdown = None
            trigger   = False
            if timer_active and timer_start is not None:
                elapsed   = time.monotonic() - timer_start
                remaining = timer_sec - elapsed
                if remaining <= 0:
                    trigger     = True
                    timer_start = None
                else:
                    countdown = remaining

            # Affichage
            frame = _draw_overlay(preview, view_idx, len(captured_plys),
                                  timer_active, countdown)
            cv2.imshow(window, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                print("Abandon par l'utilisateur.")
                break

            elif key == ord('t'):
                timer_active = not timer_active
                timer_start  = None
                status = "activé" if timer_active else "désactivé"
                print(f"Retardateur {status}.")

            elif key == ord(' '):
                if timer_active:
                    # Démarre / relance le décompte
                    timer_start = time.monotonic()
                    print(f"Retardateur {timer_sec}s…")
                else:
                    trigger = True

            # Déclenchement effectif
            if trigger:
                view_name = VIEW_KEYS[view_idx]
                print(f"Capture : {view_name}…", end=" ", flush=True)
                try:
                    color_img, pcd = _capture_frame(pipeline, align, filters, depth_scale)
                except RuntimeError as e:
                    print(f"ERREUR : {e}")
                    continue

                jpg_path = out_dir / f"{view_name}.jpg"
                ply_path = out_dir / f"{view_name}.ply"
                cv2.imwrite(str(jpg_path), cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
                o3d.io.write_point_cloud(str(ply_path), pcd)
                captured_plys.append(ply_path)
                print(f"OK ({len(pcd.points):,} pts) → {jpg_path.name} + {ply_path.name}")

                view_idx += 1

                if view_idx < 4:
                    print(f"\n→ Prochaine vue : {VIEWS[view_idx]}")
                    if timer_active:
                        timer_start = None   # reset le décompte pour la prochaine vue

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if len(captured_plys) == 4:
        print(f"\nAcquisition terminée. Résultats dans : {out_dir}")
    else:
        print(f"\nAcquisition incomplète ({len(captured_plys)}/4 vues).")


# ── entrée ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patient_id", nargs="?",
                        help="Identifiant patient (ex: P001). Crée datasets/acquisitions/<id>/")
    parser.add_argument("--timer", type=int, default=0, metavar="SEC",
                        help="Durée du retardateur en secondes (0 = mode manuel par défaut)")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "datasets" / "acquisitions",
                        help="Dossier racine pour les acquisitions")
    args = parser.parse_args()

    if args.timer < 0:
        parser.error("--timer doit être >= 0")

    if args.patient_id:
        run(args.patient_id, args.output, args.timer)
    else:
        run_gui(args.output, args.timer)


if __name__ == "__main__":
    main()
