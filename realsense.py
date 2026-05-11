#!/usr/bin/env python3
"""
Visualiseur clinique 3D — projet M26068

Panneau gauche   : liste des patients (datasets/acquisitions/<id>/)
Panneau central  : nuage de points (par vue) OU mesh SMPL avec anneaux de mesure
Panneau droit    : photo front.jpg + mesures + sliders pour repositionner les coupes

Usage :
    python3 visu.py
"""

import os
import importlib.util
import shutil
import sys
import traceback
from pathlib import Path

# PyVista/VTK est plus stable avec X11/xcb sur certaines sessions Linux.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
import trimesh
import pyvista as pv
from pyvistaqt import QtInteractor

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QTableWidget, QTableWidgetItem, QLabel,
    QPushButton, QButtonGroup, QSlider, QSizePolicy, QLineEdit,
    QTabWidget, QComboBox, QSpinBox, QMessageBox, QTreeWidget,
    QTreeWidgetItem, QAbstractItemView, QListWidgetItem,
)
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

# ── chemins projet ────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent
ACQ_ROOT    = REPO_ROOT / "datasets" / "acquisitions"
BENCH_UTILS = REPO_ROOT / "benchmark" / "utils"
sys.path.insert(0, str(BENCH_UTILS))

from smpl_measurements import (
    load_mesh,
    chest_circumference_below_arms_smpl,
    measure_from_part_seg,
    section_circumference,
    _load_smpl_part_seg,
    SMPL_ARM_PARTS,
)

# ── joints SMPL standard (indices dans le tableau 24×3) ──────────────────────
# Référence : https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
J_PELVIS        = 0
J_LEFT_HIP      = 1
J_RIGHT_HIP     = 2
J_SPINE1        = 3   # bas du dos
J_LEFT_KNEE     = 4
J_RIGHT_KNEE    = 5
J_SPINE2        = 6   # milieu du dos / taille
J_SPINE3        = 9   # haut du dos / sous-poitrine
J_LEFT_SHOULDER = 16
J_RIGHT_SHOULDER= 17

# ── constantes mesures ────────────────────────────────────────────────────────
# (label, clé smpl_measurements, couleur anneau hex)
MEASURES = [
    ("Poitrine", "chest_girth",  "#e63030"),
    ("Taille",   "waist_girth",  "#e69030"),
    ("Hanches",  "hip_girth",    "#30b030"),
    ("Cuisse",   "thigh_girth",  "#3080e6"),
]

COLOR_GREEN  = (144, 238, 144)
COLOR_ORANGE = (255, 200, 100)
COLOR_RED    = (255, 100, 100)

VIEW_KEYS = ["front", "left", "back", "right"]
VIEW_LABELS = ["Face", "Profil gauche", "Dos", "Profil droit"]


def _load_capture_module():
    path = REPO_ROOT / "00_capture_realsense.py"
    spec = importlib.util.spec_from_file_location("capture_realsense", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── calcul des mesures + hauteurs de coupe ────────────────────────────────────
def _heights_from_joints(joints: np.ndarray) -> dict[str, float]:
    """
    Dérive les hauteurs de coupe optimales depuis les 24 joints SMPL.

    Anatomie :
      Poitrine : juste sous les épaules  → Y épaule - 3 cm
      Taille   : mi-chemin spine2/spine3 → niveau le plus étroit du tronc
      Hanches  : mi-chemin pelvis/hanches → point le plus large du bassin
      Cuisse   : mi-chemin hanche/genou  → mi-cuisse
    """
    y_shoulder = float((joints[J_LEFT_SHOULDER, 1] + joints[J_RIGHT_SHOULDER, 1]) / 2)
    y_spine2   = float(joints[J_SPINE2, 1])
    y_spine3   = float(joints[J_SPINE3, 1])
    y_pelvis   = float(joints[J_PELVIS, 1])
    y_lhip     = float(joints[J_LEFT_HIP,  1])
    y_rhip     = float(joints[J_RIGHT_HIP, 1])
    y_lknee    = float(joints[J_LEFT_KNEE,  1])
    y_rknee    = float(joints[J_RIGHT_KNEE, 1])

    y_hip_avg  = (y_lhip  + y_rhip)  / 2
    y_knee_avg = (y_lknee + y_rknee) / 2

    return {
        "Poitrine": y_shoulder - 0.03,
        "Taille":   (y_spine2 + y_spine3) / 2,
        "Hanches":  (y_pelvis + y_hip_avg) / 2,
        "Cuisse":   (y_hip_avg + y_knee_avg) / 2,
    }


def compute_full(obj_path: Path) -> tuple[dict, dict, trimesh.Trimesh | None]:
    """
    Retourne (values_cm, y_heights_m, tri_mesh).
    Utilise joints.npy si disponible pour placer les coupes anatomiquement.
    """
    try:
        mesh = load_mesh(str(obj_path))
    except Exception as e:
        print(f"[MESURES] Erreur : {e}")
        return {}, {}, None

    verts = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.faces)
    # mesh.obj est déjà normalisé Y-up (sol à 0) par save_results
    mesh_norm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    values, heights = {}, {}
    values["Hauteur"] = float(verts[:, 1].max()) * 100

    # Charger les joints si disponibles
    joints_path = obj_path.parent / "joints.npy"
    if joints_path.exists():
        joints  = np.load(str(joints_path))   # (24, 3)
        heights = _heights_from_joints(joints)
        print(f"[MESURES] Hauteurs depuis joints SMPL : "
              + ", ".join(f"{k}={v*100:.1f}cm" for k, v in heights.items()))
    else:
        # Fallback : centroïdes des parties (ancienne méthode)
        print("[MESURES] joints.npy absent — fallback part_seg")
        seg = _load_smpl_part_seg()
        arm_ids = set()
        for p in SMPL_ARM_PARTS:
            arm_ids.update(seg.get(p, []).tolist())
        y_chest = float(verts[np.array(sorted(arm_ids)), 1].min()) - 0.015 \
                  if arm_ids else float(verts[:, 1].max()) * 0.79
        heights["Poitrine"] = y_chest
        for label, parts in [("Taille",["spine"]),("Hanches",["hips"]),
                              ("Cuisse",["leftUpLeg","rightUpLeg"])]:
            ids = set()
            for p in parts: ids.update(seg.get(p, []).tolist())
            pv_ = verts[np.array(sorted(ids))]
            heights[label] = float((pv_[:, 1].max() + pv_[:, 1].min()) / 2)

    # Calculer les circonférences aux hauteurs déterminées (section_circumference → mètres → *100 = cm)
    c = section_circumference(mesh_norm, heights["Poitrine"], mode="sum")
    values["Poitrine"] = c * 100 if c else None
    key_map = {"Taille":"waist_girth","Hanches":"hip_girth","Cuisse":"thigh_girth"}
    for label, key in key_map.items():
        c = measure_from_part_seg(verts, faces, key)
        values[label] = c * 100 if c else None

    return values, heights, mesh_norm


# ── section → anneau PyVista ──────────────────────────────────────────────────
def get_ring(tri_mesh: trimesh.Trimesh, y: float) -> pv.PolyData | None:
    """Coupe horizontale à hauteur y → contour PyVista (lignes)."""
    try:
        section = tri_mesh.section(
            plane_origin=[0.0, y, 0.0],
            plane_normal=[0.0, 1.0, 0.0])
    except Exception:
        return None
    if section is None or len(section.entities) == 0:
        return None

    pts = section.vertices
    lines = []
    for ent in section.entities:
        idx = list(ent.points)
        for i in range(len(idx) - 1):
            lines.extend([2, idx[i], idx[i + 1]])
        if len(idx) > 1:
            lines.extend([2, idx[-1], idx[0]])
    if not lines:
        return None

    poly = pv.PolyData()
    poly.points = pts
    poly.lines  = np.array(lines, dtype=np.int_)
    return poly


# ── chargement PLY avec crop fond ─────────────────────────────────────────────
def load_pyvista_pointcloud(ply_path: Path) -> pv.PolyData:
    cloud = pv.read(str(ply_path))
    pts   = cloud.points
    if len(pts) == 0:
        return cloud
    shifted       = pts.copy()
    shifted[:, 1] -= np.percentile(shifted[:, 1], 1)
    cx   = np.median(shifted[:, 0])
    cz   = np.median(shifted[:, 2])
    d_xz = np.sqrt((shifted[:, 0] - cx) ** 2 + (shifted[:, 2] - cz) ** 2)
    mask = (d_xz < 0.65) & (shifted[:, 1] > 0.02) & (shifted[:, 1] < 2.3)
    print(f"[PLY] Crop : {mask.sum():,} / {len(pts):,} points conservés")
    return cloud.extract_points(mask)


# ── page acquisition ─────────────────────────────────────────────────────────
class AcquisitionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.capture = None
        self.pipeline = None
        self.align = None
        self.filters = None
        self.depth_scale = None
        self.patient_id: str | None = None
        self.out_dir: Path | None = None
        self.view_idx = 0
        self.timer_active = False
        self.timer_seconds = 5
        self.timer_start: float | None = None
        self.capturing = False

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._update_preview)

        layout = QHBoxLayout(self)

        left_w = QWidget(); left_w.setFixedWidth(320)
        left = QVBoxLayout(left_w)
        left.addWidget(QLabel("ID patient"))
        self.patient_edit = QLineEdit()
        self.patient_edit.setPlaceholderText("Entrer ou sélectionner un ID")
        self.patient_edit.returnPressed.connect(self._create_or_select_patient)
        left.addWidget(self.patient_edit)

        row = QHBoxLayout()
        self.select_btn = QPushButton("Créer/choisir")
        self.select_btn.clicked.connect(self._create_or_select_patient)
        row.addWidget(self.select_btn)
        self.refresh_btn = QPushButton("Actualiser")
        self.refresh_btn.clicked.connect(self._refresh_patients)
        row.addWidget(self.refresh_btn)
        self.delete_btn = QPushButton("Supprimer")
        self.delete_btn.clicked.connect(self._delete_selected)
        row.addWidget(self.delete_btn)
        left.addLayout(row)

        left.addWidget(QLabel("Dossiers patients"))
        self.patient_list = QListWidget()
        self.patient_list.currentItemChanged.connect(self._on_patient_selected)
        left.addWidget(self.patient_list, stretch=1)

        left.addWidget(QLabel("Fichiers du dossier"))
        self.files_tree = QTreeWidget()
        self.files_tree.setHeaderLabels(["Nom", "Type", "Taille"])
        self.files_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        left.addWidget(self.files_tree, stretch=2)
        layout.addWidget(left_w)

        right = QVBoxLayout()
        self.preview = QLabel("Onglet Acquisition prêt. La caméra démarrera à l'ouverture de cet onglet.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 420)
        self.preview.setStyleSheet("background:#111;color:#ddd;")
        right.addWidget(self.preview, stretch=1)

        controls = QHBoxLayout()
        self.new_acq_btn = QPushButton("Nouvelle acquisition")
        self.new_acq_btn.clicked.connect(self._start_sequence)
        self.new_acq_btn.setEnabled(False)
        controls.addWidget(self.new_acq_btn)
        self.capture_btn = QPushButton("Capturer")
        self.capture_btn.clicked.connect(self._request_capture)
        self.capture_btn.setEnabled(False)
        controls.addWidget(self.capture_btn)
        self.timer_btn = QPushButton("Retardateur")
        self.timer_btn.clicked.connect(self._toggle_timer)
        self.timer_btn.setEnabled(False)
        controls.addWidget(self.timer_btn)
        self.timer_spin = QSpinBox()
        self.timer_spin.setRange(1, 30)
        self.timer_spin.setValue(self.timer_seconds)
        controls.addWidget(self.timer_spin)
        controls.addWidget(QLabel("sec"))
        self.retake_combo = QComboBox()
        self.retake_combo.addItems(VIEW_LABELS)
        controls.addWidget(self.retake_combo)
        self.retake_btn = QPushButton("Refaire cette vue")
        self.retake_btn.clicked.connect(self._retake_view)
        self.retake_btn.setEnabled(False)
        controls.addWidget(self.retake_btn)
        right.addLayout(controls)

        self.status = QLabel(f"Dossier racine : {ACQ_ROOT}")
        right.addWidget(self.status)
        layout.addLayout(right, stretch=1)

        self._refresh_patients()

    def closeEvent(self, event):
        self._stop_camera()
        super().closeEvent(event)

    def shutdown(self):
        self._stop_camera()

    def _start_camera(self):
        if self.pipeline is not None:
            return
        try:
            self.capture = _load_capture_module()
            self.pipeline, profile, self.align, self.filters = self.capture._start_realsense_pipeline()
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self.status.setText("Caméra prête. Choisis un patient.")
        except BaseException as e:
            traceback.print_exc()
            self.preview.setText("Caméra indisponible")
            self.status.setText(f"{type(e).__name__}: {e}")
            return
        self.new_acq_btn.setEnabled(True)
        self.timer_btn.setEnabled(True)
        self.retake_btn.setEnabled(True)
        self.preview_timer.start(30)

    def ensure_camera_started(self):
        if self.pipeline is None:
            self._start_camera()

    def _stop_camera(self):
        self.preview_timer.stop()
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None
        self.depth_scale = None

    def _update_preview(self):
        if self.pipeline is None:
            return
        try:
            frames = self.pipeline.poll_for_frames()
            if not frames:
                return
            aligned = self.align.process(frames)
            color_f = aligned.get_color_frame()
            if not color_f:
                return
            frame = np.asanyarray(color_f.get_data())
            countdown = None
            if self.timer_active and self.timer_start is not None:
                import time
                remaining = self.timer_seconds - (time.monotonic() - self.timer_start)
                if remaining <= 0:
                    self.timer_start = None
                    self._capture_current_view()
                else:
                    countdown = remaining
            display_idx = min(self.view_idx, len(VIEW_KEYS) - 1)
            frame = self.capture._draw_overlay(
                frame, display_idx, min(self.view_idx, 4), self.timer_active, countdown)
            image = QImage(frame.data, frame.shape[1], frame.shape[0],
                           frame.strides[0], QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(image.copy()).scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.preview.setPixmap(pix)
        except Exception as e:
            if "frame didn't arrive" not in str(e):
                self.status.setText(f"Preview : {e}")

    def _refresh_patients(self):
        self.patient_list.clear()
        ACQ_ROOT.mkdir(parents=True, exist_ok=True)
        patients = sorted(p for p in ACQ_ROOT.iterdir() if p.is_dir() and not p.name.startswith("."))
        for path in patients:
            n_files = sum(1 for p in path.iterdir() if p.is_file())
            item = QListWidgetItem(f"{path.name}  ({n_files} fichier(s))")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.patient_list.addItem(item)

    def _on_patient_selected(self, item):
        if item is None:
            return
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        self.patient_edit.setText(path.name)
        self.patient_id = path.name
        self.out_dir = path
        self._refresh_files()

    def _refresh_files(self):
        self.files_tree.clear()
        if not self.out_dir or not self.out_dir.exists():
            return
        for path in sorted(self.out_dir.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            kind = "dossier" if path.is_dir() else "fichier"
            size = "" if path.is_dir() else _format_size(path.stat().st_size)
            item = QTreeWidgetItem([path.name + ("/" if path.is_dir() else ""), kind, size])
            item.setData(0, Qt.ItemDataRole.UserRole, str(path))
            self.files_tree.addTopLevelItem(item)

    def _create_or_select_patient(self):
        pid = self.patient_edit.text().strip()
        if not pid:
            QMessageBox.warning(self, "ID patient manquant", "Entre ou sélectionne un ID patient.")
            return False
        self.patient_id = pid
        self.out_dir = ACQ_ROOT / pid
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._refresh_patients()
        self._refresh_files()
        self.status.setText(f"Patient sélectionné : {pid}")
        return True

    def _delete_selected(self):
        if self.capturing:
            QMessageBox.warning(self, "Suppression impossible", "Attends la fin de la capture.")
            return
        targets = []
        for item in self.files_tree.selectedItems():
            path = Path(item.data(0, Qt.ItemDataRole.UserRole))
            if path.exists():
                targets.append(path)
        if not targets and self.out_dir:
            targets = [self.out_dir]
        if not targets:
            return
        msg = "\n".join(str(p) for p in targets)
        if QMessageBox.question(self, "Confirmer", f"Supprimer ?\n\n{msg}") != QMessageBox.StandardButton.Yes:
            return
        for path in targets:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        if self.out_dir and not self.out_dir.exists():
            self.out_dir = None
            self.patient_id = None
            self.patient_edit.clear()
        self._refresh_patients()
        self._refresh_files()

    def _start_sequence(self):
        if not self._create_or_select_patient():
            return
        self.view_idx = 0
        self.capture_btn.setEnabled(True)
        self.status.setText("Capture la vue Face.")

    def _request_capture(self):
        if self.timer_active:
            import time
            self.timer_seconds = self.timer_spin.value()
            self.timer_start = time.monotonic()
            self.status.setText(f"Retardateur {self.timer_seconds}s...")
        else:
            self._capture_current_view()

    def _toggle_timer(self):
        self.timer_active = not self.timer_active
        self.timer_seconds = self.timer_spin.value()
        self.timer_start = None
        self.status.setText(
            f"Retardateur activé ({self.timer_seconds}s)." if self.timer_active
            else "Retardateur désactivé.")

    def _capture_view(self, idx: int):
        if not self.out_dir or self.pipeline is None or self.depth_scale is None:
            return False
        self.capturing = True
        self.status.setText(f"Capture {VIEW_LABELS[idx]}...")
        QApplication.processEvents()
        try:
            color_img, pcd = self.capture._capture_frame(
                self.pipeline, self.align, self.filters, self.depth_scale)
        except Exception as e:
            self.status.setText(f"Erreur capture : {e}")
            self.capturing = False
            return False
        import cv2
        import open3d as o3d
        jpg_path = self.out_dir / f"{VIEW_KEYS[idx]}.jpg"
        ply_path = self.out_dir / f"{VIEW_KEYS[idx]}.ply"
        cv2.imwrite(str(jpg_path), cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
        o3d.io.write_point_cloud(str(ply_path), pcd)
        self.capturing = False
        self._refresh_files()
        return True

    def _capture_current_view(self):
        if self.view_idx >= len(VIEW_KEYS):
            return
        if not self._capture_view(self.view_idx):
            return
        self.view_idx += 1
        if self.view_idx < len(VIEW_KEYS):
            self.status.setText(f"OK. Prochaine vue : {VIEW_LABELS[self.view_idx]}")
        else:
            self.capture_btn.setEnabled(False)
            self.status.setText("Acquisition terminée. Caméra toujours active.")

    def _retake_view(self):
        if not self._create_or_select_patient():
            return
        idx = self.retake_combo.currentIndex()
        if self._capture_view(idx):
            self.status.setText(f"Vue refaite : {VIEW_LABELS[idx]}")


# ── fenêtre principale ────────────────────────────────────────────────────────
class ClinicalViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clinical 3D Body Viewer — M26068")
        self.resize(1600, 900)

        self._current_patient: str | None      = None
        self._show_mesh  = False
        self._ply_view   = "front"
        self._tri_mesh: trimesh.Trimesh | None = None   # mesh normalisé en mémoire
        self._y_heights: dict[str, float]      = {}     # hauteurs courantes (m)
        self._ring_actors: dict[str, any]      = {}     # acteurs PyVista des anneaux

        central = QWidget()
        self.setCentralWidget(central)
        layout  = QHBoxLayout(central)

        # ── panneau gauche ────────────────────────────────────────────────────
        left_w = QWidget(); left_w.setFixedWidth(240)
        left   = QVBoxLayout(left_w)
        lbl    = QLabel("Patients")
        lbl.setStyleSheet("font-weight:bold;font-size:14px;padding:4px;")
        left.addWidget(lbl)

        self.patient_id_edit = QLineEdit()
        self.patient_id_edit.setPlaceholderText("Entrer un ID patient...")
        self.patient_id_edit.returnPressed.connect(self._load_patient_from_text)
        left.addWidget(self.patient_id_edit)

        id_btn_row = QHBoxLayout()
        self.load_patient_btn = QPushButton("Charger")
        self.load_patient_btn.clicked.connect(self._load_patient_from_text)
        id_btn_row.addWidget(self.load_patient_btn)
        self.refresh_patients_btn = QPushButton("↻")
        self.refresh_patients_btn.setToolTip("Actualiser les dossiers patients")
        self.refresh_patients_btn.setFixedWidth(36)
        self.refresh_patients_btn.clicked.connect(self._load_patient_list)
        id_btn_row.addWidget(self.refresh_patients_btn)
        left.addLayout(id_btn_row)

        self.patient_list = QListWidget()
        self.patient_list.itemClicked.connect(self._on_patient_selected)
        left.addWidget(self.patient_list)

        files_lbl = QLabel("Contenu du dossier")
        files_lbl.setStyleSheet("font-weight:bold;font-size:13px;padding:4px;margin-top:6px;")
        left.addWidget(files_lbl)
        self.files_list = QListWidget()
        self.files_list.setMinimumHeight(150)
        left.addWidget(self.files_list)
        layout.addWidget(left_w)

        # ── panneau central ───────────────────────────────────────────────────
        center_w = QWidget()
        center   = QVBoxLayout(center_w)

        # Boutons type de vue
        btn_row = QHBoxLayout()
        self._btn_group = QButtonGroup(); self._btn_group.setExclusive(True)
        self._btn_cloud = QPushButton("Nuage de points")
        self._btn_mesh  = QPushButton("Mesh SMPL")
        for btn in (self._btn_cloud, self._btn_mesh):
            btn.setCheckable(True)
            btn.setStyleSheet("padding:4px 12px;")
            self._btn_group.addButton(btn)
            btn_row.addWidget(btn)
        self._btn_cloud.setChecked(True)
        self._btn_cloud.clicked.connect(lambda: self._switch_view(False))
        self._btn_mesh.clicked.connect(lambda:  self._switch_view(True))
        center.addLayout(btn_row)

        # Boutons vue PLY
        self._ply_row_w = QWidget()
        ply_row = QHBoxLayout(self._ply_row_w); ply_row.setContentsMargins(0,0,0,0)
        self._ply_btn_group = QButtonGroup(); self._ply_btn_group.setExclusive(True)
        self._ply_btns: dict[str, QPushButton] = {}
        for key, label in [("front","Face"),("left","Profil G"),
                            ("back","Dos"),("right","Profil D"),("fused","Fusionné")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setStyleSheet("padding:3px 8px;font-size:11px;")
            self._ply_btn_group.addButton(btn)
            ply_row.addWidget(btn)
            self._ply_btns[key] = btn
            btn.clicked.connect(lambda checked, k=key: self._switch_ply_view(k))
        self._ply_btns["front"].setChecked(True)
        center.addWidget(self._ply_row_w)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        center.addWidget(self.plotter.interactor)
        layout.addWidget(center_w, stretch=3)

        # ── panneau droit ─────────────────────────────────────────────────────
        right_w = QWidget(); right_w.setFixedWidth(340)
        right   = QVBoxLayout(right_w)

        photo_lbl = QLabel("Vue de face")
        photo_lbl.setStyleSheet("font-weight:bold;font-size:13px;padding:4px;")
        right.addWidget(photo_lbl)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(280)
        self.image_label.setStyleSheet("background:#eee;border:1px solid #ccc;")
        right.addWidget(self.image_label)

        meas_lbl = QLabel("Mesures anthropométriques")
        meas_lbl.setStyleSheet("font-weight:bold;font-size:13px;padding:4px;")
        right.addWidget(meas_lbl)

        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Mesure", "Valeur (cm)"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setMaximumHeight(175)
        right.addWidget(self.metrics_table)

        # Sliders de repositionnement des coupes (un par mesure sauf Hauteur)
        sliders_lbl = QLabel("Repositionner les coupes (cm)")
        sliders_lbl.setStyleSheet("font-weight:bold;font-size:12px;padding:4px;margin-top:6px;")
        right.addWidget(sliders_lbl)

        self._sliders: dict[str, QSlider]      = {}
        self._slider_lbls: dict[str, QLabel]   = {}

        for label, _, color in MEASURES:
            row_w = QWidget()
            row   = QHBoxLayout(row_w); row.setContentsMargins(4, 1, 4, 1)

            dot = QLabel("●")
            dot.setStyleSheet(f"color:{color};font-size:14px;")
            dot.setFixedWidth(18)
            row.addWidget(dot)

            name_lbl = QLabel(label)
            name_lbl.setFixedWidth(60)
            row.addWidget(name_lbl)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 230)      # 0–230 cm → 0–2.3 m
            slider.setValue(100)
            slider.setEnabled(False)
            row.addWidget(slider, stretch=1)

            val_lbl = QLabel("—")
            val_lbl.setFixedWidth(42)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(val_lbl)

            right.addWidget(row_w)
            self._sliders[label]     = slider
            self._slider_lbls[label] = val_lbl

            # Connexion : déplacement du slider → recalcul
            slider.valueChanged.connect(
                lambda v, lbl=label: self._on_slider_moved(lbl, v / 100.0))

        right.addStretch()
        layout.addWidget(right_w)

        self._load_patient_list()

    # ── liste patients ────────────────────────────────────────────────────────
    def _load_patient_list(self):
        selected_patient = self._current_patient
        self.patient_list.clear()
        if not ACQ_ROOT.exists():
            self.patient_list.addItem("(aucune acquisition)")
            self.files_list.clear()
            self.files_list.addItem(f"{ACQ_ROOT} introuvable")
            return
        patients = sorted(p.name for p in ACQ_ROOT.iterdir()
                          if p.is_dir() and not p.name.startswith("."))
        self.patient_list.addItems(patients or ["(aucune acquisition)"])
        if selected_patient in patients:
            matches = self.patient_list.findItems(
                selected_patient, Qt.MatchFlag.MatchExactly)
            if matches:
                self.patient_list.setCurrentItem(matches[0])

    # ── sélection patient ─────────────────────────────────────────────────────
    def _on_patient_selected(self, item):
        self._select_patient(item.text())

    def _load_patient_from_text(self):
        pid = self.patient_id_edit.text().strip()
        if pid:
            self._select_patient(pid)

    def _select_patient(self, pid: str):
        if pid.startswith("("):
            return
        pat_dir = ACQ_ROOT / pid
        self._current_patient = pid
        self.patient_id_edit.setText(pid)
        matches = self.patient_list.findItems(pid, Qt.MatchFlag.MatchExactly)
        if matches:
            self.patient_list.setCurrentItem(matches[0])
        self._refresh_file_list(pid)
        self._load_mesh_data(pid)
        self._refresh_view()
        self._load_photo(pid)

    def _refresh_file_list(self, patient_id: str):
        self.files_list.clear()
        pat_dir = ACQ_ROOT / patient_id
        if not pat_dir.exists():
            self.files_list.addItem("Dossier patient introuvable")
            return

        entries = sorted(
            pat_dir.iterdir(),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
        if not entries:
            self.files_list.addItem("(dossier vide)")
            return

        for path in entries:
            if path.is_dir():
                self.files_list.addItem(f"📁 {path.name}/")
            else:
                size = path.stat().st_size
                self.files_list.addItem(f"📄 {path.name}  ({_format_size(size)})")

    def _load_mesh_data(self, patient_id: str):
        """Charge mesh.obj, calcule mesures + Y initiaux, initialise les sliders."""
        obj_path = ACQ_ROOT / patient_id / "mesh.obj"
        self._tri_mesh  = None
        self._y_heights = {}
        for s in self._sliders.values():
            s.setEnabled(False)
        for lbl in self._slider_lbls.values():
            lbl.setText("—")

        if not obj_path.exists():
            self._update_table({})
            return

        values, heights, tri_mesh = compute_full(obj_path)
        self._tri_mesh  = tri_mesh
        self._y_heights = heights.copy()

        self._update_table(values)

        # Initialiser les sliders à la hauteur calculée
        for label, _, _ in MEASURES:
            y = heights.get(label, 1.0)
            s = self._sliders[label]
            s.blockSignals(True)
            s.setValue(int(y * 100))
            s.setEnabled(tri_mesh is not None)
            s.blockSignals(False)
            v = values.get(label)
            self._slider_lbls[label].setText(f"{v:.1f}" if v else "—")

    # ── slider déplacé ────────────────────────────────────────────────────────
    def _on_slider_moved(self, label: str, y_m: float):
        if self._tri_mesh is None: return

        self._y_heights[label] = y_m

        # Recalcul circonférence à la nouvelle hauteur (mètres → cm)
        circ = section_circumference(self._tri_mesh, y_m, mode="sum")
        val  = circ * 100 if circ else None

        # Mettre à jour le label du slider
        self._slider_lbls[label].setText(f"{val:.1f}" if val else "—")

        # Mettre à jour la ligne dans le tableau
        for row in range(self.metrics_table.rowCount()):
            if self.metrics_table.item(row, 0) and \
               self.metrics_table.item(row, 0).text() == label:
                if val:
                    item = QTableWidgetItem(f"{val:.1f}")
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    item.setBackground(_measure_color(label, val))
                    self.metrics_table.setItem(row, 1, item)
                break

        # Redessiner l'anneau si on est en mode mesh
        if self._show_mesh:
            self._redraw_ring(label, y_m)

    # ── tableau des mesures ───────────────────────────────────────────────────
    def _update_table(self, values: dict):
        self.metrics_table.setRowCount(0)
        # Hauteur
        h = values.get("Hauteur")
        self.metrics_table.insertRow(0)
        self.metrics_table.setItem(0, 0, QTableWidgetItem("Hauteur"))
        self.metrics_table.setItem(0, 1,
            QTableWidgetItem(f"{h:.1f}" if h else "—"))

        for label, _, _ in MEASURES:
            val = values.get(label)
            row = self.metrics_table.rowCount()
            self.metrics_table.insertRow(row)
            self.metrics_table.setItem(row, 0, QTableWidgetItem(label))
            if val is not None:
                item = QTableWidgetItem(f"{val:.1f}")
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item.setBackground(_measure_color(label, val))
                self.metrics_table.setItem(row, 1, item)
            else:
                self.metrics_table.setItem(row, 1, QTableWidgetItem("—"))

    # ── bascule type de vue ───────────────────────────────────────────────────
    def _switch_view(self, show_mesh: bool):
        self._show_mesh = show_mesh
        self._ply_row_w.setVisible(not show_mesh)
        if self._current_patient:
            self._refresh_view()

    def _switch_ply_view(self, key: str):
        self._ply_view = key
        if self._current_patient and not self._show_mesh:
            self._refresh_view()

    # ── refresh viewer 3D ─────────────────────────────────────────────────────
    def _refresh_view(self):
        if not self._current_patient: return
        pat_dir = ACQ_ROOT / self._current_patient
        self.plotter.clear()
        self._ring_actors.clear()

        if self._show_mesh:
            obj_path = pat_dir / "mesh.obj"
            if obj_path.exists():
                mesh = pv.read(str(obj_path))
                self.plotter.add_mesh(mesh, color="#c8a882", smooth_shading=True,
                                      opacity=0.85)
                # Dessiner les anneaux de mesure
                if self._tri_mesh is not None:
                    for label, _, color in MEASURES:
                        y = self._y_heights.get(label)
                        if y is not None:
                            self._redraw_ring(label, y)
            else:
                self.plotter.add_text(
                    "mesh.obj introuvable\nLancer 01_fit_smpl_sbatch.sh",
                    font_size=11, color="red")
        else:
            fname    = "fused.ply" if self._ply_view == "fused" \
                       else f"{self._ply_view}.ply"
            ply_path = pat_dir / fname
            if ply_path.exists():
                cloud = load_pyvista_pointcloud(ply_path)
                kw = {"scalars": "RGB", "rgb": True} \
                     if "RGB" in cloud.point_data else {"color": "steelblue"}
                self.plotter.add_mesh(cloud, point_size=2,
                                      render_points_as_spheres=False, **kw)
                label = {"front":"Face","left":"Profil G","back":"Dos",
                         "right":"Profil D","fused":"Fusionné"}[self._ply_view]
                self.plotter.add_text(f"Nuage — {label}", font_size=10, color="gray")
            else:
                self.plotter.add_text(f"{fname} introuvable",
                                      font_size=11, color="red")

        self.plotter.reset_camera()

    # ── dessin d'un anneau ────────────────────────────────────────────────────
    def _redraw_ring(self, label: str, y_m: float):
        """Supprime l'ancien anneau et dessine le nouveau à hauteur y_m."""
        # Supprimer l'acteur précédent si existant
        if label in self._ring_actors:
            try:
                self.plotter.remove_actor(self._ring_actors[label])
            except Exception:
                pass

        color = next((c for l, _, c in MEASURES if l == label), "red")
        ring  = get_ring(self._tri_mesh, y_m)
        if ring is None:
            return

        actor = self.plotter.add_mesh(ring, color=color, line_width=3,
                                       render_lines_as_tubes=True)
        self._ring_actors[label] = actor
        self.plotter.render()

    # ── photo 2D ─────────────────────────────────────────────────────────────
    def _load_photo(self, patient_id: str):
        jpg = ACQ_ROOT / patient_id / "front.jpg"
        if jpg.exists():
            pixmap = QPixmap(str(jpg))
            self.image_label.setPixmap(
                pixmap.scaledToWidth(310, Qt.TransformationMode.SmoothTransformation))
        else:
            self.image_label.setText("front.jpg introuvable")


def _measure_color(label: str, val: float) -> QColor:
    ranges = {"Hauteur":(150,210),"Poitrine":(80,130),
              "Taille":(60,110),"Hanches":(80,130),"Cuisse":(40,75)}
    lo, hi = ranges.get(label, (0, 9999))
    if lo <= val <= hi:                           return QColor(*COLOR_GREEN)
    elif (lo * 0.85) <= val <= (hi * 1.15):      return QColor(*COLOR_ORANGE)
    else:                                          return QColor(*COLOR_RED)


def _format_size(size_bytes: int) -> str:
    units = ["o", "Ko", "Mo", "Go"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f} {unit}" if unit == "o" else f"{size:.1f} {unit}"
        size /= 1024


class UnifiedClinicalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clinical Body Toolkit — Acquisition & Visualisation")
        self.resize(1700, 950)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.acquisition_tab = AcquisitionWidget()
        self.viewer_tab = ClinicalViewer()
        self.tabs.addTab(self.acquisition_tab, "Acquisition")
        self.tabs.addTab(self.viewer_tab, "Visualisation")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.setCurrentWidget(self.acquisition_tab)
        self.acquisition_tab.ensure_camera_started()

    def _on_tab_changed(self, index: int):
        if self.tabs.widget(index) is self.viewer_tab:
            self.viewer_tab._load_patient_list()
        elif self.tabs.widget(index) is self.acquisition_tab:
            self.acquisition_tab.ensure_camera_started()

    def closeEvent(self, event):
        self.acquisition_tab.shutdown()
        try:
            self.viewer_tab.plotter.close()
        except Exception:
            pass
        QApplication.instance().quit()
        super().closeEvent(event)


if __name__ == "__main__":
    # L'ID patient se choisit maintenant dans l'interface.
    # On ne transmet pas les arguments CLI à Qt pour éviter qu'un ancien
    # paramètre de lancement dans l'IDE soit interprété comme nécessaire.
    app    = QApplication([sys.argv[0]])
    window = UnifiedClinicalApp()
    window.show()
    code = app.exec()
    try:
        window.acquisition_tab.shutdown()
        window.viewer_tab.plotter.close()
    except Exception:
        pass
    sys.exit(code)
