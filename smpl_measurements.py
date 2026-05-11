"""
Extraction de mesures anthropométriques depuis un mesh 3D.

Méthode : sections transversales (trimesh.section) à des fractions de hauteur
calibrées sur des meshes SMPL en pose standard.

Observation clé : les meshes SMPL exportés depuis les modèles HMR sont souvent
scindés en demi-corps gauche/droite dans la topologie OBJ. La section transversale
retourne donc typiquement 2 entités (gauche + droite) dont la SOMME donne la
circonférence totale.

Pour le tour de cuisse : on prend la moyenne des 2 plus grandes entités isolées
(chaque jambe séparée).

Unités d'entrée : supposées en mètres (SMPL standard) → sorties en cm.
"""

import json
import numpy as np
import trimesh
from pathlib import Path

# Fractions de hauteur — utilisées pour les meshes posés uniquement
HEIGHT_FRACS = {
    "chest_girth":  0.79,
    "waist_girth":  0.66,
    "hip_girth":    0.58,
    "thigh_girth":  0.46,
}

# Segmentation SMPL : vertex IDs par partie du corps
SMPL_PART_SEG_PATH = str(
    Path(__file__).parent.parent.parent
    / "modeles" / "ShapeBoost" / "extra_files" / "smpl_related" / "part_seg.json"
)

# Mapping mesure → parties SMPL à utiliser (chest géré séparément via armpit detection)
SMPL_MEASURE_PARTS = {
    "waist_girth":  ["spine"],          # spine ≈ 65% de hauteur
    "hip_girth":    ["hips"],           # hips ≈ 54% de hauteur (point le plus large du bassin)
    "thigh_girth":  ["leftUpLeg", "rightUpLeg"],
}
# Parties bras SMPL (pour déterminer le niveau aisselle)
SMPL_ARM_PARTS = ["leftArm", "rightArm"]

_smpl_part_seg_cache = None


def _load_smpl_part_seg() -> dict:
    global _smpl_part_seg_cache
    if _smpl_part_seg_cache is None:
        with open(SMPL_PART_SEG_PATH) as f:
            raw = json.load(f)
        _smpl_part_seg_cache = {k: np.array(v) for k, v in raw.items()}
    return _smpl_part_seg_cache


def chest_circumference_below_arms_smpl(verts: np.ndarray, faces: np.ndarray) -> float | None:
    """
    Mesure poitrine SMPL : coupe transversale sur le mesh complet juste en-dessous
    de l'attachement des bras (évite la contamination bras-torse en T-pose).
    """
    seg = _load_smpl_part_seg()
    arm_ids = set()
    for p in SMPL_ARM_PARTS:
        arm_ids.update(seg.get(p, []).tolist())
    if not arm_ids:
        return None
    arm_verts = verts[np.array(sorted(arm_ids))]
    y_arm_start = float(arm_verts[:, 1].min())
    y_chest = y_arm_start - 0.015  # légèrement en-dessous de l'attache du bras
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    circ = section_circumference(mesh, y_chest, mode="sum")
    return circ


def chest_circumference_below_arms_smplx(verts: np.ndarray, faces: np.ndarray) -> float | None:
    """
    Mesure poitrine SMPL-X : coupe transversale juste en-dessous de l'attache des bras.
    """
    from pathlib import Path
    if not Path(SMPLX_PARTS_SEG_PATH).exists():
        return None
    segm = _load_smplx_face_seg()
    arm_mask = np.isin(segm, [16, 17])  # upper arm parts
    arm_f = faces[arm_mask]
    if len(arm_f) == 0:
        return None
    arm_v = verts[np.unique(arm_f)]
    y_arm_start = float(arm_v[:, 1].min())
    y_chest = y_arm_start - 0.015
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    circ = section_circumference(mesh, y_chest, mode="sum")
    return circ


def measure_from_part_seg(verts: np.ndarray, faces: np.ndarray,
                           measure_name: str) -> float | None:
    """
    Mesure la circonférence en filtrant le mesh aux vertices SMPL
    du/des body part(s) concerné(s), puis coupe transversale au Y centroïde.
    Évite la contamination par les jambes (hip) en T-pose.
    Pour chest_girth, utiliser chest_circumference_below_arms_smpl().
    """
    seg = _load_smpl_part_seg()
    parts = SMPL_MEASURE_PARTS.get(measure_name, [])

    part_ids = set()
    for p in parts:
        part_ids.update(seg.get(p, []))
    if not part_ids:
        return None

    part_ids = np.array(sorted(part_ids))
    part_verts = verts[part_ids]

    # Hauteur de mesure : Y centroïde des vertices de la partie
    y_center = float((part_verts[:, 1].max() + part_verts[:, 1].min()) / 2)

    # Sous-mesh : faces dont les 3 vertices appartiennent à la partie
    id_set = set(part_ids.tolist())
    face_mask = np.array([id_set.issuperset(f) for f in faces])
    sub_faces = faces[face_mask]
    if len(sub_faces) == 0:
        return None

    # Ré-indexation
    used_ids = np.unique(sub_faces)
    id_map = {old: new for new, old in enumerate(used_ids)}
    new_faces = np.array([[id_map[v] for v in f] for f in sub_faces])
    sub_mesh = trimesh.Trimesh(
        vertices=verts[used_ids], faces=new_faces, process=False
    )

    if measure_name == "thigh_girth":
        # Mesurer chaque cuisse séparément, retourner la moyenne
        left_ids  = set(seg.get("leftUpLeg",  []).tolist())
        right_ids = set(seg.get("rightUpLeg", []).tolist())
        results = []
        for side_ids in (left_ids, right_ids):
            s_face_mask = np.array([side_ids.issuperset(f) for f in faces])
            s_sub = faces[s_face_mask]
            if len(s_sub) == 0:
                continue
            s_used = np.unique(s_sub)
            s_map = {o: n for n, o in enumerate(s_used)}
            s_new = np.array([[s_map[v] for v in f] for f in s_sub])
            s_mesh = trimesh.Trimesh(vertices=verts[s_used], faces=s_new, process=False)
            s_verts = verts[s_used]
            y_c = float((s_verts[:, 1].max() + s_verts[:, 1].min()) / 2)
            c = section_circumference(s_mesh, y_c, mode="sum")
            if c is not None:
                results.append(c)
        return float(np.mean(results)) if results else None

    return section_circumference(sub_mesh, y_center, mode="sum")


def load_mesh(obj_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        # Scene avec plusieurs meshes → prendre le plus grand
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = max(meshes, key=lambda m: len(m.vertices))
    return mesh


def normalize_orientation(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Recentre le mesh : pied à y=0, centré en x/z.
    Gère les deux conventions d'axe vertical : Y-up et Z-up.
    """
    v = mesh.vertices.copy()
    # Détecter l'axe vertical : celui avec le plus grand range
    ranges = v.max(axis=0) - v.min(axis=0)
    up_axis = int(np.argmax(ranges))

    if up_axis != 1:
        # Permuter pour que Y soit l'axe vertical
        perm = [0, 1, 2]
        perm[1], perm[up_axis] = perm[up_axis], perm[1]
        v = v[:, perm]

    # Pieds à y=0
    v[:, 1] -= v[:, 1].min()
    # Centrer x et z
    v[:, 0] -= (v[:, 0].max() + v[:, 0].min()) / 2
    v[:, 2] -= (v[:, 2].max() + v[:, 2].min()) / 2

    new_mesh = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
    return new_mesh


def _entity_length(section, ent) -> float:
    """Longueur d'une entité de section (boucle fermée ou chaîne)."""
    pts = section.vertices[ent.points]
    diffs = np.diff(pts, axis=0)
    if ent.closed and len(pts) > 1:
        diffs = np.vstack([diffs, pts[-1] - pts[0]])
    return float(np.linalg.norm(diffs, axis=1).sum())


def section_circumference(mesh: trimesh.Trimesh, y: float,
                          mode: str = "sum") -> float | None:
    """
    Calcule la circonférence (en unités du mesh) à la hauteur y.

    mode='sum'   : somme toutes les entités (meshes posés, bras le long du corps).
    mode='torso' : ne garde que les entités proches de X=0 (T-pose, filtre les bras).
    mode='thigh' : moyenne des 2 plus grandes entités (cuisses séparées).
    mode='max'   : prend uniquement la plus grande entité.
    """
    try:
        section = mesh.section(
            plane_origin=[0.0, y, 0.0],
            plane_normal=[0.0, 1.0, 0.0]
        )
    except Exception:
        return None

    if section is None or len(section.entities) == 0:
        return None

    if mode == "torso":
        # Filtre les bras en T-pose : ne garde que les entités dont le centroïde
        # est proche de X=0. Les bras sont loin du centre en T-pose.
        ent_data = []
        for ent in section.entities:
            pts = section.vertices[ent.points]
            cx = float(np.mean(pts[:, 0]))
            ent_data.append((abs(cx), _entity_length(section, ent)))
        if not ent_data:
            return None
        # Seuil adaptatif : on garde les entités dont |cx| < 2× le min observé + 0.05m
        ent_data.sort(key=lambda x: x[0])
        cx_min = ent_data[0][0]
        threshold = cx_min * 3 + 0.05  # tolérance pour les deux moitiés du torse
        torso_lengths = [l for cx, l in ent_data if cx <= threshold]
        return float(sum(torso_lengths)) if torso_lengths else None

    lengths = sorted(
        [_entity_length(section, ent) for ent in section.entities],
        reverse=True
    )

    if mode == "sum":
        return float(sum(lengths))
    elif mode == "thigh":
        if len(lengths) >= 2:
            return float((lengths[0] + lengths[1]) / 2)
        return float(lengths[0])
    else:  # max
        return float(lengths[0])


def extract_measurements(obj_path: str, scale_factor: float = 1.0) -> dict:
    """
    Extrait les mesures anthropométriques depuis un fichier OBJ.

    Args:
        obj_path: chemin vers le fichier .obj
        scale_factor: facteur d'échelle si le mesh n'est pas en mètres
                      (ex: 100.0 si le mesh est en cm)

    Returns:
        dict avec les clés : height, chest_girth, waist_girth,
                             hip_girth, thigh_girth (en cm)
        Les valeurs None indiquent un échec de mesure.
    """
    try:
        mesh = load_mesh(obj_path)
    except Exception as e:
        return {"error": str(e)}

    # Appliquer facteur d'échelle → convertir en mètres
    if scale_factor != 1.0:
        mesh.vertices *= scale_factor

    mesh = normalize_orientation(mesh)
    v = mesh.vertices

    height_m = float(v[:, 1].max() - v[:, 1].min())
    results = {"height": round(height_m * 100, 2)}  # cm

    for measure_name, frac in HEIGHT_FRACS.items():
        y = height_m * frac
        mode = "thigh" if measure_name == "thigh_girth" else "sum"
        circ_m = section_circumference(mesh, y, mode=mode)
        if circ_m is not None:
            results[measure_name] = round(circ_m * 100, 2)
        else:
            results[measure_name] = None

    return results


SMPLX_PARTS_SEG_PATH = "/projects/m26068/modeles/smplify-x/smplx_parts_segm.pkl"

# Mapping mesure → face part IDs SMPL-X (smplx_parts_segm.pkl)
# Parts identifiés par analyse de la géométrie du template SMPL-X neutral (hauteur normalisée ~1.72m) :
#   0: pelvis/fesses (Y=[0.78, 1.08])
#   1: cuisse droite (Y=[0.47, 0.88])
#   2: cuisse gauche (Y=[0.48, 0.88])
#   3: bas abdomen (Y=[1.03, 1.17])
#   6: taille (Y=[1.13, 1.27])
#   9: torse inférieur/thorax (Y=[1.19, 1.47])
#  13: épaule droite (Y=[1.30, 1.45]) → bras en T, à exclure pour chest
#  14: épaule gauche (Y=[1.29, 1.45]) → bras en T, à exclure pour chest
SMPLX_MEASURE_FACE_PARTS = {
    # chest_girth géré séparément via chest_circumference_below_arms_smplx()
    "waist_girth":  [3, 6],   # bas-abdomen + taille (1 entité connectée, Y≈65%)
    "hip_girth":    [0],      # pelvis/fesses (entité connectée, Y≈54%)
    "thigh_girth":  [1, 2],   # cuisses (1=droite, 2=gauche)
}

_smplx_face_seg_cache = None


def _load_smplx_face_seg() -> np.ndarray:
    """Retourne tableau (n_faces,) de labels de parts SMPL-X."""
    global _smplx_face_seg_cache
    if _smplx_face_seg_cache is None:
        import pickle
        with open(SMPLX_PARTS_SEG_PATH, "rb") as f:
            seg = pickle.load(f, encoding="latin1")
        _smplx_face_seg_cache = np.array(seg["segm"])
    return _smplx_face_seg_cache


def _sub_mesh_from_face_parts(verts: np.ndarray, faces: np.ndarray,
                               segm: np.ndarray, part_ids: list):
    """Extrait un sous-mesh limité aux faces dont le label est dans part_ids."""
    mask = np.isin(segm, part_ids)
    sub_faces = faces[mask]
    if len(sub_faces) == 0:
        return None, None
    used = np.unique(sub_faces)
    id_map = np.full(verts.shape[0], -1, dtype=np.int64)
    id_map[used] = np.arange(len(used))
    new_faces = id_map[sub_faces]
    sub_mesh = trimesh.Trimesh(vertices=verts[used], faces=new_faces, process=False)
    return sub_mesh, verts[used]


def measure_from_face_seg_smplx(verts: np.ndarray, faces: np.ndarray,
                                 measure_name: str) -> float | None:
    """
    Mesure circumférence SMPL-X par segmentation de faces (smplx_parts_segm.pkl).
    """
    from pathlib import Path
    if not Path(SMPLX_PARTS_SEG_PATH).exists():
        return None

    segm = _load_smplx_face_seg()
    part_ids = SMPLX_MEASURE_FACE_PARTS.get(measure_name, [])
    if not part_ids:
        return None

    if measure_name == "thigh_girth":
        # Parts 1 (droite) et 2 (gauche) séparément
        results = []
        for pid in [1, 2]:
            sub_mesh, sub_verts = _sub_mesh_from_face_parts(verts, faces, segm, [pid])
            if sub_mesh is None:
                continue
            y_c = float((sub_verts[:, 1].max() + sub_verts[:, 1].min()) / 2)
            c = section_circumference(sub_mesh, y_c, mode="sum")
            if c is not None:
                results.append(c)
        return float(np.mean(results)) if results else None

    sub_mesh, sub_verts = _sub_mesh_from_face_parts(verts, faces, segm, part_ids)
    if sub_mesh is None:
        return None

    y_center = float((sub_verts[:, 1].max() + sub_verts[:, 1].min()) / 2)
    return section_circumference(sub_mesh, y_center, mode="sum")


SMPL_PKL_PATHS = {
    "neutral": "/projects/m26068/modeles/ShapeBoost/model_files/smpl_v1.1.0/smpl/SMPL_NEUTRAL.pkl",
    "male":    "/projects/m26068/modeles/ShapeBoost/model_files/smpl_v1.1.0/smpl/SMPL_MALE.pkl",
    "female":  "/projects/m26068/modeles/ShapeBoost/model_files/smpl_v1.1.0/smpl/SMPL_FEMALE.pkl",
}
SMPLX_NPZ_PATHS = {
    "neutral": "/projects/m26068/modeles/smplify-x/data/models/smplx/SMPLX_NEUTRAL.npz",
    "male":    "/projects/m26068/modeles/smplify-x/data/models/smplx/SMPLX_MALE.npz",
    "female":  "/projects/m26068/modeles/smplify-x/data/models/smplx/SMPLX_FEMALE.npz",
}


def _tpose_from_betas(betas: np.ndarray, model_type: str, gender: str):
    """Génère vertices+faces T-pose depuis les betas. Sans dépendance smplx."""
    import pickle
    b = np.array(betas).flatten()[:10]
    if model_type == "smplx":
        path = SMPLX_NPZ_PATHS.get(gender.lower(), SMPLX_NPZ_PATHS["neutral"])
        d = np.load(path, allow_pickle=True)
        v_template = np.array(d["v_template"])
        shapedirs  = np.array(d["shapedirs"])[:, :, :10]
        faces      = np.array(d["f"])
    else:
        path = SMPL_PKL_PATHS.get(gender.lower(), SMPL_PKL_PATHS["neutral"])
        # Patch numpy pour chumpy (np.int/np.float/np.unicode supprimés ≥1.24)
        import builtins
        for _a, _v in (("bool",bool),("int",int),("float",float),
                       ("complex",complex),("object",object),
                       ("str",str),("unicode",str)):
            if not hasattr(np, _a):
                setattr(np, _a, _v)
        with open(path, "rb") as f:
            smpl = pickle.load(f, encoding="latin1")
        v_template = np.array(smpl["v_template"])
        shapedirs  = np.array(smpl["shapedirs"])[:, :, :10]
        faces      = np.array(smpl["f"])
    verts = v_template + np.einsum("ijk,k->ij", shapedirs, b)
    return verts, faces


def extract_measurements_from_betas(
    betas: np.ndarray,
    model_type: str = "smpl",
    gender: str = "neutral",
    smpl_model_path: str = "/projects/m26068/modeles/smplify-x/data/models",
) -> dict:
    """
    Reconstruit un mesh T-pose depuis les betas et extrait les mesures.
    Utilise numpy directement (pas de dépendance smplx).
    """
    try:
        verts, faces = _tpose_from_betas(betas, model_type, gender)
    except Exception as e:
        return {"error": f"T-pose generation failed: {e}"}

    # SMPL/SMPL-X est toujours Y-up. En T-pose, le span des bras (axe X) ≈ la hauteur (axe Y),
    # donc normalize_orientation() peut choisir X comme axe vertical → bug.
    # On fait une normalisation simple : pieds à y=0, centrage x/z, sans permuter les axes.
    v = np.array(verts, dtype=float)
    v[:, 1] -= v[:, 1].min()          # pieds à y=0
    v[:, 0] -= (v[:, 0].max() + v[:, 0].min()) / 2
    v[:, 2] -= (v[:, 2].max() + v[:, 2].min()) / 2
    mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
    verts_norm = mesh.vertices
    faces_arr  = np.array(mesh.faces)

    height_m = float(verts_norm[:, 1].max() - verts_norm[:, 1].min())
    results = {"height": round(height_m * 100, 2), "source": "betas_tpose"}

    # Pour SMPL (6890 vertices) : segmentation par vertex (part_seg.json)
    use_smpl_seg = (model_type == "smpl" and len(verts_norm) == 6890
                    and Path(SMPL_PART_SEG_PATH).exists())
    # Pour SMPL-X (10475 vertices) : segmentation par face (smplx_parts_segm.pkl)
    use_smplx_seg = (model_type == "smplx" and len(verts_norm) == 10475
                     and Path(SMPLX_PARTS_SEG_PATH).exists())

    for measure_name in HEIGHT_FRACS:
        if measure_name == "chest_girth":
            # Poitrine : coupe juste sous l'attache des bras (T-pose arms horizontal)
            if use_smpl_seg:
                circ_m = chest_circumference_below_arms_smpl(verts_norm, faces_arr)
            elif use_smplx_seg:
                circ_m = chest_circumference_below_arms_smplx(verts_norm, faces_arr)
            else:
                y = height_m * HEIGHT_FRACS[measure_name]
                circ_m = section_circumference(mesh, y, mode="torso")
        elif use_smpl_seg:
            circ_m = measure_from_part_seg(verts_norm, faces_arr, measure_name)
        elif use_smplx_seg:
            circ_m = measure_from_face_seg_smplx(verts_norm, faces_arr, measure_name)
        else:
            # Fallback : coupe transversale classique avec filtre torso
            y = height_m * HEIGHT_FRACS[measure_name]
            mode = "thigh" if measure_name == "thigh_girth" else "torso"
            circ_m = section_circumference(mesh, y, mode=mode)
        results[measure_name] = round(circ_m * 100, 2) if circ_m is not None else None

    return results


def apply_height_normalization(
    measurements: dict, height_gt: float
) -> dict:
    """
    Corrige l'ambiguïté d'échelle monoculaire.
    Toutes les mesures (sauf height) sont renormalisées par :
        measure_corrected = (measure_pred / height_pred) * height_gt

    Args:
        measurements: dict avec 'height' en cm et les autres mesures en cm
        height_gt: taille réelle du sujet en cm
    Returns:
        dict corrigé
    """
    corrected = measurements.copy()
    height_pred = measurements.get("height")
    if height_pred is None or height_pred <= 0:
        return corrected

    scale = height_gt / height_pred
    for key, val in measurements.items():
        if key == "height" or val is None or key == "source" or key == "error":
            continue
        corrected[key] = round(val * scale, 2)

    return corrected


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python smpl_measurements.py mesh.obj [scale_factor]")
        sys.exit(1)

    obj = sys.argv[1]
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    m = extract_measurements(obj, scale)
    for k, v in m.items():
        print(f"  {k}: {v}")
