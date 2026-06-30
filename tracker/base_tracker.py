# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np

from tracker.matching import *
from tracker.trajectory import Trajectory
from utils.utils import norm_realative_radian


# ============================================================
# nuScenes conservative modular extensions
# Added on top of the original MCTrack Base3DTracker.
# These modules are output / initialization stage only:
#   1) TRACK_AWARE_LOW_SCORE_GATE: low-score unmatched detections do not create new tracks.
#   2) OUTPUT_FILTER: suppress low-quality short / low-score / long-lost outputs.
#   3) NUSCENES_BEV_OUTPUT_FILTER: range filter for predicted/fake boxes.
#   4) OUTPUT_TRAJ_NMS: optional simple duplicate-output suppression.
#   5) DISTANCE_AWARE_SCORE: independent birth / tentative-output policy.
# ============================================================

def _cfg_by_cls(value, cls_id, default=None):
    """Read scalar or dict class-wise config safely."""
    if isinstance(value, dict):
        if cls_id in value:
            return value[cls_id]
        if str(cls_id) in value:
            return value[str(cls_id)]
        return default
    if value is None:
        return default
    return value


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _get_bbox_score(bbox, default=1.0):
    for name in ("det_score", "score", "tracking_score"):
        if hasattr(bbox, name):
            return _safe_float(getattr(bbox, name), default)
    return float(default)


def _get_bbox_category_raw(bbox):
    for name in ("category", "category_name", "tracking_name", "class_name", "cls_name"):
        if hasattr(bbox, name):
            return getattr(bbox, name)
    for name in ("category_id", "class_id", "cls_id", "category_num"):
        if hasattr(bbox, name):
            return getattr(bbox, name)
    return 0


def _get_bbox_cls_id(bbox, cfg):
    raw = _get_bbox_category_raw(bbox)
    cmap = cfg.get("CATEGORY_MAP_TO_NUMBER", {}) if isinstance(cfg, dict) else {}
    if raw in cmap:
        return int(cmap[raw])
    try:
        return int(raw)
    except Exception:
        return 0


def _get_traj_cls_id(traj, cfg):
    for name in ("category", "category_id", "class_id", "cls_id", "category_num"):
        if hasattr(traj, name):
            raw = getattr(traj, name)
            cmap = cfg.get("CATEGORY_MAP_TO_NUMBER", {}) if isinstance(cfg, dict) else {}
            if raw in cmap:
                return int(cmap[raw])
            try:
                return int(raw)
            except Exception:
                pass
    if hasattr(traj, "bboxes") and len(traj.bboxes) > 0:
        return _get_bbox_cls_id(traj.bboxes[-1], cfg)
    return 0


def _get_bbox_xyz(bbox):
    if hasattr(bbox, "global_xyz"):
        xyz = getattr(bbox, "global_xyz")
        return np.array(xyz[:3], dtype=float)
    for name in ("xyz", "translation", "center", "location"):
        if hasattr(bbox, name):
            xyz = getattr(bbox, name)
            return np.array(xyz[:3], dtype=float)
    if hasattr(bbox, "global_xyz_lwh_yaw"):
        arr = getattr(bbox, "global_xyz_lwh_yaw")
        return np.array(arr[:3], dtype=float)
    return np.zeros(3, dtype=float)


def _as_homogeneous_matrix(value):
    """Convert a list / ndarray into a valid 4x4 homogeneous matrix."""
    if value is None:
        return None

    try:
        matrix = np.asarray(value, dtype=float)
    except Exception:
        return None

    if matrix.shape == (4, 4):
        return matrix

    if matrix.shape == (3, 4):
        return np.vstack([matrix, np.array([0.0, 0.0, 0.0, 1.0])])

    if matrix.size == 16:
        return matrix.reshape(4, 4)

    if matrix.size == 12:
        matrix = matrix.reshape(3, 4)
        return np.vstack([matrix, np.array([0.0, 0.0, 0.0, 1.0])])

    return None


def _resolve_global_to_ego_matrix(transform_matrix):
    """
    Resolve the current-frame global -> LiDAR / ego transform.

    BaseVersion data normally stores this directly as a 4x4 matrix.  The
    additional dict branches keep the tracker compatible with common wrapper
    formats used by converted nuScenes data.
    """
    if transform_matrix is None:
        return None

    if isinstance(transform_matrix, dict):
        lower_key_map = {str(key).lower(): key for key in transform_matrix.keys()}

        direct_keys = (
            "global2lidar",
            "global_to_lidar",
            "global2ego",
            "global_to_ego",
            "global2sensor",
            "global_to_sensor",
            "world2lidar",
            "world_to_lidar",
            "world2ego",
            "world_to_ego",
        )
        for key in direct_keys:
            actual_key = lower_key_map.get(key)
            if actual_key is None:
                continue
            matrix = _as_homogeneous_matrix(transform_matrix[actual_key])
            if matrix is not None:
                return matrix

        inverse_keys = (
            "lidar2global",
            "lidar_to_global",
            "ego2global",
            "ego_to_global",
            "sensor2global",
            "sensor_to_global",
            "lidar2world",
            "lidar_to_world",
            "ego2world",
            "ego_to_world",
        )
        for key in inverse_keys:
            actual_key = lower_key_map.get(key)
            if actual_key is None:
                continue
            matrix = _as_homogeneous_matrix(transform_matrix[actual_key])
            if matrix is None:
                continue
            try:
                return np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                continue

        # A nested "transform_matrix" / "global_to_lidar" wrapper is also
        # accepted when the outer metadata dictionary contains other fields.
        for key in ("transform_matrix", "matrix", "global_to_lidar", "global2lidar"):
            actual_key = lower_key_map.get(key)
            if actual_key is None:
                continue
            matrix = _as_homogeneous_matrix(transform_matrix[actual_key])
            if matrix is not None:
                return matrix

        return None

    # In the original BaseVersion pipeline, the direct matrix is global -> lidar.
    return _as_homogeneous_matrix(transform_matrix)


def _distance_source(cfg):
    distance_cfg = cfg.get("DISTANCE_AWARE_SCORE", {}) if isinstance(cfg, dict) else {}
    return str(distance_cfg.get("DISTANCE_SOURCE", "EGO_BEV")).upper()


def _get_bbox_dist(bbox, transform_matrix=None, cfg=None):
    """
    Return BEV distance to the *current ego/LiDAR frame*.

    `bbox.global_xyz` is in nuScenes global coordinates, so taking
    `norm(global_xy)` is not a valid distance-to-ego measurement.  For the
    distance-aware policy we first transform the global center using the
    current frame's global->ego matrix.

    If a transform is unavailable, return None rather than silently applying
    a distance threshold in global-map coordinates.
    """
    xyz = _get_bbox_xyz(bbox)
    source = _distance_source(cfg)

    if source in ("GLOBAL_XY", "MAP_XY", "LEGACY_GLOBAL_XY"):
        return float(np.linalg.norm(xyz[:2]))

    matrix = _resolve_global_to_ego_matrix(transform_matrix)
    if matrix is None:
        return None

    xyz_h = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=float)
    ego_xyz_h = matrix @ xyz_h
    if not np.all(np.isfinite(ego_xyz_h[:2])):
        return None
    return float(np.linalg.norm(ego_xyz_h[:2]))


def _get_bbox_lw(bbox):
    """Return length/width for approximate BEV overlap. Falls back to 1x1."""
    for name in ("global_lwh", "lwh", "size", "dims", "dimension"):
        if hasattr(bbox, name):
            arr = getattr(bbox, name)
            try:
                if len(arr) >= 2:
                    return max(_safe_float(arr[0], 1.0), 1e-3), max(_safe_float(arr[1], 1.0), 1e-3)
            except Exception:
                pass
    if hasattr(bbox, "global_xyz_lwh_yaw"):
        arr = getattr(bbox, "global_xyz_lwh_yaw")
        try:
            return max(_safe_float(arr[3], 1.0), 1e-3), max(_safe_float(arr[4], 1.0), 1e-3)
        except Exception:
            pass
    return 1.0, 1.0


def _approx_bev_iou(b1, b2):
    """Axis-aligned approximate BEV IoU. Used only for optional output NMS."""
    c1 = _get_bbox_xyz(b1)
    c2 = _get_bbox_xyz(b2)
    l1, w1 = _get_bbox_lw(b1)
    l2, w2 = _get_bbox_lw(b2)
    x11, y11, x12, y12 = c1[0] - l1 / 2.0, c1[1] - w1 / 2.0, c1[0] + l1 / 2.0, c1[1] + w1 / 2.0
    x21, y21, x22, y22 = c2[0] - l2 / 2.0, c2[1] - w2 / 2.0, c2[0] + l2 / 2.0, c2[1] + w2 / 2.0
    ix1, iy1 = max(x11, x21), max(y11, y21)
    ix2, iy2 = min(x12, x22), min(y12, y22)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area1 = max(l1 * w1, 1e-6)
    area2 = max(l2 * w2, 1e-6)
    return float(inter / max(area1 + area2 - inter, 1e-6))


def _track_length(traj):
    if hasattr(traj, "bboxes"):
        try:
            return len(traj.bboxes)
        except Exception:
            pass
    for name in ("track_length", "age", "life_time"):
        if hasattr(traj, name):
            try:
                return int(getattr(traj, name))
            except Exception:
                pass
    return 1


def _unmatched_length(traj):
    for name in ("unmatch_length", "unmatched_length", "unmatched_num", "time_since_update", "lost_length"):
        if hasattr(traj, name):
            try:
                return int(getattr(traj, name))
            except Exception:
                pass
    return 0


def _traj_avg_score(traj, default=1.0):
    if not hasattr(traj, "bboxes") or len(traj.bboxes) == 0:
        return float(default)
    scores = [_get_bbox_score(b, default=default) for b in traj.bboxes]
    return float(np.mean(scores)) if len(scores) > 0 else float(default)


def _is_fake_or_predict_bbox(bbox, traj=None, cfg=None):
    if getattr(bbox, "is_fake", False):
        return True
    if traj is not None and hasattr(traj, "_is_filter_predict_box"):
        try:
            return _safe_float(getattr(bbox, "det_score"), 9999.0) == _safe_float(traj._is_filter_predict_box, -9999.0)
        except Exception:
            pass
    if traj is not None and cfg is not None:
        try:
            cls_id = _get_traj_cls_id(traj, cfg)
            fake_score = _cfg_by_cls(
                cfg.get("THRESHOLD", {}).get("TRAJECTORY_THRE", {}).get("IS_FILTER_PREDICT_BOX", {}),
                cls_id,
                None,
            )
            if fake_score is not None:
                return _safe_float(getattr(bbox, "det_score", 9999.0), 9999.0) == _safe_float(fake_score, -9999.0)
        except Exception:
            pass
    return False


_DISTANCE_AWARE_DEBUG_PRINT_COUNT = 0
_DISTANCE_AWARE_TRANSFORM_WARNING_EMITTED = False


def _distance_aware_cfg(cfg):
    return cfg.get("DISTANCE_AWARE_SCORE", {}) if isinstance(cfg, dict) else {}


def _distance_aware_class_enabled(cfg, cls_id):
    score_cfg = _distance_aware_cfg(cfg)
    if not bool(score_cfg.get("ENABLE", False)):
        return False
    return bool(_cfg_by_cls(score_cfg.get("APPLY_CLASSES", {}), cls_id, False))


def _distance_aware_bin_index(dist, bins):
    """
    Bins are configured as [near_start, mid_start, far_start, max_range].
    Values above max_range intentionally use the far score rather than being
    dropped here; range handling remains the responsibility of OUTPUT_FILTER.
    """
    try:
        bins = [float(x) for x in bins]
    except Exception:
        return None, None

    if len(bins) < 4 or not (bins[0] <= bins[1] <= bins[2] <= bins[3]):
        return None, None

    if dist < bins[1]:
        return 0, "NEAR"
    if dist < bins[2]:
        return 1, "MID"
    return 2, "FAR"


def _distance_aware_threshold(bbox, cfg, score_field, transform_matrix=None):
    """
    Return (threshold, distance, bucket) for a configured class.
    Returns (None, None, None) when the policy is disabled / unavailable.
    """
    cls_id = _get_bbox_cls_id(bbox, cfg)
    if not _distance_aware_class_enabled(cfg, cls_id):
        return None, None, None

    score_cfg = _distance_aware_cfg(cfg)
    dist = _get_bbox_dist(bbox, transform_matrix=transform_matrix, cfg=cfg)
    if dist is None:
        global _DISTANCE_AWARE_TRANSFORM_WARNING_EMITTED
        if bool(score_cfg.get("WARN_ON_MISSING_EGO_TRANSFORM", True)) and not _DISTANCE_AWARE_TRANSFORM_WARNING_EMITTED:
            print(
                "[DISTANCE_AWARE_SCORE] 未解析到当前帧 global->ego/lidar 变换；"
                "本次运行将跳过距离阈值，而不会错误地使用 global 坐标距离。"
            )
            _DISTANCE_AWARE_TRANSFORM_WARNING_EMITTED = True
        return None, None, None

    bins = _cfg_by_cls(score_cfg.get("DISTANCE_BINS", {}), cls_id, None)
    score_list = _cfg_by_cls(score_cfg.get(score_field, {}), cls_id, None)
    if bins is None or score_list is None:
        return None, None, None

    bin_idx, bucket = _distance_aware_bin_index(dist, bins)
    if bin_idx is None:
        return None, None, None

    try:
        threshold = float(score_list[bin_idx])
    except (TypeError, IndexError, ValueError):
        return None, None, None

    return threshold, float(dist), bucket


def _distance_aware_debug(cfg, stage, cls_id, score, threshold, dist, bucket, accepted):
    """Optional, capped debug output; disabled by default."""
    global _DISTANCE_AWARE_DEBUG_PRINT_COUNT
    score_cfg = _distance_aware_cfg(cfg)
    if not bool(score_cfg.get("DEBUG", False)):
        return

    max_prints = int(score_cfg.get("DEBUG_MAX_PRINTS", 50))
    if _DISTANCE_AWARE_DEBUG_PRINT_COUNT >= max_prints:
        return

    print(
        "[DISTANCE_AWARE_SCORE]",
        "stage=", stage,
        "cls=", cls_id,
        "dist=", round(float(dist), 2) if dist is not None else None,
        "bucket=", bucket,
        "score=", round(float(score), 4),
        "threshold=", round(float(threshold), 4) if threshold is not None else None,
        "pass=", bool(accepted),
    )
    _DISTANCE_AWARE_DEBUG_PRINT_COUNT += 1


def should_create_new_track_nuscenes(det_bbox, cfg):
    """
    Static low-score gate for unmatched detections.

    This function is intentionally independent from DISTANCE_AWARE_SCORE.
    It only implements TRACK_AWARE_LOW_SCORE_GATE.  When that module is
    disabled, it returns True and leaves the original MCTrack birth behavior
    unchanged.
    """
    gate_cfg = cfg.get("TRACK_AWARE_LOW_SCORE_GATE", {})
    if not bool(gate_cfg.get("ENABLE", False)):
        return True

    cls_id = _get_bbox_cls_id(det_bbox, cfg)
    score = _get_bbox_score(det_bbox, default=1.0)
    low_thre = float(_cfg_by_cls(gate_cfg.get("LOW_SCORE_THRE", {}), cls_id, 0.0))
    normal_thre = float(_cfg_by_cls(gate_cfg.get("NORMAL_SCORE_THRE", {}), cls_id, 0.0))
    weak_can_create = bool(gate_cfg.get("WEAK_DET_CREATE_NEW_TRACK", False))

    if score < low_thre:
        return False
    if score < normal_thre and not weak_can_create:
        return False
    return True


def should_create_new_track_distance_aware(det_bbox, cfg, transform_matrix=None):
    """
    Distance-aware gate for unmatched detections.

    This function reads ONLY DISTANCE_AWARE_SCORE.  It does not inspect
    TRACK_AWARE_LOW_SCORE_GATE or OUTPUT_FILTER, so it can be enabled as a
    clean standalone ablation.  If both birth modules are enabled, the caller
    applies them sequentially and both conditions must pass.

    Optional configuration:
      APPLY_NEW_TRACK_INIT: True / False
      NEW_TRACK_HARD_FLOOR: class-wise optional absolute floor.  Omit it to
        use only the distance-bin threshold.
    """
    score_cfg = _distance_aware_cfg(cfg)
    if not bool(score_cfg.get("ENABLE", False)):
        return True
    if not bool(score_cfg.get("APPLY_NEW_TRACK_INIT", True)):
        return True

    cls_id = _get_bbox_cls_id(det_bbox, cfg)
    score = _get_bbox_score(det_bbox, default=1.0)
    distance_thre, dist, bucket = _distance_aware_threshold(
        det_bbox,
        cfg,
        "NEW_TRACK_INIT_SCORE",
        transform_matrix=transform_matrix,
    )
    if distance_thre is None:
        # This class is disabled or its distance / threshold configuration is
        # unavailable.  Do not change the original birth behavior.
        return True

    required_score = float(distance_thre)
    hard_floor = _cfg_by_cls(score_cfg.get("NEW_TRACK_HARD_FLOOR", {}), cls_id, None)
    if hard_floor is not None:
        try:
            required_score = max(required_score, float(hard_floor))
        except (TypeError, ValueError):
            pass

    accepted = score >= required_score
    _distance_aware_debug(
        cfg,
        "NEW_TRACK_INIT",
        cls_id,
        score,
        required_score,
        dist,
        bucket,
        accepted,
    )
    return accepted


def should_output_by_quality_filter(traj, bbox, frame_id, cfg, transform_matrix=None):
    """
    OUTPUT_FILTER-only output-stage filter.

    This function is intentionally independent from DISTANCE_AWARE_SCORE.
    When OUTPUT_FILTER.ENABLE is False, it has no effect.
    """
    out_cfg = cfg.get("OUTPUT_FILTER", {})
    if not bool(out_cfg.get("ENABLE", False)):
        return True

    cls_id = _get_traj_cls_id(traj, cfg)
    score = _get_bbox_score(bbox, default=1.0)
    dist = _get_bbox_dist(bbox, transform_matrix=transform_matrix, cfg=cfg)
    tlen = _track_length(traj)
    lost_len = _unmatched_length(traj)
    avg_score = _traj_avg_score(traj, default=score)

    min_len = int(_cfg_by_cls(out_cfg.get("MIN_OUTPUT_TRACK_LENGTH", {}), cls_id, 1))
    min_score = float(_cfg_by_cls(out_cfg.get("MIN_OUTPUT_SCORE", {}), cls_id, 0.0))
    far_dist = float(_cfg_by_cls(out_cfg.get("FAR_DIST", {}), cls_id, 1e9))
    far_min_score = float(_cfg_by_cls(out_cfg.get("FAR_MIN_OUTPUT_SCORE", {}), cls_id, min_score))
    max_lost = int(_cfg_by_cls(out_cfg.get("MAX_LOST_OUTPUT_LENGTH", {}), cls_id, 9999))
    short_len = int(_cfg_by_cls(out_cfg.get("SHORT_TRAJ_LENGTH", {}), cls_id, 0))
    short_avg = float(_cfg_by_cls(out_cfg.get("SHORT_TRAJ_AVG_SCORE", {}), cls_id, 0.0))

    # Preserve the existing initial-frame behavior.
    if tlen < min_len and frame_id >= 3:
        return False
    if score < min_score:
        return False
    if dist is not None and dist > far_dist and score < far_min_score:
        return False
    if lost_len > max_lost:
        return False
    if short_len > 0 and tlen <= short_len and avg_score < short_avg:
        return False

    return True


def should_output_by_distance_aware_tentative_filter(
    traj,
    bbox,
    cfg,
    transform_matrix=None,
):
    """
    Distance-aware output filter for tentative / short tracks only.

    This function reads ONLY DISTANCE_AWARE_SCORE.  In particular, the
    definition of "short trajectory" comes from
    DISTANCE_AWARE_SCORE.TENTATIVE_MAX_TRACK_LENGTH and never from
    OUTPUT_FILTER.SHORT_TRAJ_LENGTH.

    Required stage switches:
      APPLY_TENTATIVE_OUTPUT: True / False
      TENTATIVE_MAX_TRACK_LENGTH: {class_id: integer}
    """
    score_cfg = _distance_aware_cfg(cfg)
    if not bool(score_cfg.get("ENABLE", False)):
        return True
    if not bool(score_cfg.get("APPLY_TENTATIVE_OUTPUT", False)):
        return True

    cls_id = _get_traj_cls_id(traj, cfg)
    max_track_len = int(
        _cfg_by_cls(
            score_cfg.get("TENTATIVE_MAX_TRACK_LENGTH", {}),
            cls_id,
            0,
        )
    )
    if max_track_len <= 0 or _track_length(traj) > max_track_len:
        return True

    score = _get_bbox_score(bbox, default=1.0)
    distance_thre, dist, bucket = _distance_aware_threshold(
        bbox,
        cfg,
        "TENTATIVE_OUTPUT_SCORE",
        transform_matrix=transform_matrix,
    )
    if distance_thre is None:
        return True

    accepted = score >= float(distance_thre)
    _distance_aware_debug(
        cfg,
        "TENTATIVE_OUTPUT",
        cls_id,
        score,
        float(distance_thre),
        dist,
        bucket,
        accepted,
    )
    return accepted


def should_output_nuscenes_bev_filter(traj, bbox, cfg, transform_matrix=None):
    """
    nuScenes BEV range filter.  The range is measured in the current
    ego/LiDAR frame, not in global map coordinates.
    """
    nusc_cfg = cfg.get("NUSCENES_BEV_OUTPUT_FILTER", {})
    if not bool(nusc_cfg.get("ENABLE", False)):
        return True

    cls_id = _get_traj_cls_id(traj, cfg)
    max_range = float(_cfg_by_cls(nusc_cfg.get("MAX_OUTPUT_RANGE", {}), cls_id, 50.0))
    margin = float(nusc_cfg.get("RANGE_MARGIN", 2.0))
    only_fake = bool(nusc_cfg.get("ONLY_FILTER_FAKE_BBOX", True))
    is_fake = _is_fake_or_predict_bbox(bbox, traj=traj, cfg=cfg)

    if only_fake and not is_fake:
        return True

    dist = _get_bbox_dist(bbox, transform_matrix=transform_matrix, cfg=cfg)
    if dist is None:
        # Never apply a false range filter in global-map coordinates.
        return True
    return dist <= max_range + margin


def _output_priority(traj, bbox):
    return _get_bbox_score(bbox, default=0.0) + 0.005 * min(_track_length(traj), 20) - 0.02 * _unmatched_length(traj)


def apply_output_traj_nms(output_trajs, all_trajs, cfg):
    """Optional simple output NMS to remove duplicate tracks at output stage only."""
    nms_cfg = cfg.get("OUTPUT_TRAJ_NMS", {})
    if not bool(nms_cfg.get("ENABLE", False)) or len(output_trajs) <= 1:
        return output_trajs

    items = []
    for tid, bbox in output_trajs.items():
        traj = all_trajs.get(tid, None)
        if traj is None:
            continue
        items.append((tid, traj, bbox, _output_priority(traj, bbox)))
    items.sort(key=lambda x: x[3], reverse=True)

    keep = []
    suppressed = set()
    for tid, traj, bbox, pri in items:
        if tid in suppressed:
            continue
        keep.append((tid, bbox))
        cls_id = _get_traj_cls_id(traj, cfg)
        iou_thre = float(_cfg_by_cls(nms_cfg.get("IOU_THRE", {}), cls_id, nms_cfg.get("IOU_THRE", 0.5)))
        center_thre = float(_cfg_by_cls(nms_cfg.get("CENTER_DIST_THRE", {}), cls_id, 0.0))
        suppress_only_weak = bool(nms_cfg.get("SUPPRESS_ONLY_WEAK_TRACKS", True))
        short_len = int(_cfg_by_cls(nms_cfg.get("SHORT_TRACK_LENGTH", {}), cls_id, 2))
        low_score_thre = float(_cfg_by_cls(nms_cfg.get("LOW_SCORE_THRE", {}), cls_id, 0.2))

        for tid2, traj2, bbox2, pri2 in items:
            if tid2 == tid or tid2 in suppressed:
                continue
            if _get_traj_cls_id(traj2, cfg) != cls_id:
                continue
            iou = _approx_bev_iou(bbox, bbox2)
            cdist = float(np.linalg.norm(_get_bbox_xyz(bbox)[:2] - _get_bbox_xyz(bbox2)[:2]))
            overlap = iou >= iou_thre or (center_thre > 0 and cdist <= center_thre)
            if not overlap:
                continue

            if suppress_only_weak:
                weak2 = _track_length(traj2) <= short_len or _get_bbox_score(bbox2, 1.0) <= low_score_thre
                if not weak2:
                    continue
            suppressed.add(tid2)

    return {tid: bbox for tid, bbox in keep}


class Base3DTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.current_frame_id = None
        self.all_trajs = {}
        self.all_dead_trajs = {}
        self.id_seed = 0
        self.cache_size = 3
        self.track_id_counter = 0

    def get_trajectory_bbox(self, all_trajs):
        track_ids = sorted(all_trajs.keys())
        trajs = []
        for i in track_ids:
            trajs.append(all_trajs[i])
        return trajs

    def predict_before_associate(self):
        for track_id, traj in self.all_trajs.items():
            traj.predict()

    def track_single_frame(self, frame_info):
        """
        Info: This function tracks objects in a single frame, performing association between predicted trajectories and detected objects.
        Parameters:
            input:
                frame_info: Object containing information about the current frame.
            output:
                output_trajs: Updated trajectories after performing tracking and matching for the current frame.
        """
        self.predict_before_associate()

        trajs = self.get_trajectory_bbox(self.all_trajs)
        trajs_cnt, dets_cnt = len(trajs), len(frame_info.bboxes)
        match_res, cost_matrix = match_trajs_and_dets(
            trajs, frame_info.bboxes, self.cfg
        )
        matched_det_indices = set(match_res[:, 1])

        unmatched_det_indices = np.array(
            [i for i in range(dets_cnt) if i not in matched_det_indices]
        )

        unmatched_trajs = {}
        for i in range(trajs_cnt):
            track_id = trajs[i].track_id
            if i in match_res[:, 0]:
                indexes = np.where(match_res[:, 0] == i)[0]
                self.all_trajs[track_id].update(
                    frame_info.bboxes[match_res[indexes, 1][0]], cost_matrix[indexes][0]
                )
            else:
                unmatched_trajs[track_id] = self.all_trajs[track_id]
                if not self.cfg["IS_RV_MATCHING"]:
                    self.all_trajs[track_id].unmatch_update(frame_info.frame_id)

        init_bboxes = frame_info.bboxes
        if self.cfg["IS_RV_MATCHING"]:
            unmatched_trajs_inbev = self.get_trajectory_bbox(unmatched_trajs)
            trajs_cnt_inbev, dets_cnt_inbev = len(unmatched_trajs_inbev), len(
                unmatched_det_indices
            )
            unmatched_dets_inbev = (
                np.array(frame_info.bboxes)[unmatched_det_indices].tolist()
                if dets_cnt_inbev > 0
                else unmatched_det_indices
            )

            match_res_inbev, cost_matrix_inbev = match_trajs_and_dets(
                unmatched_trajs_inbev,
                unmatched_dets_inbev,
                self.cfg,
                frame_info.transform_matrix,
                is_rv=True,
            )

            for i in range(trajs_cnt_inbev):
                track_id = unmatched_trajs_inbev[i].track_id
                if i in match_res_inbev[:, 0]:
                    indexes = np.where(match_res_inbev[:, 0] == i)[0]
                    trk_bbox = self.all_trajs[track_id].bboxes[-1]
                    det_bbox = unmatched_dets_inbev[
                        match_res_inbev[match_res_inbev[:, 0] == i, 1][0]
                    ]
                    diff_rot = (
                        abs(
                            norm_realative_radian(
                                trk_bbox.global_yaw - det_bbox.global_yaw
                            )
                        )
                        * 180
                        / np.pi
                    )
                    dist = np.linalg.norm(
                        np.array(trk_bbox.global_xyz) - np.array(det_bbox.global_xyz)
                    )
                    if diff_rot > 90 or dist > 5:
                        self.all_trajs[track_id].unmatch_update(frame_info.frame_id)
                        continue
                    self.all_trajs[track_id].update(
                        det_bbox, float(cost_matrix_inbev[indexes])
                    )
                else:
                    self.all_trajs[track_id].unmatch_update(frame_info.frame_id)

            matched_det_indices = set(match_res_inbev[:, 1])
            unmatched_det_indices = np.array(
                [i for i in range(dets_cnt_inbev) if i not in matched_det_indices]
            )
            init_bboxes = unmatched_dets_inbev

        for i in unmatched_det_indices:
            # Apply static and distance-aware birth gates independently.
            # Each module has its own ENABLE switch and configuration.
            if self.cfg.get("DATASET", "").lower() == "nuscenes":
                if not should_create_new_track_nuscenes(init_bboxes[i], self.cfg):
                    continue
                if not should_create_new_track_distance_aware(
                    init_bboxes[i],
                    self.cfg,
                    transform_matrix=frame_info.transform_matrix,
                ):
                    continue

            self.all_trajs[self.track_id_counter] = Trajectory(
                track_id=self.track_id_counter,
                init_bbox=init_bboxes[i],
                cfg=self.cfg,
            )
            self.track_id_counter += 1

        for track_id in list(self.all_trajs.keys()):
            if self.all_trajs[track_id].status_flag == 4:
                self.all_dead_trajs[track_id] = self.all_trajs[track_id]
                del self.all_trajs[track_id]

        output_trajs = self.get_output_trajs(frame_info.frame_id, frame_info.transform_matrix)

        return output_trajs

    def get_output_trajs(self, frame_id, transform_matrix=None):
        output_trajs = {}
        for track_id in list(self.all_trajs.keys()):
            traj = self.all_trajs[track_id]
            if traj.status_flag == 1 or frame_id < 3:
                bbox = traj.bboxes[-1]

                # Keep original MCTrack behavior: do not output filtered prediction boxes.
                if hasattr(traj, "_is_filter_predict_box") and hasattr(bbox, "det_score"):
                    if bbox.det_score == traj._is_filter_predict_box:
                        continue

                if self.cfg.get("DATASET", "").lower() == "nuscenes":
                    # OUTPUT_FILTER-only quality filter: affects output only, not
                    # association or trajectory deletion.
                    if not should_output_by_quality_filter(
                        traj,
                        bbox,
                        frame_id,
                        self.cfg,
                        transform_matrix=transform_matrix,
                    ):
                        continue

                    # Independent distance-aware filter for tentative tracks.
                    # It does not read OUTPUT_FILTER.SHORT_TRAJ_LENGTH.
                    if not should_output_by_distance_aware_tentative_filter(
                        traj,
                        bbox,
                        self.cfg,
                        transform_matrix=transform_matrix,
                    ):
                        continue

                    # BEV range filter: replacement for KITTI image-boundary filtering.
                    if not should_output_nuscenes_bev_filter(
                        traj,
                        bbox,
                        self.cfg,
                        transform_matrix=transform_matrix,
                    ):
                        continue

                output_trajs[track_id] = bbox
                traj.is_output = True

        if self.cfg.get("DATASET", "").lower() == "nuscenes":
            output_trajs = apply_output_traj_nms(output_trajs, self.all_trajs, self.cfg)

        return output_trajs

    def post_processing(self):
        trajs = {}
        for track_id in self.all_dead_trajs.keys():
            traj = self.all_dead_trajs[track_id]
            traj.filtering()
            trajs[track_id] = traj
        for track_id in self.all_trajs.keys():
            traj = self.all_trajs[track_id]
            traj.filtering()
            trajs[track_id] = traj
        return trajs