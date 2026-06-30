# ------------------------------------------------------------------------
# HSM-LTM: Historical Similar-Motion Guided Lost Trajectory Management
# Exp1 version: only reliability check and early deletion.
# No Kalman fusion is used in this version.
# ------------------------------------------------------------------------

import numpy as np


def _get_param(cfg, keys, cat=None, default=None):
    """
    Safe getter for nested yaml config.
    Example:
        _get_param(cfg, ["HSM_LTM", "HISTORY_WINDOW"], cat=0, default=5)
    """
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]

    if cat is not None and isinstance(cur, dict):
        if cat in cur:
            return cur[cat]
        if str(cat) in cur:
            return cur[str(cat)]
        return default

    return cur


def _enabled(cfg):
    return bool(_get_param(cfg, ["HSM_LTM", "ENABLE"], default=False))


def _ensure_hsm_attrs(traj):
    if not hasattr(traj, "hsm_ref_track_ids"):
        traj.hsm_ref_track_ids = []
    if not hasattr(traj, "hsm_ref_base_pos"):
        traj.hsm_ref_base_pos = {}
    if not hasattr(traj, "hsm_ref_weights"):
        traj.hsm_ref_weights = {}
    if not hasattr(traj, "hsm_lost_base_pos"):
        traj.hsm_lost_base_pos = None
    if not hasattr(traj, "hsm_lost_base_frame"):
        traj.hsm_lost_base_frame = None
    if not hasattr(traj, "hsm_bad_count"):
        traj.hsm_bad_count = 0
    if not hasattr(traj, "hsm_delete_reason"):
        traj.hsm_delete_reason = ""


def _bbox_xy(bbox):
    """
    Use fused position when available.
    """
    if hasattr(bbox, "global_xyz_lwh_yaw_fusion"):
        return np.asarray(bbox.global_xyz_lwh_yaw_fusion[:2], dtype=float)
    return np.asarray(bbox.global_xyz[:2], dtype=float)


def _bbox_score(bbox):
    return float(getattr(bbox, "det_score", 0.0))


def _last_real_bbox(traj):
    """
    Return the latest non-fake bbox.
    """
    for bbox in reversed(traj.bboxes):
        if not getattr(bbox, "is_fake", False):
            return bbox
    return None


def _recent_real_bboxes(traj, k):
    """
    Return recent k non-fake bboxes.
    """
    real = [b for b in traj.bboxes if not getattr(b, "is_fake", False)]
    return real[-k:]


def _norm(v):
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def _angle_diff_deg(v1, v2, min_speed=0.5):
    """
    Direction difference between two velocity vectors.
    Return None when speed is too small.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < min_speed or n2 < min_speed:
        return None

    cos_val = float(np.dot(v1, v2) / (n1 * n2 + 1e-6))
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def _safe_velocity_from_bbox(bbox):
    """
    Prefer velocity produced by the tracker, then raw detection velocity.
    """
    for attr in ["global_velocity_fusion", "global_velocity_diff", "global_velocity"]:
        if hasattr(bbox, attr):
            v = np.asarray(getattr(bbox, attr)[:2], dtype=float)
            if np.all(np.isfinite(v)):
                return v
    return np.zeros(2, dtype=float)


def _motion_vector(traj, history_window, frame_rate):
    """
    Estimate historical velocity from recent real bboxes.
    """
    bboxes = _recent_real_bboxes(traj, history_window)

    if len(bboxes) >= 2:
        b0 = bboxes[0]
        b1 = bboxes[-1]

        dt = (b1.frame_id - b0.frame_id) / float(frame_rate)
        if dt > 1e-6:
            return (_bbox_xy(b1) - _bbox_xy(b0)) / dt

    last = _last_real_bbox(traj)
    if last is not None:
        return _safe_velocity_from_bbox(last)

    return np.zeros(2, dtype=float)


def _current_velocity(traj):
    """
    For lost trajectory after unmatch_update(), the last bbox is fake.
    Its global_velocity_diff is calculated after fake update.
    """
    bbox = traj.bboxes[-1]
    for attr in ["global_velocity_diff", "global_velocity_fusion", "global_velocity"]:
        if hasattr(bbox, attr):
            v = np.asarray(getattr(bbox, attr)[:2], dtype=float)
            if np.all(np.isfinite(v)):
                return v
    return np.zeros(2, dtype=float)


def _speed_rel_diff(v1, v2, base_min=0.5):
    s1 = _norm(v1)
    s2 = _norm(v2)
    return abs(s1 - s2) / max(s1, s2, base_min)


def _pos_res_threshold(lost_base_pos, cfg, cat):
    dist = _norm(lost_base_pos)

    near_dist = float(_get_param(cfg, ["HSM_LTM", "CONSISTENCY", "NEAR_DIST"], cat, 30.0))
    mid_dist = float(_get_param(cfg, ["HSM_LTM", "CONSISTENCY", "MID_DIST"], cat, 50.0))

    if dist < near_dist:
        return float(_get_param(cfg, ["HSM_LTM", "CONSISTENCY", "POS_RES_THRE_NEAR"], cat, 6.0))
    elif dist < mid_dist:
        return float(_get_param(cfg, ["HSM_LTM", "CONSISTENCY", "POS_RES_THRE_MID"], cat, 8.0))
    else:
        return float(_get_param(cfg, ["HSM_LTM", "CONSISTENCY", "POS_RES_THRE_FAR"], cat, 12.0))

def _apply_group_prediction_to_fake_bbox(lost_traj, p_group, p_kf, cfg, cat):
    """
    用 HSM-LTM 群体运动预测轻量修正当前 fake bbox。
    注意：这里只改当前 fake bbox 的位置，不直接改 Kalman 内部状态。
    """
    mode = str(
        _get_param(
            cfg,
            ["HSM_LTM", "PREDICTION", "MODE"],
            default="CHECK_ONLY",
        )
    )

    enable = bool(
        _get_param(
            cfg,
            ["HSM_LTM", "PREDICTION", "ENABLE"],
            default=False,
        )
    )

    if not enable or mode != "BLEND_GROUP":
        return False

    alpha = float(
        _get_param(
            cfg,
            ["HSM_LTM", "PREDICTION", "GROUP_BLEND_ALPHA"],
            cat,
            0.2,
        )
    )

    alpha = float(np.clip(alpha, 0.0, 1.0))

    p_group = np.asarray(p_group, dtype=float)
    p_kf = np.asarray(p_kf, dtype=float)

    p_final = p_kf + alpha * (p_group - p_kf)

    bbox = lost_traj.bboxes[-1]

    if hasattr(bbox, "global_xyz_lwh_yaw"):
        bbox.global_xyz_lwh_yaw[0] = float(p_final[0])
        bbox.global_xyz_lwh_yaw[1] = float(p_final[1])

    if hasattr(bbox, "global_xyz_lwh_yaw_fusion"):
        bbox.global_xyz_lwh_yaw_fusion[0] = float(p_final[0])
        bbox.global_xyz_lwh_yaw_fusion[1] = float(p_final[1])

    # 保存调试信息
    bbox.hsm_pred_mode = "blend_group"
    bbox.hsm_group_pred_x = float(p_group[0])
    bbox.hsm_group_pred_y = float(p_group[1])
    bbox.hsm_kf_pred_x = float(p_kf[0])
    bbox.hsm_kf_pred_y = float(p_kf[1])
    bbox.hsm_final_pred_x = float(p_final[0])
    bbox.hsm_final_pred_y = float(p_final[1])
    bbox.hsm_group_blend_alpha = float(alpha)

    if bool(_get_param(cfg, ["HSM_LTM", "PREDICTION", "DEBUG"], default=False)):
        print(
            "[HSM_LTM][BLEND_GROUP]",
            "track_id=", lost_traj.track_id,
            "unmatch=", lost_traj.unmatch_length,
            "alpha=", round(alpha, 3),
            "kf=", np.round(p_kf, 3).tolist(),
            "group=", np.round(p_group, 3).tolist(),
            "final=", np.round(p_final, 3).tolist(),
        )

    return True
def _build_reference_group(lost_traj, all_trajs, cfg):
    """
    Build historical similar-motion reference group for a lost trajectory.
    """
    _ensure_hsm_attrs(lost_traj)

    cat = lost_traj.category_num
    history_window = int(_get_param(cfg, ["HSM_LTM", "HISTORY_WINDOW"], cat, 5))
    frame_rate = float(cfg.get("FRAME_RATE", 10))

    min_det_score = float(_get_param(cfg, ["HSM_LTM", "CANDIDATE", "MIN_DET_SCORE"], cat, 0.7))
    min_track_length = int(_get_param(cfg, ["HSM_LTM", "CANDIDATE", "MIN_TRACK_LENGTH"], cat, 3))
    min_speed = float(_get_param(cfg, ["HSM_LTM", "CANDIDATE", "MIN_SPEED"], cat, 0.5))
    max_distance = float(_get_param(cfg, ["HSM_LTM", "CANDIDATE", "MAX_DISTANCE"], cat, 25.0))
    max_dir_diff = float(_get_param(cfg, ["HSM_LTM", "CANDIDATE", "MAX_DIR_DIFF_DEG"], cat, 35.0))
    max_speed_rel = float(_get_param(cfg, ["HSM_LTM", "CANDIDATE", "MAX_SPEED_REL_DIFF"], cat, 0.6))

    w_dir = float(_get_param(cfg, ["HSM_LTM", "WEIGHT", "DIR"], cat, 0.4))
    w_speed = float(_get_param(cfg, ["HSM_LTM", "WEIGHT", "SPEED"], cat, 0.3))
    w_dist = float(_get_param(cfg, ["HSM_LTM", "WEIGHT", "DIST"], cat, 0.2))
    w_score = float(_get_param(cfg, ["HSM_LTM", "WEIGHT", "SCORE"], cat, 0.1))

    topk = int(_get_param(cfg, ["HSM_LTM", "SELECTION", "TOPK"], cat, 5))
    sample_num = int(_get_param(cfg, ["HSM_LTM", "SELECTION", "SAMPLE_NUM"], cat, 3))
    mode = str(_get_param(cfg, ["HSM_LTM", "SELECTION", "MODE"], default="TOPK_WEIGHTED_RANDOM"))
    seed = int(_get_param(cfg, ["HSM_LTM", "SELECTION", "RANDOM_SEED"], default=2026))

    lost_last_real = _last_real_bbox(lost_traj)
    if lost_last_real is None:
        return []

    lost_base_pos = _bbox_xy(lost_last_real)
    lost_motion = _motion_vector(lost_traj, history_window, frame_rate)

    if _norm(lost_motion) < min_speed:
        return []

    candidates = []

    for ref_id, ref_traj in all_trajs.items():
        if ref_id == lost_traj.track_id:
            continue

        if getattr(ref_traj, "category_num", None) != cat:
            continue

        if getattr(ref_traj, "track_length", 0) < min_track_length:
            continue

        if len(ref_traj.bboxes) == 0:
            continue

        # Exp1 only uses currently matched reliable tracks as references.
        # If the current bbox is fake, skip it.
        if getattr(ref_traj.bboxes[-1], "is_fake", False):
            continue

        ref_last_real = _last_real_bbox(ref_traj)
        if ref_last_real is None:
            continue

        if _bbox_score(ref_last_real) < min_det_score:
            continue

        ref_pos = _bbox_xy(ref_last_real)
        dist = _norm(ref_pos - lost_base_pos)
        if dist > max_distance:
            continue

        ref_motion = _motion_vector(ref_traj, history_window, frame_rate)
        if _norm(ref_motion) < min_speed:
            continue

        dir_diff = _angle_diff_deg(lost_motion, ref_motion, min_speed=min_speed)
        if dir_diff is None or dir_diff > max_dir_diff:
            continue

        speed_rel = _speed_rel_diff(lost_motion, ref_motion, base_min=min_speed)
        if speed_rel > max_speed_rel:
            continue

        s_dir = max(0.0, 1.0 - dir_diff / max(max_dir_diff, 1e-6))
        s_speed = max(0.0, 1.0 - speed_rel / max(max_speed_rel, 1e-6))
        s_dist = max(0.0, 1.0 - dist / max(max_distance, 1e-6))
        s_score = np.clip(_bbox_score(ref_last_real), 0.0, 1.0)

        sim = w_dir * s_dir + w_speed * s_speed + w_dist * s_dist + w_score * s_score

        candidates.append({
            "track_id": ref_id,
            "sim": float(sim),
            "base_pos": ref_pos,
        })

    if len(candidates) == 0:
        return []

    candidates = sorted(candidates, key=lambda x: x["sim"], reverse=True)
    candidates = candidates[:max(topk, 1)]

    if mode == "TOPK":
        selected = candidates[:sample_num]

    elif mode == "RANDOM":
        rng = np.random.RandomState(seed + int(lost_traj.track_id))
        idx = rng.choice(len(candidates), size=min(sample_num, len(candidates)), replace=False)
        selected = [candidates[i] for i in idx]

    else:
        # TOPK_WEIGHTED_RANDOM
        rng = np.random.RandomState(
            seed + int(lost_traj.track_id) + int(lost_last_real.frame_id)
        )
        probs = np.asarray([c["sim"] for c in candidates], dtype=float)
        probs = probs + 1e-6
        probs = probs / probs.sum()

        idx = rng.choice(
            len(candidates),
            size=min(sample_num, len(candidates)),
            replace=False,
            p=probs,
        )
        selected = [candidates[i] for i in idx]

    return selected


def hsm_after_unmatch_update(lost_traj, all_trajs, cfg):
    """
    Main function called right after traj.unmatch_update(frame_id).

    Return:
        True  -> deleted by HSM-LTM
        False -> not deleted
    """
    if not _enabled(cfg):
        return False

    _ensure_hsm_attrs(lost_traj)

    if lost_traj.status_flag == 4:
        return False

    cat = lost_traj.category_num
    start_unmatch = int(_get_param(cfg, ["HSM_LTM", "START_UNMATCH_LENGTH"], cat, 1))
    min_ref_num = int(_get_param(cfg, ["HSM_LTM", "SELECTION", "MIN_REF_NUM"], cat, 2))

    if lost_traj.unmatch_length < start_unmatch:
        return False

    # Build reference group only once when this trajectory just enters lost state.
    if lost_traj.unmatch_length == start_unmatch and len(lost_traj.hsm_ref_track_ids) == 0:
        refs = _build_reference_group(lost_traj, all_trajs, cfg)

        if len(refs) >= min_ref_num:
            lost_last_real = _last_real_bbox(lost_traj)
            lost_traj.hsm_lost_base_pos = _bbox_xy(lost_last_real)
            lost_traj.hsm_lost_base_frame = lost_last_real.frame_id

            lost_traj.hsm_ref_track_ids = [r["track_id"] for r in refs]
            lost_traj.hsm_ref_base_pos = {
                r["track_id"]: r["base_pos"] for r in refs
            }
            lost_traj.hsm_ref_weights = {
                r["track_id"]: max(float(r["sim"]), 1e-6) for r in refs
            }
            lost_traj.hsm_bad_count = 0

    # If no valid reference group, use original tracker behavior.
    if len(lost_traj.hsm_ref_track_ids) < min_ref_num:
        return False

    if lost_traj.hsm_lost_base_pos is None:
        return False

    valid_disps = []
    valid_vels = []
    valid_weights = []

    history_window = int(_get_param(cfg, ["HSM_LTM", "HISTORY_WINDOW"], cat, 5))
    frame_rate = float(cfg.get("FRAME_RATE", 10))
    min_speed = float(_get_param(cfg, ["HSM_LTM", "CANDIDATE", "MIN_SPEED"], cat, 0.5))

    for ref_id in list(lost_traj.hsm_ref_track_ids):
        ref_traj = all_trajs.get(ref_id, None)
        if ref_traj is None:
            continue

        if ref_traj.status_flag == 4:
            continue

        if len(ref_traj.bboxes) == 0:
            continue

        # If reference vehicle also becomes lost, do not use it this frame.
        if getattr(ref_traj.bboxes[-1], "is_fake", False):
            continue

        ref_last = _last_real_bbox(ref_traj)
        if ref_last is None:
            continue

        ref_base_pos = lost_traj.hsm_ref_base_pos.get(ref_id, None)
        if ref_base_pos is None:
            continue

        ref_cur_pos = _bbox_xy(ref_last)
        ref_disp = ref_cur_pos - np.asarray(ref_base_pos, dtype=float)
        ref_vel = _motion_vector(ref_traj, history_window, frame_rate)
        ref_weight = float(lost_traj.hsm_ref_weights.get(ref_id, 1.0))

        valid_disps.append(ref_disp)
        valid_vels.append(ref_vel)
        valid_weights.append(ref_weight)

    if len(valid_disps) < min_ref_num:
        return False

    weights = np.asarray(valid_weights, dtype=float)
    weights = weights / (weights.sum() + 1e-6)

    group_disp = np.sum(np.asarray(valid_disps) * weights[:, None], axis=0)
    group_vel = np.sum(np.asarray(valid_vels) * weights[:, None], axis=0)

    p_group = np.asarray(lost_traj.hsm_lost_base_pos, dtype=float) + group_disp

    p_kf = _bbox_xy(lost_traj.bboxes[-1])
    v_kf = _current_velocity(lost_traj)

    dir_thre = float(_get_param(cfg, ["HSM_LTM", "CONSISTENCY", "DIR_THRE_DEG"], cat, 60.0))
    speed_thre = float(_get_param(cfg, ["HSM_LTM", "CONSISTENCY", "SPEED_REL_THRE"], cat, 0.9))
    abnormal_item_thre = int(_get_param(cfg, ["HSM_LTM", "CONSISTENCY", "ABNORMAL_ITEM_THRE"], cat, 2))
    bad_count_thre = int(_get_param(cfg, ["HSM_LTM", "CONSISTENCY", "BAD_COUNT_THRE"], cat, 2))

    pos_thre = _pos_res_threshold(lost_traj.hsm_lost_base_pos, cfg, cat)

    abnormal_items = 0
    reasons = []

    dir_err = _angle_diff_deg(v_kf, group_vel, min_speed=min_speed)
    if dir_err is not None and dir_err > dir_thre:
        abnormal_items += 1
        reasons.append(f"dir={dir_err:.1f}>{dir_thre:.1f}")

    speed_err = _speed_rel_diff(v_kf, group_vel, base_min=min_speed)
    if speed_err > speed_thre:
        abnormal_items += 1
        reasons.append(f"speed={speed_err:.2f}>{speed_thre:.2f}")

    pos_err = _norm(p_kf - p_group)
    if pos_err > pos_thre:
        abnormal_items += 1
        reasons.append(f"pos={pos_err:.2f}>{pos_thre:.2f}")
    # ------------------------------------------------------------
    # 新增：运动目标使用群体运动轻量修正 Kalman fake bbox
    # ------------------------------------------------------------
    only_blend_when_pos_normal = bool(
        _get_param(
            cfg,
            ["HSM_LTM", "PREDICTION", "ONLY_BLEND_WHEN_POS_NORMAL"],
            default=True,
        )
    )

    can_blend = True

    # 如果位置误差已经超过阈值，说明群体预测和 Kalman 预测冲突较大，
    # 此时不融合，避免把轨迹拉偏。
    if only_blend_when_pos_normal and pos_err > pos_thre:
        can_blend = False

    if can_blend:
        _apply_group_prediction_to_fake_bbox(
            lost_traj=lost_traj,
            p_group=p_group,
            p_kf=p_kf,
            cfg=cfg,
            cat=cat,
        )

    # Save debug values on the current fake bbox.
    lost_traj.bboxes[-1].hsm_pos_error = float(pos_err)
    lost_traj.bboxes[-1].hsm_speed_error = float(speed_err)
    lost_traj.bboxes[-1].hsm_dir_error = -1.0 if dir_err is None else float(dir_err)
    lost_traj.bboxes[-1].hsm_ref_num = int(len(valid_disps))

    if abnormal_items >= abnormal_item_thre:
        lost_traj.hsm_bad_count += 1
    else:
        lost_traj.hsm_bad_count = max(0, lost_traj.hsm_bad_count - 1)

    if lost_traj.hsm_bad_count >= bad_count_thre:
        lost_traj.status_flag = 4
        lost_traj.hsm_delete_reason = ",".join(reasons)

        if bool(_get_param(cfg, ["HSM_LTM", "DEBUG", "PRINT_DELETE"], default=False)):
            print(
                f"[HSM-LTM DELETE] track={lost_traj.track_id}, "
                f"unmatch={lost_traj.unmatch_length}, "
                f"ref={len(valid_disps)}, "
                f"reason={lost_traj.hsm_delete_reason}"
            )

        return True

    return False