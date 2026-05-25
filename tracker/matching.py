# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import lap
from tracker.group_motion_switch import detect_group_motion
from tracker.cost_function import *
from utils.utils import mask_tras_dets


def Greedy(cost_matrix, thresholds):
    """
    Refer: https://github.com/lixiaoyu2000/Poly-MOT/blob/main/utils/matching.py
    Info: This function implements the Greedy matching algorithm.
    Parameters:
        input:
            cost_matrix: np.array, either 2D or 3D cost matrix with shape [N_cls, N_det, N_tra] or [N_det, N_tra].
                         - N_cls: Number of classes.
                         - N_det: Number of detections.
                         - N_tra: Number of trajectories.
                         Invalid costs are represented by np.inf.
            thresholds: dict, class-specific matching thresholds to restrict false positive matches.
        output:
            m_det: list, indices of matched detections.
            m_tra: list, indices of matched trajectories.
            um_det: np.array, indices of unmatched detections.
            um_tra: np.array, indices of unmatched trajectories.
            costs: np.array, matching costs for the matched pairs.
    """
    assert cost_matrix.ndim == 2 or cost_matrix.ndim == 3, "cost matrix must be valid."
    if cost_matrix.ndim == 2:
        cost_matrix = cost_matrix[None, :, :]
    assert (
        len(thresholds) == cost_matrix.shape[0]
    ), "the number of thresholds should be egual to cost matrix number."

    # solve cost matrix
    m_det, m_tra = [], []
    costs = []
    num_det, num_tra = cost_matrix.shape[1:]
    for cls_idx, cls_cost in enumerate(cost_matrix):
        for det_idx in range(num_det):
            tra_idx = cls_cost[det_idx].argmin()
            if cls_cost[det_idx][tra_idx] <= thresholds[cls_idx]:
                costs.append(cls_cost[det_idx, tra_idx])
                cost_matrix[cls_idx, :, tra_idx] = 1e18
                m_det.append(det_idx)
                m_tra.append(tra_idx)

    # unmatched tra and det
    if len(m_det) == 0:
        um_det, um_tra = np.arange(num_det), np.arange(num_tra)
    else:
        um_det = np.setdiff1d(np.arange(num_det), np.array(m_det))
        um_tra = np.setdiff1d(np.arange(num_tra), np.array(m_tra))

    return m_det, m_tra, um_det, um_tra, np.array(costs)


def Hungarian(cost_matrix, thresholds):
    """
    Refer: https://github.com/lixiaoyu2000/Poly-MOT/blob/main/utils/matching.py
    Info: This function implements the Hungarian algorithm using the Linear Assignment Problem solver (lapjv).
    Parameters:
        input:
            cost_matrix: np.array, either 2D or 3D cost matrix with shape [N_cls, N_det, N_tra] or [N_det, N_tra].
                         Invalid costs are represented by np.inf.
            thresholds: dict, class-specific matching thresholds to restrict false positive matches.
        output:
            m_det: list, indices of matched detections.
            m_tra: list, indices of matched trajectories.
            um_det: np.array, indices of unmatched detections.
            um_tra: np.array, indices of unmatched trajectories.
            costs: np.array, matching costs for the matched pairs.
    """
    assert cost_matrix.ndim == 2 or cost_matrix.ndim == 3, "cost matrix must be valid."
    if cost_matrix.ndim == 2:
        cost_matrix = cost_matrix[None, :, :]
    assert (
        len(thresholds) == cost_matrix.shape[0]
    ), "the number of thresholds should be equal to cost matrix number."

    # solve cost matrix
    m_det, m_tra = [], []
    costs = []
    for cls_idx, cls_cost in enumerate(cost_matrix):
        _, x, y = lap.lapjv(cls_cost, extend_cost=True, cost_limit=thresholds[cls_idx])
        for ix, mx in enumerate(x):
            if mx >= 0:
                assert (ix not in m_det) and (mx not in m_tra)
                m_det.append(ix)
                m_tra.append(mx)
                costs.append(cls_cost[ix, mx])

    # unmatched tra and det
    num_det, num_tra = cost_matrix.shape[1:]
    if len(m_det) == 0:
        um_det, um_tra = np.arange(num_det), np.arange(num_tra)
    else:
        um_det = np.setdiff1d(np.arange(num_det), np.array(m_det))
        um_tra = np.setdiff1d(np.arange(num_tra), np.array(m_tra))

    return m_det, m_tra, um_det, um_tra, np.array(costs)

def _cfg_by_cls(value, cls_id, default):
    """
    支持两种配置：
    DIR_THRE_DEG: 45.0
    DIR_THRE_DEG: {0: 45.0}
    注意：MCTrack 的 yaml key 会被转成字符串，所以也要支持 "0"。
    """
    if isinstance(value, dict):
        if cls_id in value:
            return value[cls_id]
        if str(cls_id) in value:
            return value[str(cls_id)]
        return default

    if value is None:
        return default

    return value


def _angle_diff_rad(a, b):
    """
    计算两个角度的最小差值，返回弧度。
    """
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(diff)


def _get_current_bbox(obj):
    """
    traj 用 traj.bboxes[-1]
    det 本身就是 bbox
    """
    if hasattr(obj, "bboxes") and len(obj.bboxes) > 0:
        return obj.bboxes[-1]
    return obj


def _get_xy(obj):
    bbox = _get_current_bbox(obj)

    if hasattr(bbox, "global_xyz"):
        xyz = np.asarray(bbox.global_xyz, dtype=float)
        return xyz[:2]

    return None


def _get_yaw(obj):
    bbox = _get_current_bbox(obj)

    if hasattr(bbox, "global_yaw"):
        return float(bbox.global_yaw)

    return None


def _get_speed(obj):
    bbox = _get_current_bbox(obj)

    if hasattr(bbox, "global_velocity"):
        v = np.asarray(bbox.global_velocity, dtype=float)
        if v.shape[0] >= 2:
            return float(np.linalg.norm(v[:2]))

    if hasattr(bbox, "velocity"):
        v = np.asarray(bbox.velocity, dtype=float)
        if v.shape[0] >= 2:
            return float(np.linalg.norm(v[:2]))

    return None


def apply_group_motion_consistency(cost_matrix, trajs, dets, cfg, dets_label):
    """
    实验2核心函数：
    只有 use_group_motion=True 时才会调用。

    作用：
    在群体运动场景中，对明显不合理的 traj-det 匹配设为 np.inf，
    让 Hungarian / Greedy 不再选择它。

    注意：
    这个函数不改变原始 MCTrack 的 cost 计算方式，
    只是在群体运动场景下增加一道保守审查。
    """
    if "CONSISTENCY" in cfg:
        cons_cfg = cfg["CONSISTENCY"]
    else:
        cons_cfg = {}

    dir_thre_cfg = cons_cfg.get("DIR_THRE_DEG", {"0": 45.0})
    speed_rel_thre_cfg = cons_cfg.get("SPEED_REL_THRE", {"0": 0.8})

    pos_near_cfg = cons_cfg.get("POS_RES_THRE_NEAR", {"0": 6.0})
    pos_mid_cfg = cons_cfg.get("POS_RES_THRE_MID", {"0": 8.0})
    pos_far_cfg = cons_cfg.get("POS_RES_THRE_FAR", {"0": 12.0})

    near_dist_cfg = cons_cfg.get("NEAR_DIST", {"0": 30.0})
    mid_dist_cfg = cons_cfg.get("MID_DIST", {"0": 50.0})

    new_cost_matrix = cost_matrix.copy()

    total_pairs = 0
    blocked_pairs = 0
    bad_pos_count = 0
    bad_dir_speed_count = 0

    for t, traj in enumerate(trajs):
        traj_xy = _get_xy(traj)
        traj_yaw = _get_yaw(traj)
        traj_speed = _get_speed(traj)

        if traj_xy is None:
            continue

        for d, det in enumerate(dets):
            total_pairs += 1

            if not np.isfinite(new_cost_matrix[t, d]):
                continue

            det_xy = _get_xy(det)
            det_yaw = _get_yaw(det)
            det_speed = _get_speed(det)

            if det_xy is None:
                continue

            cls_id = int(dets_label[d])

            dir_thre_deg = float(_cfg_by_cls(dir_thre_cfg, cls_id, 45.0))
            speed_rel_thre = float(_cfg_by_cls(speed_rel_thre_cfg, cls_id, 0.8))

            pos_near = float(_cfg_by_cls(pos_near_cfg, cls_id, 6.0))
            pos_mid = float(_cfg_by_cls(pos_mid_cfg, cls_id, 8.0))
            pos_far = float(_cfg_by_cls(pos_far_cfg, cls_id, 12.0))

            near_dist = float(_cfg_by_cls(near_dist_cfg, cls_id, 30.0))
            mid_dist = float(_cfg_by_cls(mid_dist_cfg, cls_id, 50.0))

            # 1. 位置残差
            pos_res = float(np.linalg.norm(traj_xy - det_xy))
            det_dist = float(np.linalg.norm(det_xy))

            if det_dist <= near_dist:
                pos_thre = pos_near
            elif det_dist <= mid_dist:
                pos_thre = pos_mid
            else:
                pos_thre = pos_far

            bad_pos = pos_res > pos_thre

            # 2. 方向差异
            bad_dir = False
            if traj_yaw is not None and det_yaw is not None:
                dir_diff_deg = _angle_diff_rad(traj_yaw, det_yaw) * 180.0 / np.pi
                bad_dir = dir_diff_deg > dir_thre_deg

            # 3. 速度差异
            bad_speed = False
            if traj_speed is not None and det_speed is not None:
                if max(traj_speed, det_speed) > 1e-6:
                    speed_rel = abs(traj_speed - det_speed) / max(traj_speed, det_speed)
                    bad_speed = speed_rel > speed_rel_thre

            # 保守屏蔽策略：
            # A. 位置残差明显过大，屏蔽
            # B. 方向和速度同时异常，屏蔽
            if bad_pos or (bad_dir and bad_speed):
                new_cost_matrix[t, d] = np.inf
                blocked_pairs += 1

                if bad_pos:
                    bad_pos_count += 1
                if bad_dir and bad_speed:
                    bad_dir_speed_count += 1

    debug = False
    if "CONSISTENCY" in cfg:
        debug = cfg["CONSISTENCY"].get("DEBUG", False)

    if debug:
        print(
            "[CONSISTENCY]",
            "total_pairs=", total_pairs,
            "blocked_pairs=", blocked_pairs,
            "bad_pos=", bad_pos_count,
            "bad_dir_speed=", bad_dir_speed_count,
        )

    return new_cost_matrix
def match_trajs_and_dets(
    trajs,
    dets,
    cfg,
    transform_matrix=None,
    is_rv=False,
    use_group_motion=False
):
    """
    Info:
        This function matches trajectories with detections using a cost matrix
        and a specified matching algorithm.

    实验2逻辑：
        use_group_motion=False:
            完全保持原始 MCTrack 匹配逻辑。

        use_group_motion=True:
            在原始 cost_matrix 基础上启用群体运动一致性审查。
    """
    if len(trajs) == 0 or len(dets) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)

    cost_matrix, trajs_category, dets_category = cost_calculate_general(
        trajs, dets, cfg, transform_matrix, is_rv
    )

    match_type = "RV" if is_rv else "BEV"

    category_map = cfg["CATEGORY_MAP_TO_NUMBER"]
    vectorized_map = np.vectorize(category_map.get)
    dets_label = vectorized_map(dets_category)
    trajs_label = vectorized_map(trajs_category)

    # ------------------------------------------------------------
    # 实验2核心开关
    # ------------------------------------------------------------
    # 没有群体运动：
    #   不改 cost_matrix，完全保持原始 MCTrack。
    #
    # 有群体运动：
    #   只在 BEV 第一次匹配中加入一致性审查。
    #   RV 二次匹配先不加，避免过度影响原始流程。
    # ------------------------------------------------------------
    cons_enable = True
    if "CONSISTENCY" in cfg:
        cons_enable = cfg["CONSISTENCY"].get("ENABLE", True)

    if use_group_motion and cons_enable and not is_rv:
        cost_matrix = apply_group_motion_consistency(
            cost_matrix=cost_matrix,
            trajs=trajs,
            dets=dets,
            cfg=cfg,
            dets_label=dets_label,
        )

    cls_num = len(cfg["CATEGORY_LIST"])
    valid_mask = mask_tras_dets(cls_num, trajs_label, dets_label)
    trans_valid_mask = valid_mask.transpose(0, 2, 1)

    trans_cost_matrix = cost_matrix.T
    trans_cost_matrix = trans_cost_matrix[None, :, :].repeat(cls_num, axis=0)
    trans_cost_matrix[np.where(~trans_valid_mask)] = np.inf

    if min(cost_matrix.shape) > 0:
        if cfg["MATCHING"][match_type]["MATCHING_MODE"] == "Hungarian":
            m_det, m_tra, um_det, um_tra, costs = Hungarian(
                trans_cost_matrix,
                cfg["THRESHOLD"][match_type]["COST_THRE"],
            )
            assert len(m_det) == len(m_tra)
            matched_indices = np.column_stack((m_tra, m_det))

        elif cfg["MATCHING"][match_type]["MATCHING_MODE"] == "Greedy":
            m_det, m_tra, um_det, um_tra, costs = Greedy(
                trans_cost_matrix,
                cfg["THRESHOLD"][match_type]["COST_THRE"],
            )
            assert len(m_det) == len(m_tra)
            matched_indices = np.column_stack((m_tra, m_det))

        else:
            matched_indices = np.empty(shape=(0, 2), dtype=int)
            costs = np.empty((0,))
    else:
        matched_indices = np.empty(shape=(0, 2), dtype=int)
        costs = np.empty((0,))

    return matched_indices, costs


def cost_calculate_general(trajs, dets, cfg, transform_matrix, is_rv=False):
    cost_matrix = np.zeros((len(trajs), len(dets)))

    def choose_cost_func(is_rv, cost_mode):
        if is_rv:
            if cost_mode == "IOU_2D":
                cal_cost_func = cal_iou_inrv
            elif cost_mode == "GIOU_2D":
                cal_cost_func = cal_giou_inrv
            elif cost_mode == "DIOU_2D":
                cal_cost_func = cal_diou_inrv
            elif cost_mode == "SDIOU_2D":
                cal_cost_func = cal_sdiou_inrv
        else:
            if cost_mode == "RO_GDIOU_3D":
                cal_cost_func = cal_rotation_gdiou_inbev
        return cal_cost_func

    for t, trk in enumerate(trajs):
        for d, det in enumerate(dets):
            trk_category = cfg["CATEGORY_MAP_TO_NUMBER"][trajs[0].bboxes[-1].category]
            match_type = "BEV"
            if is_rv:
                match_type = "RV"
            cost_mode = cfg["MATCHING"][match_type]["COST_MODE"][trk_category]
            cost_state = cfg["MATCHING"][match_type]["COST_STATE"][trk_category]
            cost_state_predict_ratio = cfg["THRESHOLD"]["COST_STATE_PREDICT_RATION"][
                trk_category
            ]
            cal_cost_func = choose_cost_func(is_rv, cost_mode)
            pred_cost = cal_cost_func(trk, det, cfg, cal_flag="Predict")
            no_pred_cost = cal_cost_func(trk, det, cfg, cal_flag="BackPredict")

            if cost_state == "Predict":
                cost_matrix[t, d] = pred_cost
            elif cost_state == "BackPredict":
                cost_matrix[t, d] = no_pred_cost
            elif cost_state == "Fusion":
                cost_matrix[t, d] = (
                    cost_state_predict_ratio * pred_cost
                    + (1 - cost_state_predict_ratio) * no_pred_cost
                )

    trajs_category = np.array([traj.bboxes[-1].category for traj in trajs])
    dets_category = np.array([det.category for det in dets])
    same_category_mask = (trajs_category[:, np.newaxis] == dets_category).astype(int)
    cost_matrix[same_category_mask == 0] = -np.inf

    return 1 - cost_matrix, trajs_category, dets_category
