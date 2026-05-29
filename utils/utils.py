# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import math

from shapely.geometry import Polygon
from pyquaternion import Quaternion


def norm_radian(radians):
    """
    Info: This function normalizes input radian values to the range [-pi, pi].
    Parameters:
        input:
            radians: Array-like or scalar radian values.
        output:
            radian_norm: Normalized radian values in the range [-pi, pi], either as a scalar or array.
    """
    radians = np.array(radians).reshape(-1)
    radians_norm = []
    for radian in radians:
        n = np.floor(radian / (2 * np.pi))
        radian = radian - n * 2 * np.pi
        radians_norm.append(radian)
    radians_norm = np.array(radians_norm)
    if len(radians_norm) == 1:
        radian_norm = radians_norm[0]
    else:
        radian_norm = radians_norm
    if radian_norm > np.pi:
        return radian_norm - np.pi
    else:
        return radian_norm


def norm_realative_radian(radians_diff):
    """
    Info: This function normalizes relative radian differences to the range [-pi, pi].
    Parameters:
        input:
            radians_diff: Array-like or scalar relative radian differences.
        output:
            radians_diff_norm: Normalized relative radian values in the range [-pi, pi], either as a scalar or array.
    """
    radians_diff = np.array(radians_diff).reshape(-1)
    radians_diff_norm = []
    for rad in radians_diff:
        if rad < -np.pi:
            rad += 2 * np.pi
        elif rad > np.pi:
            rad -= 2 * np.pi
        radians_diff_norm.append(rad)
    radians_diff_norm = np.array(radians_diff_norm)
    if len(radians_diff_norm) == 1:
        return radians_diff_norm[0]
    else:
        return radians_diff_norm


def transform_yaw2quaternion(yaw):
    """
    Info: This function converts a yaw angle (in radians) to a quaternion representation.
    Parameters:
        input:
            yaw: A scalar or array-like yaw angle in radians.
        output:
            orientation: A Quaternion object representing the orientation corresponding to the yaw angle.
    """
    orientation = Quaternion(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    return orientation


def orientation_similarity(angle1_rad, angle2_rad):
    cosine_similarity = math.cos(norm_radian(angle1_rad - angle2_rad))
    similarity = (cosine_similarity + 1) / 2

    return similarity


def mask_between_boxes(labels_a, labels_b):
    """
    Info: This function creates a mask between two sets of labels, indicating whether the labels are different or the same.
    Parameters:
        input:
            labels_a: np.array, labels of the first collection.
            labels_b: np.array, labels of the second collection.
        output:
            mask: np.array[bool], matrix where 1 denotes different labels and 0 denotes the same labels.
            flattened_mask: np.array, flattened version of the mask.
    """
    mask = labels_a.reshape(-1, 1).repeat(len(labels_b), axis=1) != labels_b.reshape(
        1, -1
    ).repeat(len(labels_a), axis=0)
    return mask, mask.reshape(-1)


def mask_tras_dets(cls_num, tra_labels, det_labels):
    """
    Info: This function generates a mask to filter invalid costs between trajectories and detections based on class labels.
    Parameters:
        input:
            cls_num: int, number of classes.
            tra_labels: np.array, labels for the trajectories.
            det_labels: np.array, labels for the detections.
        output:
            mask: np.array[bool], shape (cls_num, tra_num, det_num).
    """
    tra_num, det_num = len(tra_labels), len(det_labels)
    cls_mask = (
        np.ones(shape=(cls_num, tra_num, det_num)) * np.arange(cls_num)[:, None, None]
    )
    # [tra_num, det_num], True denotes invalid(diff cls)
    same_mask, _ = mask_between_boxes(tra_labels, det_labels)
    # [tra_num, det_num], invalid idx assign -1
    tmp_labels = det_labels[None, :].repeat(tra_num, axis=0)
    tmp_labels[np.where(same_mask)] = -1
    return tmp_labels[None, :, :].repeat(cls_num, axis=0) == cls_mask

def _cfg_by_cls(value, cls_id, default):
    if isinstance(value, dict):
        if cls_id in value:
            return value[cls_id]
        if str(cls_id) in value:
            return value[str(cls_id)]
        return default
    if value is None:
        return default
    return value


def _get_box_bev_dist_from_np_det(np_det):
    """
    np_det 格式：
    [x, y, z, w, l, h, vx, vy, qw, qx, qy, qz, det_score, class_label]

    用 x, y 计算 BEV 距离。
    """
    if np_det is None or len(np_det) < 2:
        return 0.0
    return float(np.linalg.norm(np.asarray(np_det[:2], dtype=float)))


def _get_nms_thre_by_dist(np_det, cfg, default_thre):
    """
    根据当前 box 的 BEV 距离选择 NMS 阈值。
    如果没有开启 NMS_THRE_BY_DIST，则退回原始 default_thre。
    """
    if cfg is None:
        return float(default_thre)

    threshold_cfg = cfg.get("THRESHOLD", {})
    nms_dist_cfg = threshold_cfg.get("NMS_THRE_BY_DIST", {})

    if not bool(nms_dist_cfg.get("ENABLE", False)):
        return float(default_thre)

    cls_id = int(np_det[-1])

    dist_policy = cfg.get("DISTANCE_POLICY", {})
    near_dist = float(_cfg_by_cls(dist_policy.get("NEAR_DIST", {0: 30.0}), cls_id, 30.0))
    mid_dist = float(_cfg_by_cls(dist_policy.get("MID_DIST", {0: 65.0}), cls_id, 65.0))

    dist = _get_box_bev_dist_from_np_det(np_det)

    if dist < near_dist:
        bucket = "NEAR"
    elif dist < mid_dist:
        bucket = "MID"
    else:
        bucket = "FAR"

    selected = float(_cfg_by_cls(nms_dist_cfg.get(bucket, {cls_id: default_thre}), cls_id, default_thre))

    if bool(nms_dist_cfg.get("DEBUG", False)):
        print(
            "[NMS_THRE_BY_DIST]",
            "cls=", cls_id,
            "dist=", round(dist, 2),
            "bucket=", bucket,
            "thre=", selected,
            "default=", default_thre,
        )

    return selected
def blend_nms(box_infos, metrics, thre, cfg=None):
    """
    Refer: https://github.com/lixiaoyu2000/Poly-MOT/blob/main/pre_processing/nusc_nms.py

    距离感知版本：
    - 原始逻辑：同一类别使用固定 NMS_THRE
    - 新逻辑：如果 cfg.THRESHOLD.NMS_THRE_BY_DIST.ENABLE=True，
      则根据 box 的 BEV 距离使用 NEAR/MID/FAR 阈值。

    注意：
    pairwise NMS 时使用 max(thre_current, thre_candidate)，
    这样可以保护远距离目标，避免远距离真实小目标被过强 NMS 压掉。
    """
    assert metrics in [
        "iou_bev",
        "iou_3d",
        "giou_bev",
        "giou_3d",
        "d_eucl",
    ], "unsupported NMS metrics"

    assert (
        "np_dets" in box_infos and "np_dets_bottom_corners" in box_infos
    ), "must contain specified keys"

    infos, corners = box_infos["np_dets"], box_infos["np_dets_bottom_corners"]

    sort_idxs, keep = np.argsort(-infos[:, -2]), []

    while sort_idxs.size > 0:
        i = sort_idxs[0]
        keep.append(i)

        if sort_idxs.size == 1:
            break

        current_class = int(infos[i, -1])
        default_current_thre = float(thre[current_class])

        current_thre = _get_nms_thre_by_dist(
            np_det=infos[i],
            cfg=cfg,
            default_thre=default_current_thre,
        )

        remain_idxs = sort_idxs[1:]

        left, first = [
            {"np_dets_bottom_corners": corners[idx], "np_dets": infos[idx]}
            for idx in [remain_idxs, i]
        ]

        distances = iou_bev(first, left)[0]

        # ------------------------------------------------------------
        # 距离感知 pairwise NMS threshold
        # ------------------------------------------------------------
        # 原始代码：
        #   sort_idxs = sort_idxs[1:][distances <= current_thre]
        #
        # 新逻辑：
        #   当前框和候选框分别取自己的距离感知阈值，
        #   pair 阈值使用 max(current_thre, candidate_thre)。
        #   这样 near-far 或 mid-far 组合不会因为近距离阈值过严而误删远距离目标。
        # ------------------------------------------------------------
        pair_thres = []

        for idx in remain_idxs:
            cand_class = int(infos[idx, -1])
            default_cand_thre = float(thre[cand_class])

            cand_thre = _get_nms_thre_by_dist(
                np_det=infos[idx],
                cfg=cfg,
                default_thre=default_cand_thre,
            )

            pair_thres.append(max(float(current_thre), float(cand_thre)))

        pair_thres = np.asarray(pair_thres, dtype=float)

        sort_idxs = remain_idxs[distances <= pair_thres]

    return keep


def iou_bev(boxes_a, boxes_b):
    """
    Refer: https://github.com/lixiaoyu2000/Poly-MOT/blob/main/geometry/nusc_distance.py
    Info: This function calculates the Intersection over Union (IoU) in the bird's eye view (BEV) between two collections of bounding boxes.
    Parameters:
        input:
            boxes_a: dict, contains:
                - 'np_dets': np.array, shape [n, 14], box attributes (x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label).
                - 'np_dets_bottom_corners': np.array, shape [n, 4, 2], bottom corners of the bounding boxes.
            boxes_b: dict, contains:
                - 'np_dets': np.array, shape [n, 14].
                - 'np_dets_bottom_corners': np.array, shape [n, 4, 2].
        output:
            ioubev: np.array, IoU values for the BEV projection between the two collections of boxes.
    """
    assert (
        "np_dets" in boxes_a and "np_dets_bottom_corners" in boxes_a
    ), "must contain specified keys"
    assert (
        "np_dets" in boxes_b and "np_dets_bottom_corners" in boxes_b
    ), "must contain specified keys"

    infos_a, infos_b = boxes_a["np_dets"], boxes_b["np_dets"]
    bcs_a, bcs_b = boxes_a["np_dets_bottom_corners"], boxes_b["np_dets_bottom_corners"]

    if infos_a.ndim == 1:
        infos_a, bcs_a = infos_a[None, :], bcs_a[None, :]
    if infos_b.ndim == 1:
        infos_b, bcs_b = infos_b[None, :], bcs_b[None, :]
    assert infos_a.shape[1] == 14 and infos_b.shape[1] == 14, "dim must be 14"

    bool_mask, seq_mask = mask_between_boxes(infos_a[:, -1], infos_b[:, -1])
    bool_mask, _ = logical_or_mask(bool_mask, seq_mask, boxes_a, boxes_b)

    wlh_a, wlh_b = expand_dims(infos_a[:, 3:6], len(infos_b), 1), expand_dims(
        infos_b[:, 3:6], len(infos_a), 0
    )
    wa, la, wb, lb = wlh_a[:, :, 0], wlh_a[:, :, 1], wlh_b[:, :, 0], wlh_b[:, :, 1]

    polys_a, polys_b = [Polygon(bc_a) for bc_a in bcs_a], [
        Polygon(bc_b) for bc_b in bcs_b
    ]

    inter_areas = loop_inter(polys_a, polys_b, bool_mask)
    union_areas = wa * la + wb * lb - inter_areas

    ioubev = inter_areas / union_areas
    ioubev[bool_mask] = -np.inf

    return ioubev


def mask_between_boxes(labels_a, labels_b):
    """
    Refer: https://github.com/lixiaoyu2000/Poly-MOT/
    Info: This function creates a mask matrix comparing two sets of labels, indicating whether labels are different or the same.
    Parameters:
        input:
            labels_a: np.array, labels of the first collection.
            labels_b: np.array, labels of the second collection.
        output:
            mask: np.array[bool], matrix where True denotes different labels and False denotes the same.
            flattened_mask: np.array, flattened version of the mask.
    """
    mask = labels_a.reshape(-1, 1).repeat(len(labels_b), axis=1) != labels_b.reshape(
        1, -1
    ).repeat(len(labels_a), axis=0)
    return mask, mask.reshape(-1)


def logical_or_mask(mask, seq_mask, boxes_a, boxes_b):
    """
    Refer: https://github.com/lixiaoyu2000/Poly-MOT/
    Info: This function merges masks from two collections of boxes. It applies a logical OR operation on the mask matrices, marking True where boxes are considered invalid.
    Parameters:
        input:
            mask: np.array, main mask matrix.
            seq_mask: np.array, sequence mask matrix.
            boxes_a: dict.
            boxes_b: dict.
        output:
            merged_mask: np.array, merged mask matrix after applying logical OR.
            flattened_mask: np.array, flattened version of the merged mask.
    """
    if "mask" in boxes_b or "mask" in boxes_a:
        if "mask" in boxes_b and "mask" in boxes_a:
            mask_ab = np.logical_or(boxes_a["mask"], boxes_b["mask"])
        elif "mask" in boxes_b:
            mask_ab = boxes_b["mask"]
        elif "mask" in boxes_a:
            mask_ab = boxes_a["mask"]
        else:
            raise Exception("cannot be happened")
        mask = np.logical_or(mask, mask_ab)
        return mask, mask.reshape(-1)
    else:
        return mask, seq_mask


def expand_dims(array, expand_len, dim):
    """
    Refer: https://github.com/lixiaoyu2000/Poly-MOT/
    """
    return np.expand_dims(array, dim).repeat(expand_len, axis=dim)


def loop_inter(polys1, polys2, mask):
    """
    Refer: https://github.com/lixiaoyu2000/Poly-MOT/
    Info: This function calculates the intersection area between two collections of polygons, skipping invalid comparisons based on a mask.
    Parameters:
        input:
            polys1: List[Polygon], first collection of polygons.
            polys2: List[Polygon], second collection of polygons.
            mask: np.array[bool], matrix where True denotes invalid comparisons and False denotes valid comparisons.
        output:
            inters: np.array, matrix of intersection areas between the two polygon collections.
    """
    inters = np.zeros_like(mask, float)
    for i, reca in enumerate(polys1):
        for j, recb in enumerate(polys2):
            inters[i, j] = reca.intersection(recb).area if not mask[i, j] else 0
    return inters
