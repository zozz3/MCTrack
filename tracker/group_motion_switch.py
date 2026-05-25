import numpy as np


def _angle_diff_rad(a, b):
    """
    返回两个角度之间的最小差值，范围 [0, pi]
    """
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(diff)


def _circular_mean(angles):
    """
    计算角度均值，避免 179 度和 -179 度被算成差很大
    """
    angles = np.asarray(angles, dtype=float)
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    return np.arctan2(sin_mean, cos_mean)


def _get_xy_from_bbox(bbox):
    """
    从 bbox 中取全局 xy 坐标
    """
    if hasattr(bbox, "global_xyz"):
        xyz = np.asarray(bbox.global_xyz, dtype=float)
        return xyz[:2]
    return None


def _get_velocity_from_bbox(bbox):
    """
    优先使用 bbox.global_velocity，如果没有就返回 None
    """
    if hasattr(bbox, "global_velocity"):
        v = np.asarray(bbox.global_velocity, dtype=float)
        if v.shape[0] >= 2:
            return v[:2]
    return None


def _get_motion_from_traj(traj):
    """
    从轨迹里提取运动方向和速度。
    优先用当前 bbox 的 global_velocity；
    如果没有速度，则用最近两个 bbox 的位置差估计方向。
    """
    if not hasattr(traj, "bboxes") or len(traj.bboxes) == 0:
        return None

    cur_bbox = traj.bboxes[-1]

    vel = _get_velocity_from_bbox(cur_bbox)
    if vel is not None:
        speed = float(np.linalg.norm(vel))
        if speed > 1e-6:
            yaw = float(np.arctan2(vel[1], vel[0]))
            return yaw, speed

    if len(traj.bboxes) >= 2:
        p1 = _get_xy_from_bbox(traj.bboxes[-2])
        p2 = _get_xy_from_bbox(traj.bboxes[-1])
        if p1 is not None and p2 is not None:
            diff = p2 - p1
            speed = float(np.linalg.norm(diff))
            if speed > 1e-6:
                yaw = float(np.arctan2(diff[1], diff[0]))
                return yaw, speed

    if hasattr(cur_bbox, "global_yaw"):
        yaw = float(cur_bbox.global_yaw)
        return yaw, 0.0

    return None


def detect_group_motion(trajs, cfg):
    """
    判断当前帧是否存在群体运动现象。

    返回：
        use_group_motion: bool
        info: dict
    """
    gm_cfg = cfg.get("GROUP_MOTION", {})
    if not gm_cfg.get("ENABLE", False):
        return False, {
            "reason": "disabled",
            "valid_tracks": 0,
            "dir_cons_ratio": 0.0,
            "dir_std_deg": 999.0,
        }

    min_tracks = gm_cfg.get("MIN_TRACKS", 4)
    min_speed = gm_cfg.get("MIN_SPEED", 0.5)
    dir_std_thre_deg = gm_cfg.get("DIR_STD_THRE_DEG", 25.0)
    dir_cons_ratio_thre = gm_cfg.get("DIR_CONS_RATIO", 0.65)

    angles = []
    speeds = []

    for traj in trajs:
        motion = _get_motion_from_traj(traj)
        if motion is None:
            continue

        yaw, speed = motion

        # 速度太小，不认为有可靠运动方向
        if speed < min_speed:
            continue

        angles.append(yaw)
        speeds.append(speed)

    valid_tracks = len(angles)

    if valid_tracks < min_tracks:
        return False, {
            "reason": "too_few_valid_tracks",
            "valid_tracks": valid_tracks,
            "dir_cons_ratio": 0.0,
            "dir_std_deg": 999.0,
        }

    angles = np.asarray(angles, dtype=float)

    main_dir = _circular_mean(angles)

    diffs = np.asarray([_angle_diff_rad(a, main_dir) for a in angles], dtype=float)
    diffs_deg = diffs * 180.0 / np.pi

    dir_std_deg = float(np.std(diffs_deg))
    dir_cons_ratio = float(np.mean(diffs_deg <= dir_std_thre_deg))

    use_group_motion = (
        dir_std_deg <= dir_std_thre_deg
        and dir_cons_ratio >= dir_cons_ratio_thre
    )

    reason = "group_motion_on" if use_group_motion else "group_motion_off"

    return use_group_motion, {
        "reason": reason,
        "valid_tracks": valid_tracks,
        "main_dir_deg": float(main_dir * 180.0 / np.pi),
        "dir_std_deg": dir_std_deg,
        "dir_cons_ratio": dir_cons_ratio,
        "avg_speed": float(np.mean(speeds)) if len(speeds) > 0 else 0.0,
    }