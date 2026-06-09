# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import copy

from tracker.matching import *
from tracker.trajectory import Trajectory
from tracker.hsm_ltm import hsm_after_unmatch_update
from tracker.group_motion_switch import detect_group_motion
from utils.utils import norm_realative_radian
from tracker.detection_quality_filter import (
    soft_ignore_full_image_dets,
    is_highly_occluded_newborn_det,
    output_traj_nms,
    get_det_score_safe,
)


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


def _get_bbox_score(bbox):
    for name in ["det_score", "score", "global_score", "confidence"]:
        if hasattr(bbox, name):
            try:
                return float(getattr(bbox, name))
            except Exception:
                pass
    return 1.0


def _get_bbox_dist(bbox):
    if hasattr(bbox, "global_xyz"):
        xyz = np.asarray(bbox.global_xyz, dtype=float)
        return float(np.linalg.norm(xyz[:2]))
    if hasattr(bbox, "xyz"):
        xyz = np.asarray(bbox.xyz, dtype=float)
        return float(np.linalg.norm(xyz[:2]))
    return 0.0


def _get_traj_cls_id(traj, cfg):
    """
    尽量从 traj 或 bbox 里取类别。
    KITTI car 一般就是 0。
    """
    category_map = cfg.get("CATEGORY_MAP_TO_NUMBER", {})

    for obj in [traj, traj.bboxes[-1] if hasattr(traj, "bboxes") and len(traj.bboxes) > 0 else None]:
        if obj is None:
            continue

        for name in ["category", "category_name", "det_name", "name"]:
            if hasattr(obj, name):
                cate = getattr(obj, name)
                if cate in category_map:
                    return int(category_map[cate])

        for name in ["category_id", "label", "cls_id"]:
            if hasattr(obj, name):
                try:
                    return int(getattr(obj, name))
                except Exception:
                    pass

    return 0


def should_output_traj_bbox_exp3(traj, bbox, cfg):
    """
    实验三：低质量轨迹输出抑制。
    注意：只决定是否写入结果，不删除轨迹，不影响内部状态。
    """
    out_cfg = cfg.get("OUTPUT_FILTER", {})
    if not out_cfg.get("ENABLE", False):
        return True

    cls_id = _get_traj_cls_id(traj, cfg)

    min_len = int(_cfg_by_cls(out_cfg.get("MIN_OUTPUT_TRACK_LENGTH", {0: 3}), cls_id, 3))
    min_score = float(_cfg_by_cls(out_cfg.get("MIN_OUTPUT_SCORE", {0: 0.55}), cls_id, 0.55))

    far_dist = float(_cfg_by_cls(out_cfg.get("FAR_DIST", {0: 35.0}), cls_id, 35.0))
    far_min_score = float(_cfg_by_cls(out_cfg.get("FAR_MIN_OUTPUT_SCORE", {0: 0.70}), cls_id, 0.70))

    max_lost_output = int(_cfg_by_cls(out_cfg.get("MAX_LOST_OUTPUT_LENGTH", {0: 1}), cls_id, 1))

    track_len = len(traj.bboxes) if hasattr(traj, "bboxes") else 1
    score = _get_bbox_score(bbox)
    dist = _get_bbox_dist(bbox)

    unmatched_length = 0
    for name in ["unmatched_length", "unmatch_length", "lost_time", "lost_frame", "time_since_update"]:
        if hasattr(traj, name):
            try:
                unmatched_length = int(getattr(traj, name))
                break
            except Exception:
                pass

    # 规则1：短轨迹 + 低分，不输出
    if track_len < min_len and score < min_score:
        return False

    # 规则2：远距离短轨迹更严格
    if dist > far_dist and track_len < min_len and score < far_min_score:
        return False

    # 规则3：丢失太久的预测框不输出
    # 注意：只是当前帧不输出，不删除轨迹
    if unmatched_length > max_lost_output:
        return False

    return True


class Base3DTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.current_frame_id = None
        self.all_trajs = {}
        self.all_dead_trajs = {}
        self.id_seed = 0
        self.cache_size = 3
        self.track_id_counter = 0
        self.exp5a_stats = {
            "checked": 0,
            "ref_update": 0,
            "terminated": 0,
            "no_bbox_image": 0,
            "no_ref_when_checked": 0,
            "skip_fake": 0,
            "boundary_checked": 0,
            "small_ratio_candidate": 0,
            "min_remain_ratio": 999.0,
        }
        # ------------------------------------------------------------
        # EXP5A STATIC_EGO_EXIT:
        # 仅用于“静止车辆出界判断”。
        # 不修改 GROUP_MOTION / HSM_LTM 的群体运动估计逻辑。
        # 核心思想：静止车自身速度为 0 时，用自车/相机位姿变化
        # 重新投影最后一次完整 3D 参考框，判断是否已经离开图像视野。
        # ------------------------------------------------------------
        self.exp5a_static_ego_stats = {
            "checked": 0,
            # preserved 是真正“无检测框时继续输出预测框”的次数
            "preserved": 0,
            # out_of_view_stop 是裁剪后剩余面积 <= 阈值，停止输出预测框的次数
            "out_of_view_stop": 0,
            "terminated": 0,
            "not_out": 0,
            "skip_disabled": 0,
            "skip_not_static": 0,
            "skip_no_ref3d": 0,
            "skip_no_ref2d": 0,
            "skip_not_output": 0,
            "skip_finished": 0,
            "skip_no_transform": 0,
            "projection_failed": 0,
            "min_remain_ratio": 999.0,
        }
        self.full_image_bbox_ignore_stats = {
            "checked_frames": 0,
            "ignored": 0,
        }
        self.newborn_occlusion_suppress_stats = {
            "checked": 0,
            "suppressed": 0,
        }
        self.output_traj_nms_stats = {
            "checked_frames": 0,
            "suppressed": 0,
        }
        # ------------------------------------------------------------
        # EXP5B-v2 Track-aware low-score observation gate.
        #
        # 核心边界：
        # 1) strong detection 正常参与关联，并允许初始化新轨迹；
        # 2) weak observation 只用于补救稳定老轨迹，不允许初始化新轨迹；
        # 3) low-score newborn / far-away weak detections 被拒绝，避免制造 ghost tracks。
        # ------------------------------------------------------------
        self.track_aware_low_score_gate_stats = {
            "frames": 0,
            "checked_dets": 0,
            "strong": 0,
            "weak": 0,
            "rejected_low_score": 0,
            "rejected_not_near_track": 0,
            "rejected_no_center": 0,
            "weak_matched": 0,
            "weak_unmatched": 0,
        }
        # ------------------------------------------------------------
        # EXP7D Reliability-aware selective motion handling.
        # high   : matched 正常 Kalman update；unmatched 只做基础 Kalman fake bbox。
        # medium : matched 正常 update；unmatched 在基础 Kalman 后进入 MOTION_STATE / HSM_LTM。
        # weak   : matched / unmatched 都不破坏基础生命周期，只在输出 NMS 阶段允许被压制。
        # ------------------------------------------------------------
        self.reliability_router_stats = {
            "matched_checked": 0,
            "high_kalman": 0,
            "medium_motion_state": 0,
            "medium_static": 0,
            "medium_moving": 0,
            "weak_nms_suppress": 0,
            "weak_skip_update": 0,
            "unmatched_high_kalman_only": 0,
            "unmatched_medium_motion_state_hsm": 0,
            "unmatched_weak_nms_only": 0,
            "unmatched_weak_motion_state_hsm": 0,
            "unmatched_unknown_default_medium": 0,
        }

    def _get_exp5a_cfg(self):
        return self.cfg.get("EXP5A_OUT_OF_VIEW", {})

    def _exp5a_enabled(self):
        return bool(self._get_exp5a_cfg().get("ENABLE", True))

    def _get_static_ego_exit_cfg_exp5a(self):
        """
        静止车辆自车运动出界配置。
        该配置只服务 EXP5A 的静止车辆出界判断，不影响原有 GROUP_MOTION。
        """
        cfg = self._get_exp5a_cfg()
        sub_cfg = cfg.get("STATIC_EGO_EXIT", {})
        if not isinstance(sub_cfg, dict):
            sub_cfg = {}
        return sub_cfg

    def _static_ego_exit_enabled_exp5a(self):
        cfg = self._get_static_ego_exit_cfg_exp5a()
        # 默认开启：因为这个函数只会在“静止 unmatched 轨迹”里被调用，
        # 且投影失败时不会删除轨迹。
        return bool(cfg.get("ENABLE", True))

    def _as_matrix_exp5a(self, value, shape=None):
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            return None
        if arr.ndim != 2:
            return None
        if shape is not None and arr.shape != shape:
            return None
        if not np.all(np.isfinite(arr)):
            return None
        return arr

    def _get_matrix_by_keys_exp5a(self, data, keys):
        if not isinstance(data, dict):
            return None
        lower_map = {str(k).lower(): k for k in data.keys()}
        for key in keys:
            real_key = lower_map.get(str(key).lower(), None)
            if real_key is None:
                continue
            mat = self._as_matrix_exp5a(data.get(real_key))
            if mat is not None:
                return mat
        return None

    def _safe_inv_exp5a(self, mat):
        mat = self._as_matrix_exp5a(mat)
        if mat is None:
            return None
        try:
            return np.linalg.inv(mat)
        except Exception:
            return None

    def _to_homo_exp5a(self, points_xyz):
        pts = np.asarray(points_xyz, dtype=float)
        ones = np.ones((pts.shape[0], 1), dtype=float)
        return np.concatenate([pts, ones], axis=1)

    def _transform_points_exp5a(self, points_xyz, mat):
        mat = self._as_matrix_exp5a(mat)
        if mat is None:
            return None
        pts_h = self._to_homo_exp5a(points_xyz)
        if mat.shape == (4, 4):
            out = (mat @ pts_h.T).T
            return out[:, :3]
        if mat.shape == (3, 4):
            out = (mat @ pts_h.T).T
            return out[:, :3]
        if mat.shape == (3, 3):
            out = (mat @ np.asarray(points_xyz, dtype=float).T).T
            return out[:, :3]
        return None

    def _project_camera_points_exp5a(self, points_cam, camera2image):
        """
        将 camera 坐标系下的 3D 点投影到图像平面。

        MCTrack BaseVersion 的相机投影字段叫 camera2image，
        不是 camera_intrinsic。这里同时兼容：
        - 3x3 内参矩阵 K
        - 3x4 camera2image 投影矩阵
        - 4x4 camera2image 齐次投影矩阵
        """
        P = self._as_matrix_exp5a(camera2image)
        pts = np.asarray(points_cam, dtype=float)
        if P is None:
            return None, None
        if pts.ndim != 2 or pts.shape[1] < 3:
            return None, None

        depth = pts[:, 2]
        valid = depth > 1e-4
        if not np.any(valid):
            return np.empty((0, 2), dtype=float), depth

        pts_valid = pts[valid, :3]

        if P.shape == (3, 3):
            proj = (P @ pts_valid.T).T
        elif P.shape[0] >= 3 and P.shape[1] >= 4:
            pts_h = self._to_homo_exp5a(pts_valid)
            proj = (P[:3, :4] @ pts_h.T).T
        else:
            return None, None

        if proj.shape[1] < 3:
            return None, None

        uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-6)
        return uv, depth[valid]

    def _get_bbox_global_box_exp5a(self, bbox):
        """
        读取 bbox 的全局 3D 框 [x, y, z, l, w, h, yaw]。
        优先使用 fusion，因为它是当前 tracker 内部滤波后的稳定状态。
        """
        for name in ["global_xyz_lwh_yaw_fusion", "global_xyz_lwh_yaw"]:
            if not hasattr(bbox, name):
                continue
            value = getattr(bbox, name)
            if value is None:
                continue
            try:
                arr = np.asarray(value, dtype=float).reshape(-1)
            except Exception:
                continue
            if arr.shape[0] >= 7 and np.all(np.isfinite(arr[:7])):
                return arr[:7].copy()
        return None

    def _box3d_corners_global_exp5a(self, box3d):
        """
        根据全局 3D 框生成 8 个角点。
        box3d = [x, y, z, l, w, h, yaw]
        """
        arr = np.asarray(box3d, dtype=float).reshape(-1)
        if arr.shape[0] < 7:
            return None
        x, y, z, l, w, h, yaw = arr[:7]
        if l <= 0 or w <= 0 or h <= 0:
            return None

        # 以 bbox 中心为中心，z 方向上下各 h/2。
        x_c = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
        y_c = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
        z_c = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rot = np.array([
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        corners = np.stack([x_c, y_c, z_c], axis=0)
        corners = (rot @ corners).T
        corners += np.array([x, y, z], dtype=float)
        return corners

    def _get_camera_info_exp5a(self, frame_info=None, camera_type=None):
        if frame_info is None or not hasattr(frame_info, "transform_matrix"):
            return None, None, None
        transform_matrix = getattr(frame_info, "transform_matrix")
        if not isinstance(transform_matrix, dict):
            return transform_matrix, None, None

        cameras = transform_matrix.get("cameras_transform_matrix", None)
        if not isinstance(cameras, dict) or len(cameras) == 0:
            return transform_matrix, None, None

        cam_info = None
        cam_key = None
        if camera_type in cameras:
            cam_key = camera_type
            cam_info = cameras[camera_type]
        else:
            # camera_type 不存在时，退回第一个相机，保证不会因为字段缺失直接报错。
            cam_key = list(cameras.keys())[0]
            cam_info = cameras[cam_key]

        if not isinstance(cam_info, dict):
            return transform_matrix, cam_key, None
        return transform_matrix, cam_key, cam_info

    def _compose_global_to_camera_exp5a(self, transform_matrix, cam_info):
        """
        尽量从 MCTrack/KITTI 常见 transform_matrix 结构中拼出 global -> camera。
        支持：
        1. cam_info 直接提供 global2camera；
        2. root 提供 global2ego，cam_info 提供 ego2camera；
        3. root 提供 global2lidar，cam_info 提供 lidar2camera；
        4. 对应反向矩阵存在时自动求逆。
        """
        if not isinstance(transform_matrix, dict) or not isinstance(cam_info, dict):
            return None

        direct = self._get_matrix_by_keys_exp5a(
            cam_info,
            ["global2camera", "global_to_camera", "global2cam", "global_to_cam"],
        )
        if direct is not None:
            return direct

        # global -> ego -> camera
        global2ego = self._get_matrix_by_keys_exp5a(
            transform_matrix,
            ["global2ego", "global_to_ego"],
        )
        if global2ego is None:
            ego2global = self._get_matrix_by_keys_exp5a(
                transform_matrix,
                ["ego2global", "ego_to_global"],
            )
            global2ego = self._safe_inv_exp5a(ego2global)

        ego2camera = self._get_matrix_by_keys_exp5a(
            cam_info,
            ["ego2camera", "ego_to_camera", "ego2cam", "ego_to_cam"],
        )
        if ego2camera is None:
            camera2ego = self._get_matrix_by_keys_exp5a(
                cam_info,
                ["camera2ego", "camera_to_ego", "cam2ego", "cam_to_ego"],
            )
            ego2camera = self._safe_inv_exp5a(camera2ego)

        if global2ego is not None and ego2camera is not None:
            return ego2camera @ global2ego

        # global -> lidar -> camera
        global2lidar = self._get_matrix_by_keys_exp5a(
            transform_matrix,
            ["global2lidar", "global_to_lidar"],
        )
        if global2lidar is None:
            lidar2global = self._get_matrix_by_keys_exp5a(
                transform_matrix,
                ["lidar2global", "lidar_to_global"],
            )
            global2lidar = self._safe_inv_exp5a(lidar2global)

        lidar2camera = self._get_matrix_by_keys_exp5a(
            cam_info,
            ["lidar2camera", "lidar_to_camera", "lidar2cam", "lidar_to_cam"],
        )
        if lidar2camera is None:
            camera2lidar = self._get_matrix_by_keys_exp5a(
                cam_info,
                ["camera2lidar", "camera_to_lidar", "cam2lidar", "cam_to_lidar"],
            )
            lidar2camera = self._safe_inv_exp5a(camera2lidar)

        if global2lidar is not None and lidar2camera is not None:
            return lidar2camera @ global2lidar

        return None

    def _project_global_box_to_image_exp5a(self, box3d, frame_info=None, camera_type=None):
        """
        将保存的静止目标全局 3D 框按当前帧自车/相机位姿投影到图像。
        返回：xyxy, image_w, image_h, camera_key
        如果无法获得投影链路，返回 None。
        """
        corners_global = self._box3d_corners_global_exp5a(box3d)
        if corners_global is None:
            return None

        transform_matrix, cam_key, cam_info = self._get_camera_info_exp5a(frame_info, camera_type)
        if transform_matrix is None or cam_info is None:
            return None

        # 读取图像尺寸
        image_w, image_h = self._get_image_shape_exp5a(bbox=None, frame_info=frame_info)
        if isinstance(cam_info, dict) and "image_shape" in cam_info:
            shape = cam_info.get("image_shape")
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                a, b = int(shape[0]), int(shape[1])
                if a <= b:
                    image_h, image_w = a, b
                else:
                    image_w, image_h = a, b

        # 方式 1：直接 global -> image 投影矩阵
        P_global2image = self._get_matrix_by_keys_exp5a(
            cam_info,
            ["global2image", "global_to_image", "global2img", "global_to_img"],
        )
        if P_global2image is not None and P_global2image.shape in [(3, 4), (4, 4)]:
            pts_h = self._to_homo_exp5a(corners_global)
            proj = (P_global2image @ pts_h.T).T
            if proj.shape[1] >= 3:
                valid = proj[:, 2] > 1e-4
                if not np.any(valid):
                    return [1e9, 1e9, 1e9 + 1.0, 1e9 + 1.0], image_w, image_h, cam_key
                uv = proj[valid, :2] / np.maximum(proj[valid, 2:3], 1e-6)
                return [float(np.min(uv[:, 0])), float(np.min(uv[:, 1])), float(np.max(uv[:, 0])), float(np.max(uv[:, 1]))], image_w, image_h, cam_key

        # MCTrack BaseVersion 中标准字段是 camera2image，README 中明确给出：
        # cameras_transform_matrix/CAM_*/camera2image。
        # 这里也兼容少数实现里可能出现的 intrinsic/K 命名。
        camera2image = self._get_matrix_by_keys_exp5a(
            cam_info,
            [
                "camera2image", "camera_to_image", "cam2image", "cam_to_image",
                "camera_intrinsic", "cam_intrinsic", "intrinsic", "K", "camera_K", "cam_K",
            ],
        )
        if camera2image is None:
            if bool(self._get_static_ego_exit_cfg_exp5a().get("DEBUG", False)):
                print(
                    "[EXP5A_STATIC_EGO_EXIT_NO_CAMERA2IMAGE]",
                    "cam_key=", cam_key,
                    "cam_info_keys=", list(cam_info.keys()) if isinstance(cam_info, dict) else None,
                )
            return None

        global2camera = self._compose_global_to_camera_exp5a(transform_matrix, cam_info)
        if global2camera is None:
            if bool(self._get_static_ego_exit_cfg_exp5a().get("DEBUG", False)):
                print(
                    "[EXP5A_STATIC_EGO_EXIT_NO_GLOBAL2CAMERA]",
                    "root_keys=", list(transform_matrix.keys()) if isinstance(transform_matrix, dict) else None,
                    "cam_key=", cam_key,
                    "cam_info_keys=", list(cam_info.keys()) if isinstance(cam_info, dict) else None,
                )
            return None

        corners_cam = self._transform_points_exp5a(corners_global, global2camera)
        if corners_cam is None:
            return None

        uv, depths = self._project_camera_points_exp5a(corners_cam, camera2image)
        if uv is None:
            return None
        if uv.shape[0] == 0:
            # 全部在相机后方，视为完全出界。
            return [1e9, 1e9, 1e9 + 1.0, 1e9 + 1.0], image_w, image_h, cam_key

        x1, y1 = np.min(uv[:, 0]), np.min(uv[:, 1])
        x2, y2 = np.max(uv[:, 0]), np.max(uv[:, 1])
        return [float(x1), float(y1), float(x2), float(y2)], image_w, image_h, cam_key

    def _remain_ratio_for_projected_xyxy_exp5a(self, xyxy, image_w, image_h, ref_area):
        x1, y1, x2, y2 = np.asarray(xyxy, dtype=float).reshape(4)
        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0
        ix1 = max(0.0, x1)
        iy1 = max(0.0, y1)
        ix2 = min(float(image_w - 1), x2)
        iy2 = min(float(image_h - 1), y2)
        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        inter_area = inter_w * inter_h
        remain_ratio = float(inter_area / max(float(ref_area), 1e-6))
        return remain_ratio, float(inter_area)

    def _update_static_ego_reference_exp5a(self, traj, bbox, frame_info=None, source="matched"):
        """
        只在更新“历史最大完整 2D 框”时同步保存对应的 3D 全局框。
        后续静止目标 unmatched 后，用当前帧自车/相机位姿重投影这个 3D 框。
        """
        box3d = self._get_bbox_global_box_exp5a(bbox)
        if box3d is None:
            return False
        traj.exp5a_static_ego_ref_global_box = box3d.copy()
        traj.exp5a_static_ego_ref_frame = getattr(frame_info, "frame_id", -1) if frame_info is not None else -1
        traj.exp5a_static_ego_ref_camera_type = self._get_camera_type_exp5a(bbox)
        traj.exp5a_static_ego_ref_source = source
        # 保存一份 transform 仅用于调试，不参与正常 GROUP_MOTION。
        if frame_info is not None and hasattr(frame_info, "transform_matrix"):
            try:
                traj.exp5a_static_ego_ref_transform_matrix = copy.deepcopy(frame_info.transform_matrix)
            except Exception:
                traj.exp5a_static_ego_ref_transform_matrix = None
        return True

    def _project_global_center_to_image_exp5a(self, box3d, frame_info=None, camera_type=None):
        """
        只投影静止目标历史参考 3D 框的中心点。
        注意：这里不再投影 3D 八角点生成 2D 大框，避免自车靠近时投影框异常放大。
        后续用历史最大 2D 参考框的宽高，在这个中心点上重建预测框。
        """
        arr = np.asarray(box3d, dtype=float).reshape(-1)
        if arr.shape[0] < 3 or not np.all(np.isfinite(arr[:3])):
            return None

        center_global = arr[:3].reshape(1, 3)

        transform_matrix, cam_key, cam_info = self._get_camera_info_exp5a(frame_info, camera_type)
        if transform_matrix is None or cam_info is None:
            return None

        image_w, image_h = self._get_image_shape_exp5a(bbox=None, frame_info=frame_info)
        if isinstance(cam_info, dict) and "image_shape" in cam_info:
            shape = cam_info.get("image_shape")
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                a, b = int(shape[0]), int(shape[1])
                if a <= b:
                    image_h, image_w = a, b
                else:
                    image_w, image_h = a, b

        # 方式 1：直接 global -> image
        P_global2image = self._get_matrix_by_keys_exp5a(
            cam_info,
            ["global2image", "global_to_image", "global2img", "global_to_img"],
        )
        if P_global2image is not None and P_global2image.shape in [(3, 4), (4, 4)]:
            pts_h = self._to_homo_exp5a(center_global)
            proj = (P_global2image @ pts_h.T).T
            if proj.shape[1] >= 3 and proj[0, 2] > 1e-4:
                uv = proj[0, :2] / max(float(proj[0, 2]), 1e-6)
                return [float(uv[0]), float(uv[1])], image_w, image_h, cam_key
            return None

        camera2image = self._get_matrix_by_keys_exp5a(
            cam_info,
            [
                "camera2image", "camera_to_image", "cam2image", "cam_to_image",
                "camera_intrinsic", "cam_intrinsic", "intrinsic", "K", "camera_K", "cam_K",
            ],
        )
        if camera2image is None:
            return None

        global2camera = self._compose_global_to_camera_exp5a(transform_matrix, cam_info)
        if global2camera is None:
            return None

        center_cam = self._transform_points_exp5a(center_global, global2camera)
        if center_cam is None or center_cam.shape[0] == 0:
            return None
        if float(center_cam[0, 2]) <= 1e-4:
            return None

        uv, depths = self._project_camera_points_exp5a(center_cam, camera2image)
        if uv is None or uv.shape[0] == 0:
            return None

        return [float(uv[0, 0]), float(uv[0, 1])], image_w, image_h, cam_key

    def _build_ref_xyxy_at_center_exp5a(self, ref_xyxy, center_uv):
        """
        使用历史最大完整 2D 检测框的宽高，在当前投影中心点处重建预测框。
        这是用户要求的核心：不是输出重新投影出来的大框，而是移动之前保存的最大检测框。
        """
        ref = np.asarray(ref_xyxy, dtype=float).reshape(4)
        u, v = float(center_uv[0]), float(center_uv[1])
        ref_w = max(0.0, float(ref[2] - ref[0]))
        ref_h = max(0.0, float(ref[3] - ref[1]))
        if ref_w <= 1e-6 or ref_h <= 1e-6:
            return None
        return [
            u - ref_w * 0.5,
            v - ref_h * 0.5,
            u + ref_w * 0.5,
            v + ref_h * 0.5,
        ]

    def _clip_xyxy_to_image_exp5a(self, xyxy, image_w, image_h):
        x1, y1, x2, y2 = np.asarray(xyxy, dtype=float).reshape(4)
        cx1 = max(0.0, min(float(image_w - 1), x1))
        cy1 = max(0.0, min(float(image_h - 1), y1))
        cx2 = max(0.0, min(float(image_w - 1), x2))
        cy2 = max(0.0, min(float(image_h - 1), y2))
        if cx2 <= cx1 or cy2 <= cy1:
            return [cx1, cy1, cx1, cy1], 0.0
        area = float((cx2 - cx1) * (cy2 - cy1))
        return [float(cx1), float(cy1), float(cx2), float(cy2)], area

    def _set_bbox_image_xyxy_exp5a(self, bbox, xyxy):
        """
        将裁剪后的 2D 预测框写回 bbox。
        若某些字段不存在则自动跳过，避免破坏原始 MCTrack 结构。
        """
        arr = np.asarray(xyxy, dtype=float).reshape(4).copy()
        for name in ["x1y1x2y2", "x1y1x2y2_fusion", "x1y1x2y2_predict"]:
            if hasattr(bbox, name):
                try:
                    setattr(bbox, name, arr.copy())
                except Exception:
                    pass

    def _apply_static_ego_exit_exp5a(self, traj, frame_info=None, source="unmatched_static"):
        """
        EXP5A STATIC_EGO_PREDICT：自车运动引导的静止目标 2D 参考框延伸。

        正确逻辑：
        1. matched 阶段保存历史最大完整 2D 检测框 ref_xyxy / ref_area；
        2. unmatched 且静止时，只投影历史 3D 参考框的中心点；
        3. 用历史最大 2D 框的宽高，在当前中心点重建 pred_xyxy；
        4. 将 pred_xyxy 裁剪到图像边界，得到 clip_xyxy；
        5. clip_area / ref_area > 10%：继续输出裁剪后的预测框；
        6. clip_area / ref_area <= 10%：停止输出这个静止预测框。

        注意：不再使用 3D 八角点投影出的巨大 2D 框作为输出框。
        """
        if not self._exp5a_enabled() or not self._static_ego_exit_enabled_exp5a():
            self.exp5a_static_ego_stats["skip_disabled"] = self.exp5a_static_ego_stats.get("skip_disabled", 0) + 1
            return False
        if traj is None or len(traj.bboxes) == 0:
            return False

        if bool(getattr(traj, "exp5a_static_ego_finished", False)):
            self.exp5a_static_ego_stats["skip_finished"] = self.exp5a_static_ego_stats.get("skip_finished", 0) + 1
            return False

        cfg = self._get_exp5a_cfg()
        ego_cfg = self._get_static_ego_exit_cfg_exp5a()
        cls_id = getattr(traj, "category_num", 0)
        bbox = traj.bboxes[-1]

        self.exp5a_static_ego_stats["checked"] = self.exp5a_static_ego_stats.get("checked", 0) + 1

        # 静止延伸只应该服务于已经确认输出过的老轨迹，避免给从未确认的新生轨迹制造 fake bbox。
        if bool(ego_cfg.get("REQUIRE_ALREADY_OUTPUT", True)) and not bool(getattr(traj, "is_output", False)):
            self.exp5a_static_ego_stats["skip_not_output"] = self.exp5a_static_ego_stats.get("skip_not_output", 0) + 1
            return False

        ref_box3d = getattr(traj, "exp5a_static_ego_ref_global_box", None)
        ref_xyxy = getattr(traj, "exp5a_full_vehicle_ref_xyxy", None)
        ref_area = float(getattr(traj, "exp5a_full_vehicle_ref_area", 0.0))

        if ref_box3d is None or ref_area <= 1e-6:
            self.exp5a_static_ego_stats["skip_no_ref3d"] = self.exp5a_static_ego_stats.get("skip_no_ref3d", 0) + 1
            return False
        if ref_xyxy is None:
            self.exp5a_static_ego_stats["skip_no_ref2d"] = self.exp5a_static_ego_stats.get("skip_no_ref2d", 0) + 1
            return False

        camera_type = getattr(traj, "exp5a_static_ego_ref_camera_type", None)
        center_proj = self._project_global_center_to_image_exp5a(
            ref_box3d,
            frame_info=frame_info,
            camera_type=camera_type,
        )
        if center_proj is None:
            self.exp5a_static_ego_stats["projection_failed"] = self.exp5a_static_ego_stats.get("projection_failed", 0) + 1
            if bool(ego_cfg.get("DEBUG", False)):
                print(
                    "[EXP5A_STATIC_EGO_CENTER_PROJ_FAIL]",
                    "track_id=", traj.track_id,
                    "frame=", getattr(frame_info, "frame_id", -1) if frame_info is not None else -1,
                    "camera_type=", camera_type,
                )
            return False

        center_uv, image_w, image_h, used_cam = center_proj
        pred_xyxy = self._build_ref_xyxy_at_center_exp5a(ref_xyxy, center_uv)
        if pred_xyxy is None:
            self.exp5a_static_ego_stats["skip_no_ref2d"] = self.exp5a_static_ego_stats.get("skip_no_ref2d", 0) + 1
            return False

        clip_xyxy, clip_area = self._clip_xyxy_to_image_exp5a(pred_xyxy, image_w, image_h)
        remain_ratio = float(clip_area / max(float(ref_area), 1e-6))

        self.exp5a_static_ego_stats["min_remain_ratio"] = min(
            float(self.exp5a_static_ego_stats.get("min_remain_ratio", 999.0)),
            float(remain_ratio),
        )

        stop_thre = float(
            _cfg_by_cls(
                ego_cfg.get("STOP_REMAIN_RATIO_THRE", ego_cfg.get("DELETE_REMAIN_RATIO_THRE", cfg.get("REMAIN_RATIO_THRE", {0: 0.10}))),
                cls_id,
                0.10,
            )
        )

        bbox.exp5a_static_ego_checked = True
        bbox.exp5a_static_ego_center_uv = [float(center_uv[0]), float(center_uv[1])]
        bbox.exp5a_static_ego_ref_xyxy = [float(v) for v in np.asarray(ref_xyxy, dtype=float).reshape(4)]
        bbox.exp5a_static_ego_pred_xyxy = [float(v) for v in pred_xyxy]
        bbox.exp5a_static_ego_clip_xyxy = [float(v) for v in clip_xyxy]
        bbox.exp5a_static_ego_remain_ratio = float(remain_ratio)
        bbox.exp5a_static_ego_inter_area = float(clip_area)
        bbox.exp5a_static_ego_ref_area = float(ref_area)
        bbox.exp5a_static_ego_camera = used_cam

        # 裁剪后只剩 <= 10%，说明这辆静止车基本离开当前视野，停止输出预测框。
        # 不在这里强制删除整条轨迹，生命周期交回原 MCTrack。
        if remain_ratio <= stop_thre:
            traj.exp5a_static_ego_finished = True
            bbox.det_score = traj._is_filter_predict_box
            bbox.exp5a_static_ego_preserved = False
            bbox.exp5a_static_ego_stop_output = True
            bbox.exp5a_static_ego_reason = "clipped_ref_box_area_le_threshold_stop_output"
            bbox.exp5a_is_out_of_view = True
            bbox.exp5a_out_view_source = source
            bbox.exp5a_out_view_reason = "static_ego_clipped_ref_box_area_le_threshold"

            self.exp5a_static_ego_stats["out_of_view_stop"] = self.exp5a_static_ego_stats.get("out_of_view_stop", 0) + 1
            self.exp5a_static_ego_stats["terminated"] = self.exp5a_static_ego_stats.get("terminated", 0) + 1

            if bool(ego_cfg.get("DEBUG", False)):
                print(
                    "[EXP5A_STATIC_EGO_REFBOX_STOP]",
                    "track_id=", traj.track_id,
                    "frame=", getattr(frame_info, "frame_id", -1) if frame_info is not None else -1,
                    "remain_ratio=", round(float(remain_ratio), 4),
                    "clip_area=", round(float(clip_area), 2),
                    "ref_area=", round(float(ref_area), 2),
                    "center_uv=", [round(float(center_uv[0]), 1), round(float(center_uv[1]), 1)],
                    "pred_xyxy=", [round(float(v), 1) for v in pred_xyxy],
                    "clip_xyxy=", [round(float(v), 1) for v in clip_xyxy],
                    "image_wh=", [image_w, image_h],
                    "camera=", used_cam,
                )
            return False

        # 还在视野内：输出“历史最大 2D 框移动后再裁剪”的预测框。
        self._set_bbox_image_xyxy_exp5a(bbox, clip_xyxy)

        # 3D 状态保持为静止目标历史参考框，避免 Kalman 0 速度 fake bbox 被其它更新污染。
        try:
            ref_box_arr = np.asarray(ref_box3d, dtype=float).reshape(-1)[:7].copy()
            bbox.global_xyz_lwh_yaw = ref_box_arr.copy()
            bbox.global_xyz_lwh_yaw_fusion = ref_box_arr.copy()
            bbox.global_xyz_lwh_yaw_predict = ref_box_arr.copy()
        except Exception:
            pass

        preserved_score = float(
            _cfg_by_cls(
                ego_cfg.get("PRESERVED_OUTPUT_SCORE", {0: 0.45}),
                cls_id,
                0.45,
            )
        )
        try:
            bbox.det_score = max(float(getattr(bbox, "det_score", 0.0)), preserved_score)
        except Exception:
            bbox.det_score = preserved_score

        bbox.exp5a_static_ego_preserved = True
        bbox.exp5a_static_ego_stop_output = False
        bbox.exp5a_static_ego_reason = "ego_motion_ref_2d_box_clipped_preserve"

        traj.exp5a_static_ego_keep_len = int(getattr(traj, "exp5a_static_ego_keep_len", 0)) + 1
        self.exp5a_static_ego_stats["preserved"] = self.exp5a_static_ego_stats.get("preserved", 0) + 1
        self.exp5a_static_ego_stats["not_out"] = self.exp5a_static_ego_stats.get("not_out", 0) + 1

        if bool(ego_cfg.get("DEBUG", False)):
            print(
                "[EXP5A_STATIC_EGO_REFBOX_KEEP]",
                "track_id=", traj.track_id,
                "frame=", getattr(frame_info, "frame_id", -1) if frame_info is not None else -1,
                "keep_len=", traj.exp5a_static_ego_keep_len,
                "remain_ratio=", round(float(remain_ratio), 4),
                "center_uv=", [round(float(center_uv[0]), 1), round(float(center_uv[1]), 1)],
                "pred_xyxy=", [round(float(v), 1) for v in pred_xyxy],
                "clip_xyxy=", [round(float(v), 1) for v in clip_xyxy],
                "ref_xyxy=", [round(float(v), 1) for v in np.asarray(ref_xyxy, dtype=float).reshape(4)],
                "image_wh=", [image_w, image_h],
                "camera=", used_cam,
            )

        return True

    def _get_bbox_image_xyxy_exp5a(self, bbox):
        """
        实验五A v6：直接使用检测结果自带的 2D bbox。
        这样不再依赖 camera transform / 3D 投影，避免 no_image_info 全部命中的问题。
        MCTrack 的 BBox 初始化时已经从 bbox_image 里保存了 x1y1x2y2。
        """
        for name in ["x1y1x2y2", "x1y1x2y2_fusion", "x1y1x2y2_predict"]:
            if hasattr(bbox, name):
                value = getattr(bbox, name)
                if value is None:
                    continue
                arr = np.asarray(value, dtype=float).reshape(-1)
                if arr.shape[0] >= 4 and np.all(np.isfinite(arr[:4])):
                    x1, y1, x2, y2 = arr[:4].tolist()
                    if x2 > x1 and y2 > y1:
                        return float(x1), float(y1), float(x2), float(y2)
        return None

    def _get_camera_type_exp5a(self, bbox):
        """
        尽量从 BBox 中读取 camera_type。
        MCTrack 的原始 json 是 bbox_image.camera_type，运行时 BBox 通常会展开成 bbox.camera_type。
        """
        if bbox is None:
            return None

        for name in ["camera_type", "camera_name", "cam_type"]:
            if hasattr(bbox, name):
                value = getattr(bbox, name)
                if value is not None:
                    return value

        if hasattr(bbox, "bbox_image"):
            bbox_image = getattr(bbox, "bbox_image")
            if isinstance(bbox_image, dict):
                return bbox_image.get("camera_type", None)

        return None

    def _get_image_shape_exp5a(self, bbox=None, frame_info=None):
        """
        优先从当前帧 transform_matrix.cameras_transform_matrix[camera_type].image_shape 读取真实图像尺寸。
        只有读取不到时，才退回 yaml 里的 IMAGE_WIDTH / IMAGE_HEIGHT。
        """
        cfg = self._get_exp5a_cfg()

        if frame_info is not None and hasattr(frame_info, "transform_matrix"):
            transform_matrix = getattr(frame_info, "transform_matrix")
            if isinstance(transform_matrix, dict):
                cameras = transform_matrix.get("cameras_transform_matrix", None)
                cam_type = self._get_camera_type_exp5a(bbox)

                if isinstance(cameras, dict) and len(cameras) > 0:
                    cam_info = None
                    if cam_type in cameras:
                        cam_info = cameras[cam_type]
                    elif cam_type is None:
                        # 没有 camera_type 时，退回第一个相机。
                        cam_info = cameras[list(cameras.keys())[0]]

                    if isinstance(cam_info, dict) and "image_shape" in cam_info:
                        shape = cam_info["image_shape"]
                        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                            a = int(shape[0])
                            b = int(shape[1])
                            # 常见格式是 [height, width]，例如 [375, 1242] 或 [900, 1600]。
                            # 如果反过来，也做一次兼容。
                            if a <= b:
                                image_h, image_w = a, b
                            else:
                                image_w, image_h = a, b
                            return image_w, image_h

        image_w = int(cfg.get("IMAGE_WIDTH", 1242))
        image_h = int(cfg.get("IMAGE_HEIGHT", 375))
        return image_w, image_h

    def _get_bbox_2d_area_info_exp5a(self, bbox, frame_info=None):
        xyxy = self._get_bbox_image_xyxy_exp5a(bbox)
        if xyxy is None:
            return None

        x1, y1, x2, y2 = xyxy
        image_w, image_h = self._get_image_shape_exp5a(bbox=bbox, frame_info=frame_info)

        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        cfg = self._get_exp5a_cfg()
        out_margin = float(cfg.get("OUT_BOUNDARY_MARGIN", cfg.get("BOUNDARY_MARGIN", 5.0)))
        ref_margin = float(cfg.get("REF_BOUNDARY_MARGIN", 20.0))

        touch_left = x1 <= out_margin
        touch_top = y1 <= out_margin
        touch_right = x2 >= float(image_w - 1) - out_margin
        touch_bottom = y2 >= float(image_h - 1) - out_margin
        touches_boundary = touch_left or touch_top or touch_right or touch_bottom

        ref_touch_left = x1 <= ref_margin
        ref_touch_top = y1 <= ref_margin
        ref_touch_right = x2 >= float(image_w - 1) - ref_margin
        ref_touch_bottom = y2 >= float(image_h - 1) - ref_margin
        touches_ref_boundary = ref_touch_left or ref_touch_top or ref_touch_right or ref_touch_bottom

        return {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "area": float(area),
            "image_w": int(image_w),
            "image_h": int(image_h),
            "touches_boundary": bool(touches_boundary),
            "touch_left": bool(touch_left),
            "touch_top": bool(touch_top),
            "touch_right": bool(touch_right),
            "touch_bottom": bool(touch_bottom),
            "touches_ref_boundary": bool(touches_ref_boundary),
            "ref_touch_left": bool(ref_touch_left),
            "ref_touch_top": bool(ref_touch_top),
            "ref_touch_right": bool(ref_touch_right),
            "ref_touch_bottom": bool(ref_touch_bottom),
        }

    def _update_full_vehicle_reference_exp5a(self, traj, frame_info=None, source="matched"):
        """
        记录该轨迹历史上“完整车辆”的 2D bbox 面积。
        只用真实检测框，不用 fake bbox；贴边框不作为完整参考。
        """
        if not self._exp5a_enabled():
            return False
        if traj is None or len(traj.bboxes) == 0:
            return False

        bbox = traj.bboxes[-1]
        if getattr(bbox, "is_fake", False):
            self.exp5a_stats["skip_fake"] = self.exp5a_stats.get("skip_fake", 0) + 1
            return False

        info = self._get_bbox_2d_area_info_exp5a(bbox, frame_info)
        if info is None:
            self.exp5a_stats["no_bbox_image"] = self.exp5a_stats.get("no_bbox_image", 0) + 1
            return False

        cls_id = getattr(traj, "category_num", 0)
        cfg = self._get_exp5a_cfg()
        min_ref_area = float(_cfg_by_cls(cfg.get("MIN_REF_AREA", {0: 80.0}), cls_id, 80.0))

        # 贴近边界时可能已经是不完整车辆，不能拿来记录“完整车辆大小”。
        # 注意这里用 REF_BOUNDARY_MARGIN，默认比真正删除用的 OUT_BOUNDARY_MARGIN 更大。
        if info.get("touches_ref_boundary", info["touches_boundary"]):
            return False
        if info["area"] < min_ref_area:
            return False

        # v6 核心：每条轨迹只保留一张“历史最大完整参考框”。
        # 如果当前候选框面积没有超过已有参考面积，完全不更新 ref_area / ref_frame / ref_xyxy。
        # 这样不会在轨迹里保存多张参考面积，也不会让较小框覆盖真正的最大完整车框。
        old_ref = float(getattr(traj, "exp5a_full_vehicle_ref_area", 0.0))
        cur_area = float(info["area"])

        bbox.exp5a_full_vehicle_ref_area = old_ref
        bbox.exp5a_ref_update = False

        if cur_area <= old_ref:
            return False

        traj.exp5a_full_vehicle_ref_area = cur_area
        traj.exp5a_full_vehicle_ref_frame = getattr(frame_info, "frame_id", -1) if frame_info is not None else -1
        traj.exp5a_full_vehicle_ref_source = source
        traj.exp5a_full_vehicle_ref_xyxy = [info["x1"], info["y1"], info["x2"], info["y2"]]
        traj.exp5a_full_vehicle_ref_image_wh = [info["image_w"], info["image_h"]]

        # 同步保存这个最大完整 2D 参考框对应的全局 3D 框。
        # 后续只在静止 unmatched 轨迹的出界判断中使用。
        self._update_static_ego_reference_exp5a(traj, bbox, frame_info, source=source)

        bbox.exp5a_full_vehicle_ref_area = cur_area
        bbox.exp5a_ref_update = True
        bbox.exp5a_full_vehicle_ref_xyxy = traj.exp5a_full_vehicle_ref_xyxy

        self.exp5a_stats["ref_update"] = self.exp5a_stats.get("ref_update", 0) + 1
        if bool(cfg.get("DEBUG", False)):
            print(
                "[EXP5A_REF_UPDATE_MAX]",
                "track_id=", traj.track_id,
                "frame=", getattr(frame_info, "frame_id", -1) if frame_info is not None else -1,
                "source=", source,
                "old_ref=", round(old_ref, 2),
                "new_max_ref=", round(cur_area, 2),
                "bbox_area=", round(cur_area, 2),
                "xyxy=", [round(info["x1"], 1), round(info["y1"], 1), round(info["x2"], 1), round(info["y2"], 1)],
                "image_wh=", [info["image_w"], info["image_h"]],
            )

        return True

    def _apply_out_of_view_termination_exp5a(self, traj, frame_info=None, source="matched"):
        """
        实验五A v6：完整车辆参考面积法。
        当当前 2D bbox 面积 <= 历史完整车辆面积的 10%，且当前框贴到图像边界，
        认为车辆整体约 90% 已经出界，只剩车尾/车屁股，终止轨迹并不输出当前 bbox。
        """
        if not self._exp5a_enabled():
            return False
        if traj is None or len(traj.bboxes) == 0:
            return False

        bbox = traj.bboxes[-1]

        # 这个 v4 只处理真实检测框。fake bbox 没有当前真实 2D bbox，不能用旧框误判。
        if getattr(bbox, "is_fake", False):
            self.exp5a_stats["skip_fake"] = self.exp5a_stats.get("skip_fake", 0) + 1
            return False

        self.exp5a_stats["checked"] = self.exp5a_stats.get("checked", 0) + 1

        info = self._get_bbox_2d_area_info_exp5a(bbox, frame_info)
        if info is None:
            self.exp5a_stats["no_bbox_image"] = self.exp5a_stats.get("no_bbox_image", 0) + 1
            return False

        ref_area = float(getattr(traj, "exp5a_full_vehicle_ref_area", 0.0))
        if ref_area <= 1e-6:
            self.exp5a_stats["no_ref_when_checked"] = self.exp5a_stats.get("no_ref_when_checked", 0) + 1
            return False

        cls_id = getattr(traj, "category_num", 0)
        cfg = self._get_exp5a_cfg()
        remain_ratio_thre = float(_cfg_by_cls(cfg.get("REMAIN_RATIO_THRE", {0: 0.10}), cls_id, 0.10))
        require_boundary = bool(cfg.get("REQUIRE_BOUNDARY_TOUCH", True))

        remaining_ratio = float(info["area"] / max(ref_area, 1e-6))
        boundary_ok = (not require_boundary) or bool(info["touches_boundary"])

        self.exp5a_stats["min_remain_ratio"] = min(
            float(self.exp5a_stats.get("min_remain_ratio", 999.0)),
            float(remaining_ratio),
        )
        if bool(info["touches_boundary"]):
            self.exp5a_stats["boundary_checked"] = self.exp5a_stats.get("boundary_checked", 0) + 1
        if remaining_ratio <= float(cfg.get("DEBUG_RATIO_THRE", 0.30)):
            self.exp5a_stats["small_ratio_candidate"] = self.exp5a_stats.get("small_ratio_candidate", 0) + 1
            if bool(cfg.get("DEBUG_CANDIDATE", False)):
                print(
                    "[EXP5A_CANDIDATE]",
                    "track_id=", traj.track_id,
                    "frame=", getattr(frame_info, "frame_id", -1) if frame_info is not None else -1,
                    "source=", source,
                    "current_area=", round(float(info["area"]), 2),
                    "ref_area=", round(float(ref_area), 2),
                    "remain_ratio=", round(float(remaining_ratio), 4),
                    "touch_boundary=", info["touches_boundary"],
                    "xyxy=", [round(info["x1"], 1), round(info["y1"], 1), round(info["x2"], 1), round(info["y2"], 1)],
                    "image_wh=", [info["image_w"], info["image_h"]],
                )

        bbox.exp5a_current_2d_area = float(info["area"])
        bbox.exp5a_full_vehicle_ref_area = float(ref_area)
        bbox.exp5a_full_vehicle_ref_xyxy = getattr(traj, "exp5a_full_vehicle_ref_xyxy", None)
        bbox.exp5a_remaining_ratio_to_ref = float(remaining_ratio)
        bbox.exp5a_touches_boundary = bool(info["touches_boundary"])
        bbox.exp5a_bbox_xyxy = [info["x1"], info["y1"], info["x2"], info["y2"]]

        if remaining_ratio <= remain_ratio_thre and boundary_ok:
            bbox.det_score = traj._is_filter_predict_box
            bbox.exp5a_is_out_of_view = True
            bbox.exp5a_out_view_source = source
            bbox.exp5a_out_view_reason = "2d_area_le_10_percent_of_reference_and_touch_boundary"

            traj.status_flag = 4
            traj.exp5a_delete_reason = "real_out_of_view_" + str(source)
            self.exp5a_stats["terminated"] = self.exp5a_stats.get("terminated", 0) + 1

            if bool(cfg.get("DEBUG", False)):
                print(
                    "[EXP5A_OUT_OF_VIEW]",
                    "track_id=", traj.track_id,
                    "frame=", getattr(frame_info, "frame_id", -1) if frame_info is not None else -1,
                    "source=", source,
                    "current_area=", round(float(info["area"]), 2),
                    "ref_area=", round(float(ref_area), 2),
                    "remain_ratio=", round(float(remaining_ratio), 4),
                    "touch_boundary=", info["touches_boundary"],
                    "xyxy=", [round(info["x1"], 1), round(info["y1"], 1), round(info["x2"], 1), round(info["y2"], 1)],
                    "image_wh=", [info["image_w"], info["image_h"]],
                )
            return True

        return False

    def _print_exp5a_summary(self):
        cfg = self._get_exp5a_cfg()
        if bool(cfg.get("PRINT_SUMMARY", True)):
            print(
                "[EXP5A_SUMMARY]",
                "checked=", self.exp5a_stats.get("checked", 0),
                "ref_update=", self.exp5a_stats.get("ref_update", 0),
                "terminated=", self.exp5a_stats.get("terminated", 0),
                "no_bbox_image=", self.exp5a_stats.get("no_bbox_image", 0),
                "no_ref_when_checked=", self.exp5a_stats.get("no_ref_when_checked", 0),
                "skip_fake=", self.exp5a_stats.get("skip_fake", 0),
                "boundary_checked=", self.exp5a_stats.get("boundary_checked", 0),
                "small_ratio_candidate=", self.exp5a_stats.get("small_ratio_candidate", 0),
                "min_remain_ratio=", round(float(self.exp5a_stats.get("min_remain_ratio", 999.0)), 4),
            )

        ego_cfg = self._get_static_ego_exit_cfg_exp5a()
        if bool(ego_cfg.get("PRINT_SUMMARY", cfg.get("PRINT_SUMMARY", True))):
            print(
                "[EXP5A_STATIC_EGO_EXIT_SUMMARY]",
                "checked=", self.exp5a_static_ego_stats.get("checked", 0),
                "preserved=", self.exp5a_static_ego_stats.get("preserved", 0),
                "out_of_view_stop=", self.exp5a_static_ego_stats.get("out_of_view_stop", 0),
                "terminated=", self.exp5a_static_ego_stats.get("terminated", 0),
                "not_out=", self.exp5a_static_ego_stats.get("not_out", 0),
                "skip_finished=", self.exp5a_static_ego_stats.get("skip_finished", 0),
                "skip_not_output=", self.exp5a_static_ego_stats.get("skip_not_output", 0),
                "skip_no_ref3d=", self.exp5a_static_ego_stats.get("skip_no_ref3d", 0),
                "skip_no_ref2d=", self.exp5a_static_ego_stats.get("skip_no_ref2d", 0),
                "projection_failed=", self.exp5a_static_ego_stats.get("projection_failed", 0),
                "min_remain_ratio=", round(float(self.exp5a_static_ego_stats.get("min_remain_ratio", 999.0)), 4),
            )

        full_image_cfg = self.cfg.get("FULL_IMAGE_BBOX_SOFT_IGNORE", {})
        if bool(full_image_cfg.get("PRINT_SUMMARY", True)):
            print(
                "[FULL_IMAGE_BBOX_SOFT_IGNORE_SUMMARY]",
                "checked_frames=", self.full_image_bbox_ignore_stats.get("checked_frames", 0),
                "ignored=", self.full_image_bbox_ignore_stats.get("ignored", 0),
            )

        newborn_occ_cfg = self.cfg.get("NEWBORN_OCCLUSION_SUPPRESS", {})
        if bool(newborn_occ_cfg.get("PRINT_SUMMARY", True)):
            print(
                "[NEWBORN_OCCLUSION_SUPPRESS_SUMMARY]",
                "checked=", self.newborn_occlusion_suppress_stats.get("checked", 0),
                "suppressed=", self.newborn_occlusion_suppress_stats.get("suppressed", 0),
            )

        rat_cfg = self._get_reliability_tracking_cfg()
        router_cfg = self._get_reliability_router_cfg()
        log_cfg = rat_cfg.get("LOG", {}) if isinstance(rat_cfg, dict) else {}
        if bool(router_cfg.get("PRINT_SUMMARY", log_cfg.get("PRINT_SUMMARY", True))):
            print(
                "[RELIABILITY_ROUTER_SUMMARY]",
                "matched_checked=", self.reliability_router_stats.get("matched_checked", 0),
                "high_kalman=", self.reliability_router_stats.get("high_kalman", 0),
                "medium_motion_state=", self.reliability_router_stats.get("medium_motion_state", 0),
                "medium_static=", self.reliability_router_stats.get("medium_static", 0),
                "medium_moving=", self.reliability_router_stats.get("medium_moving", 0),
                "weak_nms_suppress=", self.reliability_router_stats.get("weak_nms_suppress", 0),
                "weak_skip_update=", self.reliability_router_stats.get("weak_skip_update", 0),
                "unmatched_high_kalman_only=", self.reliability_router_stats.get("unmatched_high_kalman_only", 0),
                "unmatched_medium_motion_state_hsm=", self.reliability_router_stats.get("unmatched_medium_motion_state_hsm", 0),
                "unmatched_weak_nms_only=", self.reliability_router_stats.get("unmatched_weak_nms_only", 0),
                "unmatched_weak_motion_state_hsm=", self.reliability_router_stats.get("unmatched_weak_motion_state_hsm", 0),
                "unmatched_unknown_default_medium=", self.reliability_router_stats.get("unmatched_unknown_default_medium", 0),
            )

        gate_cfg = self._get_track_aware_low_score_gate_cfg()
        if bool(gate_cfg.get("PRINT_SUMMARY", True)):
            print(
                "[TRACK_AWARE_LOW_SCORE_GATE_SUMMARY]",
                "frames=", self.track_aware_low_score_gate_stats.get("frames", 0),
                "checked_dets=", self.track_aware_low_score_gate_stats.get("checked_dets", 0),
                "strong=", self.track_aware_low_score_gate_stats.get("strong", 0),
                "weak=", self.track_aware_low_score_gate_stats.get("weak", 0),
                "rejected_low_score=", self.track_aware_low_score_gate_stats.get("rejected_low_score", 0),
                "rejected_not_near_track=", self.track_aware_low_score_gate_stats.get("rejected_not_near_track", 0),
                "rejected_no_center=", self.track_aware_low_score_gate_stats.get("rejected_no_center", 0),
                "weak_matched=", self.track_aware_low_score_gate_stats.get("weak_matched", 0),
                "weak_unmatched=", self.track_aware_low_score_gate_stats.get("weak_unmatched", 0),
            )

        output_traj_nms_cfg = self.cfg.get("OUTPUT_TRAJ_NMS", {})
        if not isinstance(output_traj_nms_cfg, dict) or len(output_traj_nms_cfg) == 0:
            output_traj_nms_cfg = self.cfg.get("THRESHOLD", {}).get("OUTPUT_TRAJ_NMS", {})
        if bool(output_traj_nms_cfg.get("PRINT_SUMMARY", True)):
            print(
                "[OUTPUT_TRAJ_NMS_SUMMARY]",
                "checked_frames=", self.output_traj_nms_stats.get("checked_frames", 0),
                "suppressed=", self.output_traj_nms_stats.get("suppressed", 0),
            )

    def _get_reliability_tracking_cfg(self):
        cfg = self.cfg.get("RELIABILITY_AWARE_TRACKING", {})
        if not isinstance(cfg, dict):
            return {}
        return cfg

    def _get_reliability_router_cfg(self):
        cfg = self._get_reliability_tracking_cfg()
        router_cfg = cfg.get("ROUTER", {})
        if not isinstance(router_cfg, dict):
            router_cfg = {}
        return router_cfg

    def _reliability_router_enabled(self):
        cfg = self._get_reliability_tracking_cfg()
        router_cfg = self._get_reliability_router_cfg()
        return bool(cfg.get("ENABLE", False)) and bool(router_cfg.get("ENABLE", False))

    def _safe_float_exp7(self, value, default=0.0):
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.shape[0] == 0:
                return float(default)
            if not np.isfinite(arr[0]):
                return float(default)
            return float(arr[0])
        except Exception:
            return float(default)

    def _get_bbox_center_xy_exp7(self, bbox):
        if bbox is None:
            return None
        for name in [
            "global_xyz_lwh_yaw_fusion",
            "global_xyz_lwh_yaw_predict",
            "global_xyz_lwh_yaw",
            "global_xyz",
            "xyz",
        ]:
            if not hasattr(bbox, name):
                continue
            value = getattr(bbox, name)
            if value is None:
                continue
            try:
                arr = np.asarray(value, dtype=float).reshape(-1)
            except Exception:
                continue
            if arr.shape[0] >= 2 and np.all(np.isfinite(arr[:2])):
                return arr[:2].astype(float)
        return None

    def _get_track_length_exp7(self, traj):
        try:
            return int(getattr(traj, "track_length", len(traj.bboxes)))
        except Exception:
            try:
                return int(len(traj.bboxes))
            except Exception:
                return 1

    def _get_unmatch_length_exp7(self, traj):
        for name in ["unmatch_length", "unmatched_length", "lost_time", "lost_frame", "time_since_update"]:
            if hasattr(traj, name):
                try:
                    return int(getattr(traj, name))
                except Exception:
                    pass
        return 0

    def _get_distance_level_exp7(self, dist):
        cfg = self._get_reliability_tracking_cfg()
        dist_cfg = cfg.get("DISTANCE_RANGE", {}) if isinstance(cfg.get("DISTANCE_RANGE", {}), dict) else {}
        near = float(dist_cfg.get("NEAR", 30.0))
        mid = float(dist_cfg.get("MID", 65.0))
        if dist < near:
            return "near"
        if dist < mid:
            return "mid"
        return "far"

    def _get_residual_scale_exp7(self, cls_id, dist_level):
        cfg = self._get_reliability_tracking_cfg()
        trs_cfg = cfg.get("TRS", {}) if isinstance(cfg.get("TRS", {}), dict) else {}
        residual_cfg = trs_cfg.get("RESIDUAL_SCALE", {}) if isinstance(trs_cfg.get("RESIDUAL_SCALE", {}), dict) else {}
        if dist_level == "near":
            return float(_cfg_by_cls(residual_cfg.get("NEAR", {0: 2.0}), cls_id, 2.0))
        if dist_level == "mid":
            return float(_cfg_by_cls(residual_cfg.get("MID", {0: 4.0}), cls_id, 4.0))
        return float(_cfg_by_cls(residual_cfg.get("FAR", {0: 6.0}), cls_id, 6.0))

    def _history_quality_exp7(self, traj, cls_id):
        cfg = self._get_reliability_tracking_cfg()
        trs_cfg = cfg.get("TRS", {}) if isinstance(cfg.get("TRS", {}), dict) else {}
        norm_len = float(_cfg_by_cls(trs_cfg.get("HISTORY_NORM_LENGTH", {0: 10}), cls_id, 10.0))
        track_len = self._get_track_length_exp7(traj)
        return float(np.clip(track_len / max(norm_len, 1.0), 0.0, 1.0))

    def _match_quality_exp7(self, traj, det_bbox, cls_id):
        if traj is None or not hasattr(traj, "bboxes") or len(traj.bboxes) == 0:
            return 0.5, 999.0
        trk_center = self._get_bbox_center_xy_exp7(traj.bboxes[-1])
        det_center = self._get_bbox_center_xy_exp7(det_bbox)
        if trk_center is None or det_center is None:
            return 0.5, 999.0
        residual = float(np.linalg.norm(det_center - trk_center))
        dist = _get_bbox_dist(det_bbox)
        dist_level = self._get_distance_level_exp7(dist)
        scale = self._get_residual_scale_exp7(cls_id, dist_level)
        quality = float(np.exp(-residual / max(scale, 1e-6)))
        return float(np.clip(quality, 0.0, 1.0)), residual

    def _motion_quality_exp7(self, traj, det_bbox, cls_id):
        cfg = self._get_reliability_tracking_cfg()
        trs_cfg = cfg.get("TRS", {}) if isinstance(cfg.get("TRS", {}), dict) else {}
        default_quality = float(trs_cfg.get("DEFAULT_MOTION_QUALITY", 0.50))
        if traj is None or not hasattr(traj, "bboxes") or len(traj.bboxes) < 2:
            return default_quality
        prev_center = self._get_bbox_center_xy_exp7(traj.bboxes[-2])
        last_center = self._get_bbox_center_xy_exp7(traj.bboxes[-1])
        det_center = self._get_bbox_center_xy_exp7(det_bbox)
        if prev_center is None or last_center is None or det_center is None:
            return default_quality
        pred_center = last_center + (last_center - prev_center)
        err = float(np.linalg.norm(det_center - pred_center))
        dist = _get_bbox_dist(det_bbox)
        dist_level = self._get_distance_level_exp7(dist)
        scale = self._get_residual_scale_exp7(cls_id, dist_level)
        quality = float(np.exp(-err / max(scale, 1e-6)))
        return float(np.clip(quality, 0.0, 1.0))

    def _distance_quality_exp7(self, det_bbox):
        dist = _get_bbox_dist(det_bbox)
        level = self._get_distance_level_exp7(dist)
        if level == "near":
            return 1.0
        if level == "mid":
            return 0.80
        return 0.60

    def _boundary_quality_exp7(self, det_bbox, frame_info=None):
        cfg = self._get_reliability_tracking_cfg()
        b_cfg = cfg.get("BOUNDARY", {}) if isinstance(cfg.get("BOUNDARY", {}), dict) else {}
        info = self._get_bbox_2d_area_info_exp5a(det_bbox, frame_info)
        if info is None:
            return 0.80
        if info.get("area", 0.0) <= 1e-6:
            return float(b_cfg.get("OUT_OF_VIEW_QUALITY", 0.30))
        if bool(info.get("touches_boundary", False)):
            return float(b_cfg.get("NEAR_BOUNDARY_QUALITY", 0.70))
        return 1.0

    def _infer_motion_state_for_traj_exp7(self, traj):
        hsm_cfg = self.cfg.get("HSM_LTM", {})
        motion_cfg = hsm_cfg.get("MOTION_STATE", {}) if isinstance(hsm_cfg.get("MOTION_STATE", {}), dict) else {}
        cls_id = getattr(traj, "category_num", 0)
        history_window = int(_cfg_by_cls(hsm_cfg.get("HISTORY_WINDOW", {0: 3}), cls_id, 3))
        static_disp_thre = float(_cfg_by_cls(motion_cfg.get("STATIC_DISP_THRE", {0: 0.02}), cls_id, 0.02))
        require_all_static = bool(motion_cfg.get("REQUIRE_ALL_STATIC", True))

        if hasattr(traj, "is_static_before_lost"):
            try:
                if traj.is_static_before_lost(
                    history_len=history_window,
                    static_disp_thre=static_disp_thre,
                    require_all_static=require_all_static,
                ):
                    return "static"
            except Exception:
                pass

        if not hasattr(traj, "bboxes") or len(traj.bboxes) < 2:
            return "moving"
        centers = []
        for bbox in traj.bboxes[-max(history_window + 1, 2):]:
            center = self._get_bbox_center_xy_exp7(bbox)
            if center is not None:
                centers.append(center)
        if len(centers) < 2:
            return "moving"
        disps = [float(np.linalg.norm(centers[i] - centers[i - 1])) for i in range(1, len(centers))]
        if len(disps) == 0:
            return "moving"
        if require_all_static:
            return "static" if all(d <= static_disp_thre for d in disps) else "moving"
        return "static" if float(np.mean(disps)) <= static_disp_thre else "moving"

    def _compute_track_reliability_exp7(self, traj, det_bbox, cost_value=None, frame_info=None, source="matched"):
        cfg = self._get_reliability_tracking_cfg()
        trs_cfg = cfg.get("TRS", {}) if isinstance(cfg.get("TRS", {}), dict) else {}
        router_cfg = self._get_reliability_router_cfg()
        cls_id = getattr(traj, "category_num", _get_traj_cls_id(traj, self.cfg))

        weight_cfg = trs_cfg.get("WEIGHT", {}) if isinstance(trs_cfg.get("WEIGHT", {}), dict) else {}
        weights = {
            "score": float(weight_cfg.get("SCORE", 0.25)),
            "history": float(weight_cfg.get("HISTORY", 0.20)),
            "match": float(weight_cfg.get("MATCH", 0.25)),
            "motion": float(weight_cfg.get("MOTION", 0.20)),
            "distance": float(weight_cfg.get("DISTANCE", 0.10)),
            # EXP7E: boundary is no longer part of TRS by default.
            # Boundary/out-of-view is already handled by EXP5A, so keep this at 0 unless explicitly enabled in yaml.
            "boundary": float(weight_cfg.get("BOUNDARY", 0.00)),
        }

        score = float(np.clip(_get_bbox_score(det_bbox), 0.0, 1.0))
        history_q = self._history_quality_exp7(traj, cls_id)
        # MATCH still uses BEV center residual by default. cost_value is recorded only for debug.
        match_q, residual = self._match_quality_exp7(traj, det_bbox, cls_id)
        motion_q = self._motion_quality_exp7(traj, det_bbox, cls_id)
        distance_q = self._distance_quality_exp7(det_bbox)
        # If boundary weight is 0, do not let boundary affect TRS.
        boundary_q = 1.0 if weights["boundary"] <= 1e-12 else self._boundary_quality_exp7(det_bbox, frame_info)

        total_w = max(sum(weights.values()), 1e-6)
        trs = (
            weights["score"] * score
            + weights["history"] * history_q
            + weights["match"] * match_q
            + weights["motion"] * motion_q
            + weights["distance"] * distance_q
            + weights["boundary"] * boundary_q
        ) / total_w

        newborn_cfg = trs_cfg.get("NEWBORN_PENALTY", {}) if isinstance(trs_cfg.get("NEWBORN_PENALTY", {}), dict) else {}
        track_len = self._get_track_length_exp7(traj)
        if bool(newborn_cfg.get("ENABLE", True)):
            max_age = int(newborn_cfg.get("MAX_AGE", 3))
            factor = float(newborn_cfg.get("FACTOR", 0.85))
            if track_len <= max_age:
                trs *= factor

        high_thre = float(_cfg_by_cls(router_cfg.get("HIGH_TRS_THRE", {0: 0.75}), cls_id, 0.75))
        weak_thre = float(_cfg_by_cls(router_cfg.get("WEAK_TRS_THRE", {0: 0.45}), cls_id, 0.45))

        if trs >= high_thre:
            level = "high"
            action = "kalman_update"
        elif trs < weak_thre:
            level = "weak"
            action = "nms_suppress"
        else:
            level = "medium"
            action = "motion_state"

        motion_state = self._infer_motion_state_for_traj_exp7(traj) if level == "medium" else "none"

        return {
            "trs": float(np.clip(trs, 0.0, 1.0)),
            "level": level,
            "action": action,
            "motion_state": motion_state,
            "score_q": float(score),
            "history_q": float(history_q),
            "match_q": float(match_q),
            "motion_q": float(motion_q),
            "distance_q": float(distance_q),
            "boundary_q": float(boundary_q),
            "residual": float(residual),
            "dist": float(_get_bbox_dist(det_bbox)),
            "cost": self._safe_float_exp7(cost_value, 0.0),
            "match_source": "bev_center_residual",
            "source": source,
        }

    def _apply_reliability_meta_exp7(self, traj, route_meta, source="matched"):
        if traj is None or route_meta is None:
            return
        traj.reliability_level = route_meta.get("level", "unknown")
        traj.reliability_route_action = route_meta.get("action", "unknown")
        traj.reliability_trs = float(route_meta.get("trs", 0.0))
        traj.reliability_motion_state = route_meta.get("motion_state", "none")
        traj.reliability_source = source
        traj.reliability_is_weak = bool(route_meta.get("level") == "weak")

        if hasattr(traj, "bboxes") and len(traj.bboxes) > 0:
            bbox = traj.bboxes[-1]
            bbox.reliability_level = traj.reliability_level
            bbox.reliability_route_action = traj.reliability_route_action
            bbox.reliability_trs = traj.reliability_trs
            bbox.reliability_motion_state = traj.reliability_motion_state
            bbox.reliability_source = source
            bbox.reliability_is_weak = traj.reliability_is_weak
            bbox.reliability_score_q = float(route_meta.get("score_q", 0.0))
            bbox.reliability_history_q = float(route_meta.get("history_q", 0.0))
            bbox.reliability_match_q = float(route_meta.get("match_q", 0.0))
            bbox.reliability_motion_q = float(route_meta.get("motion_q", 0.0))
            bbox.reliability_distance_q = float(route_meta.get("distance_q", 0.0))
            bbox.reliability_boundary_q = float(route_meta.get("boundary_q", 0.0))
            bbox.reliability_match_source = route_meta.get("match_source", "bev_center_residual")
            bbox.reliability_residual = float(route_meta.get("residual", 999.0))
            bbox.reliability_dist = float(route_meta.get("dist", 0.0))
            if traj.reliability_level == "medium":
                bbox.hsm_motion_state = traj.reliability_motion_state
                bbox.hsm_action = "motion_state_ready"
            elif traj.reliability_level == "weak":
                bbox.hsm_action = "weak_route_output_nms"

    def _update_reliability_router_stats_exp7(self, route_meta, matched=True):
        if route_meta is None:
            return
        if matched:
            self.reliability_router_stats["matched_checked"] = self.reliability_router_stats.get("matched_checked", 0) + 1
            level = route_meta.get("level", "unknown")
            if level == "high":
                self.reliability_router_stats["high_kalman"] = self.reliability_router_stats.get("high_kalman", 0) + 1
            elif level == "medium":
                self.reliability_router_stats["medium_motion_state"] = self.reliability_router_stats.get("medium_motion_state", 0) + 1
                if route_meta.get("motion_state") == "static":
                    self.reliability_router_stats["medium_static"] = self.reliability_router_stats.get("medium_static", 0) + 1
                else:
                    self.reliability_router_stats["medium_moving"] = self.reliability_router_stats.get("medium_moving", 0) + 1
            elif level == "weak":
                self.reliability_router_stats["weak_nms_suppress"] = self.reliability_router_stats.get("weak_nms_suppress", 0) + 1

    def _update_matched_with_reliability_router(self, track_id, det_bbox, cost_value, frame_info=None, source="matched"):
        traj = self.all_trajs[track_id]
        route_meta = None
        if self._reliability_router_enabled():
            route_meta = self._compute_track_reliability_exp7(
                traj=traj,
                det_bbox=det_bbox,
                cost_value=cost_value,
                frame_info=frame_info,
                source=source,
            )
            self._update_reliability_router_stats_exp7(route_meta, matched=True)

            # EXP7D：matched 阶段所有轨迹都必须正常 update。
            # 三层路由只决定 unmatched 后是否进入 HSM/MOTION_STATE，
            # 以及 weak 是否允许在输出 NMS 阶段被压制。

        traj.update(det_bbox, cost_value)

        if route_meta is not None:
            self._apply_reliability_meta_exp7(traj, route_meta, source=source)
            router_cfg = self._get_reliability_router_cfg()
            if bool(router_cfg.get("DEBUG", False)):
                print(
                    "[RELIABILITY_ROUTER_MATCHED]",
                    "track_id=", track_id,
                    "frame=", getattr(frame_info, "frame_id", -1) if frame_info is not None else -1,
                    "source=", source,
                    "level=", route_meta.get("level"),
                    "action=", route_meta.get("action"),
                    "motion_state=", route_meta.get("motion_state"),
                    "trs=", round(float(route_meta.get("trs", 0.0)), 4),
                    "score_q=", round(float(route_meta.get("score_q", 0.0)), 4),
                    "history_q=", round(float(route_meta.get("history_q", 0.0)), 4),
                    "match_q=", round(float(route_meta.get("match_q", 0.0)), 4),
                    "residual=", round(float(route_meta.get("residual", 999.0)), 4),
                )

        return route_meta

    def unmatch_update_with_hsm(self, track_id, frame_id, frame_info=None):
        """
        EXP7D：可靠性感知的选择性运动状态处理。

        关键边界：
        1) 所有轨迹都先执行基础 Kalman unmatched 更新，保证生命周期、fake bbox、unmatch_length 不被破坏。
        2) high 轨迹：基础 Kalman 后直接返回，不进入 HSM_LTM / MOTION_STATE。
        3) medium 轨迹：基础 Kalman 后进入 MOTION_STATE / HSM_LTM。
        4) weak 轨迹：默认只给输出阶段 weak-only NMS 提供可压制标签；
           若 ROUTER.ALLOW_WEAK_HSM_LTM=True，则 weak 也带着 weak 标签进入 HSM_LTM。

        注意：matched 阶段所有轨迹仍然正常 traj.update()，不会因为 weak 而跳过 update。
        """
        traj = self.all_trajs[track_id]

        hsm_cfg = self.cfg.get("HSM_LTM", {})
        motion_cfg = hsm_cfg.get("MOTION_STATE", {})

        hsm_enable = bool(hsm_cfg.get("ENABLE", False))
        motion_state_enable = bool(motion_cfg.get("ENABLE", False))

        cls_id = getattr(traj, "category_num", 0)

        router_enabled = self._reliability_router_enabled()
        router_cfg = self._get_reliability_router_cfg() if router_enabled else {}
        allow_weak_hsm_ltm = bool(router_cfg.get("ALLOW_WEAK_HSM_LTM", False))
        default_level = str(router_cfg.get("DEFAULT_UNMATCHED_LEVEL", "medium")).lower()
        if default_level not in ["high", "medium", "weak"]:
            default_level = "medium"

        if router_enabled:
            route_level = str(getattr(traj, "reliability_level", default_level)).lower()
            if route_level not in ["high", "medium", "weak"]:
                route_level = default_level
            if not hasattr(traj, "reliability_level"):
                self.reliability_router_stats["unmatched_unknown_default_medium"] = (
                    self.reliability_router_stats.get("unmatched_unknown_default_medium", 0) + 1
                )
        else:
            # 关闭 EXP7D 时完全退回原始行为：所有轨迹允许进入 HSM_LTM。
            route_level = "medium"

        route_trs = float(getattr(traj, "reliability_trs", 0.0))

        # ------------------------------------------------------------
        # 必须在 traj.unmatch_update() 前判断静止状态。
        # 因为 unmatch_update() 会追加 Kalman fake bbox，追加后再判断会污染历史位移。
        # medium 一定需要 MOTION_STATE / HSM_LTM；weak 只有在 ALLOW_WEAK_HSM_LTM=True 时才需要。
        # ------------------------------------------------------------
        is_static_before_lost = False
        if (route_level == "medium" or (route_level == "weak" and allow_weak_hsm_ltm)) and motion_state_enable:
            history_window = int(
                _cfg_by_cls(
                    hsm_cfg.get("HISTORY_WINDOW", {0: 3}),
                    cls_id,
                    3,
                )
            )

            static_disp_thre = float(
                _cfg_by_cls(
                    motion_cfg.get("STATIC_DISP_THRE", {0: 0.02}),
                    cls_id,
                    0.02,
                )
            )

            require_all_static = bool(motion_cfg.get("REQUIRE_ALL_STATIC", True))

            is_static_before_lost = traj.is_static_before_lost(
                history_len=history_window,
                static_disp_thre=static_disp_thre,
                require_all_static=require_all_static,
            )

        # ------------------------------------------------------------
        # 所有轨迹都必须走基础 Kalman unmatched 更新。
        # 这一步负责 fake bbox、unmatch_length、生命周期等基础逻辑。
        # ------------------------------------------------------------
        traj.unmatch_update(frame_id)

        # 给刚生成的 fake bbox 写入可靠性标签，供最终输出 NMS 使用。
        if router_enabled:
            if route_level == "high":
                self.reliability_router_stats["unmatched_high_kalman_only"] = (
                    self.reliability_router_stats.get("unmatched_high_kalman_only", 0) + 1
                )
                route_meta = {
                    "level": "high",
                    "action": "kalman_unmatched_only",
                    "motion_state": "none",
                    "trs": route_trs,
                    "source": "unmatched_high_kalman_only",
                }
                self._apply_reliability_meta_exp7(traj, route_meta, source="unmatched_high_kalman_only")

            elif route_level == "weak":
                if allow_weak_hsm_ltm:
                    self.reliability_router_stats["unmatched_weak_motion_state_hsm"] = (
                        self.reliability_router_stats.get("unmatched_weak_motion_state_hsm", 0) + 1
                    )
                    route_meta = {
                        "level": "weak",
                        "action": "motion_state_hsm_after_kalman_unmatched_weak",
                        "motion_state": "static" if is_static_before_lost else "moving",
                        "trs": route_trs,
                        "source": "unmatched_weak_motion_state_hsm",
                    }
                    self._apply_reliability_meta_exp7(traj, route_meta, source="unmatched_weak_motion_state_hsm")
                else:
                    self.reliability_router_stats["unmatched_weak_nms_only"] = (
                        self.reliability_router_stats.get("unmatched_weak_nms_only", 0) + 1
                    )
                    route_meta = {
                        "level": "weak",
                        "action": "kalman_unmatched_nms_only",
                        "motion_state": "none",
                        "trs": route_trs,
                        "source": "unmatched_weak_nms_only",
                    }
                    self._apply_reliability_meta_exp7(traj, route_meta, source="unmatched_weak_nms_only")

            else:
                self.reliability_router_stats["unmatched_medium_motion_state_hsm"] = (
                    self.reliability_router_stats.get("unmatched_medium_motion_state_hsm", 0) + 1
                )
                route_meta = {
                    "level": "medium",
                    "action": "motion_state_hsm_after_kalman_unmatched",
                    "motion_state": "static" if is_static_before_lost else "moving",
                    "trs": route_trs,
                    "source": "unmatched_medium_motion_state_hsm",
                }
                self._apply_reliability_meta_exp7(traj, route_meta, source="unmatched_medium_motion_state_hsm")

        # ------------------------------------------------------------
        # EXP7D 路由分流：
        # high：只信任基础 Kalman 预测，不进入 HSM_LTM / MOTION_STATE。
        # medium：正常进入 HSM_LTM / MOTION_STATE。
        # weak：保留 weak 标签；若 ALLOW_WEAK_HSM_LTM=True，则也进入 HSM_LTM / MOTION_STATE。
        # ------------------------------------------------------------
        if router_enabled and route_level == "high":
            if len(traj.bboxes) > 0:
                traj.bboxes[-1].hsm_motion_state = "none"
                traj.bboxes[-1].hsm_action = "exp7d_high_kalman_only"
            return

        if router_enabled and route_level == "weak" and not allow_weak_hsm_ltm:
            if len(traj.bboxes) > 0:
                traj.bboxes[-1].hsm_motion_state = "none"
                traj.bboxes[-1].hsm_action = "exp7d_weak_nms_only"
            return

        # 关闭 HSM 时，medium / weak 都只保留基础 Kalman unmatched。
        if not hsm_enable:
            return

        start_unmatch_length = int(
            _cfg_by_cls(
                hsm_cfg.get("START_UNMATCH_LENGTH", {0: 1}),
                cls_id,
                1,
            )
        )

        if traj.unmatch_length < start_unmatch_length:
            return

        # ------------------------------------------------------------
        # medium 轨迹一定进入这里；weak 在 ALLOW_WEAK_HSM_LTM=True 时也进入这里。
        # 静止轨迹：不进群体运动，用静止自车参考框 / Kalman-only 检查。
        # 运动轨迹：沿用原 HSM_LTM 审查 / 修正逻辑。
        # ------------------------------------------------------------
        if motion_state_enable and is_static_before_lost:
            ego_preserved = self._apply_static_ego_exit_exp5a(
                traj,
                frame_info=frame_info,
                source="unmatched_weak_static_ego" if route_level == "weak" else "unmatched_medium_static_ego",
            )

            if motion_cfg.get("DEBUG", False):
                print(
                    "[HSM_LTM][MOTION_STATE][EXP7D_WEAK]" if route_level == "weak" else "[HSM_LTM][MOTION_STATE][EXP7D_MEDIUM]",
                    "track_id=", track_id,
                    "frame=", frame_id,
                    "state=static",
                    "action=static_ego_refbox_preserved" if ego_preserved else "kalman_only_static_ego_checked",
                    "unmatch_length=", traj.unmatch_length,
                    "trs=", round(float(route_trs), 4),
                )

            if len(traj.bboxes) > 0:
                traj.bboxes[-1].hsm_motion_state = "static"
                traj.bboxes[-1].hsm_action = "static_ego_refbox_preserved" if ego_preserved else "kalman_only_static_ego_checked"

            return

        if motion_state_enable:
            if len(traj.bboxes) > 0:
                traj.bboxes[-1].hsm_motion_state = "moving"
                traj.bboxes[-1].hsm_action = "original_hsm_ltm_weak_enabled" if route_level == "weak" else "original_hsm_ltm_medium_only"

            if motion_cfg.get("DEBUG", False):
                print(
                    "[HSM_LTM][MOTION_STATE][EXP7D_WEAK]" if route_level == "weak" else "[HSM_LTM][MOTION_STATE][EXP7D_MEDIUM]",
                    "track_id=", track_id,
                    "frame=", frame_id,
                    "state=moving",
                    "action=original_hsm_ltm",
                    "unmatch_length=", traj.unmatch_length,
                    "trs=", round(float(route_trs), 4),
                )

        hsm_after_unmatch_update(
            lost_traj=traj,
            all_trajs=self.all_trajs,
            cfg=self.cfg,
        )

    def get_trajectory_bbox(self, all_trajs):
        track_ids = sorted(all_trajs.keys())
        trajs = []
        for i in track_ids:
            trajs.append(all_trajs[i])
        return trajs

    def predict_before_associate(self):
        for track_id, traj in self.all_trajs.items():
            traj.predict()


    # ------------------------------------------------------------
    # EXP5B-v2 Track-aware low-score observation gate
    # ------------------------------------------------------------
    def _get_track_aware_low_score_gate_cfg(self):
        cfg = self.cfg.get("TRACK_AWARE_LOW_SCORE_GATE", {})
        if not isinstance(cfg, dict):
            return {}
        return cfg

    def _track_aware_low_score_gate_enabled(self):
        cfg = self._get_track_aware_low_score_gate_cfg()
        return bool(cfg.get("ENABLE", False))

    def _get_bbox_cls_id_track_aware_gate(self, bbox):
        """
        尽量读取检测框类别。KITTI car 默认为 0。
        """
        category_map = self.cfg.get("CATEGORY_MAP_TO_NUMBER", {})

        for name in ["category", "category_name", "det_name", "name"]:
            if hasattr(bbox, name):
                cate = getattr(bbox, name)
                if cate in category_map:
                    try:
                        return int(category_map[cate])
                    except Exception:
                        pass

        for name in ["category_num", "category_id", "label", "cls_id"]:
            if hasattr(bbox, name):
                try:
                    return int(getattr(bbox, name))
                except Exception:
                    pass

        return 0

    def _get_bbox_center_xy_track_aware_gate(self, bbox):
        """
        读取 bbox 的 BEV 中心。
        这里优先使用 fusion/predict/global 字段，兼容 MCTrack 的 BBox 结构。
        """
        return self._get_bbox_center_xy_exp7(bbox)

    def _get_track_center_xy_track_aware_gate(self, traj):
        """
        读取轨迹当前预测中心。
        predict_before_associate() 已经在当前帧开头执行，
        所以 traj.bboxes[-1] 通常就是当前帧用于关联的预测状态。
        """
        if traj is None or not hasattr(traj, "bboxes") or len(traj.bboxes) == 0:
            return None
        return self._get_bbox_center_xy_track_aware_gate(traj.bboxes[-1])

    def _get_distance_level_track_aware_gate(self, dist, cls_id=0):
        cfg = self._get_track_aware_low_score_gate_cfg()
        dist_cfg = cfg.get("DISTANCE_POLICY", {})
        if not isinstance(dist_cfg, dict):
            dist_cfg = {}

        near = float(_cfg_by_cls(dist_cfg.get("NEAR_DIST", {0: 30.0}), cls_id, 30.0))
        mid = float(_cfg_by_cls(dist_cfg.get("MID_DIST", {0: 50.0}), cls_id, 50.0))

        if dist < near:
            return "NEAR"
        if dist < mid:
            return "MID"
        return "FAR"

    def _get_track_aware_gate_dist_thre(self, det_dist, cls_id=0):
        cfg = self._get_track_aware_low_score_gate_cfg()
        dist_thre_cfg = cfg.get("NEAR_TRACK_DIST_THRE", {})
        if not isinstance(dist_thre_cfg, dict):
            dist_thre_cfg = {}

        level = self._get_distance_level_track_aware_gate(det_dist, cls_id)
        if level == "NEAR":
            return float(_cfg_by_cls(dist_thre_cfg.get("NEAR", {0: 3.0}), cls_id, 3.0))
        if level == "MID":
            return float(_cfg_by_cls(dist_thre_cfg.get("MID", {0: 5.0}), cls_id, 5.0))
        return float(_cfg_by_cls(dist_thre_cfg.get("FAR", {0: 7.0}), cls_id, 7.0))

    def _is_stable_track_for_low_score_gate(self, traj, cls_id=None):
        """
        weak observation 只能服务稳定老轨迹。
        默认不要求已经输出过，因为有些真实轨迹在确认前也可能需要低分补救；
        若想更保守，可在 yaml 中设置 REQUIRE_ALREADY_OUTPUT: True。
        """
        if traj is None:
            return False

        if int(getattr(traj, "status_flag", 1)) == 4:
            return False

        gate_cfg = self._get_track_aware_low_score_gate_cfg()
        track_cls = getattr(traj, "category_num", _get_traj_cls_id(traj, self.cfg))
        if cls_id is not None and int(track_cls) != int(cls_id):
            return False

        min_track_length = int(
            _cfg_by_cls(
                gate_cfg.get("MIN_TRACK_LENGTH", {0: 3}),
                int(track_cls),
                3,
            )
        )
        max_unmatched_age = int(
            _cfg_by_cls(
                gate_cfg.get("MAX_UNMATCHED_AGE", {0: 2}),
                int(track_cls),
                2,
            )
        )

        track_length = self._get_track_length_exp7(traj)
        unmatch_length = self._get_unmatch_length_exp7(traj)

        if track_length < min_track_length:
            return False

        if unmatch_length > max_unmatched_age:
            return False

        if bool(gate_cfg.get("REQUIRE_ALREADY_OUTPUT", False)) and not bool(getattr(traj, "is_output", False)):
            return False

        if self._get_track_center_xy_track_aware_gate(traj) is None:
            return False

        return True

    def _is_det_near_stable_track_for_low_score_gate(self, det_bbox, trajs, cls_id=0):
        det_center = self._get_bbox_center_xy_track_aware_gate(det_bbox)
        if det_center is None:
            self.track_aware_low_score_gate_stats["rejected_no_center"] = (
                self.track_aware_low_score_gate_stats.get("rejected_no_center", 0) + 1
            )
            return False

        det_dist = _get_bbox_dist(det_bbox)
        dist_gate = self._get_track_aware_gate_dist_thre(det_dist, cls_id)

        for traj in trajs:
            if not self._is_stable_track_for_low_score_gate(traj, cls_id=cls_id):
                continue

            track_center = self._get_track_center_xy_track_aware_gate(traj)
            if track_center is None:
                continue

            residual = float(np.linalg.norm(np.asarray(det_center, dtype=float) - np.asarray(track_center, dtype=float)))
            if residual <= dist_gate:
                try:
                    det_bbox.track_aware_near_track_id = int(traj.track_id)
                    det_bbox.track_aware_near_track_residual = float(residual)
                    det_bbox.track_aware_dist_gate = float(dist_gate)
                except Exception:
                    pass
                return True

        return False

    def _split_dets_by_track_aware_low_score_gate(self, det_bboxes, trajs, frame_info=None):
        """
        将检测分成三类：
        1) strong_bboxes：正常参与 BEV/RV 关联，并允许初始化新轨迹；
        2) weak_bboxes：低分但靠近稳定老轨迹，只允许二阶段补救关联，不允许初始化新轨迹；
        3) rejected_bboxes：极低分或低分且远离稳定老轨迹，不参与关联，不初始化新轨迹。
        """
        gate_cfg = self._get_track_aware_low_score_gate_cfg()
        if not self._track_aware_low_score_gate_enabled():
            return list(det_bboxes), [], []

        self.track_aware_low_score_gate_stats["frames"] = (
            self.track_aware_low_score_gate_stats.get("frames", 0) + 1
        )

        strong_bboxes = []
        weak_bboxes = []
        rejected_bboxes = []

        traj_list = list(trajs) if trajs is not None else []

        for det_bbox in list(det_bboxes):
            self.track_aware_low_score_gate_stats["checked_dets"] = (
                self.track_aware_low_score_gate_stats.get("checked_dets", 0) + 1
            )

            cls_id = self._get_bbox_cls_id_track_aware_gate(det_bbox)
            normal_score_thre = float(
                _cfg_by_cls(
                    gate_cfg.get("NORMAL_SCORE_THRE", {0: 0.40}),
                    cls_id,
                    0.40,
                )
            )
            low_score_thre = float(
                _cfg_by_cls(
                    gate_cfg.get("LOW_SCORE_THRE", {0: 0.20}),
                    cls_id,
                    0.20,
                )
            )

            score = float(_get_bbox_score(det_bbox))

            if score >= normal_score_thre:
                try:
                    det_bbox.track_aware_obs_level = "strong"
                    det_bbox.track_aware_can_init = True
                except Exception:
                    pass
                strong_bboxes.append(det_bbox)
                self.track_aware_low_score_gate_stats["strong"] = (
                    self.track_aware_low_score_gate_stats.get("strong", 0) + 1
                )
                continue

            if score < low_score_thre:
                try:
                    det_bbox.track_aware_obs_level = "rejected_low_score"
                    det_bbox.track_aware_can_init = False
                except Exception:
                    pass
                rejected_bboxes.append(det_bbox)
                self.track_aware_low_score_gate_stats["rejected_low_score"] = (
                    self.track_aware_low_score_gate_stats.get("rejected_low_score", 0) + 1
                )
                continue

            if self._is_det_near_stable_track_for_low_score_gate(det_bbox, traj_list, cls_id=cls_id):
                try:
                    det_bbox.track_aware_obs_level = "weak"
                    det_bbox.track_aware_can_init = False
                except Exception:
                    pass
                weak_bboxes.append(det_bbox)
                self.track_aware_low_score_gate_stats["weak"] = (
                    self.track_aware_low_score_gate_stats.get("weak", 0) + 1
                )
            else:
                try:
                    det_bbox.track_aware_obs_level = "rejected_not_near_track"
                    det_bbox.track_aware_can_init = False
                except Exception:
                    pass
                rejected_bboxes.append(det_bbox)
                self.track_aware_low_score_gate_stats["rejected_not_near_track"] = (
                    self.track_aware_low_score_gate_stats.get("rejected_not_near_track", 0) + 1
                )

        if bool(gate_cfg.get("DEBUG", False)):
            print(
                "[TRACK_AWARE_LOW_SCORE_GATE]",
                "frame=", getattr(frame_info, "frame_id", -1) if frame_info is not None else -1,
                "strong=", len(strong_bboxes),
                "weak=", len(weak_bboxes),
                "rejected=", len(rejected_bboxes),
            )

        return strong_bboxes, weak_bboxes, rejected_bboxes

    def _associate_weak_observations_track_aware_gate(
        self,
        unmatched_trajs,
        weak_bboxes,
        frame_info=None,
        use_group_motion=False,
    ):
        """
        第二阶段 weak observation 补救关联。
        只允许 unmatched stable tracks 参与，不创建新轨迹。
        """
        if not self._track_aware_low_score_gate_enabled():
            return unmatched_trajs

        if len(unmatched_trajs) == 0 or len(weak_bboxes) == 0:
            self.track_aware_low_score_gate_stats["weak_unmatched"] = (
                self.track_aware_low_score_gate_stats.get("weak_unmatched", 0) + len(weak_bboxes)
            )
            return unmatched_trajs

        candidate_trajs = []
        for traj in self.get_trajectory_bbox(unmatched_trajs):
            cls_id = getattr(traj, "category_num", _get_traj_cls_id(traj, self.cfg))
            if self._is_stable_track_for_low_score_gate(traj, cls_id=cls_id):
                candidate_trajs.append(traj)

        if len(candidate_trajs) == 0:
            self.track_aware_low_score_gate_stats["weak_unmatched"] = (
                self.track_aware_low_score_gate_stats.get("weak_unmatched", 0) + len(weak_bboxes)
            )
            return unmatched_trajs

        match_res_weak, cost_matrix_weak = match_trajs_and_dets(
            candidate_trajs,
            weak_bboxes,
            self.cfg,
            use_group_motion=use_group_motion,
        )
        match_res_weak = np.asarray(match_res_weak, dtype=int).reshape(-1, 2)

        matched_track_ids = set()
        matched_weak_det_indices = set()

        for i in range(len(candidate_trajs)):
            track_id = candidate_trajs[i].track_id
            if match_res_weak.shape[0] > 0 and i in match_res_weak[:, 0]:
                indexes = np.where(match_res_weak[:, 0] == i)[0]
                det_index = int(match_res_weak[indexes, 1][0])
                det_bbox = weak_bboxes[det_index]

                # 二阶段 weak 关联再做一次轻量安全检查，避免 weak 框跨距离误接。
                cls_id = getattr(candidate_trajs[i], "category_num", _get_traj_cls_id(candidate_trajs[i], self.cfg))
                det_center = self._get_bbox_center_xy_track_aware_gate(det_bbox)
                track_center = self._get_track_center_xy_track_aware_gate(candidate_trajs[i])
                det_dist = _get_bbox_dist(det_bbox)
                dist_gate = self._get_track_aware_gate_dist_thre(det_dist, cls_id)
                residual = 999.0
                if det_center is not None and track_center is not None:
                    residual = float(np.linalg.norm(np.asarray(det_center, dtype=float) - np.asarray(track_center, dtype=float)))

                if bool(self._get_track_aware_low_score_gate_cfg().get("SECOND_STAGE_STRICT", True)):
                    if residual > dist_gate:
                        continue

                try:
                    det_bbox.track_aware_obs_level = "weak_matched"
                    det_bbox.track_aware_can_init = False
                    det_bbox.track_aware_second_stage_residual = float(residual)
                except Exception:
                    pass

                self._update_matched_with_reliability_router(
                    track_id=track_id,
                    det_bbox=det_bbox,
                    cost_value=cost_matrix_weak[indexes][0],
                    frame_info=frame_info,
                    source="matched_weak_low_score",
                )
                self.all_trajs[track_id].exp5a_static_ego_finished = False
                self.all_trajs[track_id].exp5a_static_ego_keep_len = 0

                exp5a_deleted = self._apply_out_of_view_termination_exp5a(
                    self.all_trajs[track_id],
                    frame_info,
                    source="matched_weak_low_score",
                )
                if not exp5a_deleted:
                    self._update_full_vehicle_reference_exp5a(
                        self.all_trajs[track_id],
                        frame_info,
                        source="matched_weak_low_score",
                    )

                matched_track_ids.add(track_id)
                matched_weak_det_indices.add(det_index)
                self.track_aware_low_score_gate_stats["weak_matched"] = (
                    self.track_aware_low_score_gate_stats.get("weak_matched", 0) + 1
                )

        self.track_aware_low_score_gate_stats["weak_unmatched"] = (
            self.track_aware_low_score_gate_stats.get("weak_unmatched", 0)
            + max(0, len(weak_bboxes) - len(matched_weak_det_indices))
        )

        final_unmatched_trajs = {}
        for track_id, traj in unmatched_trajs.items():
            if track_id not in matched_track_ids:
                final_unmatched_trajs[track_id] = traj

        return final_unmatched_trajs


    def track_single_frame(self, frame_info):
        """
        Info: This function tracks objects in a single frame, performing association between predicted trajectories and detected objects.

        EXP5B-v2 clean path:
        1) 旧的全图异常框、新生遮挡抑制、输出 NMS 等逻辑仍保留，但只在 yaml ENABLE=True 时生效；
        2) 新增 Track-aware low-score observation gate；
        3) strong detections 负责正常关联和新轨迹出生；
        4) weak observations 只用于 unmatched stable tracks 的二阶段补救关联，不允许创建新轨迹。
        """
        self.predict_before_associate()

        # ------------------------------------------------------------
        # 旧逻辑清理：全图异常检测框软屏蔽默认不再自动开启。
        # 只有 yaml 显式 FULL_IMAGE_BBOX_SOFT_IGNORE.ENABLE=True 时才执行。
        # ------------------------------------------------------------
        full_image_cfg = self.cfg.get("FULL_IMAGE_BBOX_SOFT_IGNORE", {})
        if bool(full_image_cfg.get("ENABLE", False)):
            image_w, image_h = self._get_image_shape_exp5a(
                bbox=None,
                frame_info=frame_info,
            )
            image_w = float(full_image_cfg.get("IMAGE_WIDTH", image_w))
            image_h = float(full_image_cfg.get("IMAGE_HEIGHT", image_h))

            valid_bboxes, ignored_full_image_bboxes = soft_ignore_full_image_dets(
                frame_info.bboxes,
                img_w=image_w,
                img_h=image_h,
                edge_margin=float(full_image_cfg.get("EDGE_MARGIN", 5.0)),
                min_width_ratio=float(full_image_cfg.get("MIN_WIDTH_RATIO", 0.95)),
                min_height_ratio=float(full_image_cfg.get("MIN_HEIGHT_RATIO", 0.90)),
                min_area_ratio=float(full_image_cfg.get("MIN_AREA_RATIO", 0.85)),
                debug=bool(full_image_cfg.get("DEBUG", False)),
                frame_id=getattr(frame_info, "frame_id", None),
            )

            self.full_image_bbox_ignore_stats["checked_frames"] = (
                self.full_image_bbox_ignore_stats.get("checked_frames", 0) + 1
            )
            self.full_image_bbox_ignore_stats["ignored"] = (
                self.full_image_bbox_ignore_stats.get("ignored", 0)
                + len(ignored_full_image_bboxes)
            )

            frame_info.bboxes = valid_bboxes

        # 当前帧预测后的轨迹。
        trajs = self.get_trajectory_bbox(self.all_trajs)

        # ------------------------------------------------------------
        # EXP5B-v2：轨迹感知低分观测门控
        # ------------------------------------------------------------
        # strong_bboxes：正常进入第一阶段 BEV/RV 关联，允许创建新轨迹。
        # weak_bboxes：只允许在 strong 关联结束后，补救 unmatched stable tracks。
        # rejected_bboxes：不参与关联，不创建新轨迹。
        # ------------------------------------------------------------
        strong_bboxes, weak_bboxes, rejected_bboxes = self._split_dets_by_track_aware_low_score_gate(
            det_bboxes=frame_info.bboxes,
            trajs=trajs,
            frame_info=frame_info,
        )

        trajs_cnt = len(trajs)
        dets_cnt = len(strong_bboxes)

        use_group_motion, group_info = detect_group_motion(trajs, self.cfg)

        if self.cfg.get("GROUP_MOTION", {}).get("DEBUG", False):
            print(
                "[GROUP_MOTION]",
                "frame=", frame_info.frame_id,
                "use=", use_group_motion,
                "reason=", group_info.get("reason"),
                "valid_tracks=", group_info.get("valid_tracks"),
                "dir_std_deg=", round(group_info.get("dir_std_deg", 999.0), 2),
                "dir_cons_ratio=", round(group_info.get("dir_cons_ratio", 0.0), 3),
                "avg_speed=", round(group_info.get("avg_speed", 0.0), 3),
            )

        # ------------------------------------------------------------
        # 第一阶段：只用 strong detections 做原始 BEV 关联。
        # weak detections 不参与这一阶段，避免低分框污染正常关联。
        # ------------------------------------------------------------
        match_res, cost_matrix = match_trajs_and_dets(
            trajs,
            strong_bboxes,
            self.cfg,
            use_group_motion=use_group_motion,
        )
        match_res = np.asarray(match_res, dtype=int).reshape(-1, 2)

        matched_det_indices = set(match_res[:, 1].tolist()) if match_res.shape[0] > 0 else set()
        unmatched_det_indices = np.array(
            [i for i in range(dets_cnt) if i not in matched_det_indices]
        )

        unmatched_trajs = {}
        for i in range(trajs_cnt):
            track_id = trajs[i].track_id
            if match_res.shape[0] > 0 and i in match_res[:, 0]:
                indexes = np.where(match_res[:, 0] == i)[0]
                det_idx = int(match_res[indexes, 1][0])
                self._update_matched_with_reliability_router(
                    track_id=track_id,
                    det_bbox=strong_bboxes[det_idx],
                    cost_value=cost_matrix[indexes][0],
                    frame_info=frame_info,
                    source="matched_bev_strong",
                )
                self.all_trajs[track_id].exp5a_static_ego_finished = False
                self.all_trajs[track_id].exp5a_static_ego_keep_len = 0

                exp5a_deleted = self._apply_out_of_view_termination_exp5a(
                    self.all_trajs[track_id], frame_info, source="matched_bev_strong"
                )
                if not exp5a_deleted:
                    self._update_full_vehicle_reference_exp5a(
                        self.all_trajs[track_id], frame_info, source="matched_bev_strong"
                    )
            else:
                unmatched_trajs[track_id] = self.all_trajs[track_id]

        # 默认情况下，新生候选只来自 unmatched strong detections。
        init_bboxes = strong_bboxes
        init_det_indices = unmatched_det_indices
        unmatched_trajs_after_strong = unmatched_trajs

        # ------------------------------------------------------------
        # RV 二次匹配仍然只使用 unmatched strong detections。
        # weak detections 留给后面的 track-aware second stage。
        # ------------------------------------------------------------
        if self.cfg["IS_RV_MATCHING"]:
            unmatched_trajs_inbev = self.get_trajectory_bbox(unmatched_trajs)
            trajs_cnt_inbev = len(unmatched_trajs_inbev)
            dets_cnt_inbev = len(unmatched_det_indices)

            unmatched_dets_inbev = (
                np.array(strong_bboxes, dtype=object)[unmatched_det_indices].tolist()
                if dets_cnt_inbev > 0
                else []
            )

            match_res_inbev, cost_matrix_inbev = match_trajs_and_dets(
                unmatched_trajs_inbev,
                unmatched_dets_inbev,
                self.cfg,
                frame_info.transform_matrix,
                is_rv=True,
            )
            match_res_inbev = np.asarray(match_res_inbev, dtype=int).reshape(-1, 2)

            rv_matched_track_ids = set()

            for i in range(trajs_cnt_inbev):
                track_id = unmatched_trajs_inbev[i].track_id
                if match_res_inbev.shape[0] > 0 and i in match_res_inbev[:, 0]:
                    indexes = np.where(match_res_inbev[:, 0] == i)[0]
                    det_index = int(match_res_inbev[indexes, 1][0])
                    trk_bbox = self.all_trajs[track_id].bboxes[-1]
                    det_bbox = unmatched_dets_inbev[det_index]

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
                        continue

                    self._update_matched_with_reliability_router(
                        track_id=track_id,
                        det_bbox=det_bbox,
                        cost_value=float(cost_matrix_inbev[indexes]),
                        frame_info=frame_info,
                        source="matched_rv_strong",
                    )
                    self.all_trajs[track_id].exp5a_static_ego_finished = False
                    self.all_trajs[track_id].exp5a_static_ego_keep_len = 0

                    exp5a_deleted = self._apply_out_of_view_termination_exp5a(
                        self.all_trajs[track_id], frame_info, source="matched_rv_strong"
                    )
                    if not exp5a_deleted:
                        self._update_full_vehicle_reference_exp5a(
                            self.all_trajs[track_id], frame_info, source="matched_rv_strong"
                        )

                    rv_matched_track_ids.add(track_id)

            matched_det_indices_rv = set(match_res_inbev[:, 1].tolist()) if match_res_inbev.shape[0] > 0 else set()
            init_det_indices = np.array(
                [i for i in range(dets_cnt_inbev) if i not in matched_det_indices_rv]
            )
            init_bboxes = unmatched_dets_inbev

            unmatched_trajs_after_strong = {}
            for track_id, traj in unmatched_trajs.items():
                if track_id not in rv_matched_track_ids:
                    unmatched_trajs_after_strong[track_id] = traj

        # ------------------------------------------------------------
        # 第二阶段：weak observations 只补救 unmatched stable tracks。
        # 不允许 weak observations 创建新轨迹。
        # ------------------------------------------------------------
        unmatched_trajs_after_weak = self._associate_weak_observations_track_aware_gate(
            unmatched_trajs=unmatched_trajs_after_strong,
            weak_bboxes=weak_bboxes,
            frame_info=frame_info,
            use_group_motion=use_group_motion,
        )

        # 所有 strong/RV/weak 都没有匹配到的轨迹，最后才执行 unmatched update。
        for track_id in list(unmatched_trajs_after_weak.keys()):
            self.unmatch_update_with_hsm(track_id, frame_info.frame_id, frame_info)

        # ------------------------------------------------------------
        # 新生轨迹初始化：只允许 unmatched strong detections。
        # weak_bboxes 和 rejected_bboxes 都不会走到这里。
        # ------------------------------------------------------------
        newborn_occ_cfg = self.cfg.get("NEWBORN_OCCLUSION_SUPPRESS", {})
        # 旧逻辑清理：新生遮挡抑制默认不再自动开启。
        newborn_occ_enable = bool(newborn_occ_cfg.get("ENABLE", False))

        for i in init_det_indices:
            det_bbox = init_bboxes[int(i)]

            if newborn_occ_enable:
                self.newborn_occlusion_suppress_stats["checked"] = (
                    self.newborn_occlusion_suppress_stats.get("checked", 0) + 1
                )

                if is_highly_occluded_newborn_det(
                    det_bbox,
                    init_bboxes,
                    occlusion_ratio_thre=float(newborn_occ_cfg.get("OCCLUSION_RATIO_THRE", 0.75)),
                    score_tolerance=float(newborn_occ_cfg.get("SCORE_TOLERANCE", 0.05)),
                    min_area=float(newborn_occ_cfg.get("MIN_AREA", 80.0)),
                ):
                    self.newborn_occlusion_suppress_stats["suppressed"] = (
                        self.newborn_occlusion_suppress_stats.get("suppressed", 0) + 1
                    )

                    if bool(newborn_occ_cfg.get("DEBUG", False)):
                        print(
                            "[NEWBORN_OCCLUSION_SUPPRESS]",
                            "frame=", frame_info.frame_id,
                            "det_index=", int(i),
                            "score=", round(get_det_score_safe(det_bbox), 3),
                            "action=skip_init",
                        )
                    continue

            self.all_trajs[self.track_id_counter] = Trajectory(
                track_id=self.track_id_counter,
                init_bbox=det_bbox,
                cfg=self.cfg,
            )
            self._update_full_vehicle_reference_exp5a(
                self.all_trajs[self.track_id_counter], frame_info, source="new_track_strong"
            )
            self.track_id_counter += 1

        for track_id in list(self.all_trajs.keys()):
            if self.all_trajs[track_id].status_flag == 4:
                self.all_dead_trajs[track_id] = self.all_trajs[track_id]
                del self.all_trajs[track_id]

        output_trajs = self.get_output_trajs(frame_info.frame_id)

        # ------------------------------------------------------------
        # 输出阶段轨迹级 NMS 仍然只在 yaml ENABLE=True 时生效。
        # ------------------------------------------------------------
        output_traj_nms_cfg = self.cfg.get("OUTPUT_TRAJ_NMS", {})
        if not isinstance(output_traj_nms_cfg, dict) or len(output_traj_nms_cfg) == 0:
            output_traj_nms_cfg = self.cfg.get("THRESHOLD", {}).get("OUTPUT_TRAJ_NMS", {})

        if bool(output_traj_nms_cfg.get("ENABLE", False)):
            self.output_traj_nms_stats["checked_frames"] = (
                self.output_traj_nms_stats.get("checked_frames", 0) + 1
            )

            output_trajs, suppressed_output_ids = output_traj_nms(
                output_trajs=output_trajs,
                cfg=self.cfg,
                frame_id=frame_info.frame_id,
            )

            self.output_traj_nms_stats["suppressed"] = (
                self.output_traj_nms_stats.get("suppressed", 0)
                + len(suppressed_output_ids)
            )

        return output_trajs


    def get_output_trajs(self, frame_id):
        output_trajs = {}

        ego_cfg = self._get_static_ego_exit_cfg_exp5a()
        require_confirmed_for_static_ego = bool(ego_cfg.get("REQUIRE_ALREADY_OUTPUT", True))

        for track_id in list(self.all_trajs.keys()):
            traj = self.all_trajs[track_id]
            if len(traj.bboxes) == 0:
                continue

            bbox = traj.bboxes[-1]
            static_ego_preserved = bool(getattr(bbox, "exp5a_static_ego_preserved", False))

            # 原始 MCTrack 输出 confirmed 轨迹。
            # 静止自车参考框延伸是特殊 fake bbox：即使 status_flag==2，也允许输出。
            if not (traj.status_flag == 1 or frame_id < 3 or static_ego_preserved):
                continue

            already_confirmed = getattr(traj, "is_output", False)

            if static_ego_preserved and require_confirmed_for_static_ego and not already_confirmed:
                continue

            # 普通预测框继续过滤；静止自车参考框延伸例外。
            if (not static_ego_preserved) and bbox.det_score == traj._is_filter_predict_box:
                continue

            # 实验四：已经输出过的老轨迹不再走新生过滤。
            # 静止自车参考框延伸来自老轨迹，也不走 OUTPUT_FILTER.MAX_LOST_OUTPUT_LENGTH。
            if (not static_ego_preserved) and (not already_confirmed):
                if not should_output_traj_bbox_exp3(traj, bbox, self.cfg):
                    continue

            # ------------------------------------------------------------
            # 实验2：给 OUTPUT_TRAJ_NMS 提供轨迹生命周期信息
            # 这些字段只用于当前帧输出 NMS，不改变轨迹状态、不影响匹配、不影响 Kalman。
            # 注意：output_already_confirmed 必须在 traj.is_output = True 之前记录。
            # ------------------------------------------------------------
            bbox.output_track_id = int(track_id)
            bbox.output_already_confirmed = bool(already_confirmed)

            # 轨迹长度
            bbox.output_track_length = int(getattr(traj, "track_length", len(traj.bboxes)))

            # 未匹配长度，不同版本字段名可能不同，这里做兼容
            bbox.output_unmatch_length = int(
                getattr(
                    traj,
                    "unmatch_length",
                    getattr(traj, "unmatched_length", 0)
                )
            )

            bbox.output_status_flag = int(getattr(traj, "status_flag", -1))

            # 当前框是否是 fake / predict bbox
            try:
                bbox.output_is_fake = bool(bbox.det_score == traj._is_filter_predict_box)
            except Exception:
                bbox.output_is_fake = bool(getattr(bbox, "is_fake", False))

            # 静止自车参考框延伸也记录下来，方便后面保护
            bbox.output_static_ego_preserved = bool(static_ego_preserved)

            # EXP7：把三层路由结果传给输出阶段 NMS。
            # detection_quality_filter.py 若支持这些字段，就可以只压制 weak 轨迹；
            # 若旧版本未读取这些字段，也不会破坏原有输出逻辑。
            bbox.output_reliability_level = getattr(traj, "reliability_level", getattr(bbox, "reliability_level", "unknown"))
            bbox.output_reliability_action = getattr(traj, "reliability_route_action", getattr(bbox, "reliability_route_action", "unknown"))
            bbox.output_reliability_trs = float(getattr(traj, "reliability_trs", getattr(bbox, "reliability_trs", 0.0)))
            bbox.output_reliability_is_weak = bool(bbox.output_reliability_level == "weak")
            bbox.output_allow_nms_suppress = bool(bbox.output_reliability_is_weak and not already_confirmed)

            output_trajs[track_id] = bbox
            traj.is_output = True

        return output_trajs

    def post_processing(self):
        self._print_exp5a_summary()
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