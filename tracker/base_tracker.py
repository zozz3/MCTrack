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

        output_traj_nms_cfg = self.cfg.get("OUTPUT_TRAJ_NMS", {})
        if not isinstance(output_traj_nms_cfg, dict) or len(output_traj_nms_cfg) == 0:
            output_traj_nms_cfg = self.cfg.get("THRESHOLD", {}).get("OUTPUT_TRAJ_NMS", {})
        if bool(output_traj_nms_cfg.get("PRINT_SUMMARY", True)):
            print(
                "[OUTPUT_TRAJ_NMS_SUMMARY]",
                "checked_frames=", self.output_traj_nms_stats.get("checked_frames", 0),
                "suppressed=", self.output_traj_nms_stats.get("suppressed", 0),
            )

    def unmatch_update_with_hsm(self, track_id, frame_id, frame_info=None):
        traj = self.all_trajs[track_id]

        hsm_cfg = self.cfg.get("HSM_LTM", {})
        motion_cfg = hsm_cfg.get("MOTION_STATE", {})

        hsm_enable = bool(hsm_cfg.get("ENABLE", False))
        motion_state_enable = bool(motion_cfg.get("ENABLE", False))

        cls_id = getattr(traj, "category_num", 0)

        is_static_before_lost = False

        # 必须在 unmatch_update() 前判断
        # 因为 unmatch_update() 会追加 Kalman fake bbox
        if motion_state_enable:
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

            require_all_static = bool(
                motion_cfg.get("REQUIRE_ALL_STATIC", True)
            )

            is_static_before_lost = traj.is_static_before_lost(
                history_len=history_window,
                static_disp_thre=static_disp_thre,
                require_all_static=require_all_static,
            )

        # 原始 Kalman unmatched 更新
        traj.unmatch_update(frame_id)

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

        # 静止目标：不进入原 HSM_LTM 群体运动审查。
        # 仅在这里额外做“自车运动引导的静止车辆出界判断”。
        # 注意：这个逻辑只可能删除已经出界的静止轨迹，不改 GROUP_MOTION。
        if motion_state_enable and is_static_before_lost:
            ego_preserved = self._apply_static_ego_exit_exp5a(
                traj,
                frame_info=frame_info,
                source="unmatched_static_ego",
            )

            if motion_cfg.get("DEBUG", False):
                print(
                    "[HSM_LTM][MOTION_STATE]",
                    "track_id=", track_id,
                    "frame=", frame_id,
                    "state=static",
                    "action=static_ego_refbox_preserved" if ego_preserved else "kalman_only_static_ego_checked",
                    "unmatch_length=", traj.unmatch_length,
                )

            if len(traj.bboxes) > 0:
                traj.bboxes[-1].hsm_motion_state = "static"
                traj.bboxes[-1].hsm_action = "static_ego_refbox_preserved" if ego_preserved else "kalman_only_static_ego_checked"

            return

        # 运动目标：沿用原 HSM_LTM 审查/删除逻辑
        if motion_state_enable:
            if len(traj.bboxes) > 0:
                traj.bboxes[-1].hsm_motion_state = "moving"
                traj.bboxes[-1].hsm_action = "original_hsm_ltm"

            if motion_cfg.get("DEBUG", False):
                print(
                    "[HSM_LTM][MOTION_STATE]",
                    "track_id=", track_id,
                    "frame=", frame_id,
                    "state=moving",
                    "action=original_hsm_ltm",
                    "unmatch_length=", traj.unmatch_length,
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

        # ------------------------------------------------------------
        # 实验五E-1：全图异常检测框软屏蔽
        # ------------------------------------------------------------
        # 原始 VirConv 检测文件不做任何修改。
        # 这里只在 tracker 当前帧内部屏蔽异常框，使其不参与：
        # 1) BEV/RV 数据关联；
        # 2) 已有轨迹 update；
        # 3) 新生轨迹初始化；
        # 4) 最终输出。
        #
        # 屏蔽条件：
        # - bbox 几乎覆盖整张图像；
        # - 或 bbox 至少三条边贴近图像边界。
        # ------------------------------------------------------------
        full_image_cfg = self.cfg.get("FULL_IMAGE_BBOX_SOFT_IGNORE", {})
        if bool(full_image_cfg.get("ENABLE", True)):
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

            # 软屏蔽：只改当前 tracker 运行时使用的 detection 列表。
            # 不改 base_version json，不改原始 VirConv 文件。
            frame_info.bboxes = valid_bboxes

        trajs = self.get_trajectory_bbox(self.all_trajs)
        trajs_cnt, dets_cnt = len(trajs), len(frame_info.bboxes)
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

        match_res, cost_matrix = match_trajs_and_dets(
            trajs,
            frame_info.bboxes,
            self.cfg,
            use_group_motion=use_group_motion
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
                self.all_trajs[track_id].exp5a_static_ego_finished = False
                self.all_trajs[track_id].exp5a_static_ego_keep_len = 0
                # 实验五A v6：先用历史完整面积判断当前框是否已经只剩车尾/车屁股。
                # 如果没有删除，再把当前非贴边大框更新为新的完整参考面积。
                exp5a_deleted = self._apply_out_of_view_termination_exp5a(
                    self.all_trajs[track_id], frame_info, source="matched_bev"
                )
                if not exp5a_deleted:
                    self._update_full_vehicle_reference_exp5a(
                        self.all_trajs[track_id], frame_info, source="matched_bev"
                    )
            else:
                unmatched_trajs[track_id] = self.all_trajs[track_id]
                if not self.cfg["IS_RV_MATCHING"]:
                    self.unmatch_update_with_hsm(track_id, frame_info.frame_id, frame_info)

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
                        self.unmatch_update_with_hsm(track_id, frame_info.frame_id, frame_info)
                        continue
                    self.all_trajs[track_id].update(
                        det_bbox, float(cost_matrix_inbev[indexes])
                    )
                    self.all_trajs[track_id].exp5a_static_ego_finished = False
                    self.all_trajs[track_id].exp5a_static_ego_keep_len = 0
                    # 实验五A v6：RV 二次匹配后也先检查真实出界，再更新参考面积。
                    exp5a_deleted = self._apply_out_of_view_termination_exp5a(
                        self.all_trajs[track_id], frame_info, source="matched_rv"
                    )
                    if not exp5a_deleted:
                        self._update_full_vehicle_reference_exp5a(
                            self.all_trajs[track_id], frame_info, source="matched_rv"
                        )
                else:
                    self.unmatch_update_with_hsm(track_id, frame_info.frame_id, frame_info)

            matched_det_indices = set(match_res_inbev[:, 1])
            unmatched_det_indices = np.array(
                [i for i in range(dets_cnt_inbev) if i not in matched_det_indices]
            )
            init_bboxes = unmatched_dets_inbev

        newborn_occ_cfg = self.cfg.get("NEWBORN_OCCLUSION_SUPPRESS", {})
        newborn_occ_enable = bool(newborn_occ_cfg.get("ENABLE", True))

        for i in unmatched_det_indices:
            det_bbox = init_bboxes[int(i)]

            # ------------------------------------------------------------
            # 实验五E-2：新生高遮挡检测抑制
            # ------------------------------------------------------------
            # 只作用于 unmatched detection 初始化新轨迹之前。
            # 不影响已有轨迹的匹配、更新和遮挡保持。
            # 如果当前检测框的大部分 2D 区域被同一帧中置信度相近或更高的检测框覆盖，
            # 则认为该检测在当前帧可见性不足或存在重复响应风险，暂不初始化新轨迹。
            # ------------------------------------------------------------
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
            # 实验五A v6：新生轨迹如果完整可见，先记录完整车辆面积。
            self._update_full_vehicle_reference_exp5a(
                self.all_trajs[self.track_id_counter], frame_info, source="new_track"
            )
            self.track_id_counter += 1

        for track_id in list(self.all_trajs.keys()):
            if self.all_trajs[track_id].status_flag == 4:
                self.all_dead_trajs[track_id] = self.all_trajs[track_id]
                del self.all_trajs[track_id]

        output_trajs = self.get_output_trajs(frame_info.frame_id)

        # ------------------------------------------------------------
        # 实验五E-4：输出阶段轨迹级 NMS
        # ------------------------------------------------------------
        # 这是 KITTI 路径真正生效的输出阶段 NMS。
        # 它只处理当前帧 output_trajs 之间的邻居重叠，不删除轨迹，
        # 不影响匹配，不影响 Kalman，只是不输出当前帧被 NMS 抑制的框。
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