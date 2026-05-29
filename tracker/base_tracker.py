# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np

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

    def unmatch_update_with_hsm(self, track_id, frame_id):
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

        # 静止目标：只保留 Kalman，不进入 HSM_LTM
        if motion_state_enable and is_static_before_lost:
            if motion_cfg.get("DEBUG", False):
                print(
                    "[HSM_LTM][MOTION_STATE]",
                    "track_id=", track_id,
                    "frame=", frame_id,
                    "state=static",
                    "action=kalman_only",
                    "unmatch_length=", traj.unmatch_length,
                )

            if len(traj.bboxes) > 0:
                traj.bboxes[-1].hsm_motion_state = "static"
                traj.bboxes[-1].hsm_action = "kalman_only"

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
                    self.unmatch_update_with_hsm(track_id, frame_info.frame_id)

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
                        self.unmatch_update_with_hsm(track_id, frame_info.frame_id)
                        continue
                    self.all_trajs[track_id].update(
                        det_bbox, float(cost_matrix_inbev[indexes])
                    )
                    # 实验五A v6：RV 二次匹配后也先检查真实出界，再更新参考面积。
                    exp5a_deleted = self._apply_out_of_view_termination_exp5a(
                        self.all_trajs[track_id], frame_info, source="matched_rv"
                    )
                    if not exp5a_deleted:
                        self._update_full_vehicle_reference_exp5a(
                            self.all_trajs[track_id], frame_info, source="matched_rv"
                        )
                else:
                    self.unmatch_update_with_hsm(track_id, frame_info.frame_id)

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

        for track_id in list(self.all_trajs.keys()):
            traj = self.all_trajs[track_id]

            if traj.status_flag == 1 or frame_id < 3:
                bbox = traj.bboxes[-1]

                # 保留原始 MCTrack 的预测框过滤逻辑
                if bbox.det_score == traj._is_filter_predict_box:
                    continue

                # ------------------------------------------------------------
                # 实验四：新生轨迹延迟确认
                # ------------------------------------------------------------
                # 如果这条轨迹以前已经输出过，说明它已经被确认过，
                # 后续不再用 OUTPUT_FILTER 卡它，避免增加 FN 和 Frag。
                #
                # 如果这条轨迹以前从未输出过，才使用实验三的输出过滤，
                # 防止短轨迹、低分、远距离假阳性直接进入结果。
                # ------------------------------------------------------------
                already_confirmed = getattr(traj, "is_output", False)

                if not already_confirmed:
                    if not should_output_traj_bbox_exp3(traj, bbox, self.cfg):
                        continue

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