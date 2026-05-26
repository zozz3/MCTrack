# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np

from tracker.matching import *
from tracker.trajectory import Trajectory
from tracker.hsm_ltm import hsm_after_unmatch_update
from tracker.group_motion_switch import detect_group_motion
from utils.utils import norm_realative_radian



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
                else:
                    self.unmatch_update_with_hsm(track_id, frame_info.frame_id)

            matched_det_indices = set(match_res_inbev[:, 1])
            unmatched_det_indices = np.array(
                [i for i in range(dets_cnt_inbev) if i not in matched_det_indices]
            )
            init_bboxes = unmatched_dets_inbev

        for i in unmatched_det_indices:
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

        output_trajs = self.get_output_trajs(frame_info.frame_id)

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