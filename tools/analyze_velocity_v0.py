import argparse
import csv
import math
import re
from pathlib import Path
from collections import defaultdict

import numpy as np


def read_calib(calib_path):
    """
    读取 KITTI tracking calib。
    返回：
    P2: camera projection
    lidar2camera: 已经包含 R_rect 的 lidar -> rect camera 变换
    camera2lidar: rect camera -> lidar
    """
    P2 = None
    Tr = None
    R0 = None

    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("P2:"):
                vals = [float(x) for x in line.strip().split()[1:]]
                P2 = np.array(vals, dtype=np.float64).reshape(3, 4)

            elif line.startswith("Tr_velo_cam") or line.startswith("Tr_velo_cam"):
                vals = [float(x) for x in line.strip().split()[1:]]
                Tr = np.array(vals, dtype=np.float64).reshape(3, 4)
                Tr = np.vstack([Tr, np.array([0, 0, 0, 1], dtype=np.float64)])

            elif line.startswith("R0_rect:") or line.startswith("R_rect:"):
                vals = [float(x) for x in line.strip().split()[1:]]
                R0 = np.array(vals, dtype=np.float64).reshape(3, 3)
                R0_4 = np.eye(4, dtype=np.float64)
                R0_4[:3, :3] = R0
                R0 = R0_4

    if P2 is None:
        raise RuntimeError(f"P2 not found in {calib_path}")
    if Tr is None:
        raise RuntimeError(f"Tr_velo_to_cam not found in {calib_path}")
    if R0 is None:
        R0 = np.eye(4, dtype=np.float64)

    lidar2camera = R0 @ Tr
    camera2lidar = np.linalg.inv(lidar2camera)

    return P2, lidar2camera, camera2lidar


def read_pose(pose_path):
    """
    读取每一帧 lidar/ego 到 global 的 4x4 pose。
    MCTrack convert_kitti.py 里就是把 pose 当 lidar2global 用。
    """
    poses = {}

    with open(pose_path, "r", encoding="utf-8") as f:
        for frame_id, line in enumerate(f):
            vals = [float(x) for x in line.strip().split()]
            mat = np.array(vals, dtype=np.float64).reshape(3, 4)
            mat = np.vstack([mat, np.array([0, 0, 0, 1], dtype=np.float64)])
            poses[frame_id] = mat

    return poses


def parse_kitti_tracking_line(line):
    """
    KITTI tracking line:
    frame track_id type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom h w l x y z rotation_y [score]
    """
    parts = line.strip().split()
    if len(parts) < 17:
        return None

    obj_type = parts[2].lower()
    if obj_type not in ["car", "van"]:
        return None

    try:
        return {
            "frame": int(parts[0]),
            "track_id": int(parts[1]),
            "type": obj_type,

            "left": float(parts[6]),
            "top": float(parts[7]),
            "right": float(parts[8]),
            "bottom": float(parts[9]),

            "h": float(parts[10]),
            "w": float(parts[11]),
            "l": float(parts[12]),

            # KITTI camera coordinate
            "cam_x": float(parts[13]),
            "cam_y": float(parts[14]),
            "cam_z": float(parts[15]),

            "rot_y": float(parts[16]),
            "score": float(parts[17]) if len(parts) > 17 else 1.0,
        }
    except Exception:
        return None


def load_kitti_tracking_file(path):
    items = []
    path = Path(path)

    if not path.exists():
        return items

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = parse_kitti_tracking_line(line)
            if item is not None:
                items.append(item)

    return items


def camera_bottom_to_global_bev(item, camera2lidar, lidar2global):
    """
    KITTI label_02 的 x,y,z 是 3D box bottom center in camera coord。
    MCTrack convert_kitti.py 里转换 detection 时：
    camera -> lidar 后，会加 h/2，变成 box center。
    这里为了速度分析，只要同一套规则用于 GT 和 Pred 即可。
    """
    cam_xyz = np.array([item["cam_x"], item["cam_y"], item["cam_z"], 1.0], dtype=np.float64)

    lidar_xyz = camera2lidar @ cam_xyz

    # 转成 3D box center，和 MCTrack global_xyz 更一致
    lidar_xyz[2] += item["h"] / 2.0

    global_xyz = lidar2global @ lidar_xyz

    # BEV 平面使用 global x,y
    return float(global_xyz[0]), float(global_xyz[1])


def add_global_xy(items, calib_path, pose_path):
    _, _, camera2lidar = read_calib(calib_path)
    poses = read_pose(pose_path)

    new_items = []

    for item in items:
        frame_id = item["frame"]
        if frame_id not in poses:
            continue

        lidar2global = poses[frame_id]
        gx, gy = camera_bottom_to_global_bev(item, camera2lidar, lidar2global)

        item = dict(item)
        item["global_x"] = gx
        item["global_y"] = gy
        new_items.append(item)

    return new_items


def group_by_frame(items):
    frames = defaultdict(list)
    for it in items:
        frames[it["frame"]].append(it)
    return frames


def group_by_track(items):
    tracks = defaultdict(dict)
    for it in items:
        tracks[it["track_id"]][it["frame"]] = it
    return tracks


def center_dist_global_bev(a, b):
    dx = a["global_x"] - b["global_x"]
    dy = a["global_y"] - b["global_y"]
    return math.sqrt(dx * dx + dy * dy)


def greedy_match_frame_by_global_center(gt_list, pred_list, dist_thre):
    candidates = []

    for gi, gt in enumerate(gt_list):
        for pi, pred in enumerate(pred_list):
            dist = center_dist_global_bev(gt, pred)
            if dist <= dist_thre:
                candidates.append((dist, gi, pi))

    candidates.sort(key=lambda x: x[0])

    used_gt = set()
    used_pred = set()
    pairs = []

    for dist, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        pairs.append((gt_list[gi], pred_list[pi], dist))

    return pairs


def calc_speed_global(cur, prev, fps):
    dx = cur["global_x"] - prev["global_x"]
    dy = cur["global_y"] - prev["global_y"]
    return math.sqrt(dx * dx + dy * dy) * fps


def calc_velocity_vec_global(cur, prev, fps):
    vx = (cur["global_x"] - prev["global_x"]) * fps
    vy = (cur["global_y"] - prev["global_y"]) * fps
    return vx, vy


def strict_static_global(cur_gt, prev_gt, zero_eps):
    """
    严格静止定义：
    理论上 zero_eps=0.0。
    但由于 camera->lidar->global 是浮点计算，如果你想只消除浮点误差，可设 1e-9。
    注意：1e-9 不是低速阈值，不等于把低速当静止。
    """
    dx = cur_gt["global_x"] - prev_gt["global_x"]
    dy = cur_gt["global_y"] - prev_gt["global_y"]

    if abs(dx) <= zero_eps and abs(dy) <= zero_eps:
        return "static"
    return "moving"


def direction_error_deg(pred_vec, gt_vec):
    px, py = pred_vec
    gx, gy = gt_vec

    pn = math.sqrt(px * px + py * py)
    gn = math.sqrt(gx * gx + gy * gy)

    if pn <= 1e-12 or gn <= 1e-12:
        return None

    cos_val = (px * gx + py * gy) / (pn * gn)
    cos_val = max(-1.0, min(1.0, cos_val))

    return math.degrees(math.acos(cos_val))


def mean(values):
    values = [v for v in values if v is not None]
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)


def rate(n, d):
    return n / d if d > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", required=True, help="例如 data/kitti/datasets/tracking/training")
    parser.add_argument("--pred_dir", required=True, help="MCTrack result data dir")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--dist_thre", type=float, default=2.0)

    # 你要求严格速度为 0，默认就是 0
    parser.add_argument("--zero_eps", type=float, default=0.0)

    parser.add_argument("--out_csv", default="velocity_v0_global_detail.csv")
    parser.add_argument("--out_summary", default="velocity_v0_global_summary.txt")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    gt_dir = dataset_root / "label_02"
    calib_dir = dataset_root / "calib"
    pose_dir = dataset_root / "pose"
    pred_dir = Path(args.pred_dir)

    pred_files = sorted(pred_dir.glob("*.txt"))

    total_gt_items = 0
    total_pred_items = 0
    total_frame_matches = 0
    usable_velocity_pairs = 0

    static_count = 0
    moving_count = 0

    static_pred_nonzero_count = 0
    static_pred_speed_over_05_count = 0
    static_pred_speed_over_10_count = 0

    speed_abs_errors_all = []
    velocity_vec_errors_all = []
    static_speed_abs_errors = []
    moving_speed_abs_errors = []
    moving_direction_errors = []

    detail_rows = []

    for pred_file in pred_files:
        seq_name = pred_file.stem

        gt_file = gt_dir / f"{seq_name}.txt"
        calib_file = calib_dir / f"{seq_name}.txt"
        pose_file = pose_dir / f"{seq_name}.txt"

        if not gt_file.exists():
            print(f"[WARN] GT not found: {gt_file}")
            continue
        if not calib_file.exists():
            print(f"[WARN] calib not found: {calib_file}")
            continue
        if not pose_file.exists():
            print(f"[WARN] pose not found: {pose_file}")
            continue

        gt_items_raw = load_kitti_tracking_file(gt_file)
        pred_items_raw = load_kitti_tracking_file(pred_file)

        gt_items = add_global_xy(gt_items_raw, calib_file, pose_file)
        pred_items = add_global_xy(pred_items_raw, calib_file, pose_file)

        total_gt_items += len(gt_items)
        total_pred_items += len(pred_items)

        gt_by_frame = group_by_frame(gt_items)
        pred_by_frame = group_by_frame(pred_items)

        gt_tracks = group_by_track(gt_items)
        pred_tracks = group_by_track(pred_items)

        common_frames = sorted(set(gt_by_frame.keys()) & set(pred_by_frame.keys()))

        for frame_id in common_frames:
            pairs = greedy_match_frame_by_global_center(
                gt_by_frame[frame_id],
                pred_by_frame[frame_id],
                args.dist_thre,
            )

            total_frame_matches += len(pairs)

            for gt, pred, dist in pairs:
                prev_gt = gt_tracks[gt["track_id"]].get(frame_id - 1)
                prev_pred = pred_tracks[pred["track_id"]].get(frame_id - 1)

                if prev_gt is None or prev_pred is None:
                    continue

                usable_velocity_pairs += 1

                gt_state = strict_static_global(gt, prev_gt, args.zero_eps)

                gt_speed = calc_speed_global(gt, prev_gt, args.fps)
                pred_speed = calc_speed_global(pred, prev_pred, args.fps)

                gt_vec = calc_velocity_vec_global(gt, prev_gt, args.fps)
                pred_vec = calc_velocity_vec_global(pred, prev_pred, args.fps)

                speed_abs_error = abs(pred_speed - gt_speed)

                vx_err = pred_vec[0] - gt_vec[0]
                vy_err = pred_vec[1] - gt_vec[1]
                velocity_vec_error = math.sqrt(vx_err * vx_err + vy_err * vy_err)

                dir_err = direction_error_deg(pred_vec, gt_vec)

                speed_abs_errors_all.append(speed_abs_error)
                velocity_vec_errors_all.append(velocity_vec_error)

                if gt_state == "static":
                    static_count += 1
                    static_speed_abs_errors.append(speed_abs_error)

                    if pred_speed > 1e-12:
                        static_pred_nonzero_count += 1
                    if pred_speed > 0.5:
                        static_pred_speed_over_05_count += 1
                    if pred_speed > 1.0:
                        static_pred_speed_over_10_count += 1

                else:
                    moving_count += 1
                    moving_speed_abs_errors.append(speed_abs_error)
                    if dir_err is not None:
                        moving_direction_errors.append(dir_err)

                detail_rows.append({
                    "seq": seq_name,
                    "frame": frame_id,
                    "gt_id": gt["track_id"],
                    "pred_id": pred["track_id"],
                    "global_center_dist": dist,
                    "gt_state_strict_global": gt_state,

                    "gt_global_x": gt["global_x"],
                    "gt_global_y": gt["global_y"],
                    "pred_global_x": pred["global_x"],
                    "pred_global_y": pred["global_y"],

                    "gt_speed_global": gt_speed,
                    "pred_speed_global": pred_speed,
                    "speed_abs_error": speed_abs_error,
                    "velocity_vec_error": velocity_vec_error,
                    "direction_error_deg": dir_err if dir_err is not None else "",

                    "pred_score": pred["score"],
                })

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "seq", "frame", "gt_id", "pred_id",
            "global_center_dist", "gt_state_strict_global",
            "gt_global_x", "gt_global_y",
            "pred_global_x", "pred_global_y",
            "gt_speed_global", "pred_speed_global",
            "speed_abs_error", "velocity_vec_error",
            "direction_error_deg",
            "pred_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_lines = []
    summary_lines.append("========== Velocity V0 Global BEV Analysis ==========")
    summary_lines.append(f"dataset_root: {dataset_root}")
    summary_lines.append(f"gt_dir: {gt_dir}")
    summary_lines.append(f"pred_dir: {pred_dir}")
    summary_lines.append(f"fps: {args.fps}")
    summary_lines.append(f"dist_thre: {args.dist_thre} m")
    summary_lines.append(f"zero_eps: {args.zero_eps}")
    summary_lines.append("")
    summary_lines.append(f"total_gt_items: {total_gt_items}")
    summary_lines.append(f"total_pred_items: {total_pred_items}")
    summary_lines.append(f"total_frame_matches: {total_frame_matches}")
    summary_lines.append(f"usable_velocity_pairs: {usable_velocity_pairs}")
    summary_lines.append("")
    summary_lines.append("Strict global static definition:")
    summary_lines.append("static: ego-motion-compensated global BEV position exactly unchanged from previous frame")
    summary_lines.append("moving: global BEV position changed by any amount")
    summary_lines.append("")
    summary_lines.append(f"static_count: {static_count}")
    summary_lines.append(f"moving_count: {moving_count}")
    summary_lines.append("")
    summary_lines.append(f"mean_speed_abs_error_all: {mean(speed_abs_errors_all):.6f} m/s")
    summary_lines.append(f"mean_velocity_vec_error_all: {mean(velocity_vec_errors_all):.6f} m/s")
    summary_lines.append("")
    summary_lines.append(f"mean_speed_abs_error_static: {mean(static_speed_abs_errors):.6f} m/s")
    summary_lines.append(f"mean_speed_abs_error_moving: {mean(moving_speed_abs_errors):.6f} m/s")
    summary_lines.append(f"mean_direction_error_moving: {mean(moving_direction_errors):.6f} deg")
    summary_lines.append("")
    summary_lines.append(f"static_pred_nonzero_count: {static_pred_nonzero_count}")
    summary_lines.append(f"static_pred_nonzero_rate: {rate(static_pred_nonzero_count, static_count):.6f}")
    summary_lines.append("")
    summary_lines.append(f"static_pred_speed_over_0.5_count: {static_pred_speed_over_05_count}")
    summary_lines.append(f"static_pred_speed_over_0.5_rate: {rate(static_pred_speed_over_05_count, static_count):.6f}")
    summary_lines.append("")
    summary_lines.append(f"static_pred_speed_over_1.0_count: {static_pred_speed_over_10_count}")
    summary_lines.append(f"static_pred_speed_over_1.0_rate: {rate(static_pred_speed_over_10_count, static_count):.6f}")
    summary_lines.append("")
    summary_lines.append(f"detail_csv: {args.out_csv}")

    summary_text = "\n".join(summary_lines)

    with open(args.out_summary, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)


if __name__ == "__main__":
    main()

    # (MCTrack)
    # hanbaobao @ hanbaobao - Legion - Y9000P - IRX9: ~ / program / MCTrack - main$ python
    # tools / analyze_velocity_v0.py - -dataset_root
    # data / kitti / datasets / tracking / training - -pred_dir
    # results / kitti / 20260526_093001 / virconv / training /
    # data - -fps
    # 10 - -dist_thre
    # 2.0 - -zero_eps
    # 0.02 - -out_csv
    # velocity_v0_global_detail_eps1e6.csv - -out_summary
    # velocity_v0_global_summary_eps1e6.txt