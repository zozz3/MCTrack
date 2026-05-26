import argparse
import csv
from pathlib import Path
from collections import defaultdict


def to_float(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def rate(n, d):
    return n / d if d > 0 else 0.0


def get_state(row):
    """
    兼容不同版本 detail csv 的字段名。
    """
    if "gt_state_strict_global" in row:
        return row["gt_state_strict_global"]
    if "gt_state_strict" in row:
        return row["gt_state_strict"]
    return ""


def get_pred_speed(row):
    """
    兼容不同版本字段名。
    """
    if "pred_speed_global" in row:
        return to_float(row["pred_speed_global"])
    if "pred_speed" in row:
        return to_float(row["pred_speed"])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail_csv", required=True, help="velocity_v0_global_detail_xxx.csv")
    parser.add_argument("--out_csv", default="velocity_v0_scene_summary.csv")
    parser.add_argument("--out_txt", default="velocity_v0_scene_summary.txt")
    args = parser.parse_args()

    detail_path = Path(args.detail_csv)

    if not detail_path.exists():
        raise FileNotFoundError(f"detail csv not found: {detail_path}")

    scene_stats = defaultdict(lambda: {
        "total": 0,
        "static_count": 0,
        "moving_count": 0,

        "speed_abs_error_all": [],
        "velocity_vec_error_all": [],

        "speed_abs_error_static": [],
        "speed_abs_error_moving": [],
        "direction_error_moving": [],

        "static_pred_nonzero_count": 0,
        "static_pred_speed_over_0_5_count": 0,
        "static_pred_speed_over_1_0_count": 0,
    })

    with open(detail_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            seq = row.get("seq", "")
            if seq == "":
                continue

            stat = scene_stats[seq]
            stat["total"] += 1

            state = get_state(row)
            pred_speed = get_pred_speed(row)

            speed_abs_error = to_float(row.get("speed_abs_error"))
            velocity_vec_error = to_float(row.get("velocity_vec_error"))
            direction_error = to_float(row.get("direction_error_deg"))

            stat["speed_abs_error_all"].append(speed_abs_error)
            stat["velocity_vec_error_all"].append(velocity_vec_error)

            if state == "static":
                stat["static_count"] += 1
                stat["speed_abs_error_static"].append(speed_abs_error)

                if pred_speed is not None:
                    if pred_speed > 1e-12:
                        stat["static_pred_nonzero_count"] += 1
                    if pred_speed > 0.5:
                        stat["static_pred_speed_over_0_5_count"] += 1
                    if pred_speed > 1.0:
                        stat["static_pred_speed_over_1_0_count"] += 1

            elif state == "moving":
                stat["moving_count"] += 1
                stat["speed_abs_error_moving"].append(speed_abs_error)
                if direction_error is not None:
                    stat["direction_error_moving"].append(direction_error)

    rows = []

    for seq in sorted(scene_stats.keys()):
        s = scene_stats[seq]

        static_count = s["static_count"]
        moving_count = s["moving_count"]
        total = s["total"]

        row = {
            "seq": seq,
            "usable_velocity_pairs": total,

            "static_count": static_count,
            "moving_count": moving_count,

            "static_ratio": rate(static_count, total),
            "moving_ratio": rate(moving_count, total),

            "mean_speed_abs_error_all": mean(s["speed_abs_error_all"]),
            "mean_velocity_vec_error_all": mean(s["velocity_vec_error_all"]),

            "mean_speed_abs_error_static": mean(s["speed_abs_error_static"]),
            "mean_speed_abs_error_moving": mean(s["speed_abs_error_moving"]),
            "mean_direction_error_moving": mean(s["direction_error_moving"]),

            "static_pred_nonzero_count": s["static_pred_nonzero_count"],
            "static_pred_nonzero_rate": rate(s["static_pred_nonzero_count"], static_count),

            "static_pred_speed_over_0_5_count": s["static_pred_speed_over_0_5_count"],
            "static_pred_speed_over_0_5_rate": rate(s["static_pred_speed_over_0_5_count"], static_count),

            "static_pred_speed_over_1_0_count": s["static_pred_speed_over_1_0_count"],
            "static_pred_speed_over_1_0_rate": rate(s["static_pred_speed_over_1_0_count"], static_count),
        }

        rows.append(row)

    fieldnames = [
        "seq",
        "usable_velocity_pairs",

        "static_count",
        "moving_count",
        "static_ratio",
        "moving_ratio",

        "mean_speed_abs_error_all",
        "mean_velocity_vec_error_all",

        "mean_speed_abs_error_static",
        "mean_speed_abs_error_moving",
        "mean_direction_error_moving",

        "static_pred_nonzero_count",
        "static_pred_nonzero_rate",

        "static_pred_speed_over_0_5_count",
        "static_pred_speed_over_0_5_rate",

        "static_pred_speed_over_1_0_count",
        "static_pred_speed_over_1_0_rate",
    ]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    lines.append("========== Velocity V0 Per-Scene Summary ==========")
    lines.append(f"detail_csv: {detail_path}")
    lines.append(f"scene_count: {len(rows)}")
    lines.append("")

    for row in rows:
        lines.append(f"----- Scene {row['seq']} -----")
        lines.append(f"usable_velocity_pairs: {row['usable_velocity_pairs']}")
        lines.append(f"static_count: {row['static_count']}")
        lines.append(f"moving_count: {row['moving_count']}")
        lines.append(f"static_ratio: {row['static_ratio']:.6f}")
        lines.append(f"moving_ratio: {row['moving_ratio']:.6f}")
        lines.append(f"mean_speed_abs_error_all: {row['mean_speed_abs_error_all']:.6f} m/s")
        lines.append(f"mean_velocity_vec_error_all: {row['mean_velocity_vec_error_all']:.6f} m/s")
        lines.append(f"mean_speed_abs_error_static: {row['mean_speed_abs_error_static']:.6f} m/s")
        lines.append(f"mean_speed_abs_error_moving: {row['mean_speed_abs_error_moving']:.6f} m/s")
        lines.append(f"mean_direction_error_moving: {row['mean_direction_error_moving']:.6f} deg")
        lines.append(f"static_pred_nonzero_count: {row['static_pred_nonzero_count']}")
        lines.append(f"static_pred_nonzero_rate: {row['static_pred_nonzero_rate']:.6f}")
        lines.append(f"static_pred_speed_over_0.5_count: {row['static_pred_speed_over_0_5_count']}")
        lines.append(f"static_pred_speed_over_0.5_rate: {row['static_pred_speed_over_0_5_rate']:.6f}")
        lines.append(f"static_pred_speed_over_1.0_count: {row['static_pred_speed_over_1_0_count']}")
        lines.append(f"static_pred_speed_over_1.0_rate: {row['static_pred_speed_over_1_0_rate']:.6f}")
        lines.append("")

    text = "\n".join(lines)

    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    print(f"[OK] saved csv: {args.out_csv}")
    print(f"[OK] saved txt: {args.out_txt}")


if __name__ == "__main__":
    main()

    # python
    # tools / summarize_velocity_by_scene.py \
    # - -detail_csv
    # velocity_v0_global_detail_eps1e6.csv \
    # - -out_csv
    # velocity_v0_scene_summary_eps002.csv \
    # - -out_txt
    # velocity_v0_scene_summary_eps002.txt