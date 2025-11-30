import csv
import math


def load_track(csv_path):
    cones = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag = row["tag"]
                x = float(row["x"])
                y = float(row["y"])
                cones.append({
                    "tag": tag,
                    "x": x,
                    "y": y,
                })
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found {csv_path!r}")
    return cones


def compute_world_bounds(cones, margin=2.0):
    if not cones:
        return -10, 10, -10, 10

    xs = [c["x"] for c in cones]
    ys = [c["y"] for c in cones]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    return min_x - margin, max_x + margin, min_y - margin, max_y + margin


def get_start_pos(cones):
    for c in cones:
        if c["tag"] == "car_start":
            return c["x"], c["y"]
    return 0.0, 0.0


def get_goal_pos(cones):
    if not cones:
        return 10.0, 10.0

    max_dist_sq = -1
    furthest_cone = None

    for c in cones:
        dist_sq = c["x"] ** 2 + c["y"] ** 2
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            furthest_cone = c

    if furthest_cone:
        return furthest_cone["x"] * 0.9, furthest_cone["y"] * 0.9

    return 10.0, 10.0


def get_obstacles(cones, size=1.0):
    obstacles = []
    for c in cones:
        if c["tag"] != "car_start":
            obstacles.append((c["x"], c["y"], size))
    return obstacles
