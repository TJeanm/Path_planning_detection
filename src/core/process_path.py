import numpy as np
from scipy.interpolate import splprep, splev
import math


class PathProcessor:
    def __init__(self):
        pass

    def compute_track_centerline(self, yellow_cones, blue_cones, start_pos):
        """
        Calculates the centerline by taking the midpoint between each yellow cone
        and its nearest blue neighbor, then sorts them to form a coherent loop.
        """
        if not yellow_cones or not blue_cones:
            print("Error: Not enough cones to compute centerline.")
            return []

        yellow = np.array(yellow_cones)
        blue = np.array(blue_cones)

        # 1. Find midpoints (Yellow <-> Nearest Blue)
        midpoints = []
        for y_pt in yellow:
            # Euclidean distance to all blue cones
            dists = np.linalg.norm(blue - y_pt, axis=1)
            nearest_blue_idx = np.argmin(dists)
            b_pt = blue[nearest_blue_idx]

            # Calculate midpoint
            mid = (y_pt + b_pt) / 2
            midpoints.append(tuple(mid))

        midpoints = np.array(midpoints)
        if len(midpoints) == 0:
            return []

        # 2. Sort points (Greedy Nearest Neighbor) to form a path
        # Start with the point closest to the car's start position
        start_arr = np.array(start_pos)
        dists_to_start = np.linalg.norm(midpoints - start_arr, axis=1)
        current_idx = np.argmin(dists_to_start)

        sorted_indices = [current_idx]
        visited = set([current_idx])
        n_points = len(midpoints)

        while len(sorted_indices) < n_points:
            current_pos = midpoints[current_idx]

            # Distances to all points
            dists = np.linalg.norm(midpoints - current_pos, axis=1)

            # Mask visited points with infinity so they aren't picked
            dists[list(visited)] = np.inf

            next_idx = np.argmin(dists)
            min_dist = dists[next_idx]

            # If the nearest unvisited point is too far (e.g., > 20m),
            # it might be a jump across the track. For now, we continue.
            if min_dist == np.inf:
                break

            sorted_indices.append(next_idx)
            visited.add(next_idx)
            current_idx = next_idx

        # Reconstruct ordered path
        ordered_path = [tuple(midpoints[i]) for i in sorted_indices]

        # Close the loop
        if ordered_path:
            ordered_path.append(ordered_path[0])

        return ordered_path

    def smooth_path(self, path):
        """
        Smooths the path using a Cubic B-Spline.
        """
        if len(path) < 3:
            return path

        path_arr = np.array(path)

        # Filter duplicates or points that are too close (< 1cm)
        diffs = np.diff(path_arr, axis=0)
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        mask = np.concatenate(([True], dists > 0.01))
        clean_path = path_arr[mask]

        if len(clean_path) < 4:
            return clean_path.tolist()

        try:
            x = clean_path[:, 0]
            y = clean_path[:, 1]

            # Spline parameters: s=0.5 (match your working script), k=3, per=True (closed loop)
            tck, u = splprep([x, y], s=0.5, k=3, per=True)

            # Generate high density points for smooth animation
            u_new = np.linspace(0, 1, num=len(clean_path) * 10)
            x_new, y_new = splev(u_new, tck)

            return list(zip(x_new, y_new))
        except Exception as e:
            print(f"Spline smoothing failed ({e}). Using raw path.")
            return clean_path.tolist()


# Compatibility wrapper
def compute_centerline(yellow, blue, start):
    p = PathProcessor()
    return p.compute_track_centerline(yellow, blue, start)


def smooth_path(path):
    p = PathProcessor()
    return p.smooth_path(path)