import numpy as np
import math
import random
from scipy.interpolate import splprep, splev


# --- CLASSES RRT ---
class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0


class LocalRRTStar:
    def __init__(self, start, goal, obstacle_list, expand_dis=1.0, max_iter=50,
                 car_yaw=None, fov=np.pi / 2, view_dist=10.0):
        self.start = RRTNode(start[0], start[1])
        self.goal = RRTNode(goal[0], goal[1])
        self.expand_dis = expand_dis
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = [self.start]
        self.car_yaw = car_yaw
        self.fov = fov
        self.view_dist = view_dist

    def plan(self):
        effective_iter = self.max_iter
        if len(self.obstacle_list) > 150:
            effective_iter = max(40, int(self.max_iter * 0.6))

        for _ in range(effective_iter):
            rnd = self.get_random_node()
            if rnd is None: continue

            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

        last_index = self.get_best_last_index()
        # IMPORTANT: On retourne self.node_list même si aucun chemin n'est trouvé
        # C'est ce qui permet d'afficher les arbres verts
        if last_index is None: return None, self.node_list

        path = self.generate_course(last_index)
        return path, self.node_list

    def choose_parent(self, new_node, near_inds):
        if not near_inds: return new_node
        costs = []
        for i in near_inds:
            t_node = self.steer(self.node_list[i], new_node)
            if t_node and self.check_collision(t_node, self.obstacle_list):
                costs.append(self.node_list[i].cost + self.dist(self.node_list[i], new_node))
            else:
                costs.append(float("inf"))
        min_cost = min(costs)
        if min_cost == float("inf"): return new_node
        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min_cost
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            edge_node = self.steer(new_node, self.node_list[i])
            if not edge_node: continue
            edge_node.cost = new_node.cost + self.dist(new_node, self.node_list[i])
            if self.node_list[i].cost > edge_node.cost and self.check_collision(edge_node, self.obstacle_list):
                self.node_list[i] = edge_node
                self.node_list[i].parent = new_node

    def find_near_nodes(self, new_node):
        n = len(self.node_list) + 1
        r = 50.0 * math.sqrt((math.log(n) / n)) if n > 1 else 10.0
        d_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for node in self.node_list]
        return [i for i, d in enumerate(d_list) if d <= r ** 2]

    def get_random_node(self):
        if self.car_yaw is not None:
            for _ in range(10):
                if random.randint(0, 100) > 10:
                    r = math.sqrt(random.random()) * self.view_dist
                    angle_offset = random.uniform(-self.fov / 2, self.fov / 2)
                    theta = self.car_yaw + angle_offset
                    rx = self.start.x + r * math.cos(theta)
                    ry = self.start.y + r * math.sin(theta)
                    rnd = RRTNode(rx, ry)
                else:
                    rnd = RRTNode(self.goal.x, self.goal.y)

                dx = rnd.x - self.start.x
                dy = rnd.y - self.start.y
                if dx * math.cos(self.car_yaw) + dy * math.sin(self.car_yaw) > -1.0:
                    return rnd
            return None
        else:
            if random.randint(0, 100) > 20:
                dx = self.goal.x - self.start.x
                dy = self.goal.y - self.start.y
                dist = math.hypot(dx, dy)
                angle = math.atan2(dy, dx)
                rand_dist = random.uniform(-1.0, dist + 1.0)
                rand_lat = random.uniform(-4.0, 4.0)
                rx = self.start.x + rand_dist * math.cos(angle) - rand_lat * math.sin(angle)
                ry = self.start.y + rand_dist * math.sin(angle) + rand_lat * math.cos(angle)
                return RRTNode(rx, ry)
            else:
                return RRTNode(self.goal.x, self.goal.y)

    def get_nearest_node_index(self, node_list, rnd):
        dlist = [(node.x - rnd.x) ** 2 + (node.y - rnd.y) ** 2 for node in node_list]
        return dlist.index(min(dlist))

    def get_best_last_index(self):
        dists = [(node.x - self.goal.x) ** 2 + (node.y - self.goal.y) ** 2 for node in self.node_list]
        best_ind = dists.index(min(dists))
        if math.sqrt(dists[best_ind]) <= self.expand_dis * 3:
            return best_ind
        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = RRTNode(from_node.x, from_node.y)
        d, theta = self.calc_dist_angle(new_node, to_node)
        if extend_length > d: extend_length = d
        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)
        new_node.parent = from_node
        return new_node

    def check_collision(self, node, obstacle_list):
        if node is None or node.parent is None: return False
        if not obstacle_list: return True
        p1 = np.array([node.parent.x, node.parent.y])
        p2 = np.array([node.x, node.y])
        vec = p2 - p1
        seg_len = np.linalg.norm(vec)
        if seg_len == 0: return True
        vec = vec / seg_len

        for (ox, oy, size) in obstacle_list:
            if abs(ox - p1[0]) > size + seg_len and abs(ox - p2[0]) > size + seg_len: continue

            obs = np.array([ox, oy])
            proj = np.dot(obs - p1, vec)
            if proj <= 0:
                closest = p1
            elif proj >= seg_len:
                closest = p2
            else:
                closest = p1 + proj * vec

            if np.linalg.norm(obs - closest) <= size:
                return False
        return True

    def generate_course(self, goal_ind):
        path = [[self.node_list[goal_ind].x, self.node_list[goal_ind].y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            node = node.parent
            path.append([node.x, node.y])
        return path[::-1]

    def calc_dist_angle(self, from_node, to_node):
        dx, dy = to_node.x - from_node.x, to_node.y - from_node.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def dist(self, n1, n2):
        return math.hypot(n1.x - n2.x, n1.y - n2.y)


class PerceptionProcessor:
    def __init__(self):
        self.FOV = np.deg2rad(100)
        self.fov_threshold = math.cos(self.FOV / 2)
        self.VIEW_DIST = 10.0
        self.memory_grid = {}
        self.CELL_SIZE = 2.0
        self.start_pos = None
        self.has_left_start_area = False
        self.lap_completed = False
        self.global_path = None
        self.history_points = []
        self.global_path_segments = []
        self.last_processed_idx = 0
        self.BATCH_SIZE = 6

        # Rayon obstacles ajusté à 0.9m
        self.OBSTACLE_RADIUS = 0.9

    def get_visible_obstacles(self, car_pos, car_yaw, all_cones):
        visible = []
        cx, cy = car_pos
        car_dir_x = math.cos(car_yaw)
        car_dir_y = math.sin(car_yaw)
        for c in all_cones:
            vec_cone_x = c['x'] - cx
            vec_cone_y = c['y'] - cy
            dist = math.hypot(vec_cone_x, vec_cone_y)
            if dist <= self.VIEW_DIST and dist > 0.01:
                vec_cone_x /= dist
                vec_cone_y /= dist
                dot = vec_cone_x * car_dir_x + vec_cone_y * car_dir_y
                if dot > self.fov_threshold:
                    visible.append({'x': c['x'], 'y': c['y'], 'r': self.OBSTACLE_RADIUS, 'tag': c['tag']})
        return visible

    def update_memory(self, visible_cones):
        for v_cone in visible_cones:
            gx = int(v_cone['x'] // self.CELL_SIZE)
            gy = int(v_cone['y'] // self.CELL_SIZE)
            found = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (gx + dx, gy + dy)
                    if key in self.memory_grid:
                        for m_cone in self.memory_grid[key]:
                            if math.hypot(v_cone['x'] - m_cone['x'], v_cone['y'] - m_cone['y']) < 0.6:
                                m_cone['x'], m_cone['y'] = v_cone['x'], v_cone['y']
                                found = True
                                break
                    if found: break
                if found: break
            if not found:
                if (gx, gy) not in self.memory_grid: self.memory_grid[(gx, gy)] = []
                self.memory_grid[(gx, gy)].append(v_cone.copy())

    def get_relevant_obstacles(self, car_pos, radius=15.0):
        obstacles = []
        min_gx, max_gx = int((car_pos[0] - radius) // self.CELL_SIZE), int((car_pos[0] + radius) // self.CELL_SIZE)
        min_gy, max_gy = int((car_pos[1] - radius) // self.CELL_SIZE), int((car_pos[1] + radius) // self.CELL_SIZE)
        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                if (gx, gy) in self.memory_grid:
                    for c in self.memory_grid[(gx, gy)]:
                        obstacles.append((c['x'], c['y'], c['r']))
        return obstacles

    def path_pruning(self, path, obstacles):
        if path is None or len(path) < 3: return path
        pruned_path = [path[0]]
        current_idx = 0
        while current_idx < len(path) - 1:
            next_idx = current_idx + 1
            for i in range(len(path) - 1, current_idx + 1, -1):
                p1, p2 = np.array(path[current_idx]), np.array(path[i])
                vec = p2 - p1
                seg_len = np.linalg.norm(vec)
                collision = False
                if seg_len > 0.1:
                    vec = vec / seg_len
                    for ox, oy, r in obstacles:
                        if abs(ox - p1[0]) > r + seg_len and abs(ox - p2[0]) > r + seg_len: continue
                        obs = np.array([ox, oy])
                        if np.linalg.norm(obs - p1) < r: collision = True; break
                        proj = np.dot(obs - p1, vec)
                        if proj <= 0:
                            closest = p1
                        elif proj >= seg_len:
                            closest = p2
                        else:
                            closest = p1 + proj * vec
                        if np.linalg.norm(obs - closest) <= r: collision = True; break
                if not collision:
                    next_idx = i
                    break
            pruned_path.append(path[next_idx])
            current_idx = next_idx
        return pruned_path

    def _compute_rrt_segment(self, start_wp, goal_wp):
        segment_center = ((start_wp[0] + goal_wp[0]) / 2, (start_wp[1] + goal_wp[1]) / 2)
        segment_obstacles = self.get_relevant_obstacles(segment_center, radius=20.0)

        rrt = LocalRRTStar(start=start_wp, goal=goal_wp,
                           obstacle_list=segment_obstacles,
                           expand_dis=2.0, max_iter=100,
                           car_yaw=None, fov=np.pi, view_dist=15.0)

        seg_path, _ = rrt.plan()

        if seg_path is None or len(seg_path) < 2:
            seg_path = [start_wp, goal_wp]

        return self.path_pruning(seg_path, segment_obstacles)

    def finalize_global_path(self):
        print("Finalisation de la trajectoire globale (Assemblage + Spline)...")

        full_path = []
        full_path.extend(self.global_path_segments)

        if len(self.history_points) > self.last_processed_idx:
            start_wp = self.history_points[self.last_processed_idx]
            goal_wp = self.history_points[0]
            last_segment = self._compute_rrt_segment(start_wp, goal_wp)
            if last_segment:
                if full_path:
                    full_path.extend(last_segment[1:])
                else:
                    full_path.extend(last_segment)

        try:
            clean_pts = [full_path[0]]
            for pt in full_path[1:]:
                if np.linalg.norm(np.array(pt) - np.array(clean_pts[-1])) > 0.5:
                    clean_pts.append(pt)

            if np.linalg.norm(np.array(clean_pts[0]) - np.array(clean_pts[-1])) > 2.0:
                clean_pts.append(clean_pts[0])

            clean_pts = np.array(clean_pts)
            if len(clean_pts) < 4: return clean_pts.tolist()

            x, y = clean_pts[:, 0], clean_pts[:, 1]
            smooth_val = len(clean_pts) * 0.1

            tck, u = splprep([x, y], s=smooth_val, k=3, per=True)
            u_new = np.linspace(0, 1, num=len(clean_pts) * 10)
            x_new, y_new = splev(u_new, tck)

            print(f"Trajectoire générée : {len(x_new)} points.")
            return list(zip(x_new, y_new))

        except Exception as e:
            print(f"Erreur spline: {e}")
            return full_path

    def plan_local_path(self, car_pos, car_yaw, all_cones):
        if self.start_pos is None: self.start_pos = car_pos

        # Mode 2
        if self.lap_completed:
            return self.global_path, [], []

        # Mode 1
        if not self.history_points or math.hypot(car_pos[0] - self.history_points[-1][0],
                                                 car_pos[1] - self.history_points[-1][1]) > 0.5:
            self.history_points.append(car_pos)

        # Calcul Incremental
        idx_end = self.last_processed_idx + self.BATCH_SIZE
        if idx_end < len(self.history_points):
            idx_start = self.last_processed_idx
            start_p = self.history_points[idx_start]
            goal_p = self.history_points[idx_end]

            new_segment = self._compute_rrt_segment(start_p, goal_p)

            if new_segment:
                if self.global_path_segments:
                    self.global_path_segments.extend(new_segment[1:])
                else:
                    self.global_path_segments.extend(new_segment)

            self.last_processed_idx = idx_end

        currently_visible = self.get_visible_obstacles(car_pos, car_yaw, all_cones)
        self.update_memory(currently_visible)

        dist_start = math.hypot(car_pos[0] - self.start_pos[0], car_pos[1] - self.start_pos[1])
        if dist_start > 15.0: self.has_left_start_area = True
        if self.has_left_start_area and dist_start < 8.0:
            print(">>> TOUR FINI. Finalisation trajectoire... <<<")
            self.lap_completed = True
            self.global_path = self.finalize_global_path()
            return self.global_path, [], []

        far_blues = [o for o in currently_visible if
                     o['tag'] == 'blue' and math.hypot(o['x'] - car_pos[0], o['y'] - car_pos[1]) > 3.0]
        far_yellows = [o for o in currently_visible if
                       o['tag'] == 'yellow' and math.hypot(o['x'] - car_pos[0], o['y'] - car_pos[1]) > 3.0]
        if not far_blues: far_blues = [o for o in currently_visible if o['tag'] == 'blue']
        if not far_yellows: far_yellows = [o for o in currently_visible if o['tag'] == 'yellow']

        tx, ty = 0, 0
        if far_blues and far_yellows:
            ab = np.mean([[c['x'], c['y']] for c in far_blues], axis=0)
            ay = np.mean([[c['x'], c['y']] for c in far_yellows], axis=0)
            tx, ty = (ab[0] + ay[0]) / 2, (ab[1] + ay[1]) / 2
        elif far_blues:
            ab = np.mean([[c['x'], c['y']] for c in far_blues], axis=0)
            tx, ty = ab[0] + math.sin(car_yaw) * 3 + math.cos(car_yaw) * 4, ab[1] - math.cos(car_yaw) * 3 + math.sin(
                car_yaw) * 4
        elif far_yellows:
            ay = np.mean([[c['x'], c['y']] for c in far_yellows], axis=0)
            tx, ty = ay[0] - math.sin(car_yaw) * 3 + math.cos(car_yaw) * 4, ay[1] + math.cos(car_yaw) * 3 + math.sin(
                car_yaw) * 4
        else:
            tx, ty = car_pos[0] + 5 * math.cos(car_yaw), car_pos[1] + 5 * math.sin(car_yaw)

        relevant_obs = self.get_relevant_obstacles(car_pos, radius=15.0)
        rrt = LocalRRTStar(start=car_pos, goal=(tx, ty), obstacle_list=relevant_obs,
                           expand_dis=1.5, max_iter=60, car_yaw=car_yaw, fov=self.FOV, view_dist=self.VIEW_DIST)
        path, tree = rrt.plan()

        if path is None or len(path) <= 1:
            path = [car_pos, (tx, ty)]

        return path, tree, currently_visible