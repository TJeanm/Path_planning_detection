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
    def __init__(self, start, goal, obstacle_list, virtual_walls, expand_dis=1.0, max_iter=50,
                 car_yaw=None, fov=np.pi / 2, view_dist=10.0):
        self.start = RRTNode(start[0], start[1])
        self.goal = RRTNode(goal[0], goal[1])
        self.expand_dis = expand_dis
        self.max_iter = max_iter

        # Obstacles ponctuels (Cercles)
        self.obstacles_np = np.array(obstacle_list) if obstacle_list else np.empty((0, 3))

        # Murs Virtuels (Lignes [x1, y1, x2, y2])
        self.walls_np = np.array(virtual_walls) if virtual_walls else np.empty((0, 4))

        self.node_list = [self.start]
        self.car_yaw = car_yaw
        self.fov = fov
        self.view_dist = view_dist

    def plan(self):
        n_obs = len(self.obstacles_np)
        effective_iter = self.max_iter
        if n_obs > 150: effective_iter = max(40, int(self.max_iter * 0.6))

        for _ in range(effective_iter):
            rnd = self.get_random_node()
            if rnd is None: continue

            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd, self.expand_dis)

            if not self.check_collision_complete(nearest_node, new_node):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

        last_index = self.get_best_last_index()
        if last_index is None: return None, self.node_list

        path = self.generate_course(last_index)
        return path, self.node_list

    def choose_parent(self, new_node, near_inds):
        if not near_inds: return new_node
        costs = []
        for i in near_inds:
            if not self.check_collision_complete(self.node_list[i], new_node):
                d = math.hypot(new_node.x - self.node_list[i].x, new_node.y - self.node_list[i].y)
                costs.append(self.node_list[i].cost + d)
            else:
                costs.append(float("inf"))
        min_cost = min(costs)
        if min_cost == float("inf"): return new_node
        min_ind = near_inds[costs.index(min_cost)]
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min_cost
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            d = math.hypot(near_node.x - new_node.x, near_node.y - new_node.y)
            scost = new_node.cost + d
            if near_node.cost > scost:
                if not self.check_collision_complete(new_node, near_node):
                    near_node.parent = new_node
                    near_node.cost = scost

    def find_near_nodes(self, new_node):
        n = len(self.node_list) + 1
        r = 50.0 * math.sqrt((math.log(n) / n)) if n > 1 else 10.0
        r2 = r ** 2
        return [i for i, node in enumerate(self.node_list) if
                (node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 <= r2]

    def get_random_node(self):
        if self.car_yaw is not None:  # Mode Local
            for _ in range(10):
                if random.random() > 0.1:
                    r = math.sqrt(random.random()) * self.view_dist
                    angle = self.car_yaw + random.uniform(-self.fov / 2, self.fov / 2)
                    rnd = RRTNode(self.start.x + r * math.cos(angle), self.start.y + r * math.sin(angle))
                else:
                    rnd = RRTNode(self.goal.x, self.goal.y)

                # Check devant
                if (rnd.x - self.start.x) * math.cos(self.car_yaw) + (rnd.y - self.start.y) * math.sin(
                        self.car_yaw) > -1.0:
                    return rnd
            return None
        else:  # Mode Global
            if random.random() > 0.2:
                dx = self.goal.x - self.start.x
                dy = self.goal.y - self.start.y
                angle = math.atan2(dy, dx)
                dist = math.hypot(dx, dy)
                rd = random.uniform(-1.0, dist + 1.0)
                rl = random.uniform(-4.0, 4.0)
                rx = self.start.x + rd * math.cos(angle) - rl * math.sin(angle)
                ry = self.start.y + rd * math.sin(angle) + rl * math.cos(angle)
                return RRTNode(rx, ry)
            else:
                return RRTNode(self.goal.x, self.goal.y)

    def get_nearest_node_index(self, node_list, rnd):
        dlist = [(node.x - rnd.x) ** 2 + (node.y - rnd.y) ** 2 for node in node_list]
        return dlist.index(min(dlist))

    def get_best_last_index(self):
        dists = [(node.x - self.goal.x) ** 2 + (node.y - self.goal.y) ** 2 for node in self.node_list]
        best_ind = dists.index(min(dists))
        if dists[best_ind] <= (self.expand_dis * 3) ** 2: return best_ind
        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = RRTNode(from_node.x, from_node.y)
        d = math.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        if extend_length > d: extend_length = d
        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)
        new_node.parent = from_node
        return new_node

    # --- COLLISION COMPLETE (Obstacles + Murs Virtuels) ---
    def check_collision_complete(self, n1, n2):
        # 1. Vérification obstacles ponctuels (Cercles)
        if self.obstacles_np.size > 0:
            p1 = np.array([n1.x, n1.y])
            p2 = np.array([n2.x, n2.y])
            vec = p2 - p1
            seg_len_sq = np.dot(vec, vec)

            if seg_len_sq > 0.0001:
                ox, oy, r = self.obstacles_np[:, 0], self.obstacles_np[:, 1], self.obstacles_np[:, 2]
                obs_vec_x = ox - p1[0]
                obs_vec_y = oy - p1[1]
                t = (obs_vec_x * vec[0] + obs_vec_y * vec[1]) / seg_len_sq
                t = np.clip(t, 0, 1)
                closest_x = p1[0] + t * vec[0]
                closest_y = p1[1] + t * vec[1]
                dist_sq = (ox - closest_x) ** 2 + (oy - closest_y) ** 2

                if np.any(dist_sq <= r ** 2): return True  # Collision obstacle

        # 2. Vérification Murs Virtuels (Intersection Segments)
        if self.walls_np.size > 0:
            # Algorithme vectoriel d'intersection de segments
            # Segment 1: Robot (p1->p2)
            # Segment 2: Mur (w1->w2)
            # On utilise le produit vectoriel (cross product) en 2D

            w1x, w1y = self.walls_np[:, 0], self.walls_np[:, 1]
            w2x, w2y = self.walls_np[:, 2], self.walls_np[:, 3]

            # Vecteurs
            rx, ry = n2.x - n1.x, n2.y - n1.y  # Robot vector
            sx, sy = w2x - w1x, w2y - w1y  # Wall vectors

            # Dénominateur commun (cross product r x s)
            denom = rx * sy - ry * sx

            # Eviter division par zéro (parallèles)
            # On crée un masque pour les non-parallèles
            mask = np.abs(denom) > 1e-6

            if np.any(mask):
                # Numérateurs
                # (q - p) x s
                qp_cross_s = (w1x - n1.x) * sy - (w1y - n1.y) * sx
                # (q - p) x r
                qp_cross_r = (w1x - n1.x) * ry - (w1y - n1.y) * rx

                # Paramètres t et u
                t = qp_cross_s / denom
                u = qp_cross_r / denom

                # Intersection si 0 <= t <= 1 ET 0 <= u <= 1
                collision_mask = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1) & mask

                if np.any(collision_mask): return True  # Collision Mur !

        return False

    def check_collision(self, node, obstacle_list):
        if node is None or node.parent is None: return False
        return self.check_collision_complete(node.parent, node)

    def generate_course(self, goal_ind):
        path = [[self.node_list[goal_ind].x, self.node_list[goal_ind].y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            node = node.parent
            path.append([node.x, node.y])
        return path[::-1]

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
        self.OBSTACLE_RADIUS = 0.9

    def get_visible_obstacles(self, car_pos, car_yaw, all_cones):
        visible = []
        cx, cy = car_pos
        car_dir_x, car_dir_y = math.cos(car_yaw), math.sin(car_yaw)
        for c in all_cones:
            dx, dy = c['x'] - cx, c['y'] - cy
            dist_sq = dx * dx + dy * dy
            if 0.0001 < dist_sq <= self.VIEW_DIST ** 2:
                dist = math.sqrt(dist_sq)
                dot = (dx / dist) * car_dir_x + (dy / dist) * car_dir_y
                if dot > self.fov_threshold:
                    visible.append({'x': c['x'], 'y': c['y'], 'r': self.OBSTACLE_RADIUS, 'tag': c['tag']})
        return visible

    def update_memory(self, visible_cones):
        for v_cone in visible_cones:
            gx, gy = int(v_cone['x'] // self.CELL_SIZE), int(v_cone['y'] // self.CELL_SIZE)
            found = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (gx + dx, gy + dy)
                    if key in self.memory_grid:
                        for m_cone in self.memory_grid[key]:
                            if (v_cone['x'] - m_cone['x']) ** 2 + (v_cone['y'] - m_cone['y']) ** 2 < 0.36:
                                m_cone['x'], m_cone['y'] = (m_cone['x'] + v_cone['x']) / 2, (
                                            m_cone['y'] + v_cone['y']) / 2
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
                        obstacles.append([c['x'], c['y'], c['r']])
        return obstacles

    def build_virtual_walls(self, car_pos, radius=15.0):
        """
        Construit des lignes (murs) entre les cônes de même couleur proches.
        """
        walls = []
        # On récupère les cônes en tant qu'objets complets (avec tag)
        local_cones = []
        min_gx, max_gx = int((car_pos[0] - radius) // self.CELL_SIZE), int((car_pos[0] + radius) // self.CELL_SIZE)
        min_gy, max_gy = int((car_pos[1] - radius) // self.CELL_SIZE), int((car_pos[1] + radius) // self.CELL_SIZE)

        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                if (gx, gy) in self.memory_grid:
                    local_cones.extend(self.memory_grid[(gx, gy)])

        blues = [c for c in local_cones if c['tag'] == 'blue']
        yellows = [c for c in local_cones if c['tag'] == 'yellow']

        # Fonction helper pour relier les voisins
        def connect_neighbors(cones_list):
            if len(cones_list) < 2: return
            # On trie ou on cherche le plus proche voisin pour chaque cône
            # Pour faire simple et rapide : chaque cône se connecte à ses 2 plus proches voisins
            # SI la distance est raisonnable (< 6m)

            # Optimisation: conversion numpy
            coords = np.array([[c['x'], c['y']] for c in cones_list])

            for i in range(len(coords)):
                # Distances à tous les autres
                dists = np.sum((coords - coords[i]) ** 2, axis=1)
                # On prend les indices triés (le premier est lui-même, dist=0)
                nearest_idxs = np.argsort(dists)

                # On connecte au plus proche (index 1) et au second (index 2)
                # si < 36m² (6m)
                for k in [1, 2]:
                    if k < len(nearest_idxs):
                        idx = nearest_idxs[k]
                        if dists[idx] < 36.0:
                            # On ajoute le mur [x1, y1, x2, y2]
                            walls.append([coords[i][0], coords[i][1], coords[idx][0], coords[idx][1]])

        connect_neighbors(blues)
        connect_neighbors(yellows)

        return walls

    def path_pruning(self, path, obstacles, walls):
        # Mise à jour pour accepter walls
        if path is None or len(path) < 3: return path
        obs_np = np.array(obstacles) if obstacles else np.empty((0, 3))
        # Pour le pruning, on ne checke pas les murs pour aller vite (on suppose que RRT a fait le job)
        # Mais pour être sûr, on pourrait. Ici on garde simple.
        if obs_np.size == 0: return path

        pruned_path = [path[0]]
        current_idx = 0
        path_arr = np.array(path)
        while current_idx < len(path) - 1:
            next_idx = current_idx + 1
            for i in range(len(path) - 1, current_idx + 1, -1):
                p1, p2 = np.array(path[current_idx]), np.array(path[i])
                vec = p2 - p1
                seg_len_sq = np.dot(vec, vec)
                if seg_len_sq > 0.01:
                    ox, oy, r = obs_np[:, 0], obs_np[:, 1], obs_np[:, 2]
                    obs_vec_x, obs_vec_y = ox - p1[0], oy - p1[1]
                    t = np.clip((obs_vec_x * vec[0] + obs_vec_y * vec[1]) / seg_len_sq, 0, 1)
                    closest_x, closest_y = p1[0] + t * vec[0], p1[1] + t * vec[1]
                    if not np.any((ox - closest_x) ** 2 + (oy - closest_y) ** 2 <= r ** 2):
                        next_idx = i
                        break
            pruned_path.append(path[next_idx])
            current_idx = next_idx
        return pruned_path

    def _compute_rrt_segment(self, start_wp, goal_wp):
        segment_center = ((start_wp[0] + goal_wp[0]) / 2, (start_wp[1] + goal_wp[1]) / 2)
        segment_obstacles = self.get_relevant_obstacles(segment_center, radius=20.0)
        segment_walls = self.build_virtual_walls(segment_center, radius=20.0)  # Murs pour Global aussi

        rrt = LocalRRTStar(start=start_wp, goal=goal_wp,
                           obstacle_list=segment_obstacles,
                           virtual_walls=segment_walls,  # Pass walls
                           expand_dis=2.0, max_iter=100,
                           car_yaw=None, fov=np.pi, view_dist=15.0)
        seg_path, _ = rrt.plan()
        if seg_path is None or len(seg_path) < 2: seg_path = [start_wp, goal_wp]
        return self.path_pruning(seg_path, segment_obstacles, segment_walls)

    def finalize_global_path(self):
        print("Finalisation Incrémentale...")
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
        except Exception:
            return full_path

    def plan_local_path(self, car_pos, car_yaw, all_cones):
        if self.start_pos is None: self.start_pos = car_pos
        if self.lap_completed: return self.global_path, [], []

        if not self.history_points or (car_pos[0] - self.history_points[-1][0]) ** 2 + (
                car_pos[1] - self.history_points[-1][1]) ** 2 > 0.25:
            self.history_points.append(car_pos)

        idx_end = self.last_processed_idx + self.BATCH_SIZE
        if idx_end < len(self.history_points):
            idx_start = self.last_processed_idx
            new_segment = self._compute_rrt_segment(self.history_points[idx_start], self.history_points[idx_end])
            if new_segment:
                if self.global_path_segments:
                    self.global_path_segments.extend(new_segment[1:])
                else:
                    self.global_path_segments.extend(new_segment)
            self.last_processed_idx = idx_end

        currently_visible = self.get_visible_obstacles(car_pos, car_yaw, all_cones)
        self.update_memory(currently_visible)

        dist_start_sq = (car_pos[0] - self.start_pos[0]) ** 2 + (car_pos[1] - self.start_pos[1]) ** 2
        if dist_start_sq > 225.0: self.has_left_start_area = True
        if self.has_left_start_area and dist_start_sq < 64.0:
            print(">>> TOUR FINI <<<")
            self.lap_completed = True
            self.global_path = self.finalize_global_path()
            return self.global_path, [], []

        # Ciblage Sémantique
        far_blues = [o for o in currently_visible if o['tag'] == 'blue']
        far_yellows = [o for o in currently_visible if o['tag'] == 'yellow']
        tx, ty = 0, 0
        if far_blues and far_yellows:
            # Plus proches
            nb = min(far_blues, key=lambda c: (c['x'] - car_pos[0]) ** 2 + (c['y'] - car_pos[1]) ** 2)
            ny = min(far_yellows, key=lambda c: (c['x'] - car_pos[0]) ** 2 + (c['y'] - car_pos[1]) ** 2)
            mid_x, mid_y = (nb['x'] + ny['x']) / 2, (nb['y'] + ny['y']) / 2
            dx, dy = ny['x'] - nb['x'], ny['y'] - nb['y']
            px, py = -dy, dx  # Perpendiculaire
            if px * math.cos(car_yaw) + py * math.sin(car_yaw) < 0: px, py = -px, -py
            norm = math.hypot(px, py)
            if norm > 0: px, py = px / norm, py / norm
            tx, ty = mid_x + px * 5.0, mid_y + py * 5.0
        elif far_blues:
            nb = min(far_blues, key=lambda c: (c['x'] - car_pos[0]) ** 2 + (c['y'] - car_pos[1]) ** 2)
            px, py = math.sin(car_yaw), -math.cos(car_yaw)
            tx = nb['x'] + px * 3.0 + math.cos(car_yaw) * 4.0
            ty = nb['y'] + py * 3.0 + math.sin(car_yaw) * 4.0
        elif far_yellows:
            ny = min(far_yellows, key=lambda c: (c['x'] - car_pos[0]) ** 2 + (c['y'] - car_pos[1]) ** 2)
            px, py = -math.sin(car_yaw), math.cos(car_yaw)
            tx = ny['x'] + px * 3.0 + math.cos(car_yaw) * 4.0
            ty = ny['y'] + py * 3.0 + math.sin(car_yaw) * 4.0
        else:
            tx, ty = car_pos[0] + 5.0 * math.cos(car_yaw), car_pos[1] + 5.0 * math.sin(car_yaw)

        relevant_obs = self.get_relevant_obstacles(car_pos, radius=15.0)
        virtual_walls = self.build_virtual_walls(car_pos, radius=15.0)  # NOUVEAU

        rrt = LocalRRTStar(start=car_pos, goal=(tx, ty),
                           obstacle_list=relevant_obs,
                           virtual_walls=virtual_walls,  # Pass walls
                           expand_dis=1.5, max_iter=60, car_yaw=car_yaw, fov=self.FOV, view_dist=self.VIEW_DIST)
        path, tree = rrt.plan()

        if path is None or len(path) <= 1: path = [car_pos, (tx, ty)]
        return path, tree, currently_visible