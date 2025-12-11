import numpy as np
import math
import random
from scipy.interpolate import splprep, splev


# --- CLASSES RRT (Standard) ---
class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0


class LocalRRTStar:
    def __init__(self, start, goal, obstacle_list, expand_dis=1.0, max_iter=50,
                 car_yaw=0.0, fov=np.pi / 2, view_dist=10.0):
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
        if len(self.obstacle_list) > 100:
            effective_iter = max(30, int(self.max_iter * 0.7))

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
        path = self.generate_course(last_index)
        return path, self.node_list

    def choose_parent(self, new_node, near_inds):
        if not near_inds: return new_node
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacle_list):
                costs.append(near_node.cost + self.dist(near_node, new_node))
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
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node: continue
            edge_node.cost = new_node.cost + self.dist(new_node, near_node)
            if near_node.cost > edge_node.cost and self.check_collision(edge_node, self.obstacle_list):
                self.node_list[i] = edge_node
                self.node_list[i].parent = new_node

    def find_near_nodes(self, new_node):
        n = len(self.node_list) + 1
        if n < 2:
            r = 10.0
        else:
            r = 50.0 * math.sqrt((math.log(n) / n))
        d_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for node in self.node_list]
        return [i for i, d in enumerate(d_list) if d <= r ** 2]

    def get_random_node(self):
        for _ in range(10):
            if random.randint(0, 100) > 20:
                r = math.sqrt(random.random()) * self.view_dist
                angle_offset = random.uniform(-self.fov / 2, self.fov / 2)
                theta = self.car_yaw + angle_offset
                rx = self.start.x + r * math.cos(theta)
                ry = self.start.y + r * math.sin(theta)
                rnd = RRTNode(rx, ry)
            else:
                rnd = RRTNode(self.goal.x, self.goal.y)

            if self.car_yaw is not None:
                dx = rnd.x - self.start.x
                dy = rnd.y - self.start.y
                forward_dot = dx * math.cos(self.car_yaw) + dy * math.sin(self.car_yaw)
                if forward_dot < -0.5: continue
            return rnd
        return None

    def get_nearest_node_index(self, node_list, rnd):
        dlist = [(node.x - rnd.x) ** 2 + (node.y - rnd.y) ** 2 for node in node_list]
        return dlist.index(min(dlist))

    def get_best_last_index(self):
        dists = [(node.x - self.goal.x) ** 2 + (node.y - self.goal.y) ** 2 for node in self.node_list]
        return dists.index(min(dists))

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
            if abs(ox - p1[0]) > size + 2.0 and abs(ox - p2[0]) > size + 2.0: continue
            obs = np.array([ox, oy])
            if np.linalg.norm(obs - p1) < size: continue
            proj = np.dot(obs - p1, vec)
            if proj <= 0:
                closest = p1
            elif proj >= seg_len:
                closest = p2
            else:
                closest = p1 + proj * vec
            if np.linalg.norm(obs - closest) <= size: return False
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

        # Historique pour la topologie
        self.history_points = []

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
                dot_product = vec_cone_x * car_dir_x + vec_cone_y * car_dir_y
                if dot_product > self.fov_threshold:
                    visible.append({'x': c['x'], 'y': c['y'], 'r': 0.4, 'tag': c['tag']})
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
                            dist = math.hypot(v_cone['x'] - m_cone['x'], v_cone['y'] - m_cone['y'])
                            if dist < 0.6:
                                m_cone['x'] = v_cone['x']
                                m_cone['y'] = v_cone['y']
                                found = True
                                break
                    if found: break
                if found: break

            if not found:
                key = (gx, gy)
                if key not in self.memory_grid:
                    self.memory_grid[key] = []
                self.memory_grid[key].append(v_cone.copy())

    def get_relevant_obstacles(self, car_pos, radius=15.0):
        obstacles = []
        min_gx = int((car_pos[0] - radius) // self.CELL_SIZE)
        max_gx = int((car_pos[0] + radius) // self.CELL_SIZE)
        min_gy = int((car_pos[1] - radius) // self.CELL_SIZE)
        max_gy = int((car_pos[1] + radius) // self.CELL_SIZE)
        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                key = (gx, gy)
                if key in self.memory_grid:
                    for c in self.memory_grid[key]:
                        obstacles.append((c['x'], c['y'], c['r']))
        return obstacles

    def compute_global_path(self):
        """
        Calcule la trajectoire parfaite en utilisant l'historique pour l'ordre (topologie)
        et les cônes mémorisés pour la précision (géométrie).
        """
        print("Génération de la trajectoire globale (Topologie Historique)...")
        all_cones = []
        for cell in self.memory_grid.values():
            all_cones.extend(cell)

        blues = np.array([[c['x'], c['y']] for c in all_cones if c['tag'] == 'blue'])
        yellows = np.array([[c['x'], c['y']] for c in all_cones if c['tag'] == 'yellow'])

        if len(blues) == 0 or len(yellows) == 0: return []

        # 1. Nuage de points médians (sans ordre)
        # On filtre les paires aberrantes (trop éloignées)
        midpoints = []
        MAX_PAIR_DIST = 10.0  # Si bleu et jaune sont > 10m, ce n'est pas une porte valide

        for y_pt in yellows:
            dists = np.linalg.norm(blues - y_pt, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_dist = dists[nearest_idx]

            if nearest_dist < MAX_PAIR_DIST:
                nearest_blue = blues[nearest_idx]
                midpoints.append((y_pt + nearest_blue) / 2)

        midpoints = np.array(midpoints)
        if len(midpoints) < 3: return []

        # 2. PROJECTION SUR L'HISTORIQUE (ANTI COURT-CIRCUIT)
        # On utilise la trace de la voiture pour savoir dans quel ordre relier les points
        ordered_points = []

        # On échantillonne l'historique (1 point tous les ~5 itérations) pour la perf
        for car_pos in self.history_points[::5]:
            # Trouver le midpoint le plus proche de cette position passée
            dists = np.linalg.norm(midpoints - car_pos, axis=1)
            nearest_idx = np.argmin(dists)
            min_dist = dists[nearest_idx]

            # Si le midpoint est raisonnablement proche (< 6m) de la trace
            if min_dist < 6.0:
                # On l'ajoute seulement s'il est différent du dernier ajouté (évite les doublons)
                if len(ordered_points) == 0 or np.linalg.norm(midpoints[nearest_idx] - ordered_points[-1]) > 0.5:
                    ordered_points.append(midpoints[nearest_idx])

        ordered_points = np.array(ordered_points)

        # Fallback si la projection échoue (pas assez de points)
        if len(ordered_points) < 5:
            print("Attention: Projection topologique insuffisante, utilisation de l'historique brut.")
            ordered_points = np.array(self.history_points)

        try:
            # 3. Lissage Spline Fermé
            # Nettoyage des points trop proches pour que Spline ne crash pas
            clean_pts = [ordered_points[0]]
            for pt in ordered_points[1:]:
                if np.linalg.norm(pt - clean_pts[-1]) > 0.5:
                    clean_pts.append(pt)

            # Fermeture explicite si nécessaire
            if np.linalg.norm(clean_pts[0] - clean_pts[-1]) > 2.0:
                clean_pts.append(clean_pts[0])

            clean_pts = np.array(clean_pts)
            if len(clean_pts) < 4: return clean_pts.tolist()

            x = clean_pts[:, 0]
            y = clean_pts[:, 1]

            # --- MODIFICATION ICI : MOINS DE SMOOTHING ---
            # s=len*0.1 force la courbe à passer beaucoup plus près des points
            smooth_factor = len(clean_pts) * 0.1
            tck, u = splprep([x, y], s=smooth_factor, k=3, per=True)

            u_new = np.linspace(0, 1, num=len(clean_pts) * 5)
            x_new, y_new = splev(u_new, tck)

            return list(zip(x_new, y_new))
        except Exception as e:
            print(f"Erreur spline globale: {e}")
            return ordered_points.tolist()

    def plan_local_path(self, car_pos, car_yaw, all_cones):
        if self.start_pos is None:
            self.start_pos = car_pos

        # --- MODE 2 : COURSE (MÉMOIRE PURE) ---
        if self.lap_completed:
            # On retourne la trajectoire globale calculée
            # On retourne des listes vides pour tree_nodes et visible_obstacles
            # => L'interface n'affichera plus rien d'autre que la trajectoire bleue
            return self.global_path, [], []

        # --- MODE 1 : DÉCOUVERTE (PERCEPTION) ---

        # 1. Enregistrement Historique (Fil d'Ariane)
        # On enregistre si on a bougé d'au moins 0.5m
        if not self.history_points or math.hypot(car_pos[0] - self.history_points[-1][0],
                                                 car_pos[1] - self.history_points[-1][1]) > 0.5:
            self.history_points.append(car_pos)

        # 2. Perception & Mémoire
        currently_visible = self.get_visible_obstacles(car_pos, car_yaw, all_cones)
        self.update_memory(currently_visible)

        # 3. Détection Fin de Tour
        dist_to_start = math.hypot(car_pos[0] - self.start_pos[0], car_pos[1] - self.start_pos[1])

        if dist_to_start > 15.0:  # Doit s'être éloigné d'au moins 15m
            self.has_left_start_area = True

        if self.has_left_start_area and dist_to_start < 8.0:  # Revenu à moins de 8m
            print(">>> FIN DU TOUR 1. Calcul trajectoire optimale... <<<")
            self.lap_completed = True
            self.global_path = self.compute_global_path()
            return self.global_path, [], []

        # 4. Calcul Cible Local (RRT)
        blues = [o for o in currently_visible if o['tag'] == 'blue']
        yellows = [o for o in currently_visible if o['tag'] == 'yellow']
        target_x, target_y = 0, 0
        track_width = 4.0

        if blues and yellows:
            avg_b = np.mean([[b['x'], b['y']] for b in blues], axis=0)
            avg_y = np.mean([[y['x'], y['y']] for y in yellows], axis=0)
            target_x = (avg_b[0] + avg_y[0]) / 2
            target_y = (avg_b[1] + avg_y[1]) / 2
        elif blues:
            avg_b = np.mean([[b['x'], b['y']] for b in blues], axis=0)
            dir_x, dir_y = math.cos(car_yaw), math.sin(car_yaw)
            perp_x, perp_y = dir_y, -dir_x
            target_x = avg_b[0] + perp_x * (track_width / 2) + dir_x * 3
            target_y = avg_b[1] + perp_y * (track_width / 2) + dir_y * 3
        elif yellows:
            avg_y = np.mean([[y['x'], y['y']] for y in yellows], axis=0)
            dir_x, dir_y = math.cos(car_yaw), math.sin(car_yaw)
            perp_x, perp_y = -dir_y, dir_x
            target_x = avg_y[0] + perp_x * (track_width / 2) + dir_x * 3
            target_y = avg_y[1] + perp_y * (track_width / 2) + dir_y * 3
        else:
            target_x = car_pos[0] + 5 * math.cos(car_yaw)
            target_y = car_pos[1] + 5 * math.sin(car_yaw)

        # 5. RRT Local
        relevant_obstacles = self.get_relevant_obstacles(car_pos, radius=15.0)
        rrt = LocalRRTStar(start=car_pos, goal=(target_x, target_y),
                           obstacle_list=relevant_obstacles,
                           expand_dis=1.5,
                           max_iter=60,
                           car_yaw=car_yaw,
                           fov=self.FOV,
                           view_dist=self.VIEW_DIST)

        path, tree_nodes = rrt.plan()

        if len(path) <= 1:
            path = [car_pos, (target_x, target_y)]

        return path, tree_nodes, currently_visible