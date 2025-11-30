import numpy as np
import math
import random


class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0


class LocalRRTStar:
    """
    RRT* optimisé pour échantillonner directement dans le cône de vision (Polar Sampling).
    """

    def __init__(self, start, goal, obstacle_list, expand_dis=1.0, max_iter=50,
                 car_yaw=0.0, fov=np.pi / 2, view_dist=10.0):

        self.start = RRTNode(start[0], start[1])
        self.goal = RRTNode(goal[0], goal[1])

        self.expand_dis = expand_dis
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = [self.start]

        # Paramètres du cône pour l'échantillonnage
        self.car_yaw = car_yaw
        self.fov = fov
        self.view_dist = view_dist

    def plan(self):
        for _ in range(self.max_iter):
            rnd = self.get_random_node()

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
        """
        Génère un point aléatoire.
        Au lieu d'un carré, on utilise des coordonnées polaires pour rester dans le cône.
        """
        # Goal Bias : 10% de chance de viser directement le but
        if random.randint(0, 100) > 10:
            # 1. Échantillonnage de la distance (r)
            # Astuce mathématique : sqrt(random()) permet une distribution uniforme dans le triangle/cercle
            # Sinon les points s'agglutinent au centre.
            r = math.sqrt(random.random()) * self.view_dist

            # 2. Échantillonnage de l'angle (theta)
            # On tire un angle entre [-FOV/2, +FOV/2]
            angle_offset = random.uniform(-self.fov / 2, self.fov / 2)
            theta = self.car_yaw + angle_offset

            # 3. Conversion en Cartésien
            rx = self.start.x + r * math.cos(theta)
            ry = self.start.y + r * math.sin(theta)

            return RRTNode(rx, ry)
        else:
            return RRTNode(self.goal.x, self.goal.y)

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
            obs = np.array([ox, oy])
            if np.linalg.norm(obs - p1) < size: continue

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
        self.VIEW_DIST = 8.0

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

    def plan_local_path(self, car_pos, car_yaw, all_cones):
        obstacles = self.get_visible_obstacles(car_pos, car_yaw, all_cones)

        blues = [o for o in obstacles if o['tag'] == 'blue']
        yellows = [o for o in obstacles if o['tag'] == 'yellow']

        target_x, target_y = 0, 0
        track_width_estimation = 4.0

        if blues and yellows:
            avg_b = np.mean([[b['x'], b['y']] for b in blues], axis=0)
            avg_y = np.mean([[y['x'], y['y']] for y in yellows], axis=0)
            target_x = (avg_b[0] + avg_y[0]) / 2
            target_y = (avg_b[1] + avg_y[1]) / 2

        elif blues:
            avg_b = np.mean([[b['x'], b['y']] for b in blues], axis=0)
            dir_x, dir_y = math.cos(car_yaw), math.sin(car_yaw)
            perp_x, perp_y = dir_y, -dir_x
            target_x = avg_b[0] + perp_x * (track_width_estimation / 2) + dir_x * 3
            target_y = avg_b[1] + perp_y * (track_width_estimation / 2) + dir_y * 3

        elif yellows:
            avg_y = np.mean([[y['x'], y['y']] for y in yellows], axis=0)
            dir_x, dir_y = math.cos(car_yaw), math.sin(car_yaw)
            perp_x, perp_y = -dir_y, dir_x
            target_x = avg_y[0] + perp_x * (track_width_estimation / 2) + dir_x * 3
            target_y = avg_y[1] + perp_y * (track_width_estimation / 2) + dir_y * 3

        else:
            target_x = car_pos[0] + 5 * math.cos(car_yaw)
            target_y = car_pos[1] + 5 * math.sin(car_yaw)

        obs_list_rrt = [(o['x'], o['y'], o['r']) for o in obstacles]

        # On n'a plus besoin de calculer 'min_area' et 'max_area' (la boîte)
        # On passe directement les propriétés du cône au RRT

        rrt = LocalRRTStar(start=car_pos, goal=(target_x, target_y),
                           obstacle_list=obs_list_rrt,
                           expand_dis=1.5,
                           max_iter=60,
                           # Nouveaux paramètres :
                           car_yaw=car_yaw,
                           fov=self.FOV,
                           view_dist=self.VIEW_DIST)

        path, tree_nodes = rrt.plan()

        if len(path) <= 1:
            path = [car_pos, (target_x, target_y)]

        return path, tree_nodes, obs_list_rrt