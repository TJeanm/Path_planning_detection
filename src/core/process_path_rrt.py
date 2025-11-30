import numpy as np
import math
import random
from scipy.interpolate import splprep, splev


class RRTNode:
    """Nœud pour l'arbre RRT"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []
        self.path_y = []
        self.parent = None
        self.cost = 0.0


class RRTStar:
    """
    Planificateur de chemin RRT* (Rapidly-exploring Random Tree Star).
    Cherche le chemin optimal localement en évitant les obstacles (cônes).
    """

    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=2.0, path_resolution=0.5, goal_sample_rate=5,
                 max_iter=100):
        self.start = RRTNode(start[0], start[1])
        self.goal = RRTNode(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def plan(self):
        self.node_list = [self.start]

        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.goal, self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # Pas de chemin trouvé dans la limite d'itérations

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node, math.hypot(new_node.x - near_node.x, new_node.y - near_node.y))
            if t_node and self.check_collision(t_node, self.obstacle_list):
                costs.append(near_node.cost + math.hypot(new_node.x - near_node.x, new_node.y - near_node.y))
            else:
                costs.append(float("inf"))

        min_cost = min(costs)
        if min_cost == float("inf"):
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node, math.hypot(new_node.x - self.node_list[min_ind].x,
                                                                            new_node.y - self.node_list[min_ind].y))
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min_cost

        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node, math.hypot(near_node.x - new_node.x, near_node.y - new_node.y))
            if not edge_node:
                continue
            edge_node.cost = new_node.cost + math.hypot(near_node.x - new_node.x, near_node.y - new_node.y)

            if near_node.cost > edge_node.cost:
                if self.check_collision(edge_node, self.obstacle_list):
                    self.node_list[i] = edge_node
                    self.node_list[i].parent = new_node

    def find_near_nodes(self, new_node):
        n_node = len(self.node_list) + 1
        r = 50.0 * math.sqrt((math.log(n_node) / n_node))
        d_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for node in self.node_list]
        near_inds = [d_list.index(i) for i in d_list if i <= r ** 2]
        return near_inds

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = RRTNode(random.uniform(self.min_rand, self.max_rand),
                          random.uniform(self.min_rand, self.max_rand))
        else:
            rnd = RRTNode(self.goal.x, self.goal.y)
        return rnd

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = RRTNode(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)

        new_node.parent = from_node
        return new_node

    def check_collision(self, node, obstacle_list):
        if node is None:
            return False

        # On vérifie pour chaque point du segment généré
        for ix, iy in zip(node.path_x, node.path_y):
            for (ox, oy, size) in obstacle_list:
                dx = ox - ix
                dy = oy - iy
                d = dx * dx + dy * dy
                # Collision si distance < (rayon cone + marge sécurité)
                if d <= size ** 2:
                    return False  # Collision detectée
        return True  # Pas de collision (Safe)

    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path[::-1]  # Retourner le chemin inversé (Start -> Goal)

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def calc_dist_to_goal(self, x, y):
        dx = x - self.goal.x
        dy = y - self.goal.y
        return math.hypot(dx, dy)


class PathProcessor:
    def __init__(self):
        pass

    def compute_track_centerline(self, yellow_cones, blue_cones, start_pos):
        """
        Utilise RRT* pour générer une trajectoire, mais retourne le chemin brut.
        Le lissage se fera entièrement dans smooth_path.
        """
        if not yellow_cones or not blue_cones:
            print("Erreur: Pas assez de cônes.")
            return []

        yellow = np.array(yellow_cones)
        blue = np.array(blue_cones)

        # 1. Identifier les midpoints (Checkpoints)
        midpoints = []
        for y_pt in yellow:
            dists = np.linalg.norm(blue - y_pt, axis=1)
            nearest_blue_idx = np.argmin(dists)
            b_pt = blue[nearest_blue_idx]
            mid = (y_pt + b_pt) / 2
            midpoints.append(mid)

        midpoints = np.array(midpoints)
        if len(midpoints) == 0: return []

        # 2. Tri des checkpoints
        start_arr = np.array(start_pos)
        dists_to_start = np.linalg.norm(midpoints - start_arr, axis=1)
        current_idx = np.argmin(dists_to_start)

        sorted_indices = [current_idx]
        visited = set([current_idx])
        n_points = len(midpoints)

        while len(sorted_indices) < n_points:
            current_pos = midpoints[current_idx]
            dists = np.linalg.norm(midpoints - current_pos, axis=1)
            dists[list(visited)] = np.inf
            next_idx = np.argmin(dists)
            if dists[next_idx] == np.inf: break
            sorted_indices.append(next_idx)
            visited.add(next_idx)
            current_idx = next_idx

        waypoints = [midpoints[i] for i in sorted_indices]

        # 3. Préparation RRT
        obstacle_list = []
        cone_radius = 1.2  # Marge de sécurité augmentée pour éviter de raser les cônes
        for c in yellow_cones: obstacle_list.append((c[0], c[1], cone_radius))
        for c in blue_cones: obstacle_list.append((c[0], c[1], cone_radius))

        all_x = [x for x, y in yellow_cones] + [x for x, y in blue_cones]
        all_y = [y for x, y in yellow_cones] + [y for x, y in blue_cones]
        rand_area = [min(all_x) - 5, max(all_x) + 5]  # min/max simplifié

        full_path = []
        current_pos = start_pos
        targets = waypoints + [waypoints[0]]  # Boucle fermée

        # 4. Exécution RRT
        print(f"RRT* en cours sur {len(targets)} segments...")
        for i, target in enumerate(targets):
            # Paramètres RRT ajustés pour être moins chaotiques
            rrt = RRTStar(start=current_pos, goal=target,
                          obstacle_list=obstacle_list,
                          rand_area=rand_area,
                          max_iter=200,  # Un peu plus d'itérations
                          expand_dis=3.0,  # Pas plus grands
                          path_resolution=1.0)  # Résolution plus grossière pour moins de bruit

            segment = rrt.plan()

            if segment:
                if len(full_path) > 0:
                    full_path.extend(segment[1:])
                else:
                    full_path.extend(segment)
                current_pos = segment[-1]
            else:
                # Fallback ligne droite
                full_path.append(tuple(target))
                current_pos = target

        return full_path

    def smooth_path(self, path):
        """
        Lissage Robuste : Filtrage -> Moyenne Glissante -> Spline
        """
        if len(path) < 3: return path
        path_arr = np.array(path)

        # --- Étape 1 : Filtrage spatial (Supprimer les points trop proches) ---
        # On garde un point tous les X mètres pour définir la structure globale
        min_dist = 1.5
        clean_path = [path_arr[0]]
        for pt in path_arr[1:]:
            if np.linalg.norm(pt - clean_path[-1]) > min_dist:
                clean_path.append(pt)

        # Fermer la boucle proprement si nécessaire
        if np.linalg.norm(clean_path[0] - clean_path[-1]) > min_dist:
            clean_path.append(clean_path[0])  # On force la fermeture

        clean_path = np.array(clean_path)
        if len(clean_path) < 3: return path

        # --- Étape 2 : Moyenne Glissante (Iterative Smoothing) ---
        # C'est ce qui empêche la trajectoire de partir à l'opposé.
        # On lisse les angles vifs du RRT géométriquement.
        smoothed_points = np.copy(clean_path)
        iterations = 3  # Nombre de passes de lissage
        alpha = 0.3  # Force du lissage (0 = pas de changement, 1 = max)

        for _ in range(iterations):
            # Pour chaque point (sauf premier et dernier si pas boucle), on le rapproche de la moyenne de ses voisins
            # Ici on traite comme une boucle fermée
            num_pts = len(smoothed_points)
            new_points = np.copy(smoothed_points)
            for i in range(num_pts):
                prev_p = smoothed_points[(i - 1) % num_pts]
                curr_p = smoothed_points[i]
                next_p = smoothed_points[(i + 1) % num_pts]

                # Formule de lissage simple : P_new = (1-a)*P + a*(Prev + Next)/2
                new_points[i] = (1 - alpha) * curr_p + alpha * (prev_p + next_p) / 2
            smoothed_points = new_points

        # --- Étape 3 : B-Spline Finale ---
        try:
            x = smoothed_points[:, 0]
            y = smoothed_points[:, 1]

            # S=0 force la courbe à passer EXACTEMENT par nos points lissés (étape 2)
            # S élevé permet de couper. Comme on a déjà lissé à l'étape 2, on peut mettre s=0 ou petit.
            tck, u = splprep([x, y], s=0.5, k=3, per=True)

            u_new = np.linspace(0, 1, num=len(path) * 5)  # Haute résolution
            x_new, y_new = splev(u_new, tck)

            return list(zip(x_new, y_new))
        except Exception as e:
            print(f"Erreur Spline: {e}. Retour au chemin lissé géométriquement.")
            return smoothed_points.tolist()