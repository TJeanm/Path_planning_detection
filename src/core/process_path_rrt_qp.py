import numpy as np
import math
import random
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import splprep, splev


# --- Classes RRT/RRT* (Identiques à la version précédente) ---

class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []
        self.path_y = []
        self.parent = None
        self.cost = 0.0


class RRTStar:
    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=2.0, path_resolution=0.5, max_iter=100):
        self.start = RRTNode(start[0], start[1])
        self.goal = RRTNode(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
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
        return None

    def choose_parent(self, new_node, near_inds):
        if not near_inds: return None
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node, math.hypot(new_node.x - near_node.x, new_node.y - near_node.y))
            if t_node and self.check_collision(t_node, self.obstacle_list):
                costs.append(near_node.cost + math.hypot(new_node.x - near_node.x, new_node.y - near_node.y))
            else:
                costs.append(float("inf"))
        min_cost = min(costs)
        if min_cost == float("inf"): return None
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
            if not edge_node: continue
            edge_node.cost = new_node.cost + math.hypot(near_node.x - new_node.x, near_node.y - new_node.y)
            if near_node.cost > edge_node.cost:
                if self.check_collision(edge_node, self.obstacle_list):
                    self.node_list[i] = edge_node
                    self.node_list[i].parent = new_node

    def find_near_nodes(self, new_node):
        n_node = len(self.node_list) + 1
        r = 50.0 * math.sqrt((math.log(n_node) / n_node))
        d_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for node in self.node_list]
        return [d_list.index(i) for i in d_list if i <= r ** 2]

    def get_random_node(self):
        if random.randint(0, 100) > 10:  # 10% chance to aim at goal
            return RRTNode(random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand))
        return RRTNode(self.goal.x, self.goal.y)

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in node_list]
        return dlist.index(min(dlist))

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = RRTNode(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        if extend_length > d: extend_length = d
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
        if node is None: return False
        for ix, iy in zip(node.path_x, node.path_y):
            for (ox, oy, size) in obstacle_list:
                if (ox - ix) ** 2 + (oy - iy) ** 2 <= size ** 2: return False
        return True

    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path[::-1]

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def calc_dist_to_goal(self, x, y):
        return math.hypot(x - self.goal.x, y - self.goal.y)


# --- Path Processor avec Pipeline QP ---

class PathProcessor:
    def __init__(self):
        pass

    def compute_track_centerline(self, yellow_cones, blue_cones, start_pos):
        """
        Combine l'extraction de topologie et RRT* pour trouver un chemin brut.
        """
        if not yellow_cones or not blue_cones: return []

        yellow = np.array(yellow_cones)
        blue = np.array(blue_cones)

        # 1. Calcul des points de passages (Waypoints)
        midpoints = []
        for y_pt in yellow:
            dists = np.linalg.norm(blue - y_pt, axis=1)
            b_pt = blue[np.argmin(dists)]
            midpoints.append((y_pt + b_pt) / 2)

        midpoints = np.array(midpoints)
        if len(midpoints) == 0: return []

        # 2. Tri pour définir l'ordre de la piste
        start_arr = np.array(start_pos)
        current_idx = np.argmin(np.linalg.norm(midpoints - start_arr, axis=1))
        sorted_indices = [current_idx]
        visited = set([current_idx])

        while len(sorted_indices) < len(midpoints):
            current_pos = midpoints[current_idx]
            dists = np.linalg.norm(midpoints - current_pos, axis=1)
            dists[list(visited)] = np.inf
            next_idx = np.argmin(dists)
            if dists[next_idx] == np.inf: break
            sorted_indices.append(next_idx)
            visited.add(next_idx)
            current_idx = next_idx

        waypoints = [midpoints[i] for i in sorted_indices]

        # 3. RRT* entre les waypoints
        obstacle_list = [(c[0], c[1], 1.2) for c in yellow_cones] + [(c[0], c[1], 1.2) for c in blue_cones]
        all_coords = yellow_cones + blue_cones
        min_x = min(c[0] for c in all_coords) - 5
        max_x = max(c[0] for c in all_coords) + 5
        rand_area = [min_x, max_x]

        full_path = []
        current_pos = start_pos

        # On boucle sur les waypoints (+1 pour fermer la boucle)
        targets = waypoints + [waypoints[0]]

        print(f"Planification RRT* QP sur {len(targets)} segments...")

        for target in targets:
            rrt = RRTStar(start=current_pos, goal=target, obstacle_list=obstacle_list,
                          rand_area=rand_area, max_iter=200, expand_dis=3.0, path_resolution=1.0)
            segment = rrt.plan()
            if segment:
                # Évite de dupliquer les points de jonction
                if len(full_path) > 0:
                    full_path.extend(segment[1:])
                else:
                    full_path.extend(segment)
                current_pos = segment[-1]
            else:
                full_path.append(tuple(target))
                current_pos = target

        return full_path

    def smooth_path(self, path):
        """
        Pipeline QP (Quadratic Programming) Smoothing.
        Minimise: Cost = Sum((Pi - Pref_i)^2) + lambda * Sum((Pi+1 - 2Pi + Pi-1)^2)
        """
        if len(path) < 3: return path
        path = np.array(path)

        # --- Etape 1: Élagage (Downsampling) ---
        # Crucial pour la stabilité du QP. On garde 1 point tous les 1.5m
        min_dist = 1.5
        clean_path = [path[0]]
        for pt in path[1:]:
            if np.linalg.norm(pt - clean_path[-1]) > min_dist:
                clean_path.append(pt)

        # Gestion boucle fermée
        if np.linalg.norm(clean_path[0] - clean_path[-1]) > min_dist:
            clean_path.append(clean_path[0])

        clean_path = np.array(clean_path)
        n = len(clean_path)
        if n < 4: return path.tolist()

        # --- Etape 2: QP Smoothing via Sparse Matrix Solver ---
        # Formulation : (I + lambda * D'D) * P = P_ref

        # Poids de lissage (lambda).
        # Valeur haute (ex: 10.0) = Chemin très lisse mais coupe les virages.
        # Valeur basse (ex: 1.0) = Colle au chemin RRT.
        smooth_weight = 5.0

        # Matrice Identité
        I = sparse.eye(n, format='csc')

        # Matrice de Dérivée Seconde (Finite Difference)
        # D = [[1, -2, 1, 0...], [0, 1, -2, 1...]]
        e = np.ones(n)
        data = np.vstack((e, -2 * e, e))
        diags = [0, 1, 2]
        D = sparse.spdiags(data, diags, n - 2, n, format='csc')

        # Pour une boucle fermée parfaite, on doit ajuster la matrice D
        # pour qu'elle "enroule" les derniers points vers les premiers.
        # Ici on simplifie en traitant comme une trajectoire ouverte avec contrainte de position.

        # Matrice du système : A = I + lambda * D.T * D
        A = I + smooth_weight * (D.T @ D)

        # Résolution pour X et Y séparément
        try:
            x_smooth = spsolve(A, clean_path[:, 0])
            y_smooth = spsolve(A, clean_path[:, 1])
            qp_path = np.column_stack((x_smooth, y_smooth))
        except Exception as e:
            print(f"QP Solver failed: {e}")
            qp_path = clean_path

        # --- Etape 3: Interpolation finale (Upsampling) ---
        # On utilise une spline légère juste pour redonner de la densité de points pour l'affichage
        try:
            # per=True pour fermer la boucle proprement
            tck, u = splprep([qp_path[:, 0], qp_path[:, 1]], s=0, k=3, per=True)
            u_new = np.linspace(0, 1, num=len(path) * 3)
            x_new, y_new = splev(u_new, tck)
            return list(zip(x_new, y_new))
        except:
            return qp_path.tolist()


# Wrappers
def compute_centerline(yellow, blue, start):
    p = PathProcessor()
    return p.compute_track_centerline(yellow, blue, start)


def smooth_path(path):
    p = PathProcessor()
    return p.smooth_path(path)