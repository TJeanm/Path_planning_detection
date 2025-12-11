import pygame
import math
import numpy as np
from scipy.interpolate import splprep, splev

from ui.camera import Camera
from core.process_path_local import PerceptionProcessor

WIDTH, HEIGHT = 1200, 800
COLORS = {
    "yellow": (255, 255, 0),
    "blue": (50, 100, 255),
    "car_start": (0, 255, 0),
    "path_line": (0, 191, 255),
    "rrt_tree": (0, 80, 0),
    "detected": (255, 50, 50),
    "car_body": (255, 0, 255),
    "car_front": (200, 0, 200),
    "fov_fill": (80, 80, 80, 40),
    "fov_border": (120, 120, 120),
    "target_debug": (0, 255, 255)
}
BACKGROUND_COLOR = (20, 20, 20)
FPS = 60
CAR_SPEED = 6.0


def get_smooth_path(path, car_yaw):
    """
    Lissage local uniquement (pour Mode 1 - Découverte).
    Ajoute un point de contrôle pour respecter la tangente de la voiture.
    """
    if len(path) < 2: return path
    points = [list(p) for p in path]
    start_x, start_y = points[0]
    control_dist = 1.5
    control_point = [
        start_x + control_dist * math.cos(car_yaw),
        start_y + control_dist * math.sin(car_yaw)
    ]
    points.insert(1, control_point)
    if len(points) == 3:
        points.insert(2, [(points[1][0] + points[2][0]) / 2, (points[1][1] + points[2][1]) / 2])

    try:
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        clean_x, clean_y = [x[0]], [y[0]]
        for i in range(1, len(x)):
            if math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) > 0.1:
                clean_x.append(x[i])
                clean_y.append(y[i])

        if len(clean_x) < 3: return path

        tck, u = splprep([clean_x, clean_y], s=2.0, k=3)
        u_new = np.linspace(0, 1, num=25)
        x_new, y_new = splev(u_new, tck)

        return list(zip(x_new, y_new))
    except Exception:
        return path


def find_target_point_pure_pursuit(car_pos, path, lookahead_dist=4.0):
    """
    Pure Pursuit : Vise un point à une distance fixe 'lookahead_dist' devant.
    Gère la boucle fermée pour ne pas s'arrêter.
    """
    if not path or len(path) < 2: return None

    # Si c'est du RRT Local (chemin court, pas une boucle)
    if len(path) < 100:
        # On prend simplement le dernier point accessible si on ne trouve pas mieux
        return path[min(1, len(path) - 1)]

    cx, cy = car_pos
    path_arr = np.array(path)

    # 1. Trouver le point le plus proche sur la trajectoire (Nearest Neighbor)
    dists_sq = np.sum((path_arr - [cx, cy]) ** 2, axis=1)
    closest_idx = np.argmin(dists_sq)

    # 2. Chercher vers l'avant le premier point à 'lookahead_dist' mètres
    n = len(path)
    for i in range(1, n):
        # Modulo pour boucler (Loop Closure)
        idx = (closest_idx + i) % n
        pt = path[idx]

        dist_from_car = math.hypot(pt[0] - cx, pt[1] - cy)

        if dist_from_car >= lookahead_dist:
            return pt

    # Fallback : Si on est perdu, on vise un point un peu plus loin par index
    return path[(closest_idx + 10) % n]


def process_pygame(csv_file, cones, world_bounds, path=None, rrt_edges=None):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Simulation RRT* - Mode Course")
    camera = Camera(world_bounds, screen.get_size())

    planner = PerceptionProcessor()

    start_cone = next((c for c in cones if c['tag'] == 'car_start'), None)
    car_x, car_y = (start_cone['x'], start_cone['y']) if start_cone else (0, 0)
    car_yaw = 0.0

    clock = pygame.time.Clock()
    running = True
    fov_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    while running:
        # Limite dt pour éviter les sauts physiques
        dt = min(clock.tick(FPS) / 1000.0, 0.1)
        screen_size = screen.get_size()
        if fov_surface.get_size() != screen_size:
            fov_surface = pygame.Surface(screen_size, pygame.SRCALPHA)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    camera.change_zoom(1.1, pygame.mouse.get_pos(), screen_size)
                elif event.y < 0:
                    camera.change_zoom(1 / 1.1, pygame.mouse.get_pos(), screen_size)

        # 1. PLANIFICATION
        path_to_draw, tree_nodes, visible_obstacles = planner.plan_local_path((car_x, car_y), car_yaw, cones)

        # 2. CONTROLE VOITURE (PURE PURSUIT)
        # lookahead_dist=4.0 est un bon équilibre pour la vitesse
        target_point = find_target_point_pure_pursuit((car_x, car_y), path_to_draw, lookahead_dist=4.0)

        if target_point:
            target_x, target_y = target_point
            dx = target_x - car_x
            dy = target_y - car_y
            dist = math.hypot(dx, dy)

            desired_yaw = math.atan2(dy, dx)
            # Calcul de l'erreur d'angle (normalisé -pi, pi)
            diff = (desired_yaw - car_yaw + np.pi) % (2 * np.pi) - np.pi

            steering_speed = 4.0
            car_yaw += diff * steering_speed * dt

            move = min(dist, CAR_SPEED * dt)
            car_x += math.cos(car_yaw) * move
            car_y += math.sin(car_yaw) * move

        # 3. DESSIN
        screen.fill(BACKGROUND_COLOR)
        fov_surface.fill((0, 0, 0, 0))

        # A. Vision (Seulement si Mode 1 - Découverte)
        # Si visible_obstacles est vide (Mode 2), on n'affiche plus le cône gris
        if visible_obstacles:
            p_center = camera.world_to_screen(car_x, car_y, screen_size)
            left_angle = car_yaw - planner.FOV / 2
            right_angle = car_yaw + planner.FOV / 2
            lx = car_x + planner.VIEW_DIST * math.cos(left_angle)
            ly = car_y + planner.VIEW_DIST * math.sin(left_angle)
            rx = car_x + planner.VIEW_DIST * math.cos(right_angle)
            ry = car_y + planner.VIEW_DIST * math.sin(right_angle)
            p_left = camera.world_to_screen(lx, ly, screen_size)
            p_right = camera.world_to_screen(rx, ry, screen_size)
            pygame.draw.polygon(fov_surface, COLORS["fov_fill"], [p_center, p_left, p_right])
            pygame.draw.line(fov_surface, COLORS["fov_border"], p_center, p_left, 1)
            pygame.draw.line(fov_surface, COLORS["fov_border"], p_center, p_right, 1)
            screen.blit(fov_surface, (0, 0))

        # B. Target Debug (Cyan)
        if target_point:
            tx, ty = target_point
            stx, sty = camera.world_to_screen(tx, ty, screen_size)
            pygame.draw.circle(screen, COLORS["target_debug"], (stx, sty), 4)

        # C. Cônes (Tous les cônes du fichier, grisés/blancs)
        for c in cones:
            sx, sy = camera.world_to_screen(c["x"], c["y"], screen_size)
            pygame.draw.circle(screen, COLORS.get(c["tag"], (200, 200, 200)), (sx, sy), 3)

        # D. Arbre RRT (Seulement si Mode 1)
        for node in tree_nodes:
            if node.parent:
                p1 = camera.world_to_screen(node.parent.x, node.parent.y, screen_size)
                p2 = camera.world_to_screen(node.x, node.y, screen_size)
                pygame.draw.line(screen, COLORS["rrt_tree"], p1, p2, 1)

        # E. Détections (Seulement si Mode 1)
        # visible_obstacles contient des dicts {'x', 'y', ...}
        for obs in visible_obstacles:
            ox, oy = obs['x'], obs['y']
            sx, sy = camera.world_to_screen(ox, oy, screen_size)
            pygame.draw.circle(screen, COLORS["detected"], (sx, sy), 6, 2)

        # F. TRAJECTOIRE (Bleue)
        if path_to_draw and len(path_to_draw) > 1:
            # Mode 2 (Global Path) : Le chemin est long (>100 pts) et déjà lissé
            if len(path_to_draw) > 100:
                screen_pts = [camera.world_to_screen(p[0], p[1], screen_size) for p in path_to_draw]
                if len(screen_pts) > 1:
                    pygame.draw.lines(screen, COLORS["path_line"], False, screen_pts, 3)
            # Mode 1 (RRT Local) : Le chemin est court, on applique le lissage visuel
            else:
                smooth_pts = get_smooth_path(path_to_draw, car_yaw)
                screen_pts = [camera.world_to_screen(p[0], p[1], screen_size) for p in smooth_pts]
                if len(screen_pts) > 1:
                    pygame.draw.aalines(screen, COLORS["path_line"], False, screen_pts)

        # G. Voiture
        scx, scy = camera.world_to_screen(car_x, car_y, screen_size)
        car_surf = pygame.Surface((24, 12), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, COLORS["car_body"], (0, 0, 24, 12))
        pygame.draw.rect(car_surf, COLORS["car_front"], (16, 0, 8, 12))
        visual_angle = math.degrees(car_yaw)
        rot_car = pygame.transform.rotate(car_surf, visual_angle)
        rect = rot_car.get_rect(center=(scx, scy))
        screen.blit(rot_car, rect)

        pygame.display.flip()

    pygame.quit()