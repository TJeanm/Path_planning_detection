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

    # Bleu plus clair/électrique pour bien ressortir sur le bitume
    "path_line": (0, 191, 255),

    "rrt_tree": (0, 80, 0),  # Vert très sombre (discret)
    "detected": (255, 50, 50),
    "car_body": (255, 0, 255),
    "car_front": (200, 0, 200),

    # --- COULEURS DU CÔNE (Grisé comme demandé) ---
    "fov_fill": (80, 80, 80, 40),  # Gris sombre transparent
    "fov_border": (120, 120, 120),  # Bordure grise

    "target_debug": (0, 255, 255)
}
BACKGROUND_COLOR = (20, 20, 20)  # Fond un peu plus sombre
FPS = 60
CAR_SPEED = 6.0


def get_smooth_path(path, car_yaw):
    """
    Génère une courbe de Bézier/Spline réaliste qui respecte l'orientation de la voiture.
    """
    if len(path) < 2:
        return path

        # 1. Préparation des points
    # On convertit en liste de listes modifiables
    points = [list(p) for p in path]

    # --- L'ASTUCE "TRAJECTOIRE RÉALISTE" ---
    # On insère un point artificiel juste devant la voiture, aligné avec son angle (car_yaw).
    # Cela force la spline à partir "droite" avant de tourner.
    start_x, start_y = points[0]

    # Distance du point de contrôle (plus c'est grand, plus la voiture met du temps à tourner)
    control_dist = 1.5

    control_point = [
        start_x + control_dist * math.cos(car_yaw),
        start_y + control_dist * math.sin(car_yaw)
    ]

    # On insère ce point en 2ème position (index 1)
    points.insert(1, control_point)

    # Si on a pas assez de points pour une cubique, on en rajoute un milieu
    if len(points) == 3:
        mid_x = (points[1][0] + points[2][0]) / 2
        mid_y = (points[1][1] + points[2][1]) / 2
        points.insert(2, [mid_x, mid_y])

    try:
        # Séparation X et Y
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        # Nettoyage des doublons (scipy n'aime pas les points superposés)
        clean_x, clean_y = [x[0]], [y[0]]
        for i in range(1, len(x)):
            if math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) > 0.1:
                clean_x.append(x[i])
                clean_y.append(y[i])

        if len(clean_x) < 3: return path  # Fallback

        # --- GENERATION SPLINE ---
        # k=3 (Cubique) pour avoir de belles courbes en S
        # s=2.0 (Smoothing factor) : Autorise la courbe à ne pas toucher exactement les points zig-zag du RRT
        tck, u = splprep([clean_x, clean_y], s=2.0, k=3)

        # On génère beaucoup de points pour que l'affichage soit soyeux
        u_new = np.linspace(0, 1, num=50)
        x_new, y_new = splev(u_new, tck)

        return list(zip(x_new, y_new))

    except Exception as e:
        # Si la mathématique échoue (ex: ligne droite parfaite), on rend le chemin brut
        return path


def process_pygame(csv_file, cones, world_bounds, path=None, rrt_edges=None):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Simulation RRT* - Trajectoire Fluide")
    camera = Camera(world_bounds, screen.get_size())

    planner = PerceptionProcessor()

    start_cone = next((c for c in cones if c['tag'] == 'car_start'), None)
    car_x, car_y = (start_cone['x'], start_cone['y']) if start_cone else (0, 0)
    car_yaw = 0.0

    clock = pygame.time.Clock()
    running = True

    # Surface pour les effets de transparence (FOV)
    fov_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    while running:
        dt = clock.tick(FPS) / 1000.0
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
        local_path, tree_nodes, visible_obstacles = planner.plan_local_path((car_x, car_y), car_yaw, cones)

        # 2. CONTROLE VOITURE
        if len(local_path) > 1:
            target_x, target_y = local_path[1]
            dx = target_x - car_x
            dy = target_y - car_y
            dist = math.hypot(dx, dy)

            desired_yaw = math.atan2(dy, dx)
            diff = (desired_yaw - car_yaw + np.pi) % (2 * np.pi) - np.pi

            # Steering un peu plus doux pour accompagner la courbe visuelle
            steering_speed = 4.0
            car_yaw += diff * steering_speed * dt

            if dist > 0.1:
                move = min(dist, CAR_SPEED * dt)
                car_x += math.cos(car_yaw) * move
                car_y += math.sin(car_yaw) * move

        # 3. DESSIN
        screen.fill(BACKGROUND_COLOR)
        fov_surface.fill((0, 0, 0, 0))

        # A. Cône de Vision (Gris)
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

        # B. Arbre RRT (Vert foncé, lignes simples)
        for node in tree_nodes:
            if node.parent:
                p1 = camera.world_to_screen(node.parent.x, node.parent.y, screen_size)
                p2 = camera.world_to_screen(node.x, node.y, screen_size)
                pygame.draw.line(screen, COLORS["rrt_tree"], p1, p2, 1)

        # C. Cônes et Détections
        for c in cones:
            sx, sy = camera.world_to_screen(c["x"], c["y"], screen_size)
            pygame.draw.circle(screen, COLORS.get(c["tag"], (200, 200, 200)), (sx, sy), 3)

        for (ox, oy, r) in visible_obstacles:
            sx, sy = camera.world_to_screen(ox, oy, screen_size)
            pygame.draw.circle(screen, COLORS["detected"], (sx, sy), 6, 2)

        # D. TRAJECTOIRE COURBE (C'est ici que ça change tout)
        if len(local_path) > 1:
            # On passe car_yaw pour forcer la courbe à partir droit
            smooth_pts = get_smooth_path(local_path, car_yaw)

            screen_pts = [camera.world_to_screen(p[0], p[1], screen_size) for p in smooth_pts]

            if len(screen_pts) > 1:
                # Ligne épaisse et anti-aliasée
                pygame.draw.aalines(screen, COLORS["path_line"], False, screen_pts)
                # On repasse une 2ème fois légèrement décalé ou plus épais si besoin
                # pygame.draw.lines(screen, COLORS["path_line"], False, screen_pts, 2)

            # Debug : Afficher la cible finale de la courbe
            tx, ty = smooth_pts[-1]
            stx, sty = camera.world_to_screen(tx, ty, screen_size)
            pygame.draw.circle(screen, COLORS["target_debug"], (stx, sty), 3)

        # E. Voiture
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