def compute_fit_zoom(world_bounds, screen_size):
    min_x, max_x, min_y, max_y = world_bounds
    screen_w, screen_h = screen_size

    w_world = max_x - min_x
    h_world = max_y - min_y

    if w_world <= 0 or h_world <= 0:
        return 50.0

    zoom_x = screen_w / w_world
    zoom_y = screen_h / h_world

    return min(zoom_x, zoom_y)


class Camera:
    def __init__(self, world_bounds, screen_size):
        min_x, max_x, min_y, max_y = world_bounds
        self.cx = (min_x + max_x) / 2.0
        self.cy = (min_y + max_y) / 2.0
        self.base_zoom = compute_fit_zoom(world_bounds, screen_size)
        self.zoom = self.base_zoom

    def world_to_screen(self, wx, wy, screen_size):
        sw, sh = screen_size
        sx = (wx - self.cx) * self.zoom + sw / 2.0
        sy = -(wy - self.cy) * self.zoom + sh / 2.0
        return int(sx), int(sy)

    def screen_to_world(self, sx, sy, screen_size):
        sw, sh = screen_size
        wx = (sx - sw / 2.0) / self.zoom + self.cx
        wy = -(sy - sh / 2.0) / self.zoom + self.cy
        return wx, wy

    def change_zoom(self, factor, mouse_pos, screen_size):
        if factor == 0:
            return

        before = self.screen_to_world(mouse_pos[0], mouse_pos[1], screen_size)

        new_zoom = self.zoom * factor
        min_zoom = self.base_zoom * 0.1
        max_zoom = self.base_zoom * 10.0
        self.zoom = max(min_zoom, min(max_zoom, new_zoom))

        after = self.screen_to_world(mouse_pos[0], mouse_pos[1], screen_size)

        self.cx += (before[0] - after[0])
        self.cy += (before[1] - after[1])

    def pan_pixels(self, dx_px, dy_px):
        self.cx -= dx_px / self.zoom
        self.cy += dy_px / self.zoom