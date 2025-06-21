import pygame
import random
import math
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

# Configuration
WIDTH, HEIGHT = 800, 600
FPS = 60
NUM_SHAPES = 5
SHAPE_RADIUS = 40
VELOCITY_SCALE = 3

FRICTION = 0.999
GRAVITY = 0.
ELASTICITY = 0.0001  # low bouncing

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

def generate_random_polygon(center, num_points=8, radius=SHAPE_RADIUS):
    """Generate a rough polygon around a center point to simulate 2D point cloud approximation"""
    points = []
    for i in range(num_points):
        angle = i * (2 * math.pi / num_points)
        r = radius + random.uniform(-10, 10)
        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        points.append((x, y))
    return points
class Body:
    def __init__(self, center):
        self.points = generate_random_polygon(center)
        self.polygon = Polygon(self.points)
        self.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        self.velocity = [0, 0]  # initially static
        self.is_dragged = False

    def apply_physics(self):
        if self.is_dragged:
            return
        # Gravity
        self.velocity[1] += GRAVITY
        # Friction
        self.velocity[0] *= FRICTION
        self.velocity[1] *= FRICTION
        # Stop small motion
        if abs(self.velocity[0]) < 0.01:
            self.velocity[0] = 0
        if abs(self.velocity[1]) < 0.01:
            self.velocity[1] = 0

    def move(self):
        if self.is_dragged:
            return
        self.points = [(x + self.velocity[0], y + self.velocity[1]) for (x, y) in self.points]
        self.polygon = Polygon(self.points)
        self.stay_in_bounds()

    def draw(self, surface):
        pygame.draw.polygon(surface, self.color, self.points)

    def contains_point(self, pos):
        return self.polygon.contains(Point(pos))

    def set_position(self, new_center):
        current_center = self.polygon.centroid.coords[0]
        dx = new_center[0] - current_center[0]
        dy = new_center[1] - current_center[1]
        self.points = [(x + dx, y + dy) for (x, y) in self.points]
        self.polygon = Polygon(self.points)

    def check_collision(self, other):
        return self.polygon.intersects(other.polygon)

    def resolve_collision(self, other):
        if not self.check_collision(other):
            return

        intersection = self.polygon.intersection(other.polygon)
        if intersection.is_empty:
            return

        vec = (
            self.polygon.centroid.x - other.polygon.centroid.x,
            self.polygon.centroid.y - other.polygon.centroid.y
        )
        length = math.hypot(vec[0], vec[1]) + 1e-6
        direction = (vec[0] / length, vec[1] / length)
        overlap = intersection.area ** 0.5
        push = (direction[0] * overlap, direction[1] * overlap)

        for shape, factor in [(self, 0.5), (other, -0.5)]:
            shape.points = [(x + push[0] * factor, y + push[1] * factor) for (x, y) in shape.points]
            shape.polygon = Polygon(shape.points)

        # Momentum transfer (low elasticity)
        for i in range(2):
            v1, v2 = self.velocity[i], other.velocity[i]
            self.velocity[i] = (1 - ELASTICITY) * v2
            other.velocity[i] = (1 - ELASTICITY) * v1

    def stay_in_bounds(self):
        minx, miny, maxx, maxy = self.polygon.bounds
        dx, dy = 0, 0
        if minx < 0:
            dx = -minx
            self.velocity[0] *= -ELASTICITY
        elif maxx > WIDTH:
            dx = WIDTH - maxx
            self.velocity[0] *= -ELASTICITY
        if miny < 0:
            dy = -miny
            self.velocity[1] *= -ELASTICITY
        elif maxy > HEIGHT:
            dy = HEIGHT - maxy
            self.velocity[1] *= -ELASTICITY

        if dx != 0 or dy != 0:
            self.points = [(x + dx, y + dy) for (x, y) in self.points]
            self.polygon = Polygon(self.points)

shapes = [Body((random.randint(100, 700), random.randint(100, 500))) for _ in range(NUM_SHAPES)]

# Main loop
running = True
dragging_shape = None
start_drag = None

while running:
    dt = clock.tick(FPS) / 1000
    screen.fill((30, 30, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            for shape in reversed(shapes):
                if shape.contains_point(event.pos):
                    dragging_shape = shape
                    shape.is_dragged = True
                    start_drag = event.pos
                    break

        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging_shape:
                end_drag = event.pos
                dx = end_drag[0] - start_drag[0]
                dy = end_drag[1] - start_drag[1]
                dragging_shape.velocity = [dx * VELOCITY_SCALE / FPS, dy * VELOCITY_SCALE / FPS]
                dragging_shape.is_dragged = False
                dragging_shape = None
                start_drag = None

        elif event.type == pygame.MOUSEMOTION and dragging_shape:
            dragging_shape.set_position(event.pos)

    for shape in shapes:
        shape.apply_physics()
        shape.move()

    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            shapes[i].resolve_collision(shapes[j])

    for shape in shapes:
        shape.draw(screen)

    pygame.display.flip()

pygame.quit()

