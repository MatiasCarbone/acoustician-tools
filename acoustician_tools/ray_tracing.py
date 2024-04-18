import numpy as np
import math
from matplotlib import pyplot as plt
from shapely.geometry import LinearRing, LineString, Point
from shapely.plotting import plot_line, plot_points
from shapely.affinity import rotate
from acoustician_tools.utils import polar_to_cartesian


def get_angle(a, b, c):
    a = (a.x, a.y)
    b = (b.x, b.y)
    c = (c.x, c.y)
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


class RayTracing:
    def __init__(self, boundary_coords=False, emitter=False):
        if boundary_coords:
            self.set_boundaries(boundary_coords)
        if emitter:
            self.set_emitter(emitter)

        self.rays = []

    def set_boundaries(self, coords):
        self.boundaries = LinearRing(coords)

    def set_emitter(self, coords):
        self.emitter = Point(coords)

    def set_listener(self, coords, radius=0.3):
        self.listener = Point(coords)
        self.listener_radius = Point(coords).buffer(radius, resolution=1000)

    def trace_ray(self, r, angle):
        # List of individual segments of boundaries
        b_list = list(self.boundaries.coords)

        origin = self.emitter

        # Calculate endpoint
        end = polar_to_cartesian(r, angle)
        end = end[0] + origin.x, end[1] + origin.y
        end = Point(end)

        # Calculate full ray, intersection and exceeding part
        full_ray = LineString((origin, end))
        diff = full_ray.difference(self.boundaries)

        inner_ray = diff.geoms[0]
        rotation_point = Point(inner_ray.coords[-1])

        # Calculate which segment of the boundary intersects the ray
        for i in range(len(b_list) - 1):
            s = LineString((b_list[i], b_list[i + 1]))
            if Point(inner_ray.coords[-1]).distance(s) < 1e-8:
                segment = s
                break

        # Calculate incidence and inverse angles
        a = Point(segment.coords[1])
        b = rotation_point
        c = Point(inner_ray.coords[0])
        incident_angle = get_angle(a, b, c)
        inverse_angle = 360 - incident_angle

        # Rotate exceding ray segment
        exceding_ray = LineString((diff.geoms[1].coords[0], diff.geoms[-1].coords[1]))
        exceding_ray = rotate(exceding_ray, inverse_angle - incident_angle, rotation_point)

        plot_line(inner_ray, color='purple')
        plot_line(exceding_ray)

    def plot(self):
        # Boundaries
        try:
            plot_line(self.boundaries, linewidth=2.5, add_points=False, color='black')
        except:
            print('No boundaries!')

        # Emitters
        try:
            plot_points(self.emitter, marker='o', color='tab:green')
        except:
            print('No emitter!')

        # Listener
        try:
            plot_points(self.listener, marker='o', color='tab:red')
            plot_points(self.listener_radius.exterior, marker=',', color='tab:red')
        except:
            print('No listener!')

        # Rays
        try:
            for r in self.rays:
                plot_line(r)
        except:
            print('No rays!')

        plt.grid(False)
        plt.show()


if __name__ == '__main__':
    test = RayTracing()

    test.set_boundaries([(0, 0), (2, 0), (2, 2), (1, 1.5), (0, 2)])

    test.set_emitter((1.8, 1.75))
    test.set_listener((0.3, 0.3))

    test.trace_ray(1.5, 180)

    test.plot()
