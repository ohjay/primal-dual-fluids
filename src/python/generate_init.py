import imageio
import argparse
import numpy as np

import utils

"""
Usage: python generate_init.py [fn_name]
"""

# =====
# Utils
# =====

def cos(deg):
    return np.cos(np.radians(deg))

def sin(deg):
    return np.sin(np.radians(deg))

def write_field(out_path, field):
    np.save(out_path, field)
    print('Wrote `%s`.' % out_path)

# ======
# Scalar
# ======

def s_fixed(w, h):
    s = np.zeros((h, w))
    base_y = h // 2
    base_x = w // 2
    s[base_y-10:base_y+10, base_x-10:base_x+10] = 0.5
    write_field('init_s_fixed.npy', s)

def s_star(w, h):
    center = np.array([w // 2.0, h // 2.0])
    binormal = np.array([0.0, 0.0, 1.0])
    s = np.zeros((h, w))

    # star points
    # clockwise order starting from top point
    # rotate 72 degrees around center each time
    point_radius = 7.0 * h / 16.0
    x1 = center + np.array([0, -point_radius])
    x2 = center + np.array([ cos(18), -sin(18)]) * point_radius
    x3 = center + np.array([ cos(54),  sin(54)]) * point_radius
    x4 = center + np.array([-cos(54),  sin(54)]) * point_radius
    x5 = center + np.array([-cos(18), -sin(18)]) * point_radius

    # joining lines
    def fill_segment(start, end, width=30):
        # compute direction
        direction = end - start
        dist = np.linalg.norm(direction)
        direction /= dist

        # compute normal
        direction3d = np.pad(direction, (0, 1), 'constant')
        normal = np.cross(binormal, direction3d)[:2]
        normal /= np.linalg.norm(normal)

        # fill flow field for segment
        for t in np.arange(0, dist, 0.25):
            x = start + direction * t
            for u in np.arange(-width / 2, width / 2, 0.25):
                xs = x + normal * u
                xs = np.round(xs).astype(np.int)
                if (xs >= 0).all() and xs[0] < w and xs[1] < h:
                    s[xs[1], xs[0]] = 1.0

    fill_segment(x1, x3)
    fill_segment(x3, x5)
    fill_segment(x5, x2)
    fill_segment(x2, x4)
    fill_segment(x4, x1)

    write_field('init_s_star.npy', s)

# ========
# Velocity
# ========

def v_circular(w, h):
    center = np.array([w // 2.0, h // 2.0, 0.0])
    binormal = np.array([0.0, 0.0, 1.0])
    
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    coords = np.stack((xx, yy, zz), axis=-1)  # (h, w, 3)
    
    n = coords - center[np.newaxis, np.newaxis, :]
    dist_to_center = np.linalg.norm(n, axis=-1)
    dist_to_center = np.stack([dist_to_center] * 3, axis=-1)
    n[dist_to_center > 0] /= dist_to_center[dist_to_center > 0]
    tangent = np.cross(n, binormal[np.newaxis, np.newaxis, :])
    tangent[dist_to_center == 0] = 0
    flow_field = tangent[:, :, :2] * 150.0
    
    utils.plot_flow(flow_field, out_path='init_v_circular_viz.png')
    write_field('init_v_circular.npy', flow_field)

def v_straight(w, h):
    base_y = h // 2
    flow_field = np.zeros((h, w, 2))
    flow_field[base_y-15:base_y+15, :, 0] = 150.0

    write_field('init_v_straight.npy', flow_field)

def v_constant(w, h):
    flow_field = np.ones((h, w, 2)) * 150.0
    write_field('init_v_constant.npy', flow_field)

def v_star(w, h):
    center = np.array([w // 2.0, h // 2.0])
    binormal = np.array([0.0, 0.0, 1.0])
    flow_field = np.zeros((h, w, 2))

    # star points
    # clockwise order starting from top point
    # rotate 72 degrees around center each time
    point_radius = 7.0 * h / 16.0
    x1 = center + np.array([0, -point_radius])
    x2 = center + np.array([ cos(18), -sin(18)]) * point_radius
    x3 = center + np.array([ cos(54),  sin(54)]) * point_radius
    x4 = center + np.array([-cos(54),  sin(54)]) * point_radius
    x5 = center + np.array([-cos(18), -sin(18)]) * point_radius

    # joining lines
    def fill_segment(start, end, width=30):
        # compute direction
        direction = end - start
        dist = np.linalg.norm(direction)
        direction /= dist

        # compute normal
        direction3d = np.pad(direction, (0, 1), 'constant')
        normal = np.cross(binormal, direction3d)[:2]
        normal /= np.linalg.norm(normal)

        # fill flow field for segment
        for t in np.arange(0, dist, 0.25):
            x = start + direction * t
            for u in np.arange(-width / 2, width / 2, 0.25):
                xs = x + normal * u
                xs = np.round(xs).astype(np.int)
                if (xs >= 0).all() and xs[0] < w and xs[1] < h:
                    flow_field[xs[1], xs[0]] = direction * 150.0

    fill_segment(x1, x3)
    fill_segment(x3, x5)
    fill_segment(x5, x2)
    fill_segment(x2, x4)
    fill_segment(x4, x1)

    write_field('init_v_star.npy', flow_field)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fn_name', type=str)
    parser.add_argument('--width', type=int, default=400)
    parser.add_argument('--height', type=int, default=400)
    args = parser.parse_args()

    fn_name = args.fn_name
    w = args.width
    h = args.height
    
    if fn_name == 'all':
        # s
        s_fixed(w, h)
        s_star(w, h)
        # v
        v_circular(w, h)
        v_straight(w, h)
        v_constant(w, h)
        v_star(w, h)
    else:
        eval(args.fn_name)(w, h)
