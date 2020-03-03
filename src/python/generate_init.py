import imageio
import argparse
import numpy as np

import utils

"""
Usage: python generate_init.py [fn_name]
"""

def s_fixed(w, h):
    s = np.zeros((h, w))
    base_y = h // 2
    base_x = w // 2
    s[base_y-10:base_y+10, base_x-10:base_x+10] = 0.5
    out_path = 'init_s_fixed.npy'
    np.save(out_path, s)
    print('Wrote `%s`.' % out_path)

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
    out_path = 'init_v_circular.npy'
    np.save(out_path, flow_field)
    print('Wrote `%s`.' % out_path)

def v_straight(w, h):
    base_y = h // 2
    flow_field = np.zeros((h, w, 2))
    flow_field[base_y-15:base_y+15, :, 0] = 150.0

    out_path = 'init_v_straight.npy'
    np.save(out_path, flow_field)
    print('Wrote `%s`.' % out_path)

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
        s_fixed(w, h)
        v_circular(w, h)
        v_straight(w, h)
    else:
        eval(args.fn_name)(w, h)
