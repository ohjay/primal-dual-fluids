import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

# Multipliers for negating velocity components
REV_X = np.array([-1, 1])[np.newaxis, np.newaxis, :]
REV_Y = np.array([1, -1])[np.newaxis, np.newaxis, :]

def update_boundary(field, is_velocity):
    mult_topbot = REV_Y if is_velocity else 1
    mult_lefrig = REV_X if is_velocity else 1
    
    # Side values
    field[0,  :] = field[1,  :] * mult_topbot
    field[-1, :] = field[-2, :] * mult_topbot
    field[:,  0] = field[:,  1] * mult_lefrig
    field[:, -1] = field[:, -2] * mult_lefrig

    # Corner values
    field[ 0,  0] = (field[ 0,  1] + field[ 1,  0]) * 0.5
    field[ 0, -1] = (field[ 0, -2] + field[ 1, -1]) * 0.5
    field[-1,  0] = (field[-2,  0] + field[-1,  1]) * 0.5
    field[-1, -1] = (field[-1, -2] + field[-2, -1]) * 0.5

def compute_divergence(field):
    divergence = np.zeros(field.shape[:2])
    divergence[1:-1, 1:-1] = -0.5 * \
        (field[1:-1, 2:,   0] - field[1:-1,  :-2, 0] \
       + field[2:,   1:-1, 1] - field[ :-2, 1:-1, 1])
    update_boundary(divergence, False)
    return divergence

def compute_curl(field):
    curl = np.zeros(field.shape[:2])
    curl[1:-1, 1:-1] = 0.5 * \
        (field[1:-1, 2:,   1] - field[1:-1,  :-2, 1] \
       - field[2:,   1:-1, 0] + field[ :-2, 1:-1, 0])
    update_boundary(curl, False)
    return curl

def compute_gradient(field):
    gradient = np.zeros(field.shape + (2,))
    gradient[1:-1, 1:-1, 0] = (field[1:-1, 2:  ] - field[1:-1,  :-2]) * 0.5
    gradient[1:-1, 1:-1, 1] = (field[2:,   1:-1] - field[ :-2, 1:-1]) * 0.5
    return gradient

def lin_solve(soln, field_prev, a, b, is_velocity, iters):
    for i in range(iters):
        neighbor_sum = soln[2:, 1:-1] + soln[:-2, 1:-1] \
                     + soln[1:-1, 2:] + soln[1:-1, :-2]
        soln[1:-1, 1:-1] = \
            (field_prev[1:-1, 1:-1] + a * neighbor_sum) / b
        update_boundary(soln, is_velocity)

def inverse_map(xy, inverse_flow):
    # Input: (M, 2) array of (x, y) coordinates
    # Output: (M, 2) array of transformed (x, y) coordinates
    inverse_flow = inverse_flow.transpose(1, 0, 2).reshape(-1, 2)
    return xy + inverse_flow

def map_coordinates_clipped(field, coords):
    minval, maxval = field.min(), field.max()
    warped = map_coordinates(field, coords, order=5)
    return np.clip(warped, minval, maxval)

def plot_flow(flow, step=30, out_path=None):
    h, w = flow.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx = xx[::step, ::step]
    yy = yy[::step, ::step]
    
    flow = flow[::step, ::step]
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    plt.quiver(xx, yy, flow_x, flow_y)
    if out_path is not None:
        plt.savefig(out_path)
        print('Wrote `%s`.' % out_path)
    else:
        plt.show()
    plt.clf()

def grayscale(im):
    # Rec. 709 luma coefficients
    return 0.2126 * im[:, :, 0] + 0.7152 * im[:, :, 1] + 0.0722 * im[:, :, 2]
