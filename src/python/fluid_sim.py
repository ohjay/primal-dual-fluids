import os
import imageio
import numpy as np
from scipy import sparse
from skimage.transform import warp

import utils

class FluidSim(object):
    def __init__(self, config):
        self.config = config
        
        self.w            = config['width']
        self.h            = config['height']
        self.dt           = config['delta_t']
        self.viscosity_k  = config['viscosity_k']
        self.diffusion_k  = config['diffusion_k']
        self.dissipation  = config['dissipation']
        self.solver_iters = config['solver_iters']

        self.out_folder   = config['out_folder']
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)
        
        color_keys = ('color_r', 'color_g', 'color_b')
        self.color = [config[k] for k in color_keys]
        self.color = np.array(self.color)[np.newaxis, np.newaxis, :]

        # Initialize scalar field
        init_s = config.get('init_s', None)
        if os.path.exists(init_s):
            if init_s.endswith('npy'):
                self.s = np.load(init_s)
            else:
                self.s = imageio.imread(init_s) / 255.0
                if len(self.s.shape) > 2:
                    self.s = utils.grayscale(self.s)
            print('Loaded scalar field from `%s`.' % init_s)
        else:
            self.s = np.zeros((self.h, self.w))

        # Initialize velocity field
        init_v = config.get('init_v', None)
        if os.path.exists(init_v):
            self.v = np.load(init_v)
            print('Loaded velocity field from `%s`.' % init_v)
        else:
            self.v = np.zeros((self.h, self.w, 2))  # order: (vx, vy)

        # Ref: Philip Zucker (https://bit.ly/2Tx2LuE)
        def lapl(N):
            diagonals = np.array([
                -np.ones(N - 1), 2 * np.ones(N), -np.ones(N - 1)])
            return sparse.diags(diagonals, [-1, 0, 1])
        lapl2 = sparse.kronsum(lapl(self.w), lapl(self.h))
        self.project_solve = sparse.linalg.factorized(lapl2)
        
        self.frame_no = 0

    def update(self):
        self.update_velocity_boundary()
        self.project()
        self.advect()
        self.convect()

        self.frame_no += 1

    def render(self):
        frame = self.s[:, :, np.newaxis] * self.color
        out_name = 'frame%s.png' % str(self.frame_no).zfill(4)
        out_path = os.path.join(self.out_folder, out_name)
        imageio.imwrite(out_path, (frame * 255).astype(np.uint8))
        print('Wrote frame %d to `%s`.' % (self.frame_no, out_path))

    # =================
    # Boundary handling
    # =================

    def update_scalar_boundary(self):
        utils.update_boundary(self.s, False)
    
    def update_velocity_boundary(self):
        utils.update_boundary(self.v, True)

    # =============================================
    # Advection, diffusion, projection, dissipation
    # =============================================
    
    def advect(self):
        inv_flow_dict = {'inverse_flow': -self.dt * self.v}
        self.s = warp(self.s, utils.inverse_map, inv_flow_dict, order=5)
        self.update_scalar_boundary()

    def convect(self):
        inv_flow_dict = {'inverse_flow': -self.dt * self.v}
        self.v = warp(self.v, utils.inverse_map, inv_flow_dict, order=5)
        self.update_velocity_boundary()

    def diffuse_scalar(self):
        a = self.dt * self.diffusion_k * self.w * self.h
        utils.lin_solve(self.s, self.s, a, 1 + 4 * a, False, self.solver_iters)

    def diffuse_velocity(self):
        a = self.dt * self.viscosity_k * self.w * self.h
        utils.lin_solve(self.v, self.v, a, 1 + 4 * a, True, self.solver_iters)

    def project(self):
        divergence = utils.compute_divergence(self.v)
        soln = self.project_solve(divergence.flatten())
        soln = soln.reshape(self.h, self.w)
        self.v -= utils.compute_gradient(soln)
        self.update_velocity_boundary()

    def dissipate(self):
        self.s /= self.dt * self.dissipation + 1

    def confine_vorticity(self):
        w = utils.compute_curl(self.v)
        abs_w = np.abs(w)
        utils.update_boundary(abs_w, False)

        grad_abs_w = utils.compute_gradient(abs_w)
        grad_abs_w /= np.linalg.norm(grad_abs_w, axis=-1, keepdims=True) + 1e-5

        fx_conf = grad_abs_w[:, :, 1] * -w
        fy_conf = grad_abs_w[:, :, 0] *  w
        utils.update_boundary(fx_conf, False)
        utils.update_boundary(fy_conf, False)

        self.v += np.stack((fx_conf, fy_conf), -1) * self.dt
        self.update_velocity_boundary()
