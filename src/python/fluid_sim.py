import os
import imageio
import numpy as np

import utils
from skimage.transform import warp

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
            self.s = np.load(init_s)
            print('Loaded scalar field from `%s`.' % init_s)
        else:
            self.s = np.zeros((self.h, self.w))

        # Initialize velocity field
        init_v = config.get('init_v', None)
        self.force = np.load(init_v)
        if os.path.exists(init_v):
            self.v = self.force
            print('Loaded velocity field from `%s`.' % init_v)
        else:
            self.v = np.zeros((self.h, self.w, 2))  # order: (vx, vy)
            self.force = np.zeros((self.h, self.w, 2))
        
        self.frame_no = 0

    def update(self):
        self.update_scalar()
        self.update_velocity()
        self.frame_no += 1

    def render(self):
        frame = self.s[:, :, np.newaxis] * self.color
        out_name = 'frame%s.png' % str(self.frame_no).zfill(4)
        out_path = os.path.join(self.out_folder, out_name)
        imageio.imwrite(out_path, (frame * 255).astype(np.uint8))
        print('Wrote frame %d to `%s`.' % (self.frame_no, out_path))

    def update_scalar(self):
        self.advect()
        self.diffuse_scalar()
        self.dissipate()

    def update_velocity(self):
        self.v += self.force
        self.update_velocity_boundary()
        self.diffuse_velocity()
        self.project()
        self.convect()
        self.project()

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
        self.s = warp(self.s, utils.inverse_map, inv_flow_dict)
        self.update_scalar_boundary()

    def convect(self):
        inv_flow_dict = {'inverse_flow': -self.dt * self.v}
        self.v = warp(self.v, utils.inverse_map, inv_flow_dict)
        self.update_velocity_boundary()

    def diffuse_scalar(self):
        a = self.dt * self.diffusion_k * self.w * self.h
        utils.lin_solve(self.s, self.s, a, 1 + 4 * a, False, self.solver_iters)

    def diffuse_velocity(self):
        a = self.dt * self.viscosity_k * self.w * self.h
        utils.lin_solve(self.v, self.v, a, 1 + 4 * a, True, self.solver_iters)

    def project(self):
        divergence = utils.compute_divergence(self.v)
        soln = np.zeros((self.h, self.w))
        utils.lin_solve(soln, -divergence, 1, 4, False, self.solver_iters)
        self.v -= utils.compute_gradient(soln)
        self.update_velocity_boundary()

    def dissipate(self):
        self.s /= self.dt * self.dissipation + 1
