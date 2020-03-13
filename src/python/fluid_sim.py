import os
import imageio
import numpy as np
from scipy import sparse
from skimage.transform import warp
from scipy.ndimage import map_coordinates

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
        self.fast_advect  = config['fast_advect']

        self.out_folder   = config['out_folder']
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        self.write_vel      = config['write_vel']
        self.out_vel_folder = os.path.join(self.out_folder, 'vel')
        if self.write_vel and not os.path.exists(self.out_vel_folder):
            os.makedirs(self.out_vel_folder)
        
        color_keys = ('color_r', 'color_g', 'color_b')
        self.color = [config[k] for k in color_keys]
        self.color = np.array(self.color)[np.newaxis, np.newaxis, :]

        # Initialize scalar field(s)
        init_s = config.get('init_s', None)
        if os.path.exists(init_s):
            if init_s.endswith('npy'):
                self.s = np.load(init_s)
                while len(self.s.shape) < 3:
                    self.s = np.expand_dims(self.s, -1)
            else:
                self.s = imageio.imread(init_s) / 255.0
            print('Loaded scalar field from `%s`.' % init_s)
        else:
            self.s = np.zeros((self.h, self.w, 1))
        self.ns = self.s.shape[-1]  # number of scalar fields

        # Initialize velocity field
        init_v = config.get('init_v', None)
        if os.path.exists(init_v):
            self.v = np.load(init_v)
            print('Loaded velocity field from `%s`.' % init_v)
        else:
            self.v = np.zeros((self.h, self.w, 2))  # order: (vx, vy)

        # For warping
        if self.fast_advect:
            xx, yy = np.meshgrid(np.arange(self.w), np.arange(self.h))
            self.base_coords = np.stack((xx, yy), axis=-1)  # (h, w, 2)

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
        self.diffuse_scalar()
        self.diffuse_velocity()
        self.dissipate()
        self.project()
        self.advect_and_convect()

        self.frame_no += 1

    def render(self):
        frame = self.s
        if self.ns == 1:
            frame = self.s * self.color
        out_name = 'frame%s.png' % str(self.frame_no).zfill(4)
        out_path = os.path.join(self.out_folder, out_name)
        imageio.imwrite(out_path, (frame * 255).astype(np.uint8))
        print('Wrote frame %d to `%s`.' % (self.frame_no, out_path))

        if self.write_vel:
            out_vel_name = 'vel%s.npy' % str(self.frame_no).zfill(4)
            out_vel_path = os.path.join(self.out_vel_folder, out_vel_name)
            np.save(out_vel_path, self.v)

    # =================
    # Boundary handling
    # =================

    def update_scalar_boundary(self):
        for i in range(self.ns):
            utils.update_boundary(self.s[:, :, i], False)
    
    def update_velocity_boundary(self):
        utils.update_boundary(self.v, True)

    # =============================================
    # Advection, diffusion, projection, dissipation
    # =============================================
    
    def advect_and_convect(self):
        if self.fast_advect:
            coords = self.base_coords - self.dt * self.v
            coords = coords[:, :, ::-1].transpose(2, 0, 1)
            for i in range(self.ns):
                self.s[:, :, i] = map_coordinates(self.s[:, :, i], coords, order=5)
            self.v[:,:,0] = map_coordinates(self.v[:,:,0], coords, order=5)
            self.v[:,:,1] = map_coordinates(self.v[:,:,1], coords, order=5)
        else:
            inv_flow_dict = {'inverse_flow': -self.dt * self.v}
            self.s = warp(self.s, utils.inverse_map, inv_flow_dict, order=5)
            self.v = warp(self.v, utils.inverse_map, inv_flow_dict, order=5)
        self.update_scalar_boundary()
        self.update_velocity_boundary()

    def diffuse_scalar(self):
        a = self.dt * self.diffusion_k * self.w * self.h
        for i in range(self.ns):
            utils.lin_solve(self.s[:, :, i], self.s[:, :, i],
                            a, 1 + 4 * a, False, self.solver_iters)

    def diffuse_velocity(self):
        a = self.dt * self.viscosity_k * self.w * self.h
        utils.lin_solve(self.v, self.v,
                        a, 1 + 4 * a, True, self.solver_iters)

    def project(self):
        divergence = utils.compute_divergence(self.v)
        soln = self.project_solve(divergence.flatten())
        soln = soln.reshape(self.h, self.w)
        self.v -= utils.compute_gradient(soln)
        self.update_velocity_boundary()

    def dissipate(self):
        self.s /= self.dt * self.dissipation + 1
