import os
import imageio
import numpy as np
import cvxpy as cp
from scipy import sparse
from scipy.signal import convolve
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
        self.vorticity    = config['vorticity']
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

        # Set guiding attributes
        self.target_v = None
        self.guiding  = config['guiding']
        target_v_path = config['target_v']
        if os.path.exists(target_v_path):
            self.target_v = np.load(target_v_path)
            print('Loaded target velocity field from `%s`.' % target_v_path)

        self.guiding_alg = config['guiding_alg']
        self.pd_params = config.get('pd_params', {})
        self.pd_x, self.pd_y, self.pd_z = None, None, None
        self.blur_kernel, self.blur = None, None
        if self.guiding_alg == 'pd':
            self.pd_params = config['pd_params']
            self.pd_x = np.zeros_like(self.v)
            self.pd_y = np.zeros_like(self.v)
            self.pd_z = np.zeros_like(self.v)
            gauss = utils.gaussian2d(self.pd_params['blur_size'])
            self.blur_kernel = 2.0 * gauss @ gauss
            self.blur_kernel = self.blur_kernel[:, :, np.newaxis]
            self.blur = lambda field: convolve(field, self.blur_kernel, 'same')

        # Ref: Philip Zucker (https://bit.ly/2Tx2LuE)
        def lapl(N):
            diagonals = np.array([
                -np.ones(N - 1), 2 * np.ones(N), -np.ones(N - 1)])
            return sparse.diags(diagonals, [-1, 0, 1])
        lapl2 = sparse.kronsum(lapl(self.w), lapl(self.h))
        self.project_solve = sparse.linalg.factorized(lapl2)
        
        self.frame_no = 0

    def update(self):
        if self.guiding and self.guiding_alg == 'assign':
            self.v = np.copy(self.target_v)
        self.update_velocity_boundary()
        self.diffuse_scalar()
        self.diffuse_velocity()
        self.dissipate()
        if self.guiding and self.guiding_alg != 'assign':
            self.guide()
        else:
            self.project_velocity()
        self.advect_and_convect()
        self.confine_vorticity()

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

    # ==========================================================
    # Advection, projection, diffusion, dissipation, confinement
    # ==========================================================

    def advect_and_convect(self):
        if self.fast_advect:
            coords = self.base_coords - self.dt * self.v
            coords = coords[:, :, ::-1].transpose(2, 0, 1)
            for i in range(self.ns):
                self.s[:, :, i] = utils.map_coordinates_clipped(self.s[:, :, i], coords)
            self.v[:,:,0] = utils.map_coordinates_clipped(self.v[:,:,0], coords)
            self.v[:,:,1] = utils.map_coordinates_clipped(self.v[:,:,1], coords)
        else:
            inv_flow_dict = {'inverse_flow': -self.dt * self.v}
            self.s = warp(self.s, utils.inverse_map, inv_flow_dict, order=5)
            self.v = warp(self.v, utils.inverse_map, inv_flow_dict, order=5)
        self.update_scalar_boundary()
        self.update_velocity_boundary()

    def project_velocity(self):
        self.v = utils.project(self.v, self.project_solve)

    def diffuse_scalar(self):
        if self.diffusion_k != 0:
            a = self.dt * self.diffusion_k * self.w * self.h
            for i in range(self.ns):
                utils.lin_solve(self.s[:, :, i], self.s[:, :, i],
                                a, 1 + 4 * a, False, self.solver_iters)

    def diffuse_velocity(self):
        if self.viscosity_k != 0:
            a = self.dt * self.viscosity_k * self.w * self.h
            utils.lin_solve(self.v, self.v,
                            a, 1 + 4 * a, True, self.solver_iters)

    def dissipate(self):
        if self.dissipation != 0:
            self.s /= self.dt * self.dissipation + 1

    def confine_vorticity(self):
        if self.vorticity != 0:
            w = utils.compute_curl(self.v)
            grad_abs_w = 2 * utils.compute_gradient(np.abs(w))
            grad_abs_w /= np.linalg.norm(grad_abs_w, axis=-1, keepdims=True) + 1e-5

            f_conf = self.dt * self.vorticity * w[:, :, np.newaxis] * grad_abs_w
            self.v[2:-2, 2:-2] += f_conf[2:-2, 2:-2]

    # =======
    # Guiding
    # =======

    def guide(self):
        assert self.target_v is not None

        if self.guiding_alg == 'initial':
            self.v = self.initial_optim()
        elif self.guiding_alg == 'pd':
            # primal-dual optimization step
            self.v = np.copy(self.pd_optim())
        else:
            import sys
            sys.exit('unrecognized guiding alg: %s' % self.guiding_alg)

    # ================
    # Guiding: initial
    # ================

    def initial_optim(self):
        # [variables] solve for a velocity field
        v = cp.Variable(self.v.size)

        # [objective]
        objective_term1 = cp.sum_squares(v - self.v.flatten())
        objective_term2 = cp.sum_squares(v - self.target_v.flatten())
        objective = cp.Minimize(objective_term1 + objective_term2)
        problem = cp.Problem(objective)

        # optimize, perform projection
        result = problem.solve()
        v_soln = v.value.reshape(self.h, self.w, 2)
        return utils.project(v_soln, self.project_solve)

    # ====================
    # Guiding: primal-dual
    # ====================

    def pd_optim(self):
        """PD optimization step.
        Ref: Inglis et al. (https://arxiv.org/pdf/1611.03677.pdf)."""
        tau       = self.pd_params['tau']
        sigma     = self.pd_params['sigma']
        theta     = self.pd_params['theta']
        w         = self.pd_params['guiding_w']
        max_iters = self.pd_params['max_iters']

        for k in range(max_iters):
            prox_f_val = self.prox_f(sigma, (self.pd_x + self.pd_y) / sigma, w)
            self.pd_x += sigma * (self.pd_y - prox_f_val)
            pd_z_next = utils.project(self.pd_z - tau * self.pd_x, self.project_solve)
            delta_pdz = pd_z_next - self.pd_z
            self.pd_z = pd_z_next
            self.pd_y = self.pd_z + theta * delta_pdz
            # termination criterion
            eps = 1e-3 * (np.sqrt(2) + np.linalg.norm(self.pd_z))
            if np.linalg.norm(delta_pdz) <= eps:
                break

        return self.pd_z

    def prox_f(self, sigma, xi, w):
        """Ref: Inglis et al. (https://arxiv.org/pdf/1611.03677.pdf)."""
        q = self.blur(self.target_v - self.v) - sigma * self.v
        gamma = 1.0 / (2.0 * w * w + sigma)
        sxq = sigma * xi + q
        return self.v + gamma * sxq - self.blur(gamma * gamma * sxq)
