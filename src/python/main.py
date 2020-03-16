import yaml
import argparse

from fluid_sim import FluidSim
from images2video import write_video

"""Offline Eulerian fluid simulation."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    fluid_sim = FluidSim(config)

    # Main loop
    num_steps = config['num_steps']
    if config['write_init']:
        fluid_sim.render()
    for step in range(num_steps):
        fluid_sim.update()
        fluid_sim.render()

    # Video export
    export_params = config.get('video_export', None)
    if export_params:
        out_path = export_params.get('out_path', None)
        if out_path:
            images_dir = config['out_folder']
            fps        = export_params.get('fps', 24)
            write_video(images_dir, out_path, fps, ['.png'])
