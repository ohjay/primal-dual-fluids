#!/usr/bin/env python

import os
import re
import imageio
import argparse
import cv2
import numpy as np
import functools

def get_shapes(images_dir, suffixes):
    # Assumption: all images with a given suffix have the same shape.
    # Assumption: all images involving the desired suffixes have the same height.
    # Assumption: for each base name, if there is an image for one desired suffix
    #             then there is an image for all other desired suffixes.
    for image_name in os.listdir(images_dir):
        if image_name.endswith(suffixes[0]):
            shapes = []
            base_name = image_name[:-len(suffixes[0])]
            for suffix in suffixes:
                image_path = os.path.join(images_dir, base_name + suffix)
                image = imageio.imread(image_path)
                shapes.append(image.shape)
            return shapes
    raise RuntimeError('[-] No images of the desired suffix(es).')

def write_video(images_dir, out_path, fps, suffixes):
    shapes = get_shapes(images_dir, suffixes)
    out_width = functools.reduce(lambda x, y: x + y[1], shapes, 0)
    out_height = shapes[0][0]

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter()
    size = (out_width, out_height)
    success = out.open(out_path, fourcc, fps, size, True)
    if not success:
        print('[-] Failed to open the video writer.')
        return

    frames_written = 0
    for image_name in sorted(os.listdir(images_dir)):
        if image_name.endswith(suffixes[0]):
            images = []
            base_name = image_name[:-len(suffixes[0])]
            for suffix in suffixes:
                image_path = os.path.join(images_dir, base_name + suffix)
                images.append(imageio.imread(image_path)[:, :, :3])
            frame = np.concatenate(images, axis=1)  # concat along x-axis
            assert frame.shape[0] == out_height
            assert frame.shape[1] == out_width
            frame = frame[:, :, ::-1]  # RGB -> BGR
            if frame.dtype in (np.float, np.float64):
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            out.write(frame)
            frames_written += 1
            print(frames_written)
    out.release()
    print('[+] Finished writing %d frames to %s.' % (frames_written, out_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str, help='directory containing images to write')
    parser.add_argument('--out_path', '-o', type=str, default='out.mov')
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--suffixes', nargs='+', type=str, default=['.png'])
    args = parser.parse_args()

    assert len(args.suffixes) > 0, 'must include >= 1 desired suffix'

    write_video(args.images_dir,
                args.out_path,
                args.fps,
                args.suffixes)
