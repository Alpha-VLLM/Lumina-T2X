from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F

from .view_base import BaseView

def make_frame_motion(im, t):
    im_size = im.size[0]
    factor = im_size / 64 / 4.0
    frame_size = int(im_size * 1.5)
    vel = 20
    amp = 29 * factor / 2   # 29 @ 256, 29 * 4 @ 1024, need to divide by 2 b/c amp

    # Triangular wave
    offset = int(amp * 2 / np.pi * np.arcsin(np.sin(2 * np.pi * vel * t)))

    # Paste on to canvas
    frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
    frame.paste(im, (offset + (frame_size - im_size) // 2, offset + (frame_size - im_size) // 2))

    return frame

class MotionBlurView(BaseView):
    def __init__(self, size=7):
        self.size = size

    def make_frame(self, im, t):
        return make_frame_motion(im, t)

    def view(self, im):
        return im

    def inverse_view(self, noise):
        c, h, w = noise.shape
        factor = h // 64    # Account for image size

        # Make kernel on the fly
        size = self.size * factor
        size = size + ((factor + 1) % 2)    # Make sure it's odd
        self.K = torch.eye(size)[None, None] / size
        self.K = self.K.to(noise.dtype).to(noise.device)

        # Apply kernel to each channel independently
        noise[:3] = torch.cat([F.conv2d(noise[i][None], self.K, padding=size//2) for i in range(3)])

        return noise

    def save_view(self, im):
        c, h, w = im.shape
        factor = h // 64    # Account for image size

        # Make kernel on the fly
        size = self.size * factor
        size = size + ((factor + 1) % 2)    # Make sure it's odd
        self.K = torch.eye(size)[None, None] / size
        self.K = self.K.to(im.dtype).to(im.device)

        # Apply kernel to each channel independently
        im = torch.cat([F.conv2d(im[i][None], self.K, padding=size//2) for i in range(3)])

        return im



class MotionBlurResView(BaseView):
    def __init__(self, size=7):
        self.size = size

    def make_frame(self, im, t):
        return make_frame_motion(im, t)

    def view(self, im):
        return im

    def inverse_view(self, noise):
        c, h, w = noise.shape
        factor = h // 64    # Account for image size

        # Make kernel on the fly
        size = self.size * factor
        size = size + ((factor + 1) % 2)    # Make sure it's odd
        self.K = torch.eye(size)[None, None] / size
        self.K = self.K.to(noise.dtype).to(noise.device)

        # Apply kernel to each channel independently, and take residual
        noise[:3] = noise[:3] - torch.cat([F.conv2d(noise[i][None], self.K, padding=size//2) for i in range(3)])

        return noise