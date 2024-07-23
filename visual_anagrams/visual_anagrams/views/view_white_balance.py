from PIL import Image
import numpy as np

import torch
import torchvision.transforms.functional as TF

from .view_base import BaseView


class WhiteBalanceViewFailure(BaseView):
    '''
    A failing white balancing view, which simply scales the pixel values
    by some constant factor. An attempt to reproduce the "dress" illusion
    '''
    def __init__(self, factor=1.5):
        self.factor = factor

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)

        # Interpolate factor on t
        factor = 1 + (self.factor - 1) * t

        # Convert to tensor
        im = torch.tensor(np.array(im) / 255.).permute(2,0,1)

        # Adjust colors
        im = im * factor
        im = torch.clip(im, 0, 1)

        # Convert back to PIL
        im = Image.fromarray((np.array(im.permute(1,2,0)) * 255.).astype(np.uint8))

        # Paste on to canvas
        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        frame.paste(im, ((frame_size - im_size) // 2, (frame_size - im_size) // 2))

        return frame

    def view(self, im):
        return im * self.factor

    def inverse_view(self, noise):
        noise[:3] = noise[:3] / self.factor
        return noise

