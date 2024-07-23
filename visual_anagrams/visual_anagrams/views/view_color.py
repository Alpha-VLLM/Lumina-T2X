from PIL import Image
import numpy as np

import torch

from .view_base import BaseView

def make_frame_color(im, t):
    im_size = im.size[0]
    frame_size = int(im_size * 1.5)

    # Convert to tensor
    im = torch.tensor(np.array(im) / 255.).permute(2,0,1)

    # Extract color and greyscale components
    im_grey = im.clone()
    im_grey[:] = im.mean(dim=0, keepdim=True)
    im_color = im - im_grey

    # Take linear interpolation
    im = im_grey + t * im_color

    # Convert back to PIL
    im = Image.fromarray((np.array(im.permute(1,2,0)) * 255.).astype(np.uint8))

    # Paste on to canvas
    frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
    frame.paste(im, ((frame_size - im_size) // 2, (frame_size - im_size) // 2))

    return frame

class GrayscaleView(BaseView):
    def __init__(self):
        pass

    def make_frame(self, im, t):
        return make_frame_color(im, t)

    def view(self, im):
        return im

    def save_view(self, im):
        im = torch.stack([im.mean(0)] * 3)
        return im

    def inverse_view(self, noise):
        # Get grayscale component by averaging color channels
        noise[:3] = torch.stack([noise[:3].mean(0)] * 3)
        return noise


class ColorView(BaseView):
    def __init__(self):
        pass

    def make_frame(self, im, t):
        return make_frame_color(im, t)

    def view(self, im):
        return im

    def inverse_view(self, noise):
        # Get color component by taking residual
        noise[:3] = noise[:3] - torch.stack([noise[:3].mean(0)] * 3)
        return noise