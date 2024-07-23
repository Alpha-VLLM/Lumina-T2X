from PIL import Image
import numpy as np

import torch
import torchvision.transforms.functional as TF

from .view_base import BaseView


class BlurViewFailure(BaseView):
    '''
    A failing blur view, which blurs an image, in an attempt
    to synthesize hybrid images.
    '''
    def __init__(self, factor=8):
        self.factor = factor

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)
        new_size = int( im_size / (1 + (self.factor - 1) * t) )

        # Convert to tensor
        im = torch.tensor(np.array(im) / 255.).permute(2,0,1)

        # Resize to new size
        im = TF.resize(im, new_size)

        # Convert back to PIL
        im = Image.fromarray((np.array(im.permute(1,2,0)) * 255.).astype(np.uint8))

        # Paste on to canvas
        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        frame.paste(im, ((frame_size - new_size) // 2, (frame_size - new_size) // 2))

        return frame

    def view(self, im):
        im_size = im.shape[-1]

        # Downsample then upsample to "blur"
        im_small = TF.resize(im, im_size // self.factor)
        im_blur = TF.resize(im_small, im_size)

        return im_blur

    def inverse_view(self, noise):
        # The transform is technically uninvertible, so just do pass through
        return noise

