from PIL import Image
import numpy as np

import torch

from .view_base import BaseView

class NegateView(BaseView):
    def __init__(self):
        pass

    def view(self, im):
        return -im

    def inverse_view(self, noise):
        '''
        Negating the variance estimate is "weird" so just don't do it.
            This hack seems to work just fine
        '''
        invert_mask = torch.ones_like(noise)
        invert_mask[:3] = -1
        return noise * invert_mask

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)

        # map t from [0, 1] -> [1, -1]
        t = 1 - t
        t = t * 2 - 1

        # Interpolate from pixels from [0, 1] to [1, 0]
        im = np.array(im) / 255.
        im = ((2 * im - 1) * t + 1) / 2.
        im = Image.fromarray((im * 255.).astype(np.uint8))

        # Paste on to canvas
        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        frame.paste(im, ((frame_size - im_size) // 2, (frame_size - im_size) // 2))

        return frame
