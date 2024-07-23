from PIL import Image
import numpy as np

import torch

from .view_base import BaseView


class SkewView(BaseView):
    def __init__(self, skew_factor=1.5):
        self.skew_factor = skew_factor

    def skew_image(self, im, skew_factor):
        '''
        Roll each column of the image by increasing displacements.
            This is a permutation of pixels
        '''

        # Params
        c,h,w = im.shape
        h_center = h//2

        # Roll columns
        cols = []
        for i in range(w):
            d = int(skew_factor * (i - h_center))  # Displacement
            col = im[:,:,i]
            cols.append(col.roll(d, dims=1))

        # Stack rolled columns
        skewed = torch.stack(cols, dim=2)
        return skewed

    def view(self, im):
        return self.skew_image(im, self.skew_factor)

    def inverse_view(self, noise):
        return self.skew_image(noise, -self.skew_factor)

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)
        skew_factor = t * self.skew_factor

        # Convert to tensor, skew, then convert back to PIL
        im = torch.tensor(np.array(im) / 255.).permute(2,0,1)
        im = self.skew_image(im, skew_factor)
        im = Image.fromarray((np.array(im.permute(1,2,0)) * 255.).astype(np.uint8))

        # Paste on to canvas
        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        frame.paste(im, ((frame_size - im_size) // 2, (frame_size - im_size) // 2))

        return frame

