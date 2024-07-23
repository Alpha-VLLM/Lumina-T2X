from PIL import Image

import torch

from .view_base import BaseView

class FlipView(BaseView):
    def __init__(self):
        pass

    def view(self, im):
        return torch.flip(im, [1])

    def inverse_view(self, noise):
        return torch.flip(noise, [1])

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)
        theta = -t * 180

        # TODO: Technically not a flip, change this to a homography later
        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        frame.paste(im, ((frame_size - im_size) // 2, (frame_size - im_size) // 2))
        frame = frame.rotate(theta, 
                             resample=Image.Resampling.BILINEAR, 
                             expand=False, 
                             fillcolor=(255,255,255))

        return frame
