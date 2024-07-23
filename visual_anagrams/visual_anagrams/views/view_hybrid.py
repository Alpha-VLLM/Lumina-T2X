from PIL import Image
import numpy as np

import torch
import torchvision.transforms.functional as TF

from .view_base import BaseView

'''
n.b.
small sigma => low freq is easier to see
larger sigma => high freq easier to see
kernel size must be large enough to avoid edge effects
    (kernel_size = 5 x sigma seems good enough)
'''

def make_frame_hybrid(im, t, resize_factor):
    im_size = im.size[0]
    frame_size = int(im_size * 1.5)
    new_size = int( im_size / (1 + (resize_factor - 1) * t) )

    # Convert to tensor
    im = torch.tensor(np.array(im) / 255.).permute(2,0,1)

    # Resize to new size
    im = TF.resize(im, new_size)

    # Convert back to PIL
    im = (np.array(im.permute(1,2,0)) * 255.)
    im = im.astype(np.uint8)
    im = Image.fromarray(im)

    # Paste on to canvas
    frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
    coords = ((frame_size - new_size) // 2, (frame_size - new_size) // 2)
    frame.paste(im, coords)

    return frame

class HybridLowPassView(BaseView):
    def __init__(self, sigma=2, kernel_size=33):
        self.sigma = sigma
        self.kernel_size = kernel_size

    def make_frame(self, im, t, resize_factor=12):
        return make_frame_hybrid(im, t, resize_factor)

    def view(self, im):
        # For factorized diffusion, we don't change the input to the model
        return im

    def inverse_view(self, noise):
        c, h, w = noise.shape

        # To account for two stages, scale kernel size and sigma
        # based on image size (either 64x64 or 256x256)
        factor = h // 64
        k = self.kernel_size * factor + ((factor + 1) % 2)
        sigma = self.sigma * factor

        # Low pass noise estimate
        noise[:3] = TF.gaussian_blur(noise[:3], k, sigma)

        return noise


class HybridHighPassView(BaseView):
    def __init__(self, sigma=2, kernel_size=33):
        self.sigma = sigma
        self.kernel_size = kernel_size

    def make_frame(self, im, t, resize_factor=12):
        return make_frame_hybrid(im, t, resize_factor)

    def view(self, im):
        # For factorized diffusion, we don't change the input to the model
        return im

    def inverse_view(self, noise):
        c, h, w = noise.shape

        # To account for two stages, scale kernel size and sigma
        # based on image size (either 64x64 or 256x256)
        factor = h // 64
        k = self.kernel_size * factor + ((factor + 1) % 2)
        sigma = self.sigma * factor

        # High pass noise estimate
        noise[:3] = (noise[:3] - TF.gaussian_blur(noise[:3], k, sigma))

        return noise




###########################
### TRIPLE HYBRID VIEWS ###
###########################

def compute_triple_bandpass(noise, sigma_1, sigma_2, kernel_size):
    '''
    Computes all three bandpasses given two sigma values
    '''
    c, h, w = noise.shape

    # To account for two stages, scale kernel size and sigma
    # based on image size (either 64x64 or 256x256)
    factor = h // 64
    k = kernel_size * factor + ((factor + 1) % 2)
    sigma_1 = sigma_1 * factor
    sigma_2 = sigma_2 * factor

    # Compute all bandpasses
    est = noise[:3] 
    mp = TF.gaussian_blur(est, k, sigma_1)
    hp = est - mp
    lp = TF.gaussian_blur(mp, k, sigma_2)
    mp = mp - lp

    return lp, mp, hp

class TripleHybridLowPassView(BaseView):
    def __init__(self, sigma_1=1, sigma_2=2, kernel_size=25):
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.kernel_size = kernel_size

    def view(self, im):
        # For factorized diffusion, we don't change the input to the model
        return im

    def inverse_view(self, noise):
        lp, _, _ = compute_triple_bandpass(noise, 
                                           self.sigma_1, 
                                           self.sigma_2, 
                                           self.kernel_size)
        noise[:3] = lp
        return noise

class TripleHybridMediumPassView(BaseView):
    def __init__(self, sigma_1=1, sigma_2=2, kernel_size=25):
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.kernel_size = kernel_size

    def view(self, im):
        # For factorized diffusion, we don't change the input to the model
        return im

    def inverse_view(self, noise):
        _, mp, _ = compute_triple_bandpass(noise,
                                           self.sigma_1, 
                                           self.sigma_2, 
                                           self.kernel_size)
        noise[:3] = mp
        return noise

class TripleHybridHighPassView(BaseView):
    def __init__(self, sigma_1=1, sigma_2=2, kernel_size=25):
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.kernel_size = kernel_size

    def view(self, im):
        # For factorized diffusion, we don't change the input to the model
        return im

    def inverse_view(self, noise):
        _, _, hp = compute_triple_bandpass(noise,
                                           self.sigma_1, 
                                           self.sigma_2, 
                                           self.kernel_size)
        noise[:3] = hp
        return noise

