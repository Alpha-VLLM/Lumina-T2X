from PIL import Image
import numpy as np

import torch
import torchvision.transforms.functional as TF

from .permutations import make_square_hinge
from .view_permute import PermuteView

class SquareHingeView(PermuteView):
    '''
    Implements a "square hinge" view, where subsquares rotate 90 degrees.
    For an example, see https://www.youtube.com/watch?v=vrOjy-v5JgQ&t=120

    Inherits from `PermuteView`, which implements the `view` and 
    `inverse_view` functions as permutations. We just make
    the correct permutation here, and implement the `make_frame` 
    method for animation
    '''
    def __init__(self):
        '''
        Make the correct "square hinge" permutations and pass it to the
        parent class constructor.
        '''
        self.perm_64 = make_square_hinge(im_size=64)
        self.perm_256 = make_square_hinge(im_size=256)
        self.perm_1024 = make_square_hinge(im_size=1024)

        super().__init__(self.perm_64, self.perm_256, self.perm_1024)

    def paste_pil(self, image, x, y, theta, xc, yc, canvas_size=384):
        '''
        Given a PIL Image, place it on a canvas so that it's center is at 
            (x,y) and it's rotate about that center at theta degrees

        x (float) : x coordinate to place image at
        y (float) : y coordinate to place image at
        theta (float) : degrees to rotate image about center
        xc (float) : x coordinate of center of image
        yc (float) : y coordinate of center of image
        '''

        # Make canvas
        canvas = Image.new("RGBA", 
                           (canvas_size, canvas_size), 
                           (255, 255, 255, 0))

        # Paste iamge so center is at (x, y)
        canvas.paste(image, (x-xc,y-yc))

        # Rotate about (x, y)
        canvas = canvas.rotate(theta, resample=Image.BILINEAR, center=(x, y))
        return canvas

    def make_frame(self, im, t):
        # Get constants
        im_size = im.size[0]
        square_size = im_size // 3
        frame_size = int(im_size * 1.5)
        theta = -t * 90

        # Do math (find center of all squares)
        theta_rad = -theta / 180 * np.pi
        dist_btwn_centers = square_size * (np.sin(theta_rad) + \
                                           np.cos(theta_rad))

        # Convert to tensor
        im = torch.tensor(np.array(im) / 255.).permute(2,0,1)

        # Make canvas
        canvas = Image.new('RGB', (frame_size, frame_size), (255, 255, 255, 0))

        # Paste squares onto canvases
        for i in range(3):
            for j in range(3):
                # Get direction to rotate
                k = 1 if (i+j)%2 == 0 else -1

                # Get offset from center
                offset_x = (i - 1) * dist_btwn_centers
                offset_y = (j - 1) * dist_btwn_centers

                # Get square bounds
                x0 = i*square_size
                x1 = x0 + square_size
                y0 = j*square_size
                y1 = y0 + square_size

                # Get square (and add alpha channel)
                subsquare = TF.to_pil_image(im[:,y0:y1,x0:x1]).convert('RGBA')
                subsquare = TF.to_tensor(subsquare)

                # Rotate here, b/c PIL rotation sucks (only takes 
                # integer coordinates to rotate about)
                subsquare = TF.to_pil_image(TF.rotate(subsquare, theta * k, expand=True, fill=0, interpolation=TF.InterpolationMode.BILINEAR))

                # Paste square
                subsquare = self.paste_pil(subsquare, 
                                           np.round(frame_size//2 + offset_x).astype(int), 
                                           np.round(frame_size//2 + offset_y).astype(int), 
                                           0, 
                                           (square_size+1)//2, 
                                           (square_size+1)//2,
                                           canvas_size=frame_size)

                canvas.paste(subsquare, (0,0), subsquare)

        return canvas

