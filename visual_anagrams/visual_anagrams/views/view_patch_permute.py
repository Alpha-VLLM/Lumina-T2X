from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange

from .permutations import get_inv_perm
from .view_base import BaseView


class PatchPermuteView(BaseView):
    def __init__(self, num_patches=8):
        '''
        Implements random patch permutations, with `num_patches`
            patches per side

        num_patches (int) :
            Number of patches in one dimension. Total number
            of patches will be num_patches**2. Should be a power of 2.
        '''

        assert 64 % num_patches == 0 and 256 % num_patches == 0, \
            "`num_patches` must divide image side lengths of 64 and 256"

        self.num_patches = num_patches

        # Get random permutation and inverse permutation
        self.perm = torch.randperm(self.num_patches**2)
        self.perm_inv = get_inv_perm(self.perm)

    def view(self, im):
        im_size = im.shape[-1]

        # Get number of pixels on one side of a patch
        patch_size = int(im_size / self.num_patches)

        # Reshape into patches of size (c, patch_size, patch_size)
        patches = rearrange(im, 
                            'c (h p1) (w p2) -> (h w) c p1 p2', 
                            p1=patch_size, 
                            p2=patch_size)

        # Permute
        patches = patches[self.perm]

        # Reshape back into image
        im_rearr = rearrange(patches, 
                             '(h w) c p1 p2 -> c (h p1) (w p2)', 
                             h=self.num_patches, 
                             w=self.num_patches, 
                             p1=patch_size, 
                             p2=patch_size)
        return im_rearr

    def inverse_view(self, noise):
        im_size = noise.shape[-1]

        # Get number of pixels on one side of a patch
        patch_size = int(im_size / self.num_patches)

        # Reshape into patches of size (c, patch_size, patch_size)
        patches = rearrange(noise, 
                            'c (h p1) (w p2) -> (h w) c p1 p2', 
                            p1=patch_size, 
                            p2=patch_size)

        # Apply inverse permutation
        patches = patches[self.perm_inv]

        # Reshape back into image
        im_rearr = rearrange(patches, 
                             '(h w) c p1 p2 -> c (h p1) (w p2)', 
                             h=self.num_patches, 
                             w=self.num_patches, 
                             p1=patch_size, 
                             p2=patch_size)
        return im_rearr

    def make_frame(self, im, t, knot_seed=0):
        # Get useful info
        im_size = im.size[0]
        canvas_size = int(1.5 * im_size)
        offset = (canvas_size - im_size) // 2  # offset to center animation

        # Scale is a hack, because PIL for some reason doesn't support pasting
        #   at floating point coordinates. So just render at larger scale
        #   and then resize by 1/scale, to get more precise control
        if im_size == 1024:
            # Don't use scale if image is larger
            # as it is unnecessary and quite slow
            scale = 1
        else:
            scale = 4
        

        canvas_size = canvas_size * scale
        offset = offset * scale

        im = TF.to_tensor(im)

        # Get number of pixels on one side of a patch
        im_size = im.shape[-1]
        patch_size = int(im_size / self.num_patches)

        # Extract patches
        patches = rearrange(im, 
                            'c (h p1) (w p2) -> (h w) c p1 p2', 
                            p1=patch_size, 
                            p2=patch_size)

        # Get start locations (top left corner of patch)
        yy, xx = torch.meshgrid(
                        torch.arange(self.num_patches), 
                        torch.arange(self.num_patches)
                    )
        xx = xx.flatten()
        yy = yy.flatten()
        start_locs = torch.stack([xx, yy], dim=1) * patch_size * scale
        start_locs = start_locs + offset

        # Get end locations by permuting
        end_locs = start_locs[self.perm_inv]

        # Get random anchor locations
        original_state = np.random.get_state()
        np.random.seed(knot_seed)
        rand_offsets = np.random.rand(self.num_patches**2, 1) * 2 - 1
        rand_offsets = rand_offsets * 2 * scale
        eps = np.random.randn(*start_locs.shape)    # Add epsilon for divide by zero
        np.random.set_state(original_state)

        # Make spline knots by taking average of start and end, 
        # and offsetting by some amount normal from the line
        avg_locs = (start_locs + end_locs) / 2.
        norm = (end_locs - start_locs)
        norm = norm + eps
        norm = norm / np.linalg.norm(norm, axis=1, keepdims=True)
        rot_mat = np.array([[0,1], [-1,0]])
        norm = norm @ rot_mat
        rand_offsets = rand_offsets * (im_size / 4)
        knot_locs = avg_locs + norm * rand_offsets

        # Get paste locations
        spline_0 = start_locs * (1 - t) + knot_locs * t
        spline_1 = knot_locs * (1 - t) + end_locs * t
        paste_locs = spline_0 * (1 - t) + spline_1 * t
        paste_locs = paste_locs.to(int)

        # Paste patches onto canvas
        canvas = Image.new("RGBA", (canvas_size, canvas_size), (255,255,255,255))
        for patch, paste_loc in zip(patches, paste_locs):
            patch = TF.to_pil_image(patch).convert('RGBA')
            patch = patch.resize((patch_size * scale, patch_size * scale))
            paste_loc = (paste_loc[0].item(), paste_loc[1].item())
            canvas.paste(patch, paste_loc, patch)

        if scale != 1.0:
            canvas = canvas.resize((canvas_size // scale, canvas_size // scale))

        return canvas
