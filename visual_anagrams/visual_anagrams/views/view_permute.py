import torch
from einops import rearrange

from .permutations import get_inv_perm
from .view_base import BaseView

class PermuteView(BaseView):
    def __init__(self, perm_64, perm_256, perm_1024=None):
        '''
        Implements arbitrary pixel permutations, for a given permutation. 
            We need two permutations. One of size 64x64 for stage 1, and 
            one of size 256x256 for stage 2.

        perm_64 (torch.tensor) :
            Tensor of integer indexes, defining a permutation, of size 64*64

        perm_256 (torch.tensor) :
            Tensor of integer indexes, defining a permutation, of size 256*256

        perm_1024 (torch.tensor) :
            Tensor of integer indexes, defining a permutation, of size 1024*1024
        '''

        assert perm_64.shape == torch.Size([64*64]), \
            "`perm_64` must be a permutation tensor of size 64*64"

        assert perm_256.shape == torch.Size([256*256]), \
            "`perm_256` must be a permutation tensor of size 256*256"

        # Save permutation and inverse permutation for stage 1
        self.perm_64 = perm_64
        self.perm_64_inv = get_inv_perm(self.perm_64)

        # Save permutation and inverse permutation for stage 2
        self.perm_256 = perm_256
        self.perm_256_inv = get_inv_perm(self.perm_256)

        # Save permutation and inverse permutation for stage 3
        if perm_1024 is None:
            self.perm_1024 = torch.arange(1024*1024)
            self.perm_1024_inv = torch.arange(1024*1024)
        else:
            self.perm_1024 = perm_1024
            self.perm_1024_inv = get_inv_perm(self.perm_1024)

    def view(self, im):
        im_size = im.shape[-1]
        if im_size == 64:
            perm = self.perm_64
        elif im_size == 256:
            perm = self.perm_256
        elif im_size == 1024:
            perm = self.perm_1024
        else:
            raise Exception("Invalid image size. Must be size 64, 256, or 1024.")

        # Permute every pixel in the image (so num_patches == num_pixels)
        num_patches = im_size
        patch_size = 1

        # Reshape into patches of size (c, patch_size, patch_size)
        patches = rearrange(im, 
                            'c (h p1) (w p2) -> (h w) c p1 p2', 
                            p1=patch_size, 
                            p2=patch_size)

        # Permute
        patches = patches[perm]

        # Reshape back into image
        im_rearr = rearrange(patches, 
                             '(h w) c p1 p2 -> c (h p1) (w p2)', 
                             h=num_patches, 
                             w=num_patches, 
                             p1=patch_size, 
                             p2=patch_size)
        return im_rearr

    def inverse_view(self, noise):
        im_size = noise.shape[-1]
        perm_inv = self.perm_64_inv if im_size == 64 else self.perm_256_inv
        num_patches = im_size

        # Permute every pixel in the image
        patch_size = 1

        # Reshape into patches of size (c, patch_size, patch_size)
        patches = rearrange(noise, 
                            'c (h p1) (w p2) -> (h w) c p1 p2', 
                            p1=patch_size, 
                            p2=patch_size)

        # Apply inverse permutation
        patches = patches[perm_inv]

        # Reshape back into image
        im_rearr = rearrange(patches, 
                             '(h w) c p1 p2 -> c (h p1) (w p2)', 
                             h=num_patches, 
                             w=num_patches, 
                             p1=patch_size, 
                             p2=patch_size)
        return im_rearr

    def make_frame(self, im, t):
        # TODO: Implement this, as just moving pixels around
        raise NotImplementedError()


