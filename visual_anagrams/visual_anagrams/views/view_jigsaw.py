import numpy as np
from PIL import Image
import torch
from einops import einsum, rearrange

from .permutations import make_jigsaw_perm, get_inv_perm
from .view_permute import PermuteView
from .jigsaw_helpers import get_jigsaw_pieces

class JigsawView(PermuteView):
    '''
    Implements a 4x4 jigsaw puzzle view...
    '''
    def __init__(self, seed=4522):
        # Get pixel permutations, corresponding to jigsaw permutations
        self.perm_64, (jigsaw_perm) = make_jigsaw_perm(64, seed=seed)
        self.perm_256, _ = make_jigsaw_perm(256, seed=seed)
        self.perm_1024, _ = make_jigsaw_perm(1024, seed=seed)

        # keep track of jigsaw permutation used
        self.piece_perms, self.edge_swaps = jigsaw_perm

        # Init parent PermuteView, with above pixel perms
        super().__init__(self.perm_64, self.perm_256, self.perm_1024)

    def extract_pieces(self, im):
        '''
        Given an image, extract jigsaw puzzle pieces from it

        im (PIL.Image) :
            PIL Image of the jigsaw illusion
        '''
        im = np.array(im)
        size = im.shape[0]
        pieces = []

        # Get jigsaw pieces
        piece_masks = get_jigsaw_pieces(size)

        # Save pieces
        for piece_mask in piece_masks:
            # Add mask as alpha mask to image
            im_piece = np.concatenate([im, piece_mask[:,:,None] * 255], axis=2)

            # Get extents of piece, and crop
            x_min = np.nonzero(im_piece[:,:,-1].sum(0))[0].min()
            x_max = np.nonzero(im_piece[:,:,-1].sum(0))[0].max()
            y_min = np.nonzero(im_piece[:,:,-1].sum(1))[0].min()
            y_max = np.nonzero(im_piece[:,:,-1].sum(1))[0].max()
            im_piece = im_piece[y_min:y_max+1, x_min:x_max+1]

            pieces.append(Image.fromarray(im_piece))

        return pieces


    def paste_piece(self, piece, x, y, theta, xc, yc, canvas_size=384):
        '''
        Given a PIL Image of a piece, place it so that it's center is at 
            (x,y) and it's rotate about that center at theta degrees

        x (float) : x coordinate to place piece at
        y (float) : y coordinate to place piece at
        theta (float) : degrees to rotate piece about center
        xc (float) : x coordinate of center of piece
        yc (float) : y coordinate of center of piece
        '''

        # Make canvas
        canvas = Image.new("RGBA", 
                           (canvas_size, canvas_size), 
                           (255, 255, 255, 0))

        # Past piece so center is at (x, y)
        canvas.paste(piece, (x-xc,y-yc), piece)

        # Rotate about (x, y)
        canvas = canvas.rotate(theta, resample=Image.BILINEAR, center=(x, y))
        return canvas


    def make_frame(self, im, t, knot_seed=0):
        '''
        This function returns a PIL image of a frame animating a jigsaw
            permutation. Pieces move and rotate from the identity view 
            (t = 0) to the rearranged view (t = 1) along splines.

        The approach is as follows:

            1. Extract all 16 pieces
            2. Figure out start locations for each of these pieces (t=0)
            3. Figure out how these pieces permute
            4. Using these permutations, figure out end locations (t=1)
            5. Make knots for splines, randomly offset normally from the 
                    midpoint of the start and end locations
            6. Paste pieces into correct locations, determined by 
                    spline interpolation

        im (PIL.Image) :
            PIL image representing the jigsaw illusion

        t (float) :
            Interpolation parameter in [0,1] indicating what frame of the
            animation to generate

        knot_seed (int) :
            Seed for random offsets for the knots
        '''
        im_size = im.size[0]
        canvas_size = int(1.5 * im_size)

        # Extract 16 jigsaw pieces
        pieces = self.extract_pieces(im)

        # Rotate all pieces to "base" piece orientation
        pieces = [p.rotate(90 * (i % 4), 
                           resample=Image.BILINEAR, 
                           expand=1) for i, p in enumerate(pieces)]

        # Get (hardcoded) start locations for each base piece, on a 
        # 4x4 grid centered on the origin.
        corner_start_loc = np.array([-1.5, -1.5])
        inner_start_loc = np.array([-0.5, -0.5])
        edge_e_start_loc = np.array([-1.5, -0.5])
        edge_f_start_loc = np.array([-1.5, 0.5])
        base_start_locs = np.stack([corner_start_loc,
                                    inner_start_loc,
                                    edge_e_start_loc,
                                    edge_f_start_loc])

        # Construct all start locations by rotating around (0,0)
        # by 90 degrees, 4 times, and concatenating the results
        rot_mats = []
        for theta in -np.arange(4) * 90 / 180 * np.pi:
            rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
            rot_mats.append(rot_mat)
        rot_mats = np.stack(rot_mats)
        start_locs = einsum(base_start_locs, rot_mats, 
                                'start i, rot j i -> start rot j')
        start_locs = rearrange(start_locs, 
                               'start rot j -> (start rot) j')

        # Add rotation information to start locations
        thetas = np.tile(np.arange(4) * -90, 4)[:, None]
        start_locs = np.concatenate([start_locs, thetas], axis=1)

        # Get explicit permutation of pieces from permutation metadata
        perm = self.piece_perms + np.repeat(np.arange(4), 4) * 4
        for edge_idx, to_swap in enumerate(self.edge_swaps):
            if to_swap:
                # Make swap permutation array
                swap_perm = np.arange(16)
                swap_perm[8 + edge_idx], swap_perm[12 + edge_idx] = \
                    swap_perm[12 + edge_idx], swap_perm[8 + edge_idx]

                # Apply swap permutation after perm
                perm = np.array([swap_perm[perm[i]] for i in range(16)])

        # Get inverse perm (the actual permutation needed)...
        perm_inv = get_inv_perm(torch.tensor(perm))

        # ...and use it to get the final locations of pieces
        end_locs = start_locs[perm_inv]

        # Convert start and end locations to pixel coordinate system
        start_locs[:,:2] = (start_locs[:,:2] + 2) * (im_size / 4)
        end_locs[:,:2] = (end_locs[:,:2] + 2) * (im_size / 4)

        # Add offset so pieces are centered on canvas
        start_locs[:,:2] = start_locs[:,:2] + (canvas_size - im_size) // 2
        end_locs[:,:2] = end_locs[:,:2] + (canvas_size - im_size) // 2

        # Get random offsets from middle for spline knot (so path is pretty)
        # Wrapped in a set seed
        original_state = np.random.get_state()
        np.random.seed(knot_seed)
        rand_offsets = np.random.rand(16, 1) * 2 - 1
        rand_offsets = rand_offsets * 2
        eps = np.random.randn(16, 2)    # Add epsilon for divide by zero
        np.random.set_state(original_state)

        # Make spline knots by taking average of start and end, 
        # and offsetting by some amount normal from the line
        avg_locs = (start_locs[:, :2] + end_locs[:, :2]) / 2.
        norm = (end_locs[:, :2] - start_locs[:, :2])
        norm = norm + eps
        norm = norm / np.linalg.norm(norm, axis=1, keepdims=True)
        rot_mat = np.array([[0,1], [-1,0]])
        norm = norm @ rot_mat
        rand_offsets = rand_offsets * (im_size / 4)
        knot_locs = avg_locs + norm * rand_offsets

        # Paste pieces on to a canvas
        canvas = Image.new("RGBA", (canvas_size, canvas_size), (255,255,255,255))
        for i in range(16):
            # Get start and end coords
            y_0, x_0, theta_0 = start_locs[i]
            y_1, x_1, theta_1 = end_locs[i]
            y_k, x_k = knot_locs[i]

            # Take spline interpolation for x and y
            x_int_0 = x_0 * (1-t) + x_k * t
            y_int_0 = y_0 * (1-t) + y_k * t
            x_int_1 = x_k * (1-t) + x_1 * t
            y_int_1 = y_k * (1-t) + y_1 * t
            x = int(np.round(x_int_0 * (1-t) + x_int_1 * t))
            y = int(np.round(y_int_0 * (1-t) + y_int_1 * t))

            # Just take normal interpolation for theta
            theta = int(np.round(theta_0 * (1-t) + theta_1 * t))

            # Get piece in location and rotation
            xc = yc = im_size // 4 // 2
            pasted_piece = self.paste_piece(pieces[i], x, y, theta, xc, yc, canvas_size=canvas_size)

            canvas.paste(pasted_piece, (0,0), pasted_piece)

        return canvas
