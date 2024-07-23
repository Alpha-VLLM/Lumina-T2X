from pathlib import Path
import numpy as np
import torch 
import torchvision.transforms.functional as TF
from einops import rearrange, repeat

from .jigsaw_helpers import get_jigsaw_pieces

def get_inv_perm(perm):
    '''
    Get the inverse permutation of a permutation. That is, the array such that
        perm[perm_inv] = perm_inv[perm] = arange(len(perm))

    perm (torch.tensor) :
        A 1-dimensional integer array, representing a permutation. Indicates
        that element i should move to index perm[i]
    '''
    perm_inv = torch.empty_like(perm)
    perm_inv[perm] = torch.arange(len(perm))
    return perm_inv


def make_inner_circle_perm(im_size=64, r=24):
    '''
    Makes permutations for "inner circle" view. Given size of image, and
    `r`, the radius of the circle. We do this by iterating through every 
    pixel and figuring out where it should go.
    '''
    perm = []       # Permutation array

    # Iterate through all positions, in order
    for iy in range(im_size):
        for ix in range(im_size):
            # Get coordinates, with origin at (0, 0)
            x = ix - im_size // 2 + 0.5
            y = iy - im_size // 2 + 0.5

            # Do 180 deg rotation if in circle
            if x**2 + y**2 < r**2:
                x = -x
                y = -y

            # Convert back to integer coordinates
            x = int(x + im_size // 2 - 0.5)
            y = int(y + im_size // 2 - 0.5)

            # Append destination pixel index to permutation
            perm.append(x + y * im_size)
    perm = torch.tensor(perm)

    return perm



def make_square_hinge(im_size=64):
    '''
    Makes permutations for "square hinge" view, given size of image.
    For an example, see https://www.youtube.com/watch?v=vrOjy-v5JgQ&t=120
    We use a 3x3 grid of squares, so there will be 1 extra pixel
    for a size 256x256 square, which we just ignore
    '''

    # Get size of sub square
    square_size = im_size // 3

    # Make idxs
    idxs = torch.arange(im_size**2).view(im_size, im_size)

    # Rotate sub squares 90 degrees, depending on location
    for i in range(3):
        for j in range(3):
            # Get direction to rotate
            k = -1 if (i+j)%2 == 0 else 1

            # Get square bounds
            x0 = i*square_size
            x1 = x0 + square_size
            y0 = j*square_size
            y1 = y0 + square_size

            # Rotate sub square
            idxs[x0:x1,y0:y1] = \
                    torch.rot90(idxs[x0:x1,y0:y1], k=k)
    return idxs.flatten()



def make_jigsaw_perm(size, seed=0):
    '''
    Returns a permutation of pixels that is a jigsaw permutation

    There are 3 types of pieces: corner, edge, and inner pieces. These were
        created in MS Paint. They are all identical and laid out like:

        c0 e0 f0 c1
        f3 i0 i1 e1
        e3 i3 i2 f1
        c3 f2 e2 c2

        where c is "corner," i is "inner," and "e" and "f" are "edges."
        "e" and "f" pieces are identical, but labeled differently such that
        to move any piece to the next index you can apply a 90 deg rotation.

    Pieces c0, e0, f0, and i0 are defined by pngs, and will be loaded in. All
        other pieces are obtained by 90 deg rotations of these "base" pieces.

    Permutations are defined by:
        1. permutation of corner (c) pieces (length 4 perm list)
        2. permutation of inner (i) pieces (length 4 perm list)
        3. permutation of edge (e) pieces (length 4 perm list)
        4. permutation of edge (f) pieces (length 4 perm list)
        5. list of four swaps, indicating swaps between e and f 
                edge pieces along the same edge (length 4 bit list)

        Note these perm indexes will just be a "rotation index" indicating 
        how many 90 deg rotations to apply to the base pieces. The swaps 
        ensure that any edge piece can go to any edge piece, and are indexed 
        by the indexes of the "e" and "f" pieces on the edge.

    Also note, order of indexes in permutation array is raster scan order. So,
        go along x's first, then y's. This means y * size + x gives us the 
        1-D location in the permutation array. And image arrays are in 
        (y,x) order.

    Plan of attack for making a pixel permutation array that represents
        a jigsaw permutation:

        1. Iterate through all pixels (in raster scan order)
        2. Figure out which puzzle piece it is in initially
        3. Look at the permutations, and see where it should go
        4. Additionally, see if it's an edge piece, and needs to be swapped
        5. Add the new (1-D) index to the permutation array

    '''
    np.random.seed(seed)

    # Get random permutations of groups of 4, and cat
    identity = np.arange(4)
    perm_corner = np.random.permutation(identity)
    perm_inner = np.random.permutation(identity)
    perm_edge1 = np.random.permutation(identity)
    perm_edge2 = np.random.permutation(identity)
    edge_swaps = np.random.randint(2, size=4)
    piece_perms = np.concatenate([perm_corner, perm_inner, perm_edge1, perm_edge2])

    # Get all 16 jigsaw pieces (in the order above)
    pieces = get_jigsaw_pieces(size)

    # Make permutation array to fill
    perm = []

    # For each pixel, figure out where it should go
    for y in range(size):
        for x in range(size):
            # Figure out which piece (x,y) is in:
            piece_idx = pieces[:,y,x].argmax()

            # Figure out how many 90 deg rotations are on the piece
            rot_idx = piece_idx % 4

            # The perms tells us how many 90 deg rotations to apply to
            # arrive at new pixel location
            dest_rot_idx = piece_perms[piece_idx]
            angle = (dest_rot_idx - rot_idx) * 90 / 180 * np.pi

            # Center coordinates on origin
            cx = x - (size - 1) / 2.
            cy = y - (size - 1) / 2.

            # Perform rotation
            nx = np.cos(angle) * cx - np.sin(angle) * cy
            ny = np.sin(angle) * cx + np.cos(angle) * cy

            # Translate back and round coordinates to _nearest_ integer
            nx = nx + (size - 1) / 2.
            ny = ny + (size - 1) / 2.
            nx = int(np.rint(nx))
            ny = int(np.rint(ny))

            # Perform swap if piece is an edge, and swap == 1 at NEW location
            new_piece_idx = pieces[:,ny,nx].argmax()
            edge_idx = new_piece_idx % 4
            if new_piece_idx >= 8 and edge_swaps[edge_idx] == 1:
                is_f_edge = (new_piece_idx - 8) // 4    # 1 if f, 0 if e edge
                edge_type_parity = 1 - 2 * is_f_edge
                rotation_parity = 1 - 2 * (edge_idx // 2)
                swap_dist = size // 4

                # if edge_idx is even, swap in x direction, else y
                if edge_idx % 2 == 0:
                    nx = nx + swap_dist * edge_type_parity * rotation_parity
                else:
                    ny = ny + swap_dist * edge_type_parity * rotation_parity

            # append new index to permutation array
            new_idx = int(ny * size + nx)
            perm.append(new_idx)

    # sanity check
    #import matplotlib.pyplot as plt
    #missing = sorted(set(range(size*size)).difference(set(perm)))
    #asdf = np.zeros(size*size)
    #asdf[missing] = 1
    #plt.imshow(asdf.reshape(size,size))
    #plt.savefig('tmp.png')
    #plt.show()
    #print(np.sum(asdf))

    #viz = np.zeros((64,64))
    #for idx in perm:
    #    y, x = idx // 64, idx % 64
    #    viz[y,x] = 1
    #plt.imshow(viz)
    #plt.savefig('tmp.png')
    #Image.fromarray(viz * 255).convert('RGB').save('tmp.png')
    #Image.fromarray(pieces_edge1[0] * 255).convert('RGB').save('tmp.png')

    # sanity check on test image
    #im = Image.open('results/flip.campfire.man/0000/sample_64.png')
    #im = Image.open('results/flip.campfire.man/0000/sample_256.png')
    #im = np.array(im)
    #Image.fromarray(im.reshape(-1, 3)[perm].reshape(size,size,3)).save('test.png')

    return torch.tensor(perm), (piece_perms, edge_swaps)

#for i in range(100):
    #make_jigsaw_perm(64, seed=i)
#make_jigsaw_perm(256, seed=11)


def recover_patch_permute(im_0, im_1, patch_size):
    '''
    Given two views of a patch permutation illusion, recover the patch 
    permutation used.

    im_0 (PIL.Image) :
        Identity view of the illusion

    im_1 (PIL.Image) :
        Patch permuted view of the illusion

    patch_size (int) :
        Size of the patches in the image
    '''

    # Convert to tensors
    im_0 = TF.to_tensor(im_0)
    im_1 = TF.to_tensor(im_1)

    # Extract patches
    patches_0 = rearrange(im_0,
                          'c (h p1) (w p2) -> (h w) c p1 p2', 
                          p1=patch_size, 
                          p2=patch_size)
    patches_1 = rearrange(im_1,
                          'c (h p1) (w p2) -> (h w) c p1 p2', 
                          p1=patch_size, 
                          p2=patch_size)

    # Repeat patches_1 for each patch in patches_0
    patches_1_repeated = repeat(patches_1, 
                                'np c p1 p2 -> np1 np c p1 p2', 
                                np=patches_1.shape[0], 
                                np1=patches_1.shape[0], 
                                p1=patch_size, 
                                p2=patch_size)

    # Find closest patch in other image by L1 dist, and return indexes
    perm = (patches_1_repeated - patches_0[:,None]).abs().sum((2,3,4)).argmin(1)
    dist = (patches_1_repeated - patches_0[:,None]).abs().sum((2,3,4))[torch.arange(len(perm)), perm].sum()
    print(f'Total L1 distance: {dist.item()}')

    return perm
