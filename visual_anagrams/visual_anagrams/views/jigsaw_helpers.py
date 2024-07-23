from pathlib import Path
from PIL import Image
import numpy as np

def get_jigsaw_pieces(size):
    '''
    Load all pieces of the 4x4 jigsaw puzzle.

    size (int) :
        Should be 64, 256, or 1024 indicating side length of jigsaw puzzle
    '''

    # Location of pieces
    piece_dir = Path(__file__).parent / 'assets'

    # Helper function to load pieces as np arrays
    def load_pieces(path):
        '''
        Load a piece, from the given path, as a binary numpy array.
        Return a list of the "base" piece, and all four of its rotations.
        '''
        piece = Image.open(path)
        piece = np.array(piece)[:,:,0] // 255
        pieces = np.stack([np.rot90(piece, k=-i) for i in range(4)])
        return pieces

    # Load pieces and rotate to get 16 pieces, and cat
    pieces_corner = load_pieces(piece_dir / f'4x4/4x4_corner_{size}.png')
    pieces_inner = load_pieces(piece_dir / f'4x4/4x4_inner_{size}.png')
    pieces_edge1 = load_pieces(piece_dir / f'4x4/4x4_edge1_{size}.png')
    pieces_edge2 = load_pieces(piece_dir / f'4x4/4x4_edge2_{size}.png')
    pieces = np.concatenate([pieces_corner, pieces_inner, pieces_edge1, pieces_edge2])

    return pieces

