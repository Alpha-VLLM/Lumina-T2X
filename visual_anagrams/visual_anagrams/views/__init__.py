from pathlib import Path
from PIL import Image
import numpy as np

from .view_identity import IdentityView
from .view_flip import FlipView
from .view_rotate import Rotate180View, Rotate90CCWView, Rotate90CWView
from .view_negate import NegateView
from .view_skew import SkewView
from .view_patch_permute import PatchPermuteView
from .view_jigsaw import JigsawView
from .view_inner_circle import InnerCircleView, InnerCircleViewFailure
from .view_square_hinge import SquareHingeView
from .view_blur import BlurViewFailure
from .view_white_balance import WhiteBalanceViewFailure
from .view_hybrid import HybridLowPassView, HybridHighPassView, \
    TripleHybridHighPassView, TripleHybridLowPassView, \
    TripleHybridMediumPassView
from .view_color import ColorView, GrayscaleView
from .view_motion import MotionBlurResView, MotionBlurView
from .view_scale import ScaleView

VIEW_MAP = {
    'identity': IdentityView,
    'flip': FlipView,
    'rotate_cw': Rotate90CWView,
    'rotate_ccw': Rotate90CCWView,
    'rotate_180': Rotate180View,
    'negate': NegateView,
    'skew': SkewView,
    'patch_permute': PatchPermuteView,
    'pixel_permute': PatchPermuteView,
    'jigsaw': JigsawView,
    'inner_circle': InnerCircleView,
    'square_hinge': SquareHingeView,
    'inner_circle_failure': InnerCircleViewFailure,
    'blur_failure': BlurViewFailure,
    'white_balance_failure': WhiteBalanceViewFailure,
    'low_pass': HybridLowPassView,
    'high_pass': HybridHighPassView,
    'triple_low_pass': TripleHybridLowPassView,
    'triple_medium_pass': TripleHybridMediumPassView,
    'triple_high_pass': TripleHybridHighPassView,
    'grayscale': GrayscaleView,
    'color': ColorView,
    'motion': MotionBlurView,
    'motion_res': MotionBlurResView,
    'scale': ScaleView,
}

def get_anagrams_views(view_names, view_args=None):
    '''
    Bespoke function to get views (just to make command line usage easier)
    '''

    views = []
    if view_args is None:
        view_args = [None for _ in view_names]

    for view_name, view_arg in zip(view_names, view_args):
        if view_name == 'patch_permute':
            args = [8 if view_arg is None else int(view_arg)]
        elif view_name == 'pixel_permute':
            args = [64 if view_arg is None else int(view_arg)]
        elif view_name == 'skew':
            args = [1.5 if view_arg is None else float(view_arg)]
        elif view_name in ['low_pass', 'high_pass']:
            args = [2.0 if view_arg is None else float(view_arg)]
        elif view_name in ['scale']:
            args = [0.5 if view_arg is None else float(view_arg)]
        else:
            args = []

        view = VIEW_MAP[view_name](*args)
        views.append(view)

    return views
