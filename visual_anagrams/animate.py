from PIL import Image

from visual_anagrams.views import get_views
from visual_anagrams.animate import animate_two_view, animate_two_view_motion_blur
from visual_anagrams.views.view_motion import MotionBlurView


if __name__ == '__main__':
    import argparse
    import pickle
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--im_path", required=True, type=str, help='Path to the illusion to animate')
    parser.add_argument("--save_video_path", default=None, type=str, 
        help='Path to save video to. If None, defaults to `im_path`, with extension `.mp4`')
    parser.add_argument("--metadata_path", default=None, type=str, help='Path to metadata. If specified, overrides `view` and `prompt` args')
    parser.add_argument("--view", default=None, type=str, help='Name of view to use')
    parser.add_argument("--prompt_1", default='', nargs='+', type=str,
        help='Prompt for first view. Passing multiple will join them with newlines.')
    parser.add_argument("--prompt_2", default='', nargs='+', type=str,
        help='Prompt for first view. Passing multiple will join them with newlines.')
    args = parser.parse_args()


    # Load image to animate
    im_path = Path(args.im_path)
    im = Image.open(im_path)

    # Get save dir
    if args.save_video_path is None:
        save_video_path = im_path.with_suffix('.mp4')

    # Get prompts and views from metadata
    if args.metadata_path is None:
        # Join prompts with newlines
        prompt_1 = '\n'.join(args.prompt_1)
        prompt_2 = '\n'.join(args.prompt_2)

        # Get paths and views
        view = get_views([args.view])[0]
    else:
        with open(args.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        view = metadata['views'][1]
        m_args = metadata['args']
        prompt_1 = f'{m_args.style} {m_args.prompts[0]}'.strip()
        prompt_2 = f'{m_args.style} {m_args.prompts[1]}'.strip()

    # Get sizes
    im_size = im.size[0]
    frame_size = int(im_size * 1.5)

    if any([isinstance(view, MotionBlurView) for view in metadata['views']]):
        # Animate specifically motion blur views
        animate_two_view_motion_blur(
                im,
                view,
                prompt_1,
                prompt_2,
                save_video_path=save_video_path,
                hold_duration=60,
                text_fade_duration=10,
                transition_duration=2000,
                im_size=im_size,
                frame_size=frame_size,
            )
    else:
        # Animate all other views
        animate_two_view(
                im,
                view,
                prompt_1,
                prompt_2,
                save_video_path=save_video_path,
                hold_duration=120,
                text_fade_duration=10,
                transition_duration=45,
                im_size=im_size,
                frame_size=frame_size,
            )