from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

import torch
import torchvision.transforms.functional as TF

from visual_anagrams.utils import get_courier_font_path


def draw_text(image, text, fill=(0,0,0), frame_size=384, im_size=256):
    image = image.copy()

    # Font info. Use 16pt for 384 pixel image, and scale up accordingly
    font_path = get_courier_font_path()
    font_size = 16
    font_size = int(font_size * frame_size / 384)

    # Make PIL objects
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    
    # Center text horizontally, and vertically between
    # illusion bottom and frame bottom
    text_position = (0, 0)
    bbox = draw.textbbox(text_position, text, font=font, align='center')
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_left = (frame_size - text_width) // 2
    text_top = int(3/4 * frame_size + 1/4 * im_size - 1/2 * text_height)
    text_position = (text_left, text_top)

    # Draw text on image
    draw.text(text_position, text, font=font, fill=fill, align='center')
    return image


def easeInOutQuint(x):
    # From Matthew Tancik: 
    # https://github.com/tancik/Illusion-Diffusion/blob/main/IllusionDiffusion.ipynb
    if x < 0.5:
        return 4 * x**3
    else:
        return 1 - (-2 * x + 2)**3 / 2


def animate_two_view(
        im,
        view,
        prompt_1,
        prompt_2,
        save_video_path,
        hold_duration=120,
        text_fade_duration=10,
        transition_duration=60,
        im_size=256,
        frame_size=384,
        boomerang=True,
):
    '''
    Animate the transition between an image and the view of an image

    im (PIL Image):
        Image to animate

    view (view object):
        The view to transform the image by. Importantly, 
        should implement `make_frame`.

    prompt_1, prompt_2 (string):
        Prompt for the identity view and the transformed view, to be
        displayed under the images

    save_video_path (string):
        Path to the location to save the video

    hold_duration (int):
        Number of frames (@ 30 FPS) to pause the video on the
        complete image

    text_fade_duration (int):
        Number of frames (@ 30 FPS) for the text to fade in or out

    transition_duration (int):
        Number of frames (@ 30 FPS) to animate the transformations

    im_size (int):
        Size of the image to animate

    frame_size (int):
        Size of the final video

    boomerang (bool):
        If true, boomerang the clip by showing image, then transformed
        image, and finally the original image.
        If false, only show image then transformed image.
    '''

    # Make list of frames
    frames = []

    # Make frames for two views 
    frame_1 = view.make_frame(im, 0.0)
    frame_2 = view.make_frame(im, 1.0)

    # Display frame 1 with text
    frame_1_text = draw_text(frame_1, 
                             prompt_1, 
                             frame_size=frame_size, 
                             im_size=im_size)
    frames += [frame_1_text] * (hold_duration // 2)

    # Fade out text 1
    for t in np.linspace(0,1,text_fade_duration):
        c = int(t * 255)
        fill = (c,c,c)
        frame = draw_text(frame_1, 
                          prompt_1, 
                          fill=fill,
                          frame_size=frame_size, 
                          im_size=im_size)
        frames.append(frame)

    # Transition view 1 -> view 2
    print('Making frames...')
    for t in tqdm(np.linspace(0,1,transition_duration)):
        t_ease = easeInOutQuint(t)
        frames.append(view.make_frame(im, t_ease))

    # Fade in text 2
    for t in np.linspace(1,0,text_fade_duration):
        c = int(t * 255)
        fill = (c,c,c)
        frame = draw_text(frame_2,
                          prompt_2,
                          fill=fill,
                          frame_size=frame_size, 
                          im_size=im_size)
        frames.append(frame)

    # Display frame 2 with text
    frame_2_text = draw_text(frame_2, 
                             prompt_2, 
                             frame_size=frame_size, 
                             im_size=im_size)
    frames += [frame_2_text] * (hold_duration // 2)

    # "Boomerang" the clip, so we get back to view 1
    if boomerang:
        frames = frames + frames[::-1]

    # Move last bit of clip to front
    frames = frames[-hold_duration//2:] + frames[:-hold_duration//2]

    # Convert PIL images to numpy arrays
    image_array = [imageio.core.asarray(frame) for frame in frames]

    # Save as video
    print('Making video...')
    imageio.mimsave(save_video_path, image_array, fps=30)




####################################
### ANIMATION FOR MOTION HYBRIDS ###
####################################

def easedLinear(t):
    '''
    Ramps up into a linear increase
    '''
    if t < np.sqrt(1/3):
        return t**3
    else:
        return t - np.sqrt(1/3) + np.power(1/3, 3/2)

def animate_two_view_motion_blur(
        im,
        view,
        prompt_1,
        prompt_2,
        save_video_path,
        hold_duration=120,
        text_fade_duration=10,
        transition_duration=60,
        im_size=256,
        frame_size=384,
        boomerang=True,
        text_top=None,
):
    '''
    Animate the transition between an image and the view of an image

    im (PIL Image):
        Image to animate

    view (view object):
        The view to transform the image by. Importantly, 
        should implement `make_frame`.

    prompt_1, prompt_2 (string):
        Prompt for the identity view and the transformed view, to be
        displayed under the images

    save_video_path (string):
        Path to the location to save the video

    hold_duration (int):
        Number of frames (@ 30 FPS) to pause the video on the
        complete image

    text_fade_duration (int):
        Number of frames (@ 30 FPS) for the text to fade in or out

    transition_duration (int):
        Number of frames (@ 30 FPS) to animate the transformations

    im_size (int):
        Size of the image to animate

    frame_size (int):
        Size of the final video

    boomerang (bool):
        If true, boomerang the clip by showing image, then transformed
        image, and finally the original image.
        If false, only show image then transformed image.

    text_top (int):
        Adjust vertical location of text for second view. For hybrid images
    '''

    # Make list of frames
    frames = []

    # Make frames for two views 
    frame_1 = view.make_frame(im, 0.0)
    frame_2 = view.make_frame(im, 1.0)

    # Display frame 1 with text
    frame_1_text = draw_text(frame_1, 
                             prompt_1, 
                             frame_size=frame_size, 
                             im_size=im_size)
    frames += [frame_1_text] * (hold_duration // 2)

    # Fade out text 1
    for t in np.linspace(0,1,text_fade_duration):
        c = int(t * 255)
        fill = (c,c,c)
        frame = draw_text(frame_1, 
                          prompt_1, 
                          fill=fill,
                          frame_size=frame_size, 
                          im_size=im_size)
        frames.append(frame)

    # Transition view 1 -> view 2
    # 1. Make buffer of frames to blur together
    print('Making frames...')
    blur_buffer = []    
    for t in tqdm(np.linspace(0,2,transition_duration)):
        if t < 1.5:
            t_ease = easedLinear(t)
            blur_buffer.append(view.make_frame(im, t_ease))

    # 2. Make blurred frames
    n = int(transition_duration / 20.)  # T / vel
    blurred_frames = []
    for i in tqdm(range(0, len(blur_buffer) - n, 10)):
        if i <= n * 3:
            window = 1
        else:
            window = min(int((i - 3 * n) / 10.), n)

        to_blur = blur_buffer[i:i+window]
        to_blur = [TF.to_tensor(im) for im in to_blur]
        im_blurred = torch.mean(torch.stack(to_blur), dim=0)
        blurred_frames.append(TF.to_pil_image(im_blurred))
    frames.extend(blurred_frames)

    # Fade in text 2
    frame_2 = frames[-1]
    for t in np.linspace(1,0,text_fade_duration):
        c = int(t * 255)
        fill = (c,c,c)
        frame = draw_text(frame_2,
                          prompt_2,
                          fill=fill,
                          frame_size=frame_size, 
                          im_size=text_top if text_top is not None else im_size)
        frames.append(frame)

    # Display frame 2 with text
    frame_2_text = draw_text(frame_2, 
                             prompt_2, 
                             frame_size=frame_size, 
                             im_size=text_top if text_top is not None else im_size)
    frames += [frame_2_text] * (hold_duration // 2)

    # "Boomerang" the clip, so we get back to view 1
    if boomerang:
        frames = frames + frames[::-1]

    # Move last bit of clip to front
    frames = frames[-hold_duration//2:] + frames[:-hold_duration//2]

    # Convert PIL images to numpy arrays
    image_array = [imageio.core.asarray(frame) for frame in frames]

    # Save as video
    print('Making video...')
    imageio.mimsave(save_video_path, image_array, fps=30)