<p align="center">
 <img src="../assets/lumina-logo.png" width="40%"/>
 <br>
</p>

# Lumina-Pro Visual Anagrams

`Lumina-Pro Visual Anagrams` is an implementation of the paper [Visual Anagrams: Generating Multi-View Optical Illusions with Diffusion Models](https://dangeng.github.io/visual_anagrams/) based on `Lumina-Pro`.


## Installation

Please refer to the `Lumina-Pro` folder.


## Usage

To generate an illusion, replace `path_to_your_ckpt` in the file `run.sh` with your Lumina-Pro model path and run the following command:
```bash
bash run.sh
```
Here is a description of some useful arguments in the script:

- `--name`: Name for the illusion. Will save samples to `./results/{name}`.
- `--prompts`: A list of prompts for illusions
- `--style`: Optional style prompt to prepend to each of the prompts. For example, could be `"an oil painting of"`. Saves some writing.
- `--views`: A list of views to use. Must match the number of prompts. For a list of views see the `get_views` function in `visual_anagrams/views/__init__.py`. (Note: Only rotation and flip are supported so far.)
- `--num_samples`: Number of illusions to sample.