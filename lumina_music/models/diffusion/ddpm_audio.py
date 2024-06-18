from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
from tqdm import tqdm

try:
    from pytorch_lightning.utilities.distributed import rank_zero_only
except:
    from pytorch_lightning.utilities import rank_zero_only  # torch2

from omegaconf import ListConfig

from ..autoencoder1d import AutoencoderKL, IdentityFirstStage, VQModelInterface
from ..util import default, instantiate_from_config, isimage, log_txt_as_img, mean_flat
from .ddim import DDIMSampler
from .ddpm import DDPM, disabled_train
from .distributions.distributions import DiagonalGaussianDistribution, normal_kl
from .util import extract_into_tensor, noise_like

__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}


from torchdyn.core import NeuralODE


class LatentDiffusion_audio(DDPM):
    """main class"""

    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        num_timesteps_cond=None,
        mel_dim=80,
        mel_length=848,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        *args,
        **kwargs,
    ):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.mel_dim = mel_dim
        self.mel_length = mel_length
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(
        self,
    ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if (
            self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
        ):
            assert self.scale_factor == 1.0, "rather not use custom rescaling and std-rescaling simultaneously"
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()  # get latent
            del self.scale_factor
            self.register_buffer(
                "scale_factor", 1.0 / z.flatten().std()
            )  # 1/latent.std， get_first_stage_encoding returns self.scale_factor * latent
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != "__is_first_stage__"
            assert config != "__is_unconditional__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def _get_denoise_row_from_list(self, samples, desc="", force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(
                self.decode_first_stage(zd.to(self.device), force_not_quantize=force_no_decoder_quantization)
            )
        n_imgs_per_row = len(denoise_row)
        if len(denoise_row[0].shape) == 3:
            denoise_row = [x.unsqueeze(1) for x in denoise_row]
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size, null_label=None):
        if null_label is not None:
            xc = null_label
            if isinstance(xc, ListConfig):
                xc = list(xc)
            if isinstance(xc, dict) or isinstance(xc, list):
                c = self.get_learned_conditioning(xc)
            else:
                if hasattr(xc, "to"):
                    xc = xc.to(self.device)
                c = self.get_learned_conditioning(xc)
        else:
            if self.cond_stage_key in ["class_label", "cls"]:
                xc = self.cond_stage_model.get_unconditional_conditioning(batch_size, device=self.device)
                return self.get_learned_conditioning(xc)
            else:
                raise NotImplementedError("todo")
        if isinstance(c, list):  # in case the encoder gives us a list
            for i in range(len(c)):
                c[i] = repeat(c[i], "1 ... -> b ...", b=batch_size).to(self.device)
        else:
            c = repeat(c, "1 ... -> b ...", b=batch_size).to(self.device)
        return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(
            weighting,
            self.split_input_params["clip_min_weight"],
            self.split_input_params["clip_max_weight"],
        )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(
                L_weighting,
                self.split_input_params["clip_min_tie_weight"],
                self.split_input_params["clip_max_tie_weight"],
            )

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                dilation=1,
                padding=0,
                stride=(stride[0] * uf, stride[1] * uf),
            )
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                dilation=1,
                padding=0,
                stride=(stride[0] // df, stride[1] // df),
            )
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(
        self,
        batch,
        k,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        bs=None,
    ):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ["caption", "coordinates_bbox", "hybrid_feat"]:
                    xc = batch[cond_key]
                elif cond_key == "class_label":
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:  # False
                if isinstance(xc, dict) or isinstance(xc, list):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]
            # Testing #
            if cond_key == "masked_image":
                mask = super().get_input(batch, "mask")
                cc = torch.nn.functional.interpolate(mask, size=c.shape[-2:])  # [B, 1, 10, 106]
                c = torch.cat((c, cc), dim=1)  # [B, 5, 10, 106]
            # Testing #
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, "pos_x": pos_x, "pos_y": pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {"pos_x": pos_x, "pos_y": pos_y}
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1.0 / self.scale_factor * z

        if isinstance(self.first_stage_model, VQModelInterface):
            return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
        else:
            return self.first_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1.0 / self.scale_factor * z

        if isinstance(self.first_stage_model, VQModelInterface):
            return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
        else:
            return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss

    def test_step(self, batch, batch_idx):
        cond = batch[self.cond_stage_key]  #  * self.test_repeat
        cond = self.get_learned_conditioning(cond)  # c: string -> [B, T, Context_dim]
        batch_size = len(cond)
        enc_emb = self.sample(
            cond, batch_size, timesteps=self.num_timesteps
        )  # shape = [batch_size,self.channels,self.mel_dim,self.mel_length]
        xrec = self.decode_first_stage(enc_emb)
        return None

    def forward(self, x, c, *args, **kwargs):
        """
        video to audio:
        x (latent): [B, 256 (time), 20]  c (video feat): [B, 32 (time), 512]
        """
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()  # [B]
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)  # c: string -> [B, T, Context_dim]
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}
        else:
            if not isinstance(cond, list):
                cond = [cond]
            if self.model.conditioning_key == "concat":
                key = "c_concat"
            elif self.model.conditioning_key == "crossattn":
                key = "c_crossattn"
            else:
                key = "c_film"
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        mean_dims = list(range(1, len(target.shape)))
        loss_simple = self.get_loss(model_output, target, mean=False).mean(dim=mean_dims)
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        logvar_t = self.logvar[t.to(self.logvar.device)].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=mean_dims)
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_codebook_ids=return_codebook_ids,
            quantize_denoised=quantize_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
        )
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(
        self,
        cond,
        shape,
        verbose=True,
        callback=None,
        quantize_denoised=False,
        img_callback=None,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        batch_size=None,
        x_T=None,
        start_T=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: (
                        cond[key][:batch_size]
                        if not isinstance(cond[key], list)
                        else list(map(lambda x: x[:batch_size], cond[key]))
                    )
                    for key in cond
                }
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Progressive Generation", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
                return_x0=True,
                temperature=temperature[i],
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
            )
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
    ):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)  # num
            if self.shorten_cond_schedule:  # False
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(
                img, cond, ts, clip_denoised=self.clip_denoised, quantize_denoised=quantize_denoised  # False
            )  # False
            if mask is not None:  # False
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_size=16,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        shape=None,
        **kwargs,
    ):
        if shape is None:
            if self.channels > 0:
                shape = (batch_size, self.channels, self.mel_dim, self.mel_length)
            else:
                shape = (batch_size, self.mel_dim, self.mel_length)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: (
                        cond[key][:batch_size]
                        if not isinstance(cond[key], list)
                        else list(map(lambda x: x[:batch_size], cond[key]))
                    )
                    for key in cond
                }
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            timesteps=timesteps,
            quantize_denoised=quantize_denoised,
            mask=mask,
            x0=x0,
        )

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (
                (self.channels, self.mel_dim, self.mel_length) if self.channels > 0 else (self.mel_dim, self.mel_length)
            )
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs)

        return samples, intermediates

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=8,
        n_row=4,
        sample=True,
        ddim_steps=200,
        ddim_eta=1.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=False,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=True,
        **kwargs,
    ):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            bs=N,
        )  # z is latent,c is condition embedding, xc is condition(caption) list
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x if len(x.shape) == 4 else x.unsqueeze(1)
        log["reconstruction"] = xrec if len(xrec.shape) == 4 else xrec.unsqueeze(1)
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode") and self.cond_stage_key != "masked_image":
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key == "masked_image":
                log["mask"] = c[:, -1, :, :][:, None, :, :]
                xc = self.cond_stage_model.decode(c[:, : self.cond_stage_model.embed_dim, :, :])
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                pass
                # xc = log_txt_as_img((256, 256), batch["caption"])
                # log["conditioning"] = xc
            elif self.cond_stage_key == "class_label":
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log["conditioning"] = xc
            elif isimage(xc):
                log["conditioning"] = xc

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))
            if len(diffusion_row[0].shape) == 3:
                diffusion_row = [x.unsqueeze(1) for x in diffusion_row]
            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, "n b c h w -> b n c h w")
            diffusion_grid = rearrange(diffusion_grid, "b n c h w -> (b n) c h w")
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(
                    cond=c, batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta
                )
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples if len(x_samples.shape) == 4 else x_samples.unsqueeze(1)
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if (
                quantize_denoised
                and not isinstance(self.first_stage_model, AutoencoderKL)
                and not isinstance(self.first_stage_model, IdentityFirstStage)
            ):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(
                        cond=c, batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta, quantize_denoised=True
                    )
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples if len(x_samples.shape) == 4 else x_samples.unsqueeze(1)

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0.0
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples, _ = self.sample_log(
                        cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta, ddim_steps=ddim_steps, x0=z[:N], mask=mask
                    )
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask_inpainting"] = mask

                # outpaint
                mask = 1 - mask
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(
                        cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta, ddim_steps=ddim_steps, x0=z[:N], mask=mask
                    )
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples
                log["mask_outpainting"] = mask

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                shape = (
                    (self.channels, self.mel_dim, self.mel_length)
                    if self.channels > 0
                    else (self.mel_dim, self.mel_length)
                )
                img, progressives = self.progressive_denoising(c, shape=shape, batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print("Diffusion model optimizing logvar")
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert "target" in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [{"scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule), "interval": "step", "frequency": 1}]
            return [opt], scheduler
        return opt


class CFM(LatentDiffusion_audio):

    def __init__(self, **kwargs):

        super(CFM, self).__init__(**kwargs)
        self.sigma_min = 1e-4

    def p_losses(self, x_start, cond, t, noise=None):
        x1 = x_start
        x0 = default(noise, lambda: torch.randn_like(x_start))
        ut = x1 - (1 - self.sigma_min) * x0
        t_unsqueeze = t.unsqueeze(1).unsqueeze(1).float() / self.num_timesteps
        x_noisy = t_unsqueeze * x1 + (1.0 - (1 - self.sigma_min) * t_unsqueeze) * x0

        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = "train" if self.training else "val"
        target = ut

        mean_dims = list(range(1, len(target.shape)))
        loss_simple = self.get_loss(model_output, target, mean=False).mean(dim=mean_dims)
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        loss = loss_simple
        loss = self.l_simple_weight * loss.mean()
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    @torch.no_grad()
    def sample(self, cond, batch_size=16, timesteps=None, shape=None, x_latent=None, t_start=None, **kwargs):
        if shape is None:
            if self.channels > 0:
                shape = (batch_size, self.channels, self.mel_dim, self.mel_length)
            else:
                shape = (batch_size, self.mel_dim, self.mel_length)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: (
                        cond[key][:batch_size]
                        if not isinstance(cond[key], list)
                        else list(map(lambda x: x[:batch_size], cond[key]))
                    )
                    for key in cond
                }
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if kwargs.get("solver", None) is None:
            kwargs["solver"] = "euler"

        neural_ode = NeuralODE(
            self.ode_wrapper(cond), solver=kwargs["solver"], sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        t_span = torch.linspace(0, 1, 25 if timesteps is None else timesteps)
        if t_start is not None:
            t_span = t_span[t_start:]

        x0 = torch.randn(shape, device=self.device) if x_latent is None else x_latent
        eval_points, traj = neural_ode(x0, t_span)

        return traj[-1], traj

    def ode_wrapper(self, cond):
        return Wrapper(self, cond)

    @torch.no_grad()
    def sample_cfg(
        self,
        cond,
        unconditional_guidance_scale,
        unconditional_conditioning,
        batch_size=16,
        timesteps=None,
        shape=None,
        x_latent=None,
        t_start=None,
        **kwargs,
    ):
        if shape is None:
            if self.channels > 0:
                shape = (batch_size, self.channels, self.mel_dim, self.mel_length)
            else:
                shape = (batch_size, self.mel_dim, self.mel_length)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: (
                        cond[key][:batch_size]
                        if not isinstance(cond[key], list)
                        else list(map(lambda x: x[:batch_size], cond[key]))
                    )
                    for key in cond
                }
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if kwargs.get("solver", None) is None:
            kwargs["solver"] = "euler"

        neural_ode = NeuralODE(
            self.ode_wrapper_cfg(cond, unconditional_guidance_scale, unconditional_conditioning),
            solver=kwargs["solver"],
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )
        t_span = torch.linspace(0, 1, 25 if timesteps is None else timesteps)

        if t_start is not None:
            t_span = t_span[t_start:]

        x0 = torch.randn(shape, device=self.device) if x_latent is None else x_latent
        eval_points, traj = neural_ode(x0, t_span)

        return traj[-1], traj

    def ode_wrapper_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning)

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        # if use_original_steps:
        #     sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
        #     sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        # else:
        sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
        sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )


class Wrapper(nn.Module):
    def __init__(self, net, cond):
        super(Wrapper, self).__init__()
        self.net = net
        self.cond = cond

    def forward(self, t, x, args):
        t = torch.tensor([t * 1000] * x.shape[0], device=t.device).long()
        return self.net.apply_model(x, t, self.cond)


class Wrapper_cfg(nn.Module):

    def __init__(self, net, cond, unconditional_guidance_scale, unconditional_conditioning):
        super(Wrapper_cfg, self).__init__()
        self.net = net
        self.cond = cond
        self.unconditional_conditioning = unconditional_conditioning
        self.unconditional_guidance_scale = unconditional_guidance_scale

    def forward(self, t, x, args):
        x_in = torch.cat([x] * 2)
        t = torch.tensor([t * 1000] * x.shape[0], device=t.device).long()
        t_in = torch.cat([t] * 2)
        c_in = torch.cat(
            [self.unconditional_conditioning, self.cond]
        )  # c/uc shape [b,seq_len=77,dim=1024],c_in shape [b*2,seq_len,dim]
        e_t_uncond, e_t = self.net.apply_model(x_in, t_in, c_in).chunk(2)
        e_t = e_t_uncond + self.unconditional_guidance_scale * (e_t - e_t_uncond)
        return e_t
