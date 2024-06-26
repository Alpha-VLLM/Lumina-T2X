import torch as th
from torchdiffeq import odeint


def sample(x1):
    """Sampling x0 & t based on shape of x1 (if needed)
    Args:
      x1 - data point; [batch, *dim]
    """
    if isinstance(x1, (list, tuple)):
        x0 = [th.randn_like(img_start) for img_start in x1]
    else:
        x0 = th.randn_like(x1)

    t = th.rand((len(x1),))
    t = t.to(x1[0])
    return t, x0, x1


def training_losses(model, x1, model_kwargs=None):
    """Loss for training the score model
    Args:
    - model: backbone model; could be score, noise, or velocity
    - x1: datapoint
    - model_kwargs: additional arguments for the model
    """
    if model_kwargs == None:
        model_kwargs = {}

    B = len(x1)

    t, x0, x1 = sample(x1)
    if isinstance(x1, (list, tuple)):
        xt = [t[i] * x1[i] + (1 - t[i]) * x0[i] for i in range(B)]
        ut = [x1[i] - x0[i] for i in range(B)]
    else:
        dims = [1] * (len(x1.size()) - 1)
        t_ = t.view(t.size(0), *dims)
        xt = t_ * x1 + (1 - t_) * x0
        ut = x1 - x0

    model_output = model(xt, t, **model_kwargs)

    terms = {}

    if isinstance(x1, (list, tuple)):
        terms["loss"] = th.stack(
            [((ut[i] - model_output[i]) ** 2).mean() for i in range(B)],
            dim=0,
        )
    else:
        terms["loss"] = ((model_output - ut) ** 2).mean(dim=list(range(1, ut.ndim)))

    return terms


class ODE:
    """ODE solver class"""

    def __init__(
        self,
        num_steps,
        sampler_type="euler",
        time_shifting_factor=None,
        t0=0.0,
        t1=1.0,
        use_sd3=False,
        strength=1.0,
    ):
        if use_sd3:
            self.t = th.linspace(t1, t0, num_steps)
            if time_shifting_factor:
                self.t = (time_shifting_factor * self.t) / (1 + (time_shifting_factor - 1) * self.t)
        else:
            self.t = th.linspace(t0, t1, num_steps)
            if time_shifting_factor:
                self.t = self.t / (self.t + time_shifting_factor - time_shifting_factor * self.t)
        
        if strength != 1.0:
            self.t = self.t[int(num_steps * (1 - strength)):]
            
        self.use_sd3 = use_sd3
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        device = x[0].device if isinstance(x, tuple) else x.device

        if not self.use_sd3:

            def _fn(t, x):
                t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
                model_output = model(x, t, **model_kwargs)
                return model_output

        else:
            cfg_scale = model_kwargs["cfg_scale"]
            model_kwargs.pop("cfg_scale")

            def _fn(t, x):
                t = th.ones(x.size(0)).to(device) * t * 1000
                half_x = x[: len(x) // 2]
                x = th.cat([half_x, half_x], dim=0)
                model_output = model(hidden_states=x, timestep=t, **model_kwargs)[0]
                uncond, cond = model_output.chunk(2, dim=0)
                model_output = uncond + cfg_scale * (cond - uncond)
                model_output = th.cat([model_output, model_output], dim=0)
                return model_output

        t = self.t.to(device)
        samples = odeint(_fn, x, t, method=self.sampler_type)
        return samples
