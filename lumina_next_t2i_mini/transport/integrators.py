import torch as th
from torchdiffeq import odeint


class ODE:
    """ODE solver class"""

    def __init__(
        self,
        num_steps,
        sampler_type="euler",  # support fixed_grid solvers midpoint / RK4
        time_shifting_factor=None,
        t0=0.0,
        t1=1.0,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.t = th.linspace(t0, t1, num_steps)
        if time_shifting_factor:
            self.t = self.t / (self.t + time_shifting_factor - time_shifting_factor * self.t)
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output = model(x, t, **model_kwargs)
            return model_output

        t = self.t.to(device)
        samples = odeint(_fn, x, t, method=self.sampler_type)
        return samples
