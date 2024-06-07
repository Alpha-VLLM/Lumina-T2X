import torch as th


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
        xt = t * x1 + (1 - t) * x0
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
