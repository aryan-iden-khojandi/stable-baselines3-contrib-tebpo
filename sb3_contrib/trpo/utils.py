import funcy as f
import numpy as np
import torch as th
import torch.nn.functional as F


def get_flat_grads(model, pred=f.constantly(True)):
    "Extracts the model's currently stored gradients into a flat vector."
    params = [param.grad.view(-1)
              for name, param in model.named_parameters()
              if pred(name)]
    # for name, param in model.named_parameters():
    #     if pred(name):
    #         import pdb; pdb.set_trace()
    flat_params = th.cat(params)
    return flat_params


def get_flat_params(model, pred=f.constantly(True)):
    "Extracts the model parameters into a flat vector."
    return th.cat([param.data.view(-1)
                   for name, param in model.named_parameters()
                   if pred(name)])


def is_actor(name):
    return "value" not in name


def set_flat_params(model, flat_params, pred=f.constantly(True)):
    prev_ind = 0
    for name, param in model.named_parameters():
        if pred(name):
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size]
                .view(param.size()))
            prev_ind += flat_size
    return model


def discounted_cumsum(x: th.Tensor, gamma: float):
    """
    x is expected to be one dimensional
    """
    x_pad = F.pad(x.view(1, 1, -1),
                 (0, len(x) - 1), "constant", 0.)
    weights = (gamma ** th.arange(len(x), dtype=x.dtype)).view(1, 1, -1)
    return th.nn.functional.conv1d(x_pad, weights).view(-1)
