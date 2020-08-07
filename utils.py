import numpy as np
import torch
from torch._six import inf

# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
class RunningStdMean():
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = 1e-4

    def update(self, input_batch_b):
        mean_b = np.mean(input_batch_b, axis=0)
        var_b = np.var(input_batch_b, axis=0)
        count_b = input_batch_b.shape[0]
        self.var, self.mean = self.parallel_variance(self.mean, self.count,
                                                     self.var, mean_b, count_b,
                                                     var_b)
        self.count = count_b + self.count

    def update_from_mean_std(self, mean_b, var_b, count_b):
        self.var, self.mean = self.parallel_variance(self.mean, self.count,
                                                     self.var, mean_b, count_b,
                                                     var_b)
        self.count = count_b + self.count

    def parallel_variance(self, avg_a, count_a, var_a, avg_b, count_b, var_b):
        delta = avg_b - avg_a
        m_a = var_a * (count_a)
        m_b = var_b * (count_b)
        M2 = m_a + m_b + np.square(delta) * count_a * count_b / (
                    count_a + count_b)
        new_mean = avg_a + delta * count_b / (count_a + count_b)
        new_var = M2 / (count_a + count_b)
        return new_var, new_mean

# this part was not mentioned in the paper and is inspired from the
# original openai implementation:
# https://github.com/openai/random-network-distillation


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems



def global_grad_norm_(parameters, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm
