import abc

import torch

STD_MIN = 1e-6


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self):
        """Construct an SDE."""
        super().__init__()

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        """
        sde: A function that returns the drift and diffusion coefficients of the SDE.

        return (drift $f(x,t)$, diffusion $g(t)$)
        """
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """
        Parameters to determine the marginal distribution of the SDE, $p_t(x)$.

        return mean, std
        """
        pass

    @abc.abstractmethod
    def diffusion_coeff(self, t):
        """
        diffusion_coeff: A function that returns the diffusion coefficient of the SDE.

        return $g(t)$
        """
        pass

    @abc.abstractmethod
    def marginal_prob_std(self, t):
        """
        Parameters to determine the marginal distribution of the SDE, $p_t(x)$.

        return std
        """
        pass


class VPSDE_linear(SDE):
    def __init__(self, beta_max=20.0, beta_min=0.1):
        """
        VP SDE

        SDE:
        $ \mathrm{d}x = -\frac{\beta(t)}{2} x \mathrm{d}t + \sqrt{\beta(t)} \mathrm{d}W_t $
        """
        super().__init__()

        self._beta_max = beta_max
        self._beta_min = beta_min

    @property
    def T(self):
        return 1.0

    def sde(self, x, t):
        """
        SDE of diffusion process

        drift = $-\frac{\beta(t)}{2} x$
        diffusion = $\sqrt{\beta(t)}$
        """
        shape = x.shape
        reshape = [-1] + [
            1,
        ] * (len(shape) - 1)
        t = t.reshape(reshape)

        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)

        return drift, diffusion

    def marginal_prob(self, x, t):
        """
        Parameters to determine the marginal distribution of the SDE, $p_t(x)$.
        """
        shape = x.shape
        reshape = [-1] + [
            1,
        ] * (len(shape) - 1)
        t = t.reshape(reshape)
        mean_log_coeff = -0.25 * t**2 * (self._beta_max - self._beta_min) - 0.5 * self._beta_min * t

        mean = torch.exp(mean_log_coeff) * x
        std = torch.sqrt(1 - torch.exp(2.0 * mean_log_coeff))
        return mean, std

    def diffusion_coeff(self, t):
        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min
        diffusion = torch.sqrt(beta_t)
        return diffusion

    def marginal_prob_std(self, t):
        discount = torch.exp(-0.5 * t**2 * (self._beta_max - self._beta_min) - self._beta_min * t)
        std = torch.sqrt(1 - discount)
        return std


class subVPSDE_exp(SDE):
    def __init__(self, sigma=25.0):
        """
        subVPSDE

        $beta(t) = sigma^t$
        """
        raise NotImplementedError
        super().__init__()

        self._sigma = sigma

    @property
    def T(self):
        return 1.0

    def sde(self, x, t):
        shape = x.shape
        reshape = [-1] + [
            1,
        ] * (len(shape) - 1)
        t = t.reshape(reshape)

        beta_t = self._sigma**t
        drift = -0.5 * beta_t * x
        discount = torch.exp(-2 * (beta_t - 1) / torch.log(self._sigma))
        diffusion = torch.sqrt(beta_t * (1.0 - discount))

        return drift, diffusion

    def marginal_prob(self, x, t):
        shape = x.shape
        reshape = [-1] + [
            1,
        ] * (len(shape) - 1)
        t = t.reshape(reshape)
        discount = torch.exp(-(self._sigma**t - 1) / torch.log(self._sigma))
        mean = discount * x
        std = torch.clamp(1 - discount, min=STD_MIN)
        return mean, std

    def diffusion_coeff(self, t):
        beta_t = self._sigma**t
        discount = torch.exp(-2 * (beta_t - 1) / torch.log(self._sigma))
        diffusion = torch.sqrt(beta_t * (1.0 - discount))
        return diffusion

    def marginal_prob_std(self, t):
        discount = torch.exp(-(self._sigma**t - 1) / torch.log(self._sigma))
        std = torch.clamp(1 - discount, min=STD_MIN)
        return std
