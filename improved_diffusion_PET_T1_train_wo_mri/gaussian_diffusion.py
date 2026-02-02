"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = ModelMeanType.START_X #model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t) for both PET and MRI pathways,
        as well as predictions of the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                    as input.
        :param x: the [N x C x ...] tensor at time t (PET input).
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys for both PET and MRI:
                - 'mean': the model mean output.
                - 'variance': the model variance output.
                - 'log_variance': the log of 'variance'.
                - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        #print(f"t.shape: {t.shape}, B: {(B,)}")
        assert t.shape == (B,)
        
        # Run the model and get separate outputs for PET and MRI
        pet_output, mri_output = model(x, self._scale_timesteps(t), **model_kwargs)

        # Define a helper function to process a single pathway (PET or MRI)
        def process_pathway(output):
            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                assert output.shape == (B, C * 2, *x.shape[2:])
                output, var_values = th.split(output, C, dim=1)
                if self.model_var_type == ModelVarType.LEARNED:
                    log_variance = var_values
                    variance = th.exp(log_variance)
                else:
                    min_log = _extract_into_tensor(
                        self.posterior_log_variance_clipped, t, x.shape
                    )
                    max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                    frac = (var_values + 1) / 2
                    log_variance = frac * max_log + (1 - frac) * min_log
                    variance = th.exp(log_variance)
            else:
                variance, log_variance = {
                    ModelVarType.FIXED_LARGE: (
                        np.append(self.posterior_variance[1], self.betas[1:]),
                        np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                    ),
                    ModelVarType.FIXED_SMALL: (
                        self.posterior_variance,
                        self.posterior_log_variance_clipped,
                    ),
                }[self.model_var_type]
                variance = _extract_into_tensor(variance, t, x.shape)
                log_variance = _extract_into_tensor(log_variance, t, x.shape)

            def process_xstart(x):
                if denoised_fn is not None:
                    x = denoised_fn(x)
                if clip_denoised:
                    return x.clamp(-1, 1)
                return x

            if self.model_mean_type == ModelMeanType.PREVIOUS_X:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_xprev(x_t=x, t=t, xprev=output)
                )
                mean = output
            elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
                if self.model_mean_type == ModelMeanType.START_X:
                    pred_xstart = process_xstart(output)
                else:
                    pred_xstart = process_xstart(
                        self._predict_xstart_from_eps(x_t=x, t=t, eps=output)
                    )
                mean, _, _ = self.q_posterior_mean_variance(
                    x_start=pred_xstart, x_t=x, t=t
                )
            else:
                raise NotImplementedError(self.model_mean_type)

            assert mean.shape == log_variance.shape == pred_xstart.shape == x.shape
            return {"mean": mean, "variance": variance, "log_variance": log_variance, "pred_xstart": pred_xstart}

        # Process PET and MRI outputs separately
        pet_results = process_pathway(pet_output)
        mri_results = process_pathway(mri_output)

        return {"pet": pet_results, "mri": mri_results}


    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Sample x_{t-1} for both PET and MRI pathways from the model at the given timestep.

        :return: A dict containing the following keys for both pathways:
                - 'pet': {'sample', 'pred_xstart'} for the PET pathway.
                - 'mri': {'sample', 'pred_xstart'} for the MRI pathway.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        # Compute samples for both PET and MRI pathways
        pet_sample = out["pet"]["mean"] + nonzero_mask * th.exp(0.5 * out["pet"]["log_variance"]) * noise
        mri_sample = out["mri"]["mean"] + nonzero_mask * th.exp(0.5 * out["mri"]["log_variance"]) * noise

        return {
            "pet": {"sample": pet_sample, "pred_xstart": out["pet"]["pred_xstart"]},
            "mri": {"sample": mri_sample, "pred_xstart": out["mri"]["pred_xstart"]},
        }

    def p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
        ):
        """
        Generate samples from the model for both PET and MRI pathways.

        :return: A dict containing final samples for both pathways:
                - 'pet': final sample for the PET pathway.
                - 'mri': final sample for the MRI pathway.
        """
        final_pet = None
        final_mri = None

        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final_pet = sample["pet"]["sample"]
            final_mri = sample["mri"]["sample"]

        return {"pet": final_pet, "mri": final_mri}

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
        ):
        """
        Generate samples progressively from the model for both PET and MRI pathways.

        :return: A generator yielding intermediate samples for both pathways:
                - 'pet': intermediate sample for the PET pathway.
                - 'mri': intermediate sample for the MRI pathway.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm  # Lazy import for progress bar
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                #img = th.cat([out["pet"]["sample"], out["mri"]["sample"]], dim=0)  # Combine samples for progression
                img = out["pet"]["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} for both PET and MRI pathways using DDIM.

        :return: A dict containing results for both PET and MRI pathways.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        # Re-derive epsilon
        pet_eps = self._predict_eps_from_xstart(x, t, out["pet"]["pred_xstart"])
        mri_eps = self._predict_eps_from_xstart(x, t, out["mri"]["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)

        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # Compute samples for both pathways
        pet_noise = th.randn_like(x)
        mri_noise = th.randn_like(x)

        pet_mean_pred = (
            out["pet"]["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * pet_eps
        )
        mri_mean_pred = (
            out["mri"]["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * mri_eps
        )

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        pet_sample = pet_mean_pred + nonzero_mask * sigma * pet_noise
        mri_sample = mri_mean_pred + nonzero_mask * sigma * mri_noise

        return {
            "pet": {"sample": pet_sample, "pred_xstart": out["pet"]["pred_xstart"]},
            "mri": {"sample": mri_sample, "pred_xstart": out["mri"]["pred_xstart"]},
        }

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate final samples from the model for both PET and MRI pathways using DDIM.

        :return: A dict containing final samples for both PET and MRI pathways.
        """
        final_pet = None
        final_mri = None

        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final_pet = sample["pet"]["sample"]
            final_mri = sample["mri"]["sample"]

        return {"pet": final_pet, "mri": final_mri}

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample for both PET and MRI pathways and yield intermediate samples.

        :return: A generator yielding intermediate samples for both PET and MRI pathways.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = th.cat([out["pet"]["sample"], out["mri"]["sample"]], dim=0)  # Combine samples for progression


    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound for both PET and MRI pathways.

        :return: A dict with separate keys for PET and MRI pathways:
                - 'pet': {'output': tensor, 'pred_xstart': tensor}.
                - 'mri': {'output': tensor, 'pred_xstart': tensor}.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        def compute_vb_terms(pathway):
            kl = normal_kl(
                true_mean, true_log_variance_clipped, out[pathway]["mean"], out[pathway]["log_variance"]
            )
            kl = mean_flat(kl) / np.log(2.0)

            decoder_nll = -discretized_gaussian_log_likelihood(
                x_start, means=out[pathway]["mean"], log_scales=0.5 * out[pathway]["log_variance"]
            )
            assert decoder_nll.shape == x_start.shape
            decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

            output = th.where((t == 0), decoder_nll, kl)
            return {"output": output, "pred_xstart": out[pathway]["pred_xstart"]}

        return {"pet": compute_vb_terms("pet"), "mri": compute_vb_terms("mri")}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for both PET and MRI pathways.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: A dict containing losses for both pathways:
                - 'pet_loss': Total loss for the PET pathway.
                - 'mri_loss': Total loss for the MRI pathway.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {"pet_loss": None, "mri_loss": None, "pet_vb": None, "mri_vb": None}

        # KL or Rescaled KL Loss
        if self.loss_type in [LossType.KL, LossType.RESCALED_KL]:
            vb_terms = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            terms["pet_loss"] = vb_terms["pet"]["output"]
            terms["mri_loss"] = vb_terms["mri"]["output"]

            if self.loss_type == LossType.RESCALED_KL:
                terms["pet_loss"] *= self.num_timesteps
                terms["mri_loss"] *= self.num_timesteps

        # MSE or Rescaled MSE Loss
        elif self.loss_type in [LossType.MSE, LossType.RESCALED_MSE]:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            def compute_mse_loss(pathway):
                target = {
                    ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                        x_start=x_start, x_t=x_t, t=t
                    )[0],
                    ModelMeanType.START_X: x_start,
                    ModelMeanType.EPSILON: noise,
                }[self.model_mean_type]
                if pathway == "pet":
                    idx = 0
                elif pathway == "mri":
                    idx = 1
                return mean_flat((target - model_output[idx]) ** 2)

            terms["pet_mse"] = compute_mse_loss("pet")
            terms["mri_mse"] = compute_mse_loss("mri")

            # Variational Bound (VB) Terms
            vb_terms = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            terms["pet_vb"] = vb_terms["pet"]["output"]
            terms["mri_vb"] = vb_terms["mri"]["output"]

            # Combine MSE and VB terms
            terms["pet_loss"] = terms["pet_mse"]
            terms["mri_loss"] = terms["mri_mse"]

            if self.loss_type == LossType.RESCALED_MSE:
                terms["pet_loss"] += terms["pet_vb"] * (self.num_timesteps / 1000.0)
                terms["mri_loss"] += terms["mri_vb"] * (self.num_timesteps / 1000.0)

        else:
            raise NotImplementedError(self.loss_type)

        return terms


    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound for both PET and MRI pathways.

        :param model: the BiTaskUNetModel to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys for both PET and MRI pathways:
                - total_bpd: the total variational lower-bound, per batch element.
                - prior_bpd: the prior term in the lower-bound.
                - vb: an [N x T] tensor of terms in the lower-bound.
                - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        pet_vb, mri_vb = [], []
        pet_xstart_mse, mri_xstart_mse = [], []
        pet_mse, mri_mse = [], []

        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)

            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )

            # PET pathway
            pet_vb.append(out["pet"]["output"])
            pet_xstart_mse.append(mean_flat((out["pet"]["pred_xstart"] - x_start) ** 2))
            pet_eps = self._predict_eps_from_xstart(x_t, t_batch, out["pet"]["pred_xstart"])
            pet_mse.append(mean_flat((pet_eps - noise) ** 2))

            # MRI pathway
            mri_vb.append(out["mri"]["output"])
            mri_xstart_mse.append(mean_flat((out["mri"]["pred_xstart"] - x_start) ** 2))
            mri_eps = self._predict_eps_from_xstart(x_t, t_batch, out["mri"]["pred_xstart"])
            mri_mse.append(mean_flat((mri_eps - noise) ** 2))

        pet_vb = th.stack(pet_vb, dim=1)
        mri_vb = th.stack(mri_vb, dim=1)
        pet_xstart_mse = th.stack(pet_xstart_mse, dim=1)
        mri_xstart_mse = th.stack(mri_xstart_mse, dim=1)
        pet_mse = th.stack(pet_mse, dim=1)
        mri_mse = th.stack(mri_mse, dim=1)

        # Prior KL term for PET and MRI pathways
        pet_prior_bpd = self._prior_bpd(x_start)
        mri_prior_bpd = self._prior_bpd(x_start)

        # Total BPD for PET and MRI
        pet_total_bpd = pet_vb.sum(dim=1) + pet_prior_bpd
        mri_total_bpd = mri_vb.sum(dim=1) + mri_prior_bpd

        return {
            "pet": {
                "total_bpd": pet_total_bpd,
                "prior_bpd": pet_prior_bpd,
                "vb": pet_vb,
                "xstart_mse": pet_xstart_mse,
                "mse": pet_mse,
            },
            "mri": {
                "total_bpd": mri_total_bpd,
                "prior_bpd": mri_prior_bpd,
                "vb": mri_vb,
                "xstart_mse": mri_xstart_mse,
                "mse": mri_mse,
            },
        }

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
