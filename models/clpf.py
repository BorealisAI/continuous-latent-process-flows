# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
from torch import distributions, nn
import torch.nn.functional as F
import torchsde

from models.ctfp_tools import build_augmented_model_tabular
from models.train_misc import set_cnf_options, create_regularization_fns


def _stable_sign(b):
    b = b.sign()
    b[b == 0] = 1
    return b


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(
        b.abs().detach() > epsilon,
        b,
        torch.full_like(b, fill_value=epsilon) * _stable_sign(b),
    )

    return a / b


def network_factory(network_dims, non_linearity="softplus"):
    module_list = []
    network_dims = list(network_dims)
    if non_linearity.lower() == "softplus":
        non_linearity = nn.Softplus
    elif non_linearity.lower() == "sigmoid":
        non_linearity = nn.Sigmoid
    elif non_linearity.lower() == "relu":
        non_linearity = nn.ReLu
    for i in range(len(network_dims) - 2):
        module_list.append(nn.Linear(network_dims[i], network_dims[i + 1]))
        module_list.append(non_linearity())
    module_list.append(nn.Linear(network_dims[-2], network_dims[-1]))
    return nn.Sequential(*module_list)


def time_embedding(t):
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    return torch.cat([t, sin_t, cos_t], 1)


def sample_normal(mean, logvar, stdv=None):
    if stdv is None:
        stdv = torch.exp(0.5 * logvar)
    return torch.randn(mean.shape).to(stdv) * stdv + mean


class ConstantVariance(nn.Module):
    def __init__(
        self, input_dim, model_dims, time_embedding_dim=3, activation="softplus"
    ):
        super(ConstantVariance, self).__init__()
        if activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError
        self.variance = nn.Parameter(torch.zeros(1, input_dim))

    def forward(self, t, y):
        batch_size = y.shape[0]
        return self.activation(self.variance).repeat(batch_size, 1)


class DiagonalVariance(nn.Module):
    def __init__(
        self, input_dim, model_dims, time_embedding_dim=3, activation="softplus"
    ):
        super(DiagonalVariance, self).__init__()
        self.context_tensor = None
        self.variance_networks = nn.ModuleList(
            [
                network_factory([1 + time_embedding_dim] + model_dims)
                for i in range(input_dim)
            ]
        )
        if activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, t, y):
        ## t: time index of size batch_size x time_embedding_dim,
        ## y: size batch_size x latent or data_size
        y_split = torch.split(y, split_size_or_sections=1, dim=1)
        result = torch.cat(
            [
                network(torch.cat([y_, t], 1))
                for (network, y_) in zip(self.variance_networks, y_split)
            ],
            1,
        )
        return self.activation(result)


class GeneralVariance(nn.Module):
    def __init__(
        self, input_dim, model_dims, time_embedding_dim=3, activation="softplus"
    ):
        super(GeneralVariance, self).__init__()
        self.context_tensor = None
        self.variance_networks = network_factory(
            [input_dim + time_embedding_dim] + model_dims
        )

        if activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, t, y):
        ## t: time index of size batch_size x time_embedding_dim,
        ## y: size batch_size x latent or data_size
        result = self.variance_networks(torch.cat([y, t], 1))
        return self.activation(result)


class latentSDE(torchsde.SDEIto):
    def __init__(
        self, latent_dim, context_dim, drift_network_dims, variance_network_dims, args
    ):
        super(latentSDE, self).__init__(noise_type=args.noise_type)
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.context_tensor_lst = []
        self.noise_type_original = args.noise_type
        self.prior_drift = network_factory(
            [latent_dim + args.time_embedding_dim] + drift_network_dims
        )
        self.posterior_drift = network_factory(
            [latent_dim + args.time_embedding_dim + context_dim] + drift_network_dims
        )
        self.variance_presoftplus = nn.Parameter(torch.rand(1, self.latent_dim))

        self.method = args.method
        self.dt = args.dt
        self.dt_min = args.dt_min
        self.dt_test = args.dt_test
        self.dt_min_test = args.dt_min_test
        self.adaptive = args.adaptive
        if args.noise_type == "general":
            self.diagonal_variance = GeneralVariance(
                latent_dim, variance_network_dims, activation=args.variance_act
            )
        elif args.noise_type == "additive":
            self.diagonal_variance = ConstantVariance(
                latent_dim, variance_network_dims, activation=args.variance_act
            )
        elif args.noise_type == "diagonal":
            self.diagonal_variance = DiagonalVariance(
                latent_dim, variance_network_dims, activation=args.variance_act
            )
        else:
            raise NotImplementedError

        self.rtol = args.rtol
        self.atol = args.atol
        ## Get the index for context
        self.context_index = 0
        self.sdeint_noadjoint = torchsde.sdeint

        self.register_buffer("time_zero", torch.tensor([0]))

    def update_context(self, context_tensor, context_time):
        self.context_tensor_lst.append((context_tensor, context_time))

    def clear_context(self):
        self.context_tensor_lst = []
        self.context_index = 0

    def f(self, t, y):  # Approximate posterior drift.
        context_index = -1
        t = t.unsqueeze(0).unsqueeze(0)
        t = time_embedding(t)
        t = t.repeat(y.shape[0], 1)
        return self.posterior_drift(
            torch.cat([t, self.context_tensor_lst[context_index][0], y], 1)
        )

    def g(self, t, y):  # Shared diffusion.
        t = t.unsqueeze(0).unsqueeze(0)
        t = time_embedding(t)
        t = t.repeat(y.shape[0], 1)
        return self.diagonal_variance(t, y)

    def g_diag(self, t, y):
        # Return G as a diagonal matrix
        batch_size = y.shape[0]
        return torch.diag_embed(
            F.softplus(self.variance_presoftplus).repeat(batch_size, 1)
        )

    def h(self, t, y):  # Prior drift.
        t = t.unsqueeze(0).unsqueeze(0)
        t = time_embedding(t)
        t = t.repeat(y.shape[0], 1)
        return self.prior_drift(torch.cat([t, y], 1))

    def h_diag(self, t, y):
        return torch.diag_embed(self.h(t, y))

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, :-1]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = 0.5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def h_aug(self, t, y):
        y = y[:, :-1]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = 0.5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([h, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, :-1]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        g_logqp = _stable_division(f - h, g)
        g = torch.diag_embed(g)
        ## size of g is batch_size x latent x latent
        return torch.cat([g, g_logqp.unsqueeze(1)], dim=1)

    def g_prior(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        g = self.g(t, y)
        return torch.diag_embed(g)

    def forward(self, y_t0, t1, context_tensor_time=None, first_time_forward=False):
        if context_tensor_time is not None:
            self.update_context(*context_tensor_time)
        batch_size = y_t0.shape[0]
        bm = torchsde.BrownianInterval(
            t0=context_tensor_time[1], t1=t1, size=y_t0.shape, device=y_t0.device
        )
        dt = self.dt
        aug_adaptive = self.adaptive
        dt_min = self.dt_min

        ## Running the augmented SDE using euler maruyama scheme to get the logpq
        ## Always set noit type to general and use euler-maruyama method to run
        ## stochastic integral.

        self.noise_type = "general"
        aug_y0 = torch.cat([y_t0, torch.zeros(batch_size, 1).to(y_t0)], dim=1)
        aug_ys = self.sdeint_noadjoint(
            sde=self,
            y0=aug_y0,
            ts=torch.cat([context_tensor_time[1].unsqueeze(0), t1]),
            method="euler",
            dt=dt,
            adaptive=aug_adaptive,
            rtol=self.rtol,
            atol=self.atol,
            dt_min=dt_min,
            names={"drift": "f_aug", "diffusion": "g_aug"},
        )
        self.noise_type = self.noise_type_original
        return aug_ys[1]

    def sample_from_posterior(self, y_t0, ts, context_tensor_time=None):
        if context_tensor_time is not None:
            self.update_context(*context_tensor_time)
        self.noise_type = "general"
        batch_size = y_t0.shape[0]
        dt = self.dt
        dt_min = self.dt_min

        aug_y0 = torch.cat([y_t0, torch.zeros(batch_size, 1).to(y_t0)], dim=1)
        aug_ys = self.sdeint_noadjoint(
            sde=self,
            y0=aug_y0,
            ts=torch.cat([context_tensor_time[1].unsqueeze(0), ts]),
            method="euler",
            dt=dt,
            adaptive=True,
            rtol=self.rtol,
            atol=self.atol,
            dt_min=dt_min,
            names={"drift": "f_aug", "diffusion": "g_aug"},
        )
        self.noise_type = self.noise_type_original

        return aug_ys[1:]

    def sample_from_prior(self, y_t0, ts):
        dt = self.dt
        dt_min = self.dt_min
        batch_size = y_t0.shape[0]
        bm = torchsde.BrownianInterval(
            t0=torch.zeros_like(ts[-1]), t1=ts[-1], size=y_t0.shape, device=y_t0.device
        )
        self.noise_type = "general"
        ys = self.sdeint_noadjoint(
            sde=self,
            y0=y_t0,
            ts=ts,
            method="euler",
            dt=dt,
            adaptive=True,
            rtol=self.rtol,
            atol=self.atol,
            dt_min=dt_min,
            names={"drift": "h", "diffusion": "g_prior"},
        )
        self.noise_type = self.noise_type_original
        return ys


class ANODEConfig(object):
    def __init__(
        self,
        dims,
        aug_hidden_dims,
        effective_shape,
        aug_dim,
        aug_mapping,
        num_blocks=1,
        layer_type="ignore",
        nonlinearity="softplus",
        divergence_fn="approximate",
        residual=False,
        rademacher=True,
        time_length=1.0,
        train_T=False,
        solver="dopri5",
        rtol=1e-5,
        atol=1e-5,
        batch_norm=False,
        bn_lag=0.0,
        step_size=None,
        test_solver=None,
        test_rtol=None,
        test_atol=None,
        l1int=None,
        l2int=None,
        dl2int=None,
        JFrobint=None,
        JdiagFrobint=None,
        JoffdiagFrobint=None,
        time_penalty=None,
    ):
        self.dims = dims
        self.aug_hidden_dims = aug_hidden_dims
        self.effective_shape = effective_shape
        self.layer_type = layer_type
        self.nonlinearity = nonlinearity
        self.aug_dim = aug_dim
        self.aug_mapping = aug_mapping
        self.divergence_fn = divergence_fn
        self.residual = residual
        self.rademacher = rademacher
        self.time_length = time_length
        self.train_T = train_T
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.num_blocks = num_blocks
        self.batch_norm = batch_norm
        self.bn_lag = bn_lag
        self.step_size = step_size
        self.test_solver = test_solver
        self.test_atol = test_atol
        self.test_rtol = test_rtol
        self.l1int = l1int
        self.l2int = l2int
        self.dl2int = dl2int
        self.JFrobint = JFrobint
        self.JdiagFrobint = JdiagFrobint
        self.JoffdiagFrobint = JoffdiagFrobint
        self.time_penalty = time_penalty


class iResNetConfig(object):
    def __init__(self, args):
        for attr in vars(args):
            if attr.startswith("ires_"):
                value = getattr(args, attr)
                attr = attr[5:]
                setattr(self, attr, value)
        self.dims = [int(i) for i in self.dims.split(",")]
        if self.aug_block_dims is not None:
            self.aug_block_dims = [int(i) for i in self.aug_block_dims.split(",")]
        self.aug_proj_dims = [int(i) for i in self.aug_proj_dims.split(",")]


class IndependentDecoderConfig(object):
    def __init__(self, args):
        self.decoder_dims = [int(i) for i in args.indecoder_dims.split(",")]


class IndependentDecoder(nn.Module):
    def __init__(self, config, effective_size, aug_size):
        super(IndependentDecoder, self).__init__()
        decoder_dims = [aug_size] + config.decoder_dims + [effective_size * 2]
        self.decoder = network_factory(decoder_dims)
        self.effective_size = effective_size

    def forward(self, x_cat, y_cat, t_cat):
        mean_logvar = self.decoder(y_cat)
        mean, logvar = torch.split(mean_logvar, self.effective_size, 1)
        stdv = torch.exp(logvar * 0.5)
        output_distri = torch.distributions.Normal(mean, stdv)
        ll = output_distri.log_prob(x_cat)
        return ll, torch.zeros_like(ll).sum(-1), None

    def density_calculation(self, z_current, t_current, z_prev, t_prev):
        return z_current.sum(-1)

    def sample_from_base_process(
        self,
        t_current,
        t_prev=None,
        z_prev=None,
        t_future=None,
        z_future=None,
        batch_size=None,
    ):
        raise NotImplementedError

    def base_to_observe(self, t_current, y_current, z_current):
        if batch_size is None:
            batch_size = 1
        data_shape = (batch_size, self.effective_size)


class CTFPDecoder(nn.Module):
    def __init__(
        self,
        config,
        effective_size,
        aug_size,
        flow_type="anode",
        base_process_type="ou",
        exact_training_ou_std=False,
    ):
        super(CTFPDecoder, self).__init__()

        self.base_process_type = base_process_type
        self.effective_size = effective_size
        self.aug_size = aug_size
        self.flow_type = flow_type
        self.exact_training_ou_std = exact_training_ou_std

        if flow_type == "anode":
            regularization_fns, regularization_coeffs = create_regularization_fns(
                config
            )
            self.indexd_flow = build_augmented_model_tabular(
                config,
                self.effective_size + self.aug_size,
                regularization_fns=regularization_fns,
            )
            self.regularization_coeffs = regularization_coeffs
            set_cnf_options(config, self.indexd_flow)
        elif flow_type == "iresnet":
            import ires_lib.layers.base as base_layers
            import ires_lib.layers as layers

            ACTIVATION_FNS = {
                "relu": torch.nn.ReLU,
                "tanh": torch.nn.Tanh,
                "elu": torch.nn.ELU,
                "selu": torch.nn.SELU,
                "fullsort": base_layers.FullSort,
                "maxmin": base_layers.MaxMin,
                "swish": base_layers.Swish,
                "lcube": base_layers.LipschitzCube,
            }

            def parse_vnorms():
                ps = []
                for p in config.vnorms:
                    if p == "f":
                        ps.append(float("inf"))
                    else:
                        ps.append(float(p))
                return ps[:-1], ps[1:]

            def build_nnet(dims, activation_fn=ACTIVATION_FNS[config.act]):
                nnet = []
                domains, codomains = parse_vnorms()
                for i, (in_dim, out_dim, domain, codomain) in enumerate(
                    zip(dims[:-1], dims[1:], domains, codomains)
                ):
                    nnet.append(activation_fn())
                    nnet.append(
                        base_layers.get_linear(
                            in_dim,
                            out_dim,
                            coeff=config.coeff,
                            n_iterations=config.n_lipschitz_iters,
                            atol=config.atol,
                            rtol=config.rtol,
                            domain=domain,
                            codomain=codomain,
                            zero_init=(out_dim == 2),
                            update_during_training=config.update_during_training,
                        )
                    )
                return torch.nn.Sequential(*nnet)

            dims = [self.effective_size] + config.dims + [self.effective_size]
            blocks = []
            if config.actnorm:
                blocks.append(layers.ActNorm1d(self.effective_size))
            for _ in range(config.num_blocks):
                blocks.append(
                    layers.iResBlock(
                        build_nnet(dims),
                        n_dist=config.n_dist,
                        n_power_series=config.n_power_series,
                        exact_trace=config.exact_trace,
                        brute_force=config.brute_force,
                        n_samples=config.n_samples,
                        neumann_grad=config.neumann_grad,
                        grad_in_forward=config.grad_in_forward,
                    )
                )
                if config.actnorm:
                    blocks.append(layers.ActNorm1d(self.effective_size))
                if config.batchnorm:
                    blocks.append(layers.MovingBatchNorm1d(self.effective_size))

            aug_blocks = None
            if config.aug_block_dims is not None:
                aug_blocks = [
                    network_factory([self.aug_size] + config.aug_block_dims)
                ] + [
                    network_factory(config.aug_block_dims[-1:] + config.aug_block_dims)
                    for _ in range(config.num_blocks)
                ]
                projection_blocks = [
                    network_factory(
                        [config.aug_block_dims[-1]]
                        + config.aug_proj_dims
                        + [self.effective_size * 2]
                    )
                    for _ in range(config.num_blocks + 1)
                ]
            else:
                projection_blocks = [
                    network_factory(
                        [self.aug_size]
                        + config.aug_proj_dims
                        + [self.effective_size * 2]
                    )
                    for _ in range(config.num_blocks + 1)
                ]
            self.indexd_flow = layers.AffineAugSequentialFlow(
                blocks, projection_blocks, self.effective_size, aug_blocks
            )

        else:
            raise NotImplementedError
        if base_process_type == "ou":
            ## Marginal of OU process under Ito's interpretation
            ## latent_dim number of independent OU processes
            self.log_theta = nn.Parameter(torch.zeros(1, effective_size))
            self.mu = nn.Parameter(torch.zeros(1, effective_size))
            self.pre_softplus_sigma = nn.Parameter(torch.zeros(1, effective_size))
        elif base_process_type == "wiener":
            pass
        else:
            raise NotImplementedError

    def forward(self, x, y, t):
        flow_input = torch.cat([x, y, t], 1)
        batch_size = flow_input.shape[0]
        base_values, flow_logdet = self.indexd_flow(
            flow_input, torch.zeros(batch_size, 1).to(flow_input)
        )
        error = None
        if not self.training and self.flow_type == "anode":
            inverted_input, _ = self.indexd_flow(
                base_values, torch.zeros(batch_size, 1).to(flow_input), reverse=True
            )
            error = torch.abs(inverted_input - flow_input)
            error = error[:, : self.effective_size]

        base_values = base_values[:, : self.effective_size]
        return base_values, flow_logdet, error

    def helper_std_function(self, param):
        ## Helper function for computing torch.sqrt(1-torch.exp(0-2*param))
        result = torch.exp(
            0.5
            * torch.log(torch.exp(2 * param + 1) - torch.exp(torch.ones_like(param)))
            - 0.5
            - param
        )
        result = torch.maximum(result, torch.ones_like(result) * 0.0002)
        return result

    def density_calculation(self, z_current, t_current, z_prev=None, t_prev=None):
        """
        t_current, t_prev: a single time_step number
        y_curent: latent tensor of batch_size x latent_dim
        x_current: observable tensor of batch_size x input_dim
        z_prev: previous step base variable
        """
        if self.base_process_type == "ou":
            self.sigma = nn.functional.softplus(self.pre_softplus_sigma)
            self.theta = torch.exp(self.log_theta)
            self.theta_sqrt = torch.exp(0.5 * self.log_theta)

        batch_size = z_current.shape[0]
        if z_prev is None:
            ## The first prediction to make
            if self.base_process_type == "wiener":
                mean = torch.zeros_like(z_current)
                std = torch.ones_like(z_current) * torch.sqrt(t_current)
            elif self.base_process_type == "ou":
                mean = torch.ones_like(z_current) * self.mu

                std = (
                    torch.ones_like(z_current)
                    * (self.sigma)
                    / torch.sqrt(2 * self.theta)
                )
        else:
            time_diff = t_current - t_prev
            if self.base_process_type == "wiener":
                mean = torch.ones_like(z_current) * z_prev
                std = torch.ones_like(mean) * torch.sqrt(time_diff)
            elif self.base_process_type == "ou":
                mean = self.mu * (
                    1 - torch.exp(0 - self.theta * time_diff)
                ) + torch.ones_like(z_current) * z_prev * torch.exp(
                    0 - self.theta * time_diff
                )
                if not (self.training and self.exact_training_ou_std):
                    # always use approximation of SDE during evaluation to avoid NaN
                    std = (
                        torch.ones_like(z_current)
                        * (self.sigma)
                        / torch.sqrt(2 * self.theta)
                        * self.helper_std_function(self.theta * time_diff)
                    )
                else:

                    std = (
                        torch.ones_like(z_current)
                        * (self.sigma)
                        / torch.sqrt(2 * self.theta)
                        * torch.sqrt(1 - torch.exp(0 - 2 * self.theta * time_diff))
                    )

        normal_distr = torch.distributions.Normal(mean, std)
        ll = normal_distr.log_prob(z_current).sum(-1)
        return ll

    def base_to_observe(self, t_current, y_current, z_current):
        batch_size = z_current.shape[0]
        t_to_cat = torch.ones(batch_size, 1).to(y_current) * t_current
        flow_input = torch.cat([z_current, y_current, t_to_cat], 1)
        observe_values, _ = self.indexd_flow(
            flow_input, torch.zeros(batch_size, 1).to(flow_input), reverse=True
        )
        observe_values = observe_values[:, : self.effective_size]
        return observe_values

    def sample_from_base_process(
        self,
        t_current,
        t_prev=None,
        z_prev=None,
        t_future=None,
        z_future=None,
        batch_size=None,
    ):
        """
        return samples at a given time t_current
        If t_prev is not providede, sample from prior no matter t_future is provided or not.
        If t_prev and t_future are provided, do interpolation.
        if t_prev is providede and t_future is not, do extrapolation.
        t_current: a single time point to sample from
        t_prev: a single time point of the previous observation
        t_future: a single time point of the future observation
        z_prev: observations at t_prev of shape batch_size x data_shape
        z_future: observations at t_future of the same hsape as z_prev
        batch_size: number of samples only used when sample from prior.
        When sample from posterior, the number of samples is the same as batch_size of z_prev and z_future.
        """
        if t_prev is None:
            """
            Only implemented for OU process. Sample from the stationary distribution.
            """
            if batch_size is None:
                batch_size = 1
            data_shape = (batch_size, self.effective_size)
            if self.base_process_type == "ou":
                self.sigma = nn.functional.softplus(self.pre_softplus_sigma)
                self.theta = torch.exp(self.log_theta)
                self.theta_sqrt = torch.exp(0.5 * self.log_theta)

                mean = torch.ones(data_shape).to(self.mu) * self.mu

                std = (
                    torch.ones(data_shape).to(self.sigma)
                    * self.sigma
                    / torch.sqrt(2 * self.theta)
                )
            else:
                raise NotImplementedError
            stationary_distribution = torch.distributions.Normal(mean, std)
            return stationary_distribution.sample()
        time_diff = t_current - t_prev

        if self.base_process_type == "wiener":
            raise NotImplementedError
        elif self.base_process_type == "ou":
            self.sigma = nn.functional.softplus(self.pre_softplus_sigma)
            self.theta = torch.exp(self.log_theta)
            self.theta_sqrt = torch.exp(0.5 * self.log_theta)
            if t_future is None:
                # predictiom
                mean = self.mu * (
                    1 - torch.exp(0 - self.theta * time_diff)
                ) + torch.ones_like(z_prev) * z_prev * torch.exp(
                    0 - self.theta * time_diff
                )
                std = (
                    torch.ones_like(z_prev)
                    * (self.sigma)
                    / torch.sqrt(2 * self.theta)
                    * torch.sqrt(1 - torch.exp(0 - 2 * self.theta * time_diff))
                )
            else:
                # interpolation
                time_diff_2 = t_future - t_prev

                mean_1 = self.mu * (
                    1 - torch.exp(0 - self.theta * time_diff)
                ) + torch.ones_like(z_prev) * z_prev * torch.exp(
                    0 - self.theta * time_diff
                )
                mean_2 = self.mu * (
                    1 - torch.exp(0 - self.theta * time_diff_2)
                ) + torch.ones_like(z_prev) * z_prev * torch.exp(
                    0 - self.theta * time_diff_2
                )

                std11 = (
                    torch.ones_like(z_prev)
                    * (self.sigma)
                    / torch.sqrt(2 * self.theta)
                    * torch.sqrt(1 - torch.exp(0 - 2 * self.theta * time_diff))
                )
                sigma11 = std11 ** 2

                std22 = (
                    torch.ones_like(z_prev)
                    * (self.sigma)
                    / torch.sqrt(2 * self.theta)
                    * torch.sqrt(1 - torch.exp(0 - 2 * self.theta * time_diff_2))
                )
                sigma22 = std22 ** 2
                sigma12 = (
                    (self.sigma ** 2)
                    / (2 * self.theta)
                    * (
                        torch.exp(0 - self.theta * time_diff_2)
                        - torch.exp(0 - self.theta * (time_diff_2 + time_diff))
                    )
                    * torch.ones_like(z_prev)
                )

                mean = mean_1 + sigma12 / sigma22 * (z_future - mean_2)
                std = torch.std(sigma11 - sigma12 / sigma22 * sigma12)

        else:
            raise NotImplementedError

        base_distributions = torch.distributions.Normal(mean, std)
        samples = base_distributions.sample()
        return samples


class RNNEncoder(nn.Module):
    def __init__(self, rnn_input_dim, hidden_dim, encoder_network=None):
        super(RNNEncoder, self).__init__()
        self.encoder_network = encoder_network
        self.rnn_cell = nn.GRUCell(rnn_input_dim, hidden_dim)

    def forward(self, h, x_current, y_prev, t_current, t_prev):
        t_current = torch.ones(x_current.shape[0], 1).to(t_current) * t_current
        t_prev = torch.ones_like(t_current) * t_prev

        if self.encoder_network is None:
            t_diff = t_current - t_prev
            t_current = time_embedding(t_current)
            t_prev = time_embedding(t_prev)
            input = torch.cat([x_current, y_prev, t_current, t_prev, t_diff], 1)
        else:
            input = self.encoder_network(x_current, y_prev, t_current, t_prev)
        return self.rnn_cell(input, h)


class qy0Network(nn.Module):
    def __init__(self, input_dim, network_dims, time_embedding_dim=3):
        super(qy0Network, self).__init__()
        self.input_dim = input_dim
        self.output_dim = network_dims[-1]
        self.network = network_factory([time_embedding_dim + input_dim] + network_dims)

    def forward(self, t, x_0):
        t = t.unsqueeze(0).unsqueeze(0)
        t = time_embedding(t)
        t = t.repeat(x_0.shape[0], 1)
        q0_mean_logvar = self.network(torch.cat([x_0, t], 1))

        return (
            q0_mean_logvar[:, : self.output_dim // 2],
            q0_mean_logvar[:, self.output_dim // 2 :],
        )


class CLPF(nn.Module):
    def __init__(
        self,
        latent_dim,
        latent_sde,
        rnn_encoder,
        decoder,
        qy0_network,
        hidden_dim,
        hidden_network_dims=None,
        num_iwae=1,
    ):
        super(CLPF, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_iwae = num_iwae

        self.latent_sde = latent_sde
        self.py0_mean = nn.Parameter(torch.zeros(1, latent_dim))
        self.py0_logvar = nn.Parameter(torch.zeros(1, latent_dim))
        self.qy0_network = qy0_network

        self.rnn_encoder = rnn_encoder
        self.hidden_state_proj = nn.Identity()
        if hidden_network_dims is not None:
            self.hidden_state_proj = network_factory([hidden_dim] + hidden_network_dims)

        self.decoder = decoder

    def inference_one_synch_batch(self, ts, inputs):
        """
        ts: a one dimensional vector
        inputs: a vector of shape batch_size x num_steps x observation_dim
        """
        self.latent_sde.clear_context()
        inputs = inputs.repeat_interleave(self.num_iwae, 0)
        num_observations = inputs.shape[1]
        batch_size = inputs.shape[0]
        h = torch.zeros((batch_size, self.hidden_dim)).to(inputs)

        py0_mean_repeated = self.py0_mean.repeat_interleave(batch_size, 0)
        py0_logvar_repeated = self.py0_logvar.repeat_interleave(batch_size, 0)
        qy0_mean, qy0_logvar = self.qy0_network(ts[0], inputs[:, 0])
        py0_distribution = distributions.Normal(
            py0_mean_repeated, torch.exp(0.5 * py0_logvar_repeated)
        )

        qy0_distribution = distributions.Normal(qy0_mean, torch.exp(0.5 * qy0_logvar))

        y = sample_normal(qy0_mean, qy0_logvar)
        prev_time = torch.from_numpy(np.array(0).astype(np.float32)).to(inputs)
        # Compute the KL Divergence for y0
        log_pqs = py0_distribution.log_prob(y) - qy0_distribution.log_prob(y)
        log_pqs = log_pqs.sum(-1)

        z = None

        y_lst = []
        x_lst = []
        t_lst = []
        for i in range(num_observations):
            current_time = ts[i]
            # Update the rnn_code
            h = self.rnn_encoder(h, inputs[:, i], y, current_time, prev_time)
            context = self.hidden_state_proj(h)
            aug_y = self.latent_sde(y, current_time.unsqueeze(0), (context, prev_time))
            y = aug_y[:, :-1]
            log_pq = -aug_y[:, -1]
            log_pqs += log_pq
            y_lst.append(y)
            x_lst.append(inputs[:, i])
            prev_time = current_time
        return y_lst, x_lst, log_pqs

    def inference_one_synch_batch_masked(self, ts, inputs, masks):
        """
        ts: a one dimensional vector
        inputs: a vector of shape batch_size x num_steps x observation_dim
        masks: a vector of shape batch_size x num_steps
        """
        self.latent_sde.clear_context()
        inputs = inputs.repeat_interleave(self.num_iwae, 0)
        num_observations = inputs.shape[1]
        batch_size = inputs.shape[0]
        h = torch.zeros((batch_size, self.hidden_dim)).to(inputs)

        py0_mean_repeated = self.py0_mean.repeat_interleave(batch_size, 0)
        py0_logvar_repeated = self.py0_logvar.repeat_interleave(batch_size, 0)
        qy0_mean, qy0_logvar = self.qy0_network(ts[0], inputs[:, 0])
        py0_distribution = distributions.Normal(
            py0_mean_repeated, torch.exp(0.5 * py0_logvar_repeated)
        )

        qy0_distribution = distributions.Normal(qy0_mean, torch.exp(0.5 * qy0_logvar))

        y = sample_normal(qy0_mean, qy0_logvar)
        prev_time = torch.from_numpy(np.array(0).astype(np.float32)).to(inputs)
        # Compute the KL Divergence for
        log_pqs = py0_distribution.log_prob(y) - qy0_distribution.log_prob(y)
        log_pqs = log_pqs.sum(-1)
        z = None
        y_lst = []
        x_lst = []
        t_lst = []
        for i in range(num_observations):
            current_time = ts[i]
            # Update the rnn_code
            h = self.rnn_encoder(h, inputs[:, i], y, current_time, prev_time)
            context = self.hidden_state_proj(h)
            aug_y = self.latent_sde(y, current_time.unsqueeze(0), (context, prev_time))
            y = aug_y[:, :-1]
            log_pq = -aug_y[:, -1] * masks[i]
            log_pqs += log_pq
            y_lst.append(y)
            x_lst.append(inputs[:, i])
            prev_time = current_time
        return y_lst, x_lst, log_pqs

    def forward(self, ts, inputs, lengths=None, masks=None):
        """
        The function takes ts a sequence of observation time stamps and inputs
        a batch of observations and return a IWAE bound of the negative log liklidhoos induced by CLPF.
        ts: batch_size x num_observs
        inputs: batch_size x num_observs x data_dims
        mask: a binary tensor indicating if an observation is observed for an batch of sequences with variable length. It is None if all the sequences in the batch has the same length.
        """
        if masks is not None:
            max_error = None
            masks = masks.transpose(0, 1)
            ts = ts[0]
            ## Masks is of size num_observs * batch_size
            batch_size = inputs.shape[0]
            num_observs = inputs.shape[1]
            observation_dim = inputs.shape[-1]
            self.latent_sde.clear_context()
            masks_iwae = masks.repeat_interleave(self.num_iwae, 1)

            y_lst, x_lst, log_pqys = self.inference_one_synch_batch_masked(
                ts, inputs, masks_iwae
            )

            y_cat = torch.cat(y_lst, 0)
            x_cat = torch.cat(x_lst, 0)
            ## size (num_observs x batch_size x num_iwae) x observation_dim
            t_cat = ts.repeat_interleave(batch_size * self.num_iwae).unsqueeze(1)
            base_values, log_dets, error = self.decoder(x_cat, y_cat, t_cat)
            if (not self.training) and (error is not None):
                error = error.reshape(
                    num_observs, batch_size * self.num_iwae, observation_dim
                )
                error = error * masks_iwae.unsqueeze(2)
                if max_error is None:
                    max_error = torch.max(error)
                else:
                    max_error = max(max_error, error)

            base_values = base_values.reshape(
                num_observs, batch_size * self.num_iwae, observation_dim
            )
            z_prev = None
            t_prev = None
            ll_lst = 0
            for i in range(num_observs):
                z_current = base_values[i]
                t_current = ts[i]
                ll = self.decoder.density_calculation(
                    z_current, t_current, z_prev, t_prev
                )
                ll_lst += ll * masks_iwae[i]
                z_prev = z_current
                t_prev = t_current
            ll = ll_lst - (
                log_dets.reshape(num_observs, batch_size * self.num_iwae) * masks_iwae
            ).sum(0)
            weights = ll + log_pqys
            weights = weights.reshape(batch_size, self.num_iwae)
            loss = -torch.logsumexp(weights, 1) + np.log(self.num_iwae)
            loss_training = -(torch.softmax(weights.detach(), 1) * weights).sum(1)
            return loss.sum(), loss_training.sum(), max_error

        else:
            max_error = None
            ts = ts[0]
            batch_size = inputs.shape[0]
            num_observs = inputs.shape[1]
            observation_dim = inputs.shape[-1]
            self.latent_sde.clear_context()
            y_lst, x_lst, log_pqys = self.inference_one_synch_batch(ts, inputs)
            y_cat = torch.cat(y_lst, 0)
            x_cat = torch.cat(x_lst, 0)
            ## size (num_observs x batch_size x num_iwae) x observation_dim
            t_cat = ts.repeat_interleave(batch_size * self.num_iwae).unsqueeze(1)
            base_values, log_dets, error = self.decoder(x_cat, y_cat, t_cat)
            if (not self.training) and (error is not None):
                if max_error is None:
                    max_error = torch.max(error)
                else:
                    max_error = max(torch.max(error), max_error)

            base_values = base_values.reshape(
                num_observs, batch_size * self.num_iwae, observation_dim
            )

            z_prev = None
            t_prev = None
            ll_lst = 0
            for i in range(num_observs):
                z_current = base_values[i]
                t_current = ts[i]
                ll = self.decoder.density_calculation(
                    z_current, t_current, z_prev, t_prev
                )
                ll_lst += ll
                z_prev = z_current
                t_prev = t_current
            ll = ll_lst - log_dets.reshape(num_observs, batch_size * self.num_iwae).sum(
                0
            )
            weights = ll + log_pqys
            weights = weights.reshape(batch_size, self.num_iwae)
            loss = -torch.logsumexp(weights, 1) + np.log(self.num_iwae)
            loss_training = -(torch.softmax(weights.detach(), 1) * weights).sum(1)
            return loss.sum(), loss_training.sum(), max_error

    def sample(
        self, target_ts, batch_size=None, ts=None, inputs=None, observe_sample_size=1
    ):
        # batch_size is the sample size of the latent variable
        # observe_sample_size is the sample size of the observations
        """
        target_ts: a tensor of observation time points to sample at. Only used
                   when ts is None. The shape of target_tx is number of time
                   points.

        batch_size: A varible determining the sample size.

        ts: a None value or list of tuples of format (observation time,
            interpolation times, extrapolations times). Observation
            time is a single tensor. Interpolation time is a tensor
            of the times for interpolation and target times are the
            times for extrapolations. If None, sample from prior and sample size
            is batch_size (number of latent samples) x batch_size (number of
            observation samples). If not None, sample from posterior. The ith element in the list tell the function to sample at interpolation times and extrapolations time, given the firth i observations from inputs and the ith observation in inputs takes place at observation time.

        inputs: a input tensor of observations of shape data batch size x number of observations x data dimensions. The number of observations is the same of the length of ts.

        observe_sample_size: for each latent variable, sample observe_sample_size observations.

        """
        if ts is None:
            # target_ts is a single tensor
            if batch_size is None:
                batch_size = 1
            py0_mean_repeated = self.py0_mean.repeat_interleave(batch_size, 0)
            py0_logvar_repeated = self.py0_logvar.repeat_interleave(batch_size, 0)
            py0_distribution = distributions.Normal(
                py0_mean_repeated, torch.exp(0.5 * py0_logvar_repeated)
            )

            y0 = py0_distribution.sample()
            prior_ys = self.latent_sde.sample_from_prior(y0, target_ts)

            t_prev = None
            z_prev = None
            base_zs = []
            for i in range(target_ts.shape[0]):
                z_prev = self.decoder.sample_from_base_process(
                    target_ts[i], t_prev=t_prev, z_prev=z_prev, batch_size=batch_size
                )
                t_prev = target_ts[i]
                base_zs.append(z_prev)

            xs = []

            for i in range(target_ts.shape[0]):
                print(i, "/", target_ts.shape[0])
                x_current = self.decoder.base_to_observe(
                    target_ts[i], prior_ys[i], base_zs[i]
                )
                xs.append(x_current.data.cpu())
            return xs
        else:
            # Assume the first point of target_ts is zero
            # ts is a list of tuples (observation time,
            # interpolation times, extrapolations times). Observation
            # time is a single tensor. Interpolation time is a tensor
            # of the times for interpolation and target times are the
            # times for extrapolations.
            # Batch size is the sample of latent trajectories
            # observe sample size if the sample size of observed trajectories
            data_batch_size = inputs.shape[0]
            self.latent_sde.clear_context()
            inputs = inputs.repeat_interleave(batch_size, 0)
            num_observations = len(ts)
            h = torch.zeros((data_batch_size * batch_size, self.hidden_dim)).to(inputs)

            qy0_mean, qy0_logvar = self.qy0_network(ts[0][0], inputs[:, 0])
            qy0_distribution = distributions.Normal(
                qy0_mean, torch.exp(0.5 * qy0_logvar)
            )

            # sample y0

            y = sample_normal(qy0_mean, qy0_logvar)
            # Assume the first element of first item in target_ts is zero (time)
            prev_time = torch.zeros_like(ts[0][0])
            x_lst = []
            end_idx = 0
            # Sample z0
            prev_z_repeated = self.decoder.sample_from_base_process(
                prev_time,
                t_prev=None,
                batch_size=data_batch_size * batch_size * observe_sample_size,
            )
            for i in range(num_observations):
                one_x_list = []
                ## Observation of the current time
                current_time = ts[i][0]
                current_time_pointer = ts[i][2]

                h = self.rnn_encoder(h, inputs[:, i], y, current_time, prev_time)
                context = self.hidden_state_proj(h)

                pred_times = ts[i][1]

                ## the time stamp to propagate the latent SDEs to
                aug_y = self.latent_sde.sample_from_posterior(
                    y, pred_times, (context, prev_time)
                )

                y = aug_y[current_time_pointer][:, :-1]
                # Size of batch_size x latent_dim

                t_cat = (
                    current_time.unsqueeze(0)
                    .unsqueeze(0)
                    .repeat_interleave(inputs.shape[0], 0)
                )
                current_z, _, _ = self.decoder(inputs[:, i], y, t_cat)
                # Do interpolations
                # Sample from the base process
                current_z_repeated = current_z.repeat_interleave(observe_sample_size, 0)
                interpolation_times = pred_times[:current_time_pointer]
                interpolation_latents = aug_y[:current_time_pointer, :, :-1]

                sample_z_list = []
                prev_time_for_interp = prev_time

                for target_time, target_latent in zip(
                    interpolation_times, interpolation_latents
                ):
                    sampled_z = self.decoder.sample_from_base_process(
                        target_time,
                        t_prev=prev_time_for_interp,
                        z_prev=prev_z_repeated,
                        t_future=current_time,
                        z_future=current_z_repeated,
                    )
                    target_latent_repeated = target_latent.repeat_interleave(
                        observe_sample_size, 0
                    )
                    one_x_list.append(
                        self.decoder.base_to_observe(
                            target_time, target_latent_repeated, sampled_z
                        ).reshape(data_batch_size, batch_size * observe_sample_size, -1)
                    )
                    prev_z_repeated = sampled_z
                    prev_time_for_interp = target_time

                # Do extrapolations
                prev_z_repeated = current_z_repeated
                prev_time_for_extrap = current_time

                extrapolation_times = pred_times[current_time_pointer + 1 :]
                extrapolation_latents = aug_y[current_time_pointer + 1 :, :, :-1]
                for target_time, target_latent in zip(
                    extrapolation_times, extrapolation_latents
                ):
                    sampled_z = self.decoder.sample_from_base_process(
                        target_time, t_prev=prev_time_for_extrap, z_prev=prev_z_repeated
                    )
                    target_latent_repeated = target_latent.repeat_interleave(
                        observe_sample_size, 0
                    )
                    one_x_list.append(
                        self.decoder.base_to_observe(
                            target_time, target_latent_repeated, sampled_z
                        ).reshape(data_batch_size, batch_size * observe_sample_size, -1)
                    )

                x_lst = x_lst + one_x_list
                prev_time = current_time
        return x_lst


def model_builder(args):
    ## Process all the dimensions
    args.drift_network_dims = [int(i) for i in args.drift_network_dims.split(",")] + [
        args.latent_dim
    ]
    args.variance_network_dims = [int(i) for i in args.variance_network_dims.split(",")]
    if args.noise_type == "diagonal":
        args.variance_network_dims = args.variance_network_dims + [1]
    elif args.noise_type == "general":
        args.variance_network_dims = args.variance_network_dims + [args.latent_dim]
    elif args.noise_type == "additive":
        pass
    else:
        raise NotImplementedError
    args.q0_network_dims = [int(i) for i in args.q0_network_dims.split(",")] + [
        args.latent_dim * 2
    ]
    if args.hidden_projection_dims is not None:
        args.hidden_projection_dims = [
            int(i) for i in args.hidden_projection_dims.split(",")
        ]
    ## Config the arguments for the latent SDE
    class SDEArgs(object):
        def __init__(self):
            self.method = args.method
            self.dt = args.dt
            self.dt_min = args.dt_min
            self.dt_test = args.dt_test
            self.dt_min_test = args.dt_min_test
            self.noise_type = args.noise_type
            self.adaptive = args.adaptive
            self.rtol = args.rtol
            self.atol = args.atol
            self.variance_act = args.variance_act
            self.time_embedding_dim = args.time_embedding_dim

    sde_args = SDEArgs()

    # Config for the decoder
    if args.indexed_flow_type == "anode":
        ctfp_args = ANODEConfig(
            args.anode_dims,
            args.anode_aug_hidden_dims,
            args.anode_effective_shape,
            args.anode_aug_dim,
            args.anode_aug_mapping,
            num_blocks=args.anode_num_blocks,
            layer_type=args.anode_layer_type,
            nonlinearity=args.anode_nonlinearity,
            divergence_fn=args.anode_divergence_fn,
            residual=args.anode_residual,
            rademacher=args.anode_rademacher,
            time_length=args.anode_time_length,
            train_T=args.anode_train_T,
            solver=args.anode_solver,
            rtol=args.anode_rtol,
            atol=args.anode_atol,
            batch_norm=args.anode_batch_norm,
            step_size=args.anode_step_size,
            test_solver=args.anode_test_solver,
            test_rtol=args.anode_test_rtol,
            test_atol=args.anode_test_atol,
            l1int=args.anode_l1int,
            l2int=args.anode_l2int,
            dl2int=args.anode_dl2int,
            JFrobint=args.anode_JFrobint,
            JdiagFrobint=args.anode_JdiagFrobint,
            JoffdiagFrobint=args.anode_JoffdiagFrobint,
            time_penalty=args.anode_time_penalty,
        )
    if args.indexed_flow_type == "iresnet":
        ctfp_args = iResNetConfig(args)

    if args.indexed_flow_type == "independent":
        ctfp_args = IndependentDecoderConfig(args)

    context_dim = args.hidden_dim
    if args.hidden_projection_dims is not None:
        context_dim = args.hidden_projection_dims[-1]
    latent_sde = latentSDE(
        args.latent_dim,
        context_dim,
        args.drift_network_dims,
        args.variance_network_dims,
        sde_args,
    )
    qy0_network = qy0Network(
        args.observation_dim, args.q0_network_dims, args.time_embedding_dim
    )
    if args.indexed_flow_type == "independent":
        ctfp_decoder = IndependentDecoder(
            ctfp_args, args.observation_dim, args.latent_dim
        )
    else:
        ctfp_decoder = CTFPDecoder(
            ctfp_args,
            args.observation_dim,
            args.latent_dim + 1,
            flow_type=args.indexed_flow_type,
            base_process_type=args.base_process_type,
            exact_training_ou_std=args.exact_training_ou_std,
        )
    rnn_encoder = RNNEncoder(
        args.observation_dim + args.latent_dim + 7, args.hidden_dim
    )

    clpf = CLPF(
        args.latent_dim,
        latent_sde,
        rnn_encoder,
        ctfp_decoder,
        qy0_network,
        args.hidden_dim,
        hidden_network_dims=args.hidden_projection_dims,
        num_iwae=args.num_iwae,
    )
    return clpf


# Test the code
if __name__ == "__main__":
    latent_dim = 16
    hidden_dim = 16
    observation_dim = 2
    # ## Define Digaonal Covariance
    ## Define the Latent SDE
    class sdeArgs(object):
        def __init__(self):
            self.method = "euler"
            self.dt = 1e-3
            self.adaptive = False
            self.rtol = 1e-5
            self.atol = 1e-5

    sde_args = sdeArgs()

    latent_sde = latentSDE(latent_dim, latent_dim, [32, 32, 16], [32, 32, 1], sde_args)
    ## Define qy0 network
    qy0_network = qy0Network(observation_dim, [32, 32, latent_dim * 2])
    ## Define decoder

    ctfp_args = ANODEConfig("8,32,32,8", None, observation_dim, 0, False)
    ctfp_decoder = CTFPDecoder(ctfp_args, observation_dim, latent_dim + 1)
    ## Define rnn_encoder
    rnn_encoder = RNNEncoder(observation_dim + hidden_dim + 7, hidden_dim)
    ## Define the CLPF
    CLPF = CLPF(
        latent_dim,
        latent_sde,
        rnn_encoder,
        ctfp_decoder,
        qy0_network,
        hidden_dim,
        num_iwae=128,
    )
    ## Define a synchrounous batch data
    ts = torch.from_numpy(np.array([0.5, 1.0, 2.0]).astype(np.float32))
    data = torch.from_numpy(np.random.rand(2, 3, 2).astype(np.float32))
    results = CLPF(ts, data)
