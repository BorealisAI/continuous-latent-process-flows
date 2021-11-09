# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os.path as osp

ODE_SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams"]


def str2bool(v):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_arguments():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser("Latent SDE RNN")
    parser.add_argument("--eval", action="store_true", help="Running evaluation")
    parser.add_argument(
        "--data_type",
        type=str,
        default="synthetic",
        choices=["synthetic", "real", "unequal"],
    )
    parser.add_argument("--data_path", type=str, default="data/gbm_2_batch.pkl")
    parser.add_argument(
        "--noise_std",
        type=float,
        default=None,
        help="Standard deviation of noise added to training data",
    )
    parser.add_argument(
        "--noise_std_test",
        type=float,
        default=None,
        help="Standard deviation of noise added to test data",
    )
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--num_iwae",
        type=int,
        default=3,
        help="Number of samples to train IWAE encoder",
    )
    parser.add_argument(
        "--niwae_test", type=int, default=25, help="Numver of IWAE samples during test"
    )
    parser.add_argument("--decoder_frequency", type=int, default=3)
    parser.add_argument("--aggressive", action="store_true")

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--begin_epoch", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save", type=str, default="debug")
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument(
        "--no_tb_log", action="store_true", help="Do not use tensorboard logging"
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        choices=["train", "test", "val"],
        help="The split of dataset to evaluate the model on",
    )
    parser.add_argument("--np_seed", type=int, default=None)
    parser.add_argument("--torch_seed", type=int, default=None)
    ############################################
    ### Experiment Parameter for predictions ###
    ############################################
    parser.add_argument("--save_np", type=str, default="sample.npy")
    parser.add_argument(
        "--pred_mode", type=str, default="pred", choices=["pred", "interp"]
    )
    parser.add_argument(
        "--observe_sample_size",
        type=int,
        default=1,
        help="number of samples to sample from observation distributions",
    )

    ############################################
    ### Experiment Parameter for predictions ###
    ############################################

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--test_batch_size", type=int, default=50)
    parser.add_argument(
        "--sample_interval",
        type=float,
        default=1e-2,
        help="a time grid with fixed interval to sample observations from",
    )

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument(
        "--amsgrad", action="store_true", help="use amsgrad for adam optimizer"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum value for sgd optimizer"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1e10,
        help="Max norm of graidents (default is just stupidly high to avoid any clipping)",
    )
    parser.add_argument(
        "--observation_dim", type=int, default=None, help="the size of observations"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=16, help="the size of latent trajectory"
    )
    parser.add_argument(
        "--time_embedding_dim", type=int, default=3, help="size of time embedding"
    )
    parser.add_argument(
        "--indexed_flow_type",
        type=str,
        default="anode",
        choices=["anode", "iresnet", "independent"],
    )
    parser.add_argument(
        "--base_process_type", type=str, default="ou", choices=["ou", "wiener"]
    )
    parser.add_argument(
        "--exact_training_ou_std",
        action="store_true",
        help="use exact std for training. Approximate std is used during evaluation. ",
    )

    ################################
    ## Real World Data Parameters ##
    ################################
    parser.add_argument(
        "--max_time",
        type=float,
        default=30,
        help="Maximum time length of sampled sequences",
    )
    parser.add_argument(
        "--max_length",
        type=float,
        default=650,
        help="Manually cap the maximum length of sequences in the dataset",
    )
    parser.add_argument(
        "--time_bias", type=float, default=0.2, help="Bias added to the sampled time"
    )
    parser.add_argument(
        "--observ_scale",
        type=float,
        default=2.0,
        help="Assume the observation times are sampled from a poisson process",
    )
    parser.add_argument(
        "--observ_interval",
        type=float,
        default=None,
        help="regular observation interval",
    )
    parser.add_argument("--interp_observ", action="store_true")
    parser.add_argument(
        "--observe_subsample_rate",
        type=int,
        default=None,
        help="create regular time grid by subsample the gt sequence",
    )

    parser.add_argument("--train_subsample_rate", type=int, default=None)
    parser.add_argument("--train_random_subsample_rate", type=int, default=None)
    parser.add_argument("--train_subsample_random_start", action="store_true")
    parser.add_argument("--eval_subsample_rate", type=int, default=None)
    parser.add_argument("--eval_random_subsample_rate", type=int, default=None)
    parser.add_argument("--eval_subsample_random_start", action="store_true")

    #################
    ## RNN Encoder ##
    #################
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Number of units per layer in each of GRU update networks in encoder",
    )

    #############################
    ## SDE networks dimensions ##
    #############################
    parser.add_argument("--drift_network_dims", type=str, default="32,32")
    parser.add_argument("--variance_network_dims", type=str, default="32,32")
    parser.add_argument("--q0_network_dims", type=str, default="32,32")
    parser.add_argument(
        "--hidden_projection_dims",
        type=str,
        default="32, 16",
        help="dimensions of networks that project hidden state to a context tensor",
    )

    ####################
    ## SDE Parameters ##
    ####################
    parser.add_argument(
        "--adaptive", type=str2bool, default=False, const=True, nargs="?"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="euler",
        choices=("euler", "milstein", "srk"),
        help="Name of numerical solver. We have only tested ",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="additive",
        choices=("additive", "diagonal", "general"),
    )
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--dt_min", type=float, default=1e-5)
    parser.add_argument("--dt_test", type=float, default=1e-4)
    parser.add_argument("--dt_min_test", type=float, default=1e-5)

    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument(
        "--variance_act", type=str, default="softplus", choices=["softplus", "sigmoid"]
    )

    ######################
    ## ANODE Parameters ##
    ######################
    parser.add_argument("--anode_dims", type=str, default="8,32,32,8")
    parser.add_argument(
        "--anode_aug_hidden_dims",
        type=str,
        default=None,
        help="The hiddden dimension of the odenet taking care of augmented dimensions",
    )
    parser.add_argument(
        "--anode_aug_dim",
        type=int,
        default=0,
        help="The dimension along which input is augmented. 0 for 1-d input",
    )
    parser.add_argument("--anode_strides", type=str, default="2,2,1,-2,-2")
    parser.add_argument(
        "--anode_num_blocks", type=int, default=2, help="Number of stacked CNFs."
    )
    parser.add_argument("--anode_conv", type=eval, default=True, choices=[True, False])
    parser.add_argument(
        "--anode_layer_type",
        type=str,
        default="ignore",
        choices=[
            "ignore",
            "concat",
            "concat_v2",
            "squash",
            "concatsquash",
            "concatcoord",
            "hyper",
            "blend",
        ],
    )
    parser.add_argument(
        "--anode_divergence_fn",
        type=str,
        default="approximate",
        choices=["brute_force", "approximate"],
    )
    parser.add_argument(
        "--anode_nonlinearity",
        type=str,
        default="softplus",
        choices=["tanh", "relu", "softplus", "elu", "swish"],
    )
    parser.add_argument(
        "--anode_solver", type=str, default="dopri5", choices=ODE_SOLVERS
    )
    parser.add_argument("--anode_atol", type=float, default=1e-5)
    parser.add_argument("--anode_rtol", type=float, default=1e-5)
    parser.add_argument(
        "--anode_step_size", type=float, default=None, help="Optional fixed step size."
    )

    parser.add_argument(
        "--anode_test_solver", type=str, default=None, choices=ODE_SOLVERS + [None]
    )
    parser.add_argument("--anode_test_atol", type=float, default=None)
    parser.add_argument("--anode_test_rtol", type=float, default=None)
    parser.add_argument("--anode_alpha", type=float, default=1e-6)
    parser.add_argument("--anode_time_length", type=float, default=1.0)
    parser.add_argument("--anode_train_T", type=eval, default=True)
    parser.add_argument("--anode_aug_mapping", action="store_true")
    parser.add_argument(
        "--anode_activation",
        type=str,
        default="exp",
        choices=["exp", "softplus", "identity"],
    )
    # ANODE Regularizations
    parser.add_argument("--anode_l1int", type=float, default=None, help="int_t ||f||_1")
    parser.add_argument("--anode_l2int", type=float, default=None, help="int_t ||f||_2")
    parser.add_argument(
        "--anode_dl2int", type=float, default=None, help="int_t ||f^T df/dt||_2"
    )
    parser.add_argument(
        "--anode_JFrobint", type=float, default=None, help="int_t ||df/dx||_F"
    )
    parser.add_argument(
        "--anode_JdiagFrobint", type=float, default=None, help="int_t ||df_i/dx_i||_F"
    )
    parser.add_argument(
        "--anode_JoffdiagFrobint",
        type=float,
        default=None,
        help="int_t ||df/dx - df_i/dx_i||_F",
    )
    parser.add_argument(
        "--anode_time_penalty",
        type=float,
        default=0,
        help="Regularization on the end_time.",
    )
    parser.add_argument(
        "--anode_residual", type=eval, default=False, choices=[True, False]
    )
    parser.add_argument(
        "--anode_autoencode", type=eval, default=False, choices=[True, False]
    )
    parser.add_argument(
        "--anode_rademacher", type=eval, default=True, choices=[True, False]
    )
    parser.add_argument(
        "--anode_multiscale", type=eval, default=False, choices=[True, False]
    )
    parser.add_argument(
        "--anode_parallel", type=eval, default=False, choices=[True, False]
    )
    parser.add_argument(
        "--anode_batch_norm", type=eval, default=False, choices=[True, False]
    )

    ########################
    ## iResNet Parameters ##
    ########################
    parser.add_argument("--ires_num_blocks", type=int, default=5)
    parser.add_argument("--ires_aug_block_dims", type=str, default=None)
    parser.add_argument("--ires_aug_proj_dims", type=str, default="64,64,64")
    parser.add_argument("--ires_coeff", type=float, default=0.9)
    parser.add_argument("--ires_vnorms", type=str, default="222222")
    parser.add_argument("--ires_n_lipschitz_iters", type=int, default=5)
    parser.add_argument("--ires_atol", type=float, default=None)
    parser.add_argument("--ires_rtol", type=float, default=None)
    parser.add_argument("--ires_mixed", type=eval, choices=[True, False], default=True)

    parser.add_argument("--ires_dims", type=str, default="64,64,64")
    parser.add_argument(
        "--ires_act",
        type=str,
        choices=["relu", "tanh", "elu", "selu", "fullsort", "maxmin", "swish", "lcube"],
        default="swish",
    )
    # parser.add_argument('--ires_nblocks', type=int, default=10)
    parser.add_argument(
        "--ires_brute_force", type=eval, choices=[True, False], default=False
    )
    parser.add_argument(
        "--ires_actnorm", type=eval, choices=[True, False], default=False
    )
    parser.add_argument(
        "--ires_batchnorm", type=eval, choices=[True, False], default=False
    )
    parser.add_argument(
        "--ires_exact_trace", type=eval, choices=[True, False], default=False
    )
    parser.add_argument("--ires_n_power_series", type=int, default=None)
    parser.add_argument("--ires_n_samples", type=int, default=1)
    parser.add_argument(
        "--ires_n_dist", choices=["geometric", "poisson"], default="geometric"
    )
    parser.add_argument(
        "--ires_neumann_grad", type=eval, choices=[True, False], default=True
    )
    parser.add_argument(
        "--ires_grad_in_forward", type=eval, choices=[True, False], default=True
    )
    parser.add_argument(
        "--ires_update_during_training", action="store_true", default=False
    )

    ####################################
    ## Independent Decoder Parameters ##
    ####################################
    parser.add_argument("--indecoder_dims", type=str, default="16,64,64,16")

    args = parser.parse_args()
    args.save = osp.join("experiments", args.save)

    args.anode_effective_shape = args.observation_dim
    return args
