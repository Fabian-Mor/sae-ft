import os
import argparse

import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. CIFAR101,CIFAR102."
             " Note that same model used for all datasets, so much have same classnames"
             "for zero shot.",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Which prompt template is used. Leave as None for linear probe, etc.",
    )
    parser.add_argument(
        "--classnames",
        type=str,
        default="openai",
        help="Which class names to use.",
    )
    parser.add_argument(
        "--alpha",
        default=[0.5],
        nargs='*',
        type=float,
        help=(
            'Interpolation coefficient for ensembling. '
            'Users should specify N-1 values, where N is the number of '
            'models being ensembled. The specified numbers should sum to '
            'less than 1. Note that the order of these values matter, and '
            'should be the same as the order of the classifiers being ensembled.'
        )
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--freeze-encoder",
        default=False,
        action="store_true",
        help="Whether or not to freeze the image encoder. Only relevant for fine-tuning."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--fisher",
        type=lambda x: x.split(","),
        default=None,
        help="TODO",
    )
    parser.add_argument(
        "--fisher_floor",
        type=float,
        default=1e-8,
        help="TODO",
    )
    parser.add_argument(
        "--load_zeroshot",
        type=str,
        default=None,
        help="location of the zeroshot model",
    )
    parser.add_argument(
        "--load_finetuned",
        type=str,
        default=None,
        help="location of the directory of the loaded fine-tuned models",
    )
    parser.add_argument(
        "--amount_finetuned",
        type=int,
        default=1,
        help="the amount of fine-tuned models to load",
    )
    parser.add_argument(
        "--jump_epochs",
        type=int,
        default=1000,
        help="the amount of epochs after which to jump to the interpolation point",
    )
    parser.add_argument(
        "--learning_rate_dirs",
        type=lambda x: x.split(","),
        default=None,
        help="Directories for learning rates",
    )
    parser.add_argument("--evaluation",
                    type=str,
                    default="quick")
    parser.add_argument(
        "--name",
        type=str,
    )
    parser.add_argument(
        "--scheduler",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--reg_type",
        type=str,
        default="",
    )
    parser.add_argument(
        "--reg_constraint_c",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--lambda_l1",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--sae_path",
        type=str,
        default="autoencoders/sa_16_0.pt",
    )
    parser.add_argument(
        "--representation_dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--sae_mult",
        type=float,
        default=4,
    )
    parser.add_argument(
        "--pca_path",
        type=str,
        default="autoencoders/pca_B16__k16_n1281167.pt",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="B16_"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
    )
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                        help='Dataset split to evaluate on.')

    parser.add_argument('--lambda_mask', type=float, default=10.0)
    parser.add_argument('--mu', type=float, default=1.0)

    parser.add_argument("--wise_ft", default=False, type=bool)
    parser.add_argument("--normalize", default=False, type=bool)
    parser.add_argument("--corruption", default=None,
                        type=str)
    parser.add_argument('--num_splits', type=int, default=1, help='Number of data splits to create')
    parser.add_argument('--split_index', type=int, default=0, help='Index of split to use (0-based)')
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
