# SAE-FT: Robust and Interpretable Fine-tuning of CLIP models via Sparse Autoencoders

This repository extends [WiSE-FT](https://arxiv.org/abs/2109.01903) (Wortsman et al., 2021) with a collection of **representation-space regularizers** applied during fine-tuning of CLIP-style zero-shot classifiers. The goal is to retain zero-shot generalization while improving in-distribution accuracy on a target dataset.

The main entry point is [`src/training.py`](src/training.py).

## Idea

When fine-tuning a CLIP encoder, the image features can drift far from the original zero-shot manifold, which hurts out-of-distribution robustness. We penalize this drift with an auxiliary loss between the current encoder's features and the frozen zero-shot encoder's features:

```
loss = cross_entropy(logits, labels) + reg_lambda * R(f_theta(x), f_0(x))
```

`R` is configurable via `--reg_type`. Several variants are implemented — MSE / L1, nuclear-norm penalties, layerwise feature distillation (LDIFS), PCA projections, and several sparse-autoencoder (SAE) variants — see [Regularizers](#regularizers) below.

## Install

```bash
conda env create -f environment.yml
conda activate wiseft_v2
export PYTHONPATH="$PYTHONPATH:$PWD"
```

## Data

Dataset loaders live in [`src/datasets/`](src/datasets). See [`datasets.md`](datasets.md) for download instructions for ImageNet distribution shifts, WILDS, and CIFAR variants. Other supported datasets (DTD, FGVCAircraft, EuroSAT, StanfordCars, Flowers102, Caltech101, dSprites, STL-10, Pets) follow the standard `torchvision` layout under `--data-location`.

## Quick start

Standard fine-tuning (no regularization) on ImageNet with ViT-B/16:

```bash
python src/training.py \
    --train-dataset=ImageNet \
    --model=ViT-B/16 \
    --template=openai_imagenet_template \
    --epochs=10 --lr=1e-5 --batch-size=32 \
    --data-location=/path/to/imagenet \
    --save=models/B16_baseline
```

SAE-mask regularized fine-tuning (the main method in this repo):

```bash
python src/training.py \
    --train-dataset=ImageNet \
    --model=ViT-B/16 \
    --template=openai_imagenet_template \
    --epochs=10 --lr=1e-5 --batch-size=32 \
    --data-location=/path/to/imagenet \
    --reg_type=sae_mask --reg_lambda=70 \
    --sae_path=autoencoders/sae_4_32.pt \
    --sae_mult=4 --k=32 \
    --save=models/B16_sae_mask
```

The script first builds a zero-shot classifier from CLIP text embeddings (saved to `<save>/zeroshot.pt`), then fine-tunes the encoder. Fine-tuned checkpoints are written to `<save>/finetuned/checkpoint_<epoch>.pt`, and a resumable `checkpoint_continue.pt` is written every epoch.

## Regularizers

Selected with `--reg_type`. All regularizers compare current features `f_theta(x)` to frozen zero-shot features `f_0(x)` on the same batch. Set `--reg_lambda` to weight them; `0` disables regularization.

| `--reg_type`        | What it penalizes                                                                 | Extra flags                                  |
| ------------------- | --------------------------------------------------------------------------------- | -------------------------------------------- |
| `mse`               | L2 between current and zero-shot features                                         | `--normalize` (unit-normalize first)         |
| `l1`                | L1 between current and zero-shot features                                         | `--normalize`                                |
| `nuclear`           | Nuclear norm of the feature-difference matrix over the batch                      | —                                            |
| `quadratic_nuclear` | Squared hinge on the nuclear norm (margin `c`)                                    | `--reg_constraint_c`                         |
| `ldifs`             | Layerwise feature distillation across CLIP transformer blocks                     | (uses `ImageEncoderAugmented` automatically) |
| `pca`               | Sparsity + residual error in a precomputed PCA basis of zero-shot features        | `--pca_path`, `--lambda_l1`, `--mu`          |
| `sae`               | Sparsity + residual error through a precomputed Top-K sparse autoencoder          | `--sae_path`, `--sae_mult`, `--k`, `--lambda_l1`, `--mu` |
| `sae_w`             | `sae` + Sinkhorn–Wasserstein between SAE codes (decoder-correlation cost matrix)  | `sae` flags + `--lambda_wass`                |
| `sae_kl`            | `sae` + KL between normalized SAE code distributions                              | `sae` flags + `--lambda_kl`                  |
| `sae_mask`          | Penalizes activating dictionary atoms that were inactive in the zero-shot code    | `sae` flags + `--lambda_mask`                |
| `sae_add_remove`    | `sae_mask` plus a one-sided penalty on removing previously active atoms           | `sae_mask` flags + `--lambda_remove`         |

The `sae` and `pca` variants require a precomputed autoencoder / basis. These can be generated with `src/testing_models.py`:

```bash
# Store training-set representations
python src/testing_models.py \
    --load_zeroshot=models/<run>/zeroshot.pt \
    --load_finetuned=models/<run>/finetuned \
    --eval-datasets=ImageNet --index=0 \
    --evaluation=store_representations

# Train a Top-K SAE on those representations
python src/testing_models.py \
    --eval-datasets=ImageNet --prefix=B16_ --name=<run> \
    --index=0 --sae_mult=4 --k=32 \
    --evaluation=autoencoder

# Or compute a PCA basis
python src/testing_models.py \
    --eval-datasets=ImageNet --prefix=B16_ --name=<run> \
    --index=0 --evaluation=pca_basis
```

## Evaluation

After training, evaluate a single checkpoint with `src/testing_models.py`:

```bash
python src/testing_models.py \
    --load_zeroshot=models/<run>/zeroshot.pt \
    --load_finetuned=models/<run>/finetuned \
    --eval-datasets=ImageNetR \
    --data-location=/path/to/imagenet-r \
    --index=10 --metric=accuracy \
    --evaluation=evaluate
```

`--evaluation` supports `evaluate`, `store_representations`, `store_evaluate`, `autoencoder`, and `pca_basis`. `--metric` supports `accuracy`, `f1`, and `worst_region` (for WILDS-style datasets with `region` metadata).

The WiSE-FT weight-space interpolation between zero-shot and fine-tuned checkpoints is selected with `--wise_ft=True` (uses α=0.5 by default).

## Profiling

`src/profile_training.py` measures per-step time and peak GPU memory of standard FT vs. SAE-mask FT on the same args:

```bash
python src/profile_training.py \
    --train-dataset=ImageNet --model=ViT-B/16 \
    --template=openai_imagenet_template --batch-size=32 \
    --data-location=/path/to/imagenet \
    --sae_path=autoencoders/sae_4_32.pt --sae_mult=4 --k=32
```

## Repository layout

```
src/
  training.py            # main fine-tuning entry point
  testing_models.py      # evaluation, representation dumping, SAE/PCA training
  profile_training.py    # step-time / memory benchmark
  representation.py      # encoding utilities used by testing_models
  args.py                # CLI flags
  utils.py
  datasets/              # dataset loaders (ImageNet variants, WILDS, CIFAR, DTD, ...)
  models/
    modeling.py          # ImageEncoder, ImageClassifier, ImageEncoderAugmented (for LDIFS)
    zeroshot.py          # builds zero-shot classifier head from CLIP text embeddings
    sparse_autoencoder.py# Top-K / Batch-Top-K SAE
    eval.py
    utils.py
  templates/             # text-prompt templates for zero-shot classification
clip/                    # vendored OpenAI CLIP
utils/                   # WILDS dataset download helper
```

## Acknowledgements

The fine-tuning loop and dataset infrastructure are derived from the [official WiSE-FT repository](https://github.com/mlfoundations/wise-ft). The CLIP code in [`clip/`](clip) is the original [OpenAI CLIP](https://github.com/openai/CLIP).

