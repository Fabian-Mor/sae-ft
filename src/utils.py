import os
import torch
import torch.nn.functional as F
import src.datasets as datasets
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models import utils
from contextlib import redirect_stdout
import copy
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
import numpy as np

def get_first_dataset(model, args):
    dataset_class = getattr(datasets, args.eval_datasets[0])
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        if args.corruption is not None:
            dataset = dataset_class(
                model.val_preprocess,
                corruption=args.corruption,
                batch_size=args.batch_size
            )
        else:
            dataset = dataset_class(
                model.val_preprocess,
                batch_size=args.batch_size
            )
    return dataset


def get_param_vector(model):
    return torch.cat([param.view(-1) for param in model.parameters()])

def get_angle(vec_base, vec_1, vec_2):
    """
    Computes the angel between the parameters as defined in the paper "Landscaping Linear Mode Connectivity".
    Args:
        vec_base: the parameters as a vector, relative to which the angle is computed.
        vec_1: parameters of the first model as a vector.
        vec_2: parameters of the second model as a vector.

    Returns:
        the angles in radians and degrees.
    """
    delta_1 = vec_1 - vec_base
    delta_2 = vec_2 - vec_base
    print(torch.sum(delta_1), torch.sum(delta_2))
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(delta_1, delta_2, dim=0, eps=1e-8)  # shape: scalar
    # Clamp to avoid NaNs from numerical issues
    # cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    # Compute angle in radians
    angle_rad = torch.acos(cos_sim)
    # Optionally convert to degrees
    angle_deg = torch.rad2deg(angle_rad)
    return angle_rad, angle_deg


def eval_single_loss(image_classifier, dataset, args, train_data=False):
    model = image_classifier
    input_key = 'images'
    image_enc = None

    model.eval()
    dataloader = get_dataloader(
        dataset, is_train=train_data, args=args, image_encoder=image_enc)
    batched_data = enumerate(dataloader)
    device = args.device

    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in batched_data:
            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            total_loss += F.cross_entropy(logits, y, reduction='sum').item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def compute_loss(image_classifier, args, train_data=False):
    if args.eval_datasets is None:
        return
    losses = []
    accuracies = []
    for i, dataset_name in enumerate(args.eval_datasets):
        dataset_class = getattr(datasets, dataset_name)
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            if args.corruption is not None:
                dataset = dataset_class(
                    image_classifier.val_preprocess,
                    corruption=args.corruption,
                    location=args.data_location,
                    batch_size=args.batch_size
                )
            else:
                dataset = dataset_class(
                    image_classifier.val_preprocess,
                    location=args.data_location,
                    batch_size=args.batch_size
                )
        avg_loss, accuracy = eval_single_loss(image_classifier, dataset, args, train_data=train_data)
        losses.append(avg_loss)
        accuracies.append(accuracy)
    return losses, accuracies


def load_checkpoint(path):
    return ImageClassifier.load(path)


def interpolate_models(model_a, model_b, alphas):
    theta_0 = model_a.state_dict()
    theta_1 = model_b.state_dict()
    models = []
    vecs = []

    for alpha in alphas:
        theta_interp = {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0
        }
        model = copy.deepcopy(model_b)
        model.load_state_dict(theta_interp)
        models.append(model)
        vecs.append(get_param_vector(model))

    return models, vecs

def interpolate_more_models(models, alphas=None):
    if alphas is None:
        alphas = [1/len(models) for _ in range(len(models))]
    model_thetas = [model.state_dict() for model in models]
    theta_interp = {
        key: val.clone().mul_(alphas[0]) for key, val in model_thetas[0].items()
    }

    # Accumulate other state dicts
    for i, theta in enumerate(model_thetas[1:]):
        for key in theta_interp:
            theta_interp[key] += alphas[i] * theta[key]
    model = copy.deepcopy(models[0])
    model.load_state_dict(theta_interp)
    return model


def evaluate_models(models, args):
    losses, accs = [], []
    for model in models:
        loss, acc = compute_loss(model, args, train_data=False)
        losses.append(loss[0])
        accs.append(acc[0])
    return losses, accs


def explore_between_models(model_a, model_b, interpolations, args):
    alphas = np.linspace(0, 1, interpolations) if interpolations > 1 else [0.5]
    models, vecs = interpolate_models(model_a, model_b, alphas)
    losses, accs = evaluate_models(models, args)
    return vecs, losses, accs