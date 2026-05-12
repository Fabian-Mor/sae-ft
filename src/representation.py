import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import gc
from itertools import islice
import time


from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models import utils
from src.utils import *
import src.models.sparse_autoencoder as sa

def get_models(args, indices):
    models = []
    for i in indices:
        if i == 0:
            pass
        else:
            models.append(load_checkpoint(f'{args.load_finetuned}/checkpoint_{i}.pt').to(args.device))
    if 0 in indices and len(models) > 1:
        zeroshot_model = load_checkpoint(args.load_zeroshot)
        zeroshot = copy.deepcopy(models[0])
        zeroshot.load_state_dict(zeroshot_model.state_dict())
        zeroshot.to(args.device)
        models.insert(0, zeroshot)
    elif 0 in indices:
        zeroshot_model = load_checkpoint(args.load_zeroshot)
        #model = load_checkpoint(f'{args.load_finetuned}/checkpoint_{1}.pt')
        zeroshot = copy.deepcopy(zeroshot_model)
        zeroshot.load_state_dict(zeroshot_model.state_dict())
        zeroshot.to(args.device)
        models.insert(0, zeroshot)

    return models

def get_encodings_indices(args, indices, vit=False, save_path=None, is_train=False, feature_dim=512):
    models = get_models(args, indices)
    dataset = get_first_dataset(models[0], args)
    dataloader = get_dataloader(dataset, is_train=is_train, args=args, image_encoder=None)
    encodings = []
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    for index, ckpt in zip(indices, models):
        path = save_path + f'{index}.dat'
        print(f'writing to {path}')
        encodings.append(get_encodings(ckpt, dataloader, args.device, vit=vit, save_path=path, feature_dim=feature_dim))
    return models, encodings


def get_encodings(model, dataloader, device, batch_number=None, vit=False, save_path=None, feature_dim=512):
    start_time = time.time()
    if vit:
        encoder = copy.deepcopy(model.image_encoder.model.visual)
        encoder.proj = None
        feature_dim = 768
    else:
        encoder = model.image_encoder
    print('got encoder')
    encoder.eval()
    batched_data = enumerate(dataloader)
    num_samples = len(dataloader.dataset)
    if save_path is None:
        save_path = 'encodings.dat'

    file_exists = os.path.exists(save_path)

    if file_exists:
        encodings = np.memmap(save_path, dtype='float32', mode='r+', shape=(num_samples, feature_dim))
        written_mask = np.any(encodings != 0, axis=1)
        start = int(written_mask.sum())
        print(f"Resuming from {start}/{num_samples} samples")
    else:
        encodings = np.memmap(save_path, dtype='float32', mode='w+', shape=(num_samples, feature_dim))
        start = 0
        print(f"Starting fresh file with shape {(num_samples, feature_dim)}")

    print('set up: ', time.time() - start_time)

    if batch_number is None:
        start_time = time.time()
        start = 0
        with torch.no_grad():
            # skip already-processed batches
            batched_data = islice(batched_data, start // dataloader.batch_size, None)
            for i, data in batched_data:
                data = maybe_dictionarize(data)
                x = data['images'].to(device)
                enc = utils.get_logits(x, encoder)
                end = start + enc.size(0)
                encodings[start:end] = enc.cpu().numpy()
                start = end
        print('evaluations', time.time() - start_time)
        encodings.flush()
    else:
        with torch.no_grad():
            batch_number = batch_number % len(dataloader)
            batch = next(islice(dataloader, batch_number, None))
            data = maybe_dictionarize(batch)
            x = data['images'].to(device)
            print(x.shape)
            encodings = utils.get_logits(x, encoder)
            print(encodings.shape)
    start_time = time.time()
    print('loading encodings again')
    encodings = torch.from_numpy(
        np.memmap(save_path, dtype='float32', mode='r', shape=(num_samples, feature_dim)))
    print('loading done', time.time() - start_time)
    return encodings


def logits_from_encodings(encodings, model, device, save_path=None):
    classifier = model.classification_head
    encodings = encodings.to(device)
    encodings = F.normalize(encodings, p=2, dim=-1)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        W = classifier.weight.detach().cpu().numpy()
        b = classifier.bias.detach().cpu().numpy()
        np.savez(save_path, W=W, b=b)

        W_torch = torch.from_numpy(W).to(device)
        b_torch = torch.from_numpy(b).to(device)
        logits = encodings @ W_torch.T + b_torch
    else:
        classifier.to(device)
        logits = classifier(encodings)
    return logits


def center_gram(gram, unbiased=False):
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram

def cka(x, y):
    # Code for this and center_gram taken from:
    # https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=MkucRi3yn7UJ
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    x = x.T
    y = y.T
    gram_x = x.dot(x.T)
    gram_y = y.dot(y.T)
    gram_x = center_gram(gram_x)
    gram_y = center_gram(gram_y)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)

def _sym_decorrelation(B):
    """Return B * (B^T B)^{-1/2} to orthonormalize columns of B (d x d)."""
    # eigen-decompose the symmetric matrix B^T B
    s, u = np.linalg.eigh(B.T @ B)
    # numerical safeguard
    s[s < 1e-15] = 1e-15
    inv_sqrt = np.diag(1.0 / np.sqrt(s))
    return B @ (u @ inv_sqrt @ u.T)


def fastica_kurtosis(X, tol=1e-6, max_iter=200, random_state=None, whiten=True, verbose=False):
    """
    Symmetric FastICA using kurtosis contrast (g(u)=u^3).
    Returns:
        S: n x d matrix of estimated independent components (columns = components)
        B: d x d unmixing matrix acting on whitened data (S = Z @ B)
        whitening: d x d whitening matrix applied to centered X (None if whiten=False)
        X_mean: 1 x d mean used for centering
    Parameters:
        X : (n, d) data matrix (rows = samples)
        tol : convergence tolerance
        max_iter : maximum iterations
        random_state : int or None
        whiten : whether to whiten data first (recommended)
        verbose : print convergence info
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    X_mean = X.mean(axis=0)
    Xc = X - X_mean

    # Whitening
    whitening = np.eye(d)
    if whiten:
        cov = (Xc.T @ Xc) / n
        # eigen-decompose covariance
        eigvals, eigvecs = np.linalg.eigh(cov)
        # keep all components but guard tiny eigenvalues
        eigvals[eigvals < 1e-15] = 1e-15
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        whitening = eigvecs @ D_inv_sqrt @ eigvecs.T
        Z = Xc @ whitening  # n x d
    else:
        Z = Xc

    rng = np.random.default_rng(random_state)
    B = rng.normal(size=(d, d))
    B = _sym_decorrelation(B)

    for it in range(max_iter):
        Y = Z @ B                # (n x d)  each column is a component time series
        G = Y ** 3               # (n x d)  kurtosis nonlinearity elementwise
        B_new = (Z.T @ G) / n - 3.0 * B  # (d x d), vectorized update
        B_new = _sym_decorrelation(B_new)

        # convergence: max | |b_new_i^T b_old_i| - 1 |
        # diagonal of B_new.T @ B contains the cosines between corresponding columns
        diag_dot = np.abs(np.diag(B_new.T @ B))
        err = np.max(np.abs(diag_dot - 1.0))

        if verbose:
            print(f"iter {it:3d}  err={err:.3e}")

        B = B_new
        if err < tol:
            break

    S = Z @ B  # n x d, estimated source signals (columns are components)
    return S, B, whitening, X_mean


def kurtosis_objective(B, h):
    """
    Calculates your custom kurtosis objective.
    Kurtosis(Bh) = mean_{n} [ mean_{i} [z_ni^4] ]
    """
    # Project the data
    # h is (N, d_H), B is (d_H, d_H)
    # y = h @ B.T results in y being (N, d_H)
    y = h @ B.T

    # Standardize each sample (row)
    # Add epsilon to prevent division by zero for constant rows
    mean_y = np.mean(y, axis=1, keepdims=True)
    std_y = np.std(y, axis=1, keepdims=True) + 1e-8
    z = (y - mean_y) / std_y

    # Calculate fourth moment for each sample
    kurtosis_per_sample = np.mean(z ** 4, axis=1)

    # Return the average across all samples
    return np.mean(kurtosis_per_sample)


def kurtosis_gradient(B, h):
    """
    Calculates the Euclidean gradient of the objective with respect to B.
    ∇_B J = (1/N) * H^T @ G
    """
    N, d_H = h.shape

    # Project the data
    y = h @ B.T

    # Standardize each sample (row)
    mean_y = np.mean(y, axis=1, keepdims=True)
    std_y = np.std(y, axis=1, keepdims=True) + 1e-8
    z = (y - mean_y) / std_y

    # Pre-calculate powers and sums for the gradient expression
    z3 = z ** 3
    z4 = z ** 4

    # Sum of z^3 and z^4 for each sample (row)
    S3 = np.sum(z3, axis=1, keepdims=True)
    S4 = np.sum(z4, axis=1, keepdims=True)

    # This is the gradient of the objective w.r.t. y (g_n in the derivation)
    # The term is ∇_y J = (4 / (d_H * σ)) * [ z^3 - (S3/d_H) - (z*S4/d_H) ]
    grad_wrt_y = (4 / (d_H * std_y)) * (z3 - (S3 / d_H) - z * (S4 / d_H))

    # The final Euclidean gradient w.r.t. B is (1/N) * h.T @ grad_wrt_y
    # In our formulation y=hB^T, so ∇_B J = (1/N) * grad_wrt_y.T @ h
    euclidean_grad = (1.0 / N) * grad_wrt_y.T @ h

    return euclidean_grad


def maximize_kurtosis_projection(h, n_iter=100, learning_rate=0.1):
    """
    Finds the orthogonal matrix B that maximizes the custom kurtosis objective
    using gradient ascent on the Stiefel manifold.
    """
    N, d_H = h.shape

    # 1. Initialize B with a random orthogonal matrix
    B = np.eye(d_H) #np.linalg.qr(np.random.randn(d_H, d_H))[0]

    history = []

    print("Starting optimization...")
    initial_kurtosis = kurtosis_objective(B, h)
    print(f"Initial Kurtosis: {initial_kurtosis:.4f}")
    history.append(initial_kurtosis)

    # 2. Iterate using gradient ascent
    for i in range(n_iter):
        # Calculate the Euclidean gradient
        grad = kurtosis_gradient(B, h)

        # Update B in the direction of the gradient
        B_updated = B + learning_rate * grad

        # 3. Retraction step: Use QR decomposition to pull the updated matrix
        # back onto the manifold of orthogonal matrices.
        B, _ = np.linalg.qr(B_updated)

        # Optional: Log progress
        if (i + 1) % 100 == 0:
            current_kurtosis = kurtosis_objective(B, h)
            history.append(current_kurtosis)
            print(f"Iteration {i + 1}/{n_iter}, Kurtosis: {current_kurtosis:.4f}")

    final_kurtosis = kurtosis_objective(B, h)
    print(f"\nOptimization finished. Final Kurtosis: {final_kurtosis:.4f}")

    return B, history

def compare_representations(model1, model2, args):
    dataset = get_first_dataset(model1, args)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    encodings1 = get_encodings(model1, dataloader, args.device)
    print(encodings1.shape)
    encodings2 = get_encodings(model2, dataloader, args.device)
    print(encodings2.shape)
    return cka(encodings1, encodings2)

def kurtosis(encodings):
    std, mean = torch.std_mean(encodings, dim=1, keepdim=True)
    z = (encodings - mean) / std
    acc = (z ** 4).mean()
    return acc

def kurtosis_distributions(encodings):
    std, mean = torch.std_mean(encodings, dim=1, keepdim=True)
    z = (encodings - mean) / std
    z = z **4
    print(z.shape)


def train_sparse_autoencoder(encodings, args, epochs=100, batch_size=1024, init_sae=None, lr=1e-3, k=32, size_mult=4):
    encodings = encodings.cpu()
    enc_dataset = TensorDataset(encodings)
    enc_dataloader = DataLoader(enc_dataset, batch_size=batch_size, shuffle=True)
    size = encodings.shape[1]
    if init_sae is None:
        autoencoder = sa.SparseAutoencoderTopK(size, int(size_mult * size), int(k), batch_top_k=False)
    else:
        autoencoder = init_sae
    autoencoder.to(args.device)
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0
        for (batch,) in enc_dataloader:
            batch = batch.to(args.device)

            loss = sa.calculate_loss_top_k(autoencoder, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)
            total_samples += batch.size(0)

        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

    return autoencoder

