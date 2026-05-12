import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from src.models.modeling import ImageEncoder, ImageClassifier, ImageEncoderAugmented
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments
import src.datasets as datasets
import time
import copy
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.utils import cosine_lr, LabelSmoothing
from src.models.sparse_autoencoder import SparseAutoencoderTopK
from torch.optim.lr_scheduler import _LRScheduler

class PCAWrapper:
    def __init__(self, basis_path, device='cuda'):
        self.basis = torch.load(basis_path, map_location=device)
        self.basis.requires_grad_(False)

    def encode(self, x):
        return x @ self.basis

    def decode(self, s):
        return s @ self.basis.T


def sinkhorn_wasserstein(p0, p1, cost_matrix, epsilon=0.05, n_iters=50):
    """
    Differentiable Sinkhorn distance between two batches of distributions p0, p1.
    p0, p1: (batch, n)
    cost_matrix: (n, n) – pairwise cost between SAE features
    epsilon: entropic regularization coefficient
    n_iters: number of Sinkhorn iterations
    """
    p0 = p0 / (p0.sum(dim=-1, keepdim=True) + 1e-8)
    p1 = p1 / (p1.sum(dim=-1, keepdim=True) + 1e-8)
    K = torch.exp(-cost_matrix / epsilon)  # (n, n)
    K = K / (K.sum(dim=-1, keepdim=True) + 1e-8)
    u = torch.ones_like(p0)
    v = torch.ones_like(p1)
    for _ in range(n_iters):
        u = p0 / (K @ v.unsqueeze(-1)).squeeze(-1)
        v = p1 / (K.T @ u.unsqueeze(-1)).squeeze(-1)
    T = u.unsqueeze(-1) * K * v.unsqueeze(-2)
    wasserstein = torch.sum(T * cost_matrix, dim=(-2, -1))
    return wasserstein


class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, start_lr=1e-5, end_lr=1e-4, total_epochs=10, last_epoch=-1):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.total_epochs // 2:
            lr = self.start_lr
        else:
            lr = self.end_lr
        return [lr for _ in self.optimizer.param_groups]


def finetuning(zeroshot, args):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    assert args.train_dataset is not None, "Please provide a training dataset."
    torch.cuda.empty_cache()
    image_classifier = ImageClassifier.load(args.load)

    model = image_classifier
    input_key = 'images'
    preprocess_fn = image_classifier.train_preprocess
    image_enc = None
    image_classifier.process_images = True
    print_every = 100

    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    if args.num_splits > 1:
        total_len = len(dataset.train_loader.dataset)
        indices = np.arange(total_len)
        np.random.seed(0)
        np.random.shuffle(indices)

        split_size = total_len // args.num_splits
        start = args.split_index * split_size
        end = (args.split_index + 1) * split_size if args.split_index < args.num_splits - 1 else total_len

        split_indices = indices[start:end]

        dataset.train_loader = torch.utils.data.DataLoader(
            Subset(dataset.train_loader.dataset, split_indices),
            batch_size=dataset.train_loader.batch_size,
            shuffle=True,
            num_workers=dataset.train_loader.num_workers,
            pin_memory=True
        )

    if args.train_dataset == 'dSprites':
        print("For dSprites, an 'epoch' is 1/10th of the data. Adjusting epochs accordingly.")
        args.epochs *= 10

        train_dataset_obj = dataset.train_loader.dataset
        if isinstance(train_dataset_obj, Subset):
            base_dataset = train_dataset_obj.dataset
            available_indices = np.array(train_dataset_obj.indices)
        else:
            base_dataset = train_dataset_obj
            available_indices = np.arange(len(base_dataset))

        snapshot_size = len(available_indices) // 10
        assert snapshot_size > 0, "Dataset is too small to be split into 10 parts per epoch."
        original_loader = dataset.train_loader

    num_batches = len(dataset.train_loader)

    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    if args.ls > 0:
        print(f'using label smoothing: {args.ls}')
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    if "sae" in args.reg_type:
        #sae_path = os.path.join(args.sae_path)
        sae = SparseAutoencoderTopK(int(args.representation_dim), int(args.sae_mult * args.representation_dim), int(args.k), batch_top_k=False)
        sae.load_state_dict(torch.load(args.sae_path))
        sae = sae.to(args.device)
        print(args.k)
        print(args.sae_mult)
        print(sae)
    elif "pca" in args.reg_type:
        pca_path = os.path.join(args.pca_path)
        pca = PCAWrapper(pca_path)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    if args.scheduler:
        scheduler = CustomScheduler(optimizer, start_lr=args.lr, total_epochs=args.epochs)
    else:
        total_steps_10_epoch = 10 * num_batches
        warmup_steps = min(args.warmup_length, total_steps_10_epoch)
        cosine_scheduler = cosine_lr(optimizer, args.lr, warmup_steps, total_steps_10_epoch)
        def combined_scheduler(step):
            if step < total_steps_10_epoch:
                cosine_scheduler(step)
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
        scheduler = combined_scheduler

    device = next(image_classifier.parameters()).device

    zeroshot_frozen_encoder = None
    if args.reg_lambda > 0:
        print(f"Enabling representation regularization of type {args.reg_type} with lambda={args.reg_lambda}")
        zeroshot_frozen_encoder = copy.deepcopy(zeroshot.image_encoder).cuda().eval()

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        model.to(device)

        if args.train_dataset == 'dSprites':
            current_indices = np.random.choice(available_indices, snapshot_size, replace=False)
            subset_for_epoch = Subset(base_dataset, current_indices)
            dataset.train_loader = torch.utils.data.DataLoader(
                subset_for_epoch,
                batch_size=original_loader.batch_size,
                shuffle=True,
                num_workers=original_loader.num_workers,
                pin_memory=True
            )

        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=image_enc)

        for i, batch in enumerate(data_loader):
            if not args.scheduler:
                step = i + epoch * num_batches
                scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time

            image_features = model.module.image_encoder(inputs)
            logits = model.module.classification_head(image_features)

            total_loss = loss_fn(logits, labels)
            representation_loss = None

            if args.reg_lambda > 0 and zeroshot_frozen_encoder is not None:
                with torch.no_grad():
                    old_features = zeroshot_frozen_encoder(inputs)

                if args.normalize:
                    target_norm = old_features.norm(dim=-1, keepdim=True) + 1e-8
                    old_features = (old_features / target_norm).detach()
                    current_norm = image_features.norm(dim=-1, keepdim=True) + 1e-8
                    image_features = image_features / current_norm

                if args.reg_type == "mse":
                    representation_loss = F.mse_loss(image_features, old_features)
                elif args.reg_type == "l1":
                    representation_loss = F.l1_loss(image_features, old_features)
                elif args.reg_type == "nuclear":
                    diff = image_features - old_features
                    representation_loss = torch.linalg.norm(diff, ord='nuc')
                elif args.reg_type == "quadratic_nuclear":
                    diff = image_features - old_features
                    nuclear_norm = torch.linalg.norm(diff, ord='nuc')
                    violation = F.relu(nuclear_norm - args.reg_constraint_c)
                    representation_loss = (violation ** 2)
                elif args.reg_type == "pca":
                    with torch.no_grad():
                        s_0 = pca.encode(old_features)
                    s_theta = pca.encode(image_features)
                    delta_s = s_theta - s_0
                    lambda_l1 = args.lambda_l1
                    l_sparse = lambda_l1 * delta_s.abs().mean(dim=-1).mean()
                    z_recon_0 = pca.decode(s_0)
                    z_recon_theta = pca.decode(s_theta)
                    delta_z = (image_features - old_features)
                    delta_z_recon = (z_recon_theta - z_recon_0)
                    residual = delta_z - delta_z_recon
                    mu = getattr(args, "mu", 1.0)
                    if args.normalize:
                        mu = getattr(args, "mu", 100.0)
                    l_resid = mu * residual.pow(2).mean()
                    representation_loss = l_sparse + l_resid

                elif args.reg_type == "ldifs":
                    with torch.no_grad():
                        old_features_list = zeroshot_frozen_encoder.get_features(inputs)
                    current_features_list = model.module.image_encoder.get_features(inputs)
                    diffs = []
                    for layer_idx in range(len(old_features_list)):
                        old_feat = old_features_list[layer_idx]
                        cur_feat = current_features_list[layer_idx]
                        old_feat_norm = old_feat / (old_feat.norm(dim=-1, keepdim=True) + 1e-8)
                        cur_feat_norm = cur_feat / (cur_feat.norm(dim=-1, keepdim=True) + 1e-8)
                        diff = ((old_feat_norm - cur_feat_norm) ** 2)
                        if len(diff.shape) == 3:
                            diff = diff.mean(dim=-1).mean(dim=-1)
                        elif len(diff.shape) == 2:
                            diff = diff.mean(dim=-1)
                        diffs.append(diff)
                    representation_loss = torch.stack(diffs, dim=1).mean(dim=-1).mean(dim=0)

                elif args.reg_type == "sae":
                    with torch.no_grad():
                        s_0 = sae.encode(old_features)
                    s_theta = sae.encode(image_features)
                    delta_s = s_theta - s_0
                    lambda_l1 = args.lambda_l1
                    l_sparse = lambda_l1 * delta_s.abs().mean(dim=-1).mean()
                    z_recon_0 = sae.decode(s_0)
                    z_recon_theta = sae.decode(s_theta)
                    delta_z = (image_features - old_features)
                    delta_z_recon = (z_recon_theta - z_recon_0)
                    residual = delta_z - delta_z_recon
                    mu = getattr(args, "mu", 1.0)
                    if args.normalize:
                        mu = getattr(args, "mu", 100.0)
                    l_resid = mu * residual.pow(2).mean()
                    representation_loss = l_sparse + l_resid

                elif args.reg_type == "sae_w":
                    with torch.no_grad():
                        s_0 = sae.encode(old_features)
                    s_theta = sae.encode(image_features)
                    delta_s = s_theta - s_0
                    lambda_l1 = getattr(args, "lambda_l1", 1e-3)
                    l_sparse = lambda_l1 * delta_s.abs().mean(dim=-1).mean()
                    z_recon_0 = sae.decode(s_0)
                    z_recon_theta = sae.decode(s_theta)
                    delta_z = image_features - old_features
                    delta_z_recon = z_recon_theta - z_recon_0
                    residual = delta_z - delta_z_recon
                    mu = getattr(args, "mu", 1.0)
                    if args.normalize:
                        mu = getattr(args, "mu", 100.0)
                    l_resid = mu * residual.pow(2).mean()
                    lambda_wass = getattr(args, "lambda_wass", 1e-1)
                    W = F.normalize(sae.decoder.weight, dim=0)

                    C = 1.0 - (W.T @ W)

                    C = C.clamp(min=0)
                    p0 = torch.relu(s_0)
                    ptheta = torch.relu(s_theta)
                    p0 = p0 / (p0.sum(dim=-1, keepdim=True) + 1e-8)
                    ptheta = ptheta / (ptheta.sum(dim=-1, keepdim=True) + 1e-8)
                    l_wass = lambda_wass * sinkhorn_wasserstein(p0, ptheta, C).mean()
                    representation_loss = l_resid + l_wass + l_sparse

                elif args.reg_type == "sae_kl":
                    with torch.no_grad():
                        s_0 = sae.encode(old_features)

                    s_theta = sae.encode(image_features)
                    delta_s = s_theta - s_0
                    lambda_l1 = getattr(args, "lambda_l1", 1e-3)
                    l_sparse = lambda_l1 * delta_s.abs().mean(dim=-1).mean()
                    z_recon_0 = sae.decode(s_0)
                    z_recon_theta = sae.decode(s_theta)
                    delta_z = image_features - old_features
                    delta_z_recon = z_recon_theta - z_recon_0
                    residual = delta_z - delta_z_recon
                    mu = getattr(args, "mu", 1.0)
                    if args.normalize:
                        mu = getattr(args, "mu", 100.0)
                    l_resid = mu * residual.pow(2).mean()
                    p0 = s_0 / (s_0.sum(dim=-1, keepdim=True) + 1e-8)
                    ptheta = s_theta / (s_theta.sum(dim=-1
                                                    , keepdim=True) + 1e-8)
                    lambda_kl = getattr(args, "lambda_kl", 1e-1)
                    kl_div = F.kl_div((ptheta + 1e-8).log(), p0, reduction='batchmean')
                    l_kl = lambda_kl * kl_div
                    representation_loss = l_resid + l_sparse + l_kl


                elif args.reg_type == "sae_mask":
                    with torch.no_grad():
                        s_0 = sae.encode(old_features)
                    s_theta = sae.encode(image_features)
                    lambda_mask = getattr(args, "lambda_mask", 10.0)
                    mask = (s_0 != 0).float()
                    l_mask = lambda_mask * ((1 - mask) * s_theta.abs()).mean()
                    z_recon_0 = sae.decode(s_0)
                    z_recon_theta = sae.decode(s_theta)
                    delta_z = (image_features - old_features)
                    delta_z_recon = (z_recon_theta - z_recon_0)
                    residual = delta_z - delta_z_recon
                    mu = getattr(args, "mu", 1.0)
                    if args.normalize:
                        mu = getattr(args, "mu", 100.0)
                    l_resid = mu * residual.pow(2).mean()
                    representation_loss = l_mask + l_resid


                elif args.reg_type == "sae_add_remove":
                    with torch.no_grad():
                        s_0 = sae.encode(old_features)
                    s_theta = sae.encode(image_features)
                    lambda_mask = getattr(args, "lambda_mask", 10.0)
                    lambda_remove = getattr(args, "lambda_remove", 0.02)
                    mask = (s_0 != 0).float()
                    l_mask = lambda_mask * ((1 - mask) * s_theta.abs()).mean()
                    l_remove = lambda_remove * (mask * torch.relu(s_0 - s_theta)).mean()
                    z_recon_0 = sae.decode(s_0)
                    z_recon_theta = sae.decode(s_theta)
                    delta_z = image_features - old_features
                    delta_z_recon = z_recon_theta - z_recon_0
                    residual = delta_z - delta_z_recon
                    mu = getattr(args, "mu", 1.0)
                    if args.normalize:
                        mu = getattr(args, "mu", 100.0)
                    l_resid = mu * residual.pow(2).mean()
                    representation_loss = l_mask + l_remove + l_resid

                if representation_loss is not None:
                    total_loss += args.reg_lambda * representation_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if i % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                log_msg = (
                    f"Train Epoch: {epoch} [{percent_complete:.0f}%]\t"
                    f"Total Loss: {total_loss.item():.6f}\t"
                    f"Batch (t) {batch_time:.3f}"
                )
                if representation_loss is not None:
                    log_msg += f"\tReg Loss: {representation_loss.item():.6f}"
                if args.reg_type == "quadratic_nuclear":
                    log_msg += f"\tReg Norm: {nuclear_norm:.3f}, constant: {args.reg_constraint_c:.3f}"
                print(log_msg, flush=True)

        if args.scheduler:
            print(f"[Epoch {epoch}] Scheduler LR: {scheduler.get_last_lr()[0]:.6e}")
            scheduler.step()
            print(f"[Epoch {epoch}] Scheduler LR: {scheduler.get_last_lr()[0]:.6e}")

        print(f'One epoch takes {time.time() - start_time} seconds')
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)

            if (epoch+1) % 10 == 0:
            #if True:
                model_path = os.path.join(args.save, f'checkpoint_{epoch + 1}.pt')
                print('Saving model to', model_path)
                image_classifier.save(model_path)

            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            model_path = os.path.join(args.save, "checkpoint_continue.pt")
            print('Saving model to', model_path)
            torch.save(checkpoint, model_path)

        args.current_epoch = epoch
    torch.cuda.empty_cache()
    if args.save is not None:
        return model_path


def robust_finetuning(args):
    assert args.save is not None, 'Please provide a path to store models'
    if args.reg_type == 'ldifs':
        print("Initializing ImageEncoderAugmented for LDIFS regularization...")
        image_encoder = ImageEncoderAugmented(args, keep_lang=True)
    else:
        image_encoder = ImageEncoder(args, keep_lang=True)
    classification_head = get_zeroshot_classifier(args, image_encoder.model)
    print(classification_head)
    if hasattr(image_encoder.model, 'transformer'):
        del image_encoder.model.transformer
        print("Deleted 'transformer' attribute.")
    elif hasattr(image_encoder.model, 'text'):
        del image_encoder.model.text
        print("Deleted 'text' attribute.")
    classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
    zeroshot_checkpoint = os.path.join(args.save, 'zeroshot.pt')
    classifier.save(zeroshot_checkpoint)
    zeroshot = ImageClassifier.load(zeroshot_checkpoint)
    args.load = zeroshot_checkpoint
    args.save = os.path.join(args.save, 'finetuned')
    finetuning(zeroshot, args)


if __name__ == '__main__':
    args = parse_arguments()
    robust_finetuning(args)