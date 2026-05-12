"""
Profile overhead of SAE-FT (sae_mask) vs standard fine-tuning.
Measures per-step time, peak GPU memory, and SAE storage cost.

Usage — run twice with different args and compare:
  # Standard fine-tuning (no reg)
  python profile_training.py --reg-lambda 0 [your other args]

  # SAE-FT (sae_mask)
  python profile_training.py --reg-lambda 1.0 --reg-type sae_mask --sae-path <path> [your other args]

Or just run this script directly — it profiles both settings back-to-back
using the same args (overriding reg_lambda and reg_type internally).
"""

import os
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F

from src.args import parse_arguments
from src.models.modeling import ImageEncoder, ImageClassifier
from src.models.zeroshot import get_zeroshot_classifier
from src.models.sparse_autoencoder import SparseAutoencoderTopK
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.utils import cosine_lr, LabelSmoothing
import src.datasets as datasets


def bytes_to_mb(b):
    return b / (1024 ** 2)


def profile_single_config(args, reg_mode="none", num_warmup=5, num_profile=50):
    """
    Run a number of training steps and measure time + memory.

    Args:
        args: parsed arguments (should have all model/data args set)
        reg_mode: "none" (standard FT), "l2" (MSE reg), or "sae_mask"
        num_warmup: warmup steps (not measured)
        num_profile: steps to measure

    Returns:
        dict with profiling results
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- Setup model ---
    image_encoder = ImageEncoder(args, keep_lang=True)
    classification_head = get_zeroshot_classifier(args, image_encoder.model)
    if hasattr(image_encoder.model, 'transformer'):
        del image_encoder.model.transformer
    elif hasattr(image_encoder.model, 'text'):
        del image_encoder.model.text

    classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
    zeroshot = copy.deepcopy(classifier)

    model = classifier.cuda()
    devices = list(range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    # --- Setup dataset ---
    preprocess_fn = classifier.train_preprocess
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(preprocess_fn, location=args.data_location, batch_size=args.batch_size)
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)

    # --- Setup loss + optimizer ---
    loss_fn = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # --- Setup regularization ---
    zeroshot_frozen_encoder = None
    sae = None

    if reg_mode in ("l2", "sae_mask"):
        zeroshot_frozen_encoder = copy.deepcopy(zeroshot.image_encoder).cuda().eval()

    if reg_mode == "sae_mask":
        sae = SparseAutoencoderTopK(
            int(args.representation_dim),
            int(args.sae_mult * args.representation_dim),
            int(args.k),
            batch_top_k=False
        )
        sae.load_state_dict(torch.load(args.sae_path))
        sae = sae.to(args.device)
        sae.eval()

    # Memory after setup (before any forward pass)
    torch.cuda.synchronize()
    mem_after_setup = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    # --- Profiling loop ---
    step_times = []
    data_iter = iter(data_loader)

    for step in range(num_warmup + num_profile):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        batch = maybe_dictionarize(batch)
        inputs = batch['images'].cuda()
        labels = batch['labels'].cuda()

        torch.cuda.synchronize()
        t0 = time.time()

        optimizer.zero_grad()
        image_features = model.module.image_encoder(inputs)
        logits = model.module.classification_head(image_features)
        total_loss = loss_fn(logits, labels)

        if reg_mode in ("l2", "sae_mask") and zeroshot_frozen_encoder is not None:
            with torch.no_grad():
                old_features = zeroshot_frozen_encoder(inputs)

            if args.normalize:
                target_norm = old_features.norm(dim=-1, keepdim=True) + 1e-8
                old_features = (old_features / target_norm).detach()
                current_norm = image_features.norm(dim=-1, keepdim=True) + 1e-8
                image_features = image_features / current_norm

            if reg_mode == "l2":
                representation_loss = F.mse_loss(image_features, old_features)
                total_loss += args.reg_lambda * representation_loss

            elif reg_mode == "sae_mask":
                with torch.no_grad():
                    s_0 = sae.encode(old_features)
                s_theta = sae.encode(image_features)
                lambda_mask = getattr(args, "lambda_mask", 10.0)
                mask = (s_0 != 0).float()
                l_mask = lambda_mask * ((1 - mask) * s_theta.abs()).mean()

                z_recon_0 = sae.decode(s_0)
                z_recon_theta = sae.decode(s_theta)
                delta_z = image_features - old_features
                delta_z_recon = z_recon_theta - z_recon_0
                residual = delta_z - delta_z_recon
                mu = getattr(args, "mu", 1.0)
                if args.normalize:
                    mu = getattr(args, "mu", 100.0)
                l_resid = mu * residual.pow(2).mean()
                representation_loss = l_mask + l_resid
                total_loss += args.reg_lambda * representation_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.time()

        if step >= num_warmup:
            step_times.append(t1 - t0)

    peak_mem = torch.cuda.max_memory_allocated()

    # SAE checkpoint size
    sae_size_mb = 0.0
    if reg_mode == "sae_mask" and os.path.exists(args.sae_path):
        sae_size_mb = bytes_to_mb(os.path.getsize(args.sae_path))

    # Cleanup
    del model, optimizer, sae, zeroshot_frozen_encoder, classifier, zeroshot
    torch.cuda.empty_cache()

    return {
        'mean_step_time_ms': np.mean(step_times) * 1000,
        'std_step_time_ms': np.std(step_times) * 1000,
        'median_step_time_ms': np.median(step_times) * 1000,
        'peak_gpu_mb': bytes_to_mb(peak_mem),
        'setup_gpu_mb': bytes_to_mb(mem_after_setup),
        'sae_checkpoint_mb': sae_size_mb,
        'num_steps_profiled': len(step_times),
    }


def main():
    args = parse_arguments()

    configs = [
        ("Standard FT", "none"),
        ("L2 reg", "l2"),
        ("SAE-FT (mask)", "sae_mask"),
    ]

    results = {}
    for label, mode in configs:
        print("\n" + "=" * 70)
        print(f"Profiling: {label}")
        print("=" * 70)

        if mode == "sae_mask":
            assert args.sae_path is not None, "Provide --sae-path for sae_mask profiling"

        # Ensure reg_lambda is set for regularized runs
        orig_reg_lambda = args.reg_lambda
        if mode != "none" and args.reg_lambda == 0:
            args.reg_lambda = 1.0
            print(f"  (reg_lambda was 0, setting to {args.reg_lambda} for profiling)")

        results[label] = profile_single_config(args, reg_mode=mode)
        args.reg_lambda = orig_reg_lambda

    # --- Print comparison ---
    labels = [l for l, _ in configs]
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    header = f"{'Metric':<30}" + "".join(f"{l:>15}" for l in labels) + f"{'SAE vs L2':>15}"
    print(header)
    print("-" * len(header))

    times = [results[l]['mean_step_time_ms'] for l in labels]
    row = f"{'Step time (ms)':<30}" + "".join(f"{t:>15.1f}" for t in times)
    row += f"{(times[2]/times[1] - 1)*100:>14.1f}%"
    print(row)

    mems = [results[l]['peak_gpu_mb'] for l in labels]
    row = f"{'Peak GPU memory (MB)':<30}" + "".join(f"{m:>15.1f}" for m in mems)
    row += f"{mems[2] - mems[1]:>14.1f}MB"
    print(row)

    setups = [results[l]['setup_gpu_mb'] for l in labels]
    row = f"{'GPU after setup (MB)':<30}" + "".join(f"{s:>15.1f}" for s in setups)
    row += f"{setups[2] - setups[1]:>14.1f}MB"
    print(row)

    sae_mb = results['SAE-FT (mask)']['sae_checkpoint_mb']
    print(f"{'SAE checkpoint (MB)':<30}{'N/A':>15}{'N/A':>15}{sae_mb:>15.1f}")

    print(f"\n{'Step time std (ms)':<30}" + "".join(f"{results[l]['std_step_time_ms']:>15.1f}" for l in labels))
    print(f"{'Steps profiled':<30}" + "".join(f"{results[l]['num_steps_profiled']:>15}" for l in labels))


if __name__ == '__main__':
    main()
