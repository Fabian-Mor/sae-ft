import pickle as pkl
from sklearn.metrics import f1_score
from tqdm import tqdm
from src.args import parse_arguments

from src.representation import *
from src.utils import *


def autoencoder(args):
    dim = args.representation_dim
    encodings_path = f'representations/{args.eval_datasets[0]}/train/{args.prefix}{args.name}/{args.index}.dat'
    mem = torch.from_numpy(
        np.memmap(encodings_path, dtype='float32', mode='r', shape=(1281167, dim))
    )
    encodings = torch.from_numpy(np.array(mem, copy=True))
    start_time = time.time()
    sparse_autoencoder_1 = train_sparse_autoencoder(encodings, args, epochs=100, k=args.k, size_mult=args.sae_mult)
    save_path = os.path.join(f'autoencoders/sae_DTD_{int(args.sae_mult)}_{int(args.k)}.pt')
    torch.save(sparse_autoencoder_1.state_dict(), save_path)
    print(f"finished autoencoder, saved at {save_path}, in {time.time() - start_time} seconds.")


def pca_basis(args, k=16, max_samples=1281167):
    encodings_path = f'representations/{args.eval_datasets[0]}/train/{args.prefix}{args.name}/{args.index}.dat'
    mem = np.memmap(
        encodings_path,
        dtype='float32',
        mode='r',
        shape=(1281167, args.dim)
    )
    N = mem.shape[0]
    idx = np.random.choice(N, size=min(max_samples, N), replace=False)
    encodings = torch.from_numpy(np.array(mem[idx], copy=True))
    encodings = encodings - encodings.mean(dim=0, keepdim=True)
    print(f"Computing PCA on {encodings.shape[0]} samples...")
    U, S, V = torch.pca_lowrank(encodings, q=k)
    U_k = V[:, :k]
    save_path = os.path.join(f'autoencoders/pca_{args.prefix}_k{k}_n{encodings.shape[0]}.pt')
    torch.save(U_k, save_path)
    print(f"Finished PCA, saved at {save_path}")


def store_evaluate(args):
    is_train = False
    wise_ft = False
    prefix = args.prefix
    indices = [args.index]
    dim = args.representation_dim
    split_val = getattr(args, 'split', 'test')

    if is_train:
        subset = 'train'
    else:
        subset = split_val

    path = f'representations/{args.eval_datasets[0]}/{subset}/{prefix}{args.name}/'

    if wise_ft:
        models = get_models(args, [0, 10])
        inter_model = interpolate_models(models[0], models[1], [0.5])[0][0]
        dataset = get_first_dataset(models[0], args)
        if split_val == 'val' and hasattr(dataset, 'val_loader'):
            dataset.test_loader = dataset.val_loader
            dataset.test_dataset = dataset.val_dataset
        dataloader = get_dataloader(dataset, is_train=is_train, args=args, image_encoder=None)
        os.makedirs(path, exist_ok=True)
        path = f'representations/{args.eval_datasets[0]}/{subset}/{prefix}{args.name}/0-10.dat'
        enc = get_encodings(inter_model, dataloader, args.device, save_path=path, feature_dim=dim)
    else:
        models, encodings = get_encodings_indices(args, indices, save_path=path, is_train=is_train, feature_dim=dim)

    dataset_name = args.eval_datasets[0]
    labels_path = f'labels/{dataset_name}/{subset}/labels.npy'
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)

    print("Fetching dataset and label information...")
    sample_model = get_models(args, [0])[0]
    dataset = get_first_dataset(sample_model, args)
    if split_val == 'val' and hasattr(dataset, 'val_loader'):
        dataset.test_loader = dataset.val_loader
        dataset.test_dataset = dataset.val_dataset
    labels = get_all_labels(dataset, args, is_train=is_train)
    np.save(labels_path, labels.cpu().numpy())
    total = labels.size(0)
    print(f"Loaded {total} labels for dataset '{dataset_name}'.")

    for index in indices:
        print(f"\n--- Evaluating index {index} ---")
        encodings_path = f'representations/{dataset_name}/{subset}/{prefix}{args.name}/{index}.dat'
        classifier_path = f'classifier/{dataset_name}/{prefix}{args.name}/{index}.npz'
        encodings = torch.from_numpy(
            np.memmap(encodings_path, dtype='float32', mode='r', shape=(total, dim))
        )
        if wise_ft:
            models = get_models(args, [0, 10])
            model = interpolate_models(models[0], models[1], [0.5])[0][0]
        else:
            model = get_models(args, [index])[0]

        logits = logits_from_encodings(encodings, model, args.device, save_path=classifier_path)

        projection_fn = getattr(dataset, 'project_logits', None)
        if projection_fn is not None:
            logits = projection_fn(logits, args.device)

        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        accuracy = correct / total
        labels_cpu = labels.cpu().numpy()
        preds_cpu = preds.cpu().numpy()
        f1 = f1_score(labels_cpu, preds_cpu, average='macro')
        print(f"Accuracy for index {index}: {accuracy:.4f} ({correct}/{total})")
        print(f"macro F1 Score for index {index}: {f1:.4f}")


def store_representations(args):
    is_train = True
    interpolate = False
    tsa = False
    avg_all = False

    prefix = args.prefix
    indices = [args.index]
    dim = args.representation_dim
    split_val = getattr(args, 'split', 'test')
    start_time = time.time()

    if is_train:
        path = f'representations/{args.eval_datasets[0]}/train/{prefix}{args.name}/'
    else:
        path = f'representations/{args.eval_datasets[0]}/{split_val}/{prefix}{args.name}/'

    if interpolate:
        models = get_models(args, [0, 10])
        inter_model = interpolate_models(models[0], models[1], [0.5])[0][0]
        dataset = get_first_dataset(models[0], args)
        if not is_train and split_val == 'val' and hasattr(dataset, 'val_loader'):
            dataset.test_loader = dataset.val_loader
            dataset.test_dataset = dataset.val_dataset
        dataloader = get_dataloader(dataset, is_train=is_train, args=args, image_encoder=None)
        if path is not None:
            os.makedirs(path, exist_ok=True)
        path = path + '0-10.dat'
        enc = get_encodings(inter_model, dataloader, args.device, save_path=path, feature_dim=dim)
    elif tsa:
        models = get_models(args, [0, 1, 9, 10])
        early_model = interpolate_models(models[0], models[1], [0.5])[0][0]
        late_model = interpolate_models(models[2], models[3], [0.5])[0][0]
        inter_model = interpolate_models(early_model, late_model, [0.5])[0][0]
        dataset = get_first_dataset(models[0], args)
        if not is_train and split_val == 'val' and hasattr(dataset, 'val_loader'):
            dataset.test_loader = dataset.val_loader
            dataset.test_dataset = dataset.val_dataset
        dataloader = get_dataloader(dataset, is_train=is_train, args=args, image_encoder=None)
        if path is not None:
            os.makedirs(path, exist_ok=True)
        path = path + 'tsa.dat'
        enc = get_encodings(inter_model, dataloader, args.device, save_path=path, feature_dim=dim)
    elif avg_all:
        models = get_models(args, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        inter_model = interpolate_more_models(models)
        dataset = get_first_dataset(models[0], args)
        if not is_train and split_val == 'val' and hasattr(dataset, 'val_loader'):
            dataset.test_loader = dataset.val_loader
            dataset.test_dataset = dataset.val_dataset
        dataloader = get_dataloader(dataset, is_train=is_train, args=args, image_encoder=None)
        if path is not None:
            os.makedirs(path, exist_ok=True)
        path = path + 'avg_all.dat'
        enc = get_encodings(inter_model, dataloader, args.device, save_path=path, feature_dim=dim)
    else:
        print(path)
        models, encodings = get_encodings_indices(args, indices, save_path=path, is_train=is_train, feature_dim=dim)
    print(f'stored representations {time.time() - start_time} seconds')


def evaluate(args):
    metric = args.metric
    is_train = False
    split_val = getattr(args, 'split', 'test')
    subset = 'train' if is_train else split_val

    print(f"Preparing {subset} dataloader for evaluation...")
    dummy_model = get_models(args, [0])[0]
    dataset = get_first_dataset(dummy_model, args)

    if split_val == 'val' and hasattr(dataset, 'val_loader'):
        dataset.test_loader = dataset.val_loader
        dataset.test_dataset = dataset.val_dataset

    dataloader = get_dataloader(dataset, is_train=is_train, args=args, image_encoder=None)

    print("Preparing model...")
    if getattr(args, 'wise_ft', False):
        alpha = 0.5
        print(f'wise-ft, {alpha}')
        models = get_models(args, [0, 10])
        model = interpolate_models(models[0], models[1], [alpha])[0][0]
    else:
        indices = [args.index]
        model = get_models(args, indices)[0]

    model.eval()
    device = args.device
    model.to(device)

    encoder = model.image_encoder

    all_preds = []
    all_labels = []
    all_metadata = []

    print(f"Running inference (Metric: {metric})...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize(batch)

            x = batch['images'].to(device)
            labels = batch['labels'].to(device)
            metadata = batch.get('metadata', None)

            enc = utils.get_logits(x, encoder)

            logits = logits_from_encodings(enc, model, device)
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            if metadata is not None:
                all_metadata.append(metadata)

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_labels)
    if all_metadata and isinstance(all_metadata[0], torch.Tensor):
        metadata = torch.cat(all_metadata)
    else:
        metadata = None

    if metric == 'accuracy':
        correct = (y_pred == y_true).sum().item()
        total = len(y_true)
        acc = correct / total
        print(f"Accuracy: {acc:.4f} ({correct}/{total})")
        return acc

    elif metric == 'f1':
        f1 = f1_score(y_true.numpy(), y_pred.numpy(), average='macro')
        print(f"Macro F1 Score: {f1:.4f}")
        return f1
    elif metric == 'worst_region':
        if metadata is None:
            print("Warning: No metadata found for worst region.")
            return 0.0

        try:
            def get_metadata_source(dset):
                if hasattr(dset, 'metadata_map') and hasattr(dset, 'metadata_fields'):
                    return dset
                if hasattr(dset, 'dataset'):
                    return get_metadata_source(dset.dataset)
                if hasattr(dset, 'dset'):
                    return get_metadata_source(dset.dset)
                return None

            wilds_dataset = get_metadata_source(dataset)

            if wilds_dataset is None:
                print(f"Could not find metadata_map in dataset: {type(dataset)}")
                region_col = 0
                mapping = None
            else:
                if 'region' in wilds_dataset.metadata_fields:
                    region_col = wilds_dataset.metadata_fields.index('region')
                else:
                    region_col = 0
                mapping = wilds_dataset.metadata_map.get('region', None)

            region_metadata = metadata[:, region_col].cpu()
            unique_regions = torch.unique(region_metadata)
            region_accuracies = {}

            print(f"Found {len(unique_regions)} unique regions.")

            for r_idx in unique_regions:
                mask = (region_metadata == r_idx)
                if mask.sum() == 0: continue

                reg_preds = y_pred[mask].cpu()
                reg_true = y_true[mask].cpu()
                reg_acc = (reg_preds == reg_true).float().mean().item()

                r_name = str(r_idx.item())
                if mapping is not None:
                    idx_val = r_idx.item()
                    if idx_val in mapping:
                        r_name = mapping[idx_val]

                region_accuracies[r_name] = reg_acc

            if not region_accuracies:
                print("No region accuracies computed.")
                return 0.0

            worst_region = min(region_accuracies, key=region_accuracies.get)
            worst_acc = region_accuracies[worst_region]

            print(f"Worst Region: {worst_region} | Accuracy: {worst_acc:.4f}")
            return worst_acc

        except Exception as e:
            print(f"Error computing worst region: {e}")
            import traceback
            traceback.print_exc()
            return 0.0


def get_all_labels(dataset, args, is_train=False):
    dataloader = get_dataloader(dataset, is_train=is_train, args=args, image_encoder=None)
    labels = []
    for data in dataloader:
        data = maybe_dictionarize(data)
        labels.append(data["labels"])
    return torch.cat(labels, dim=0).to(args.device)


if __name__ == '__main__':
    args = parse_arguments()
    opt = args.evaluation
    if opt == "autoencoder":
        autoencoder(args)
    elif opt == "store_representations":
        store_representations(args)
    elif opt == 'store_evaluate':
        store_evaluate(args)
    elif opt == 'evaluate':
        evaluate(args)
    elif opt == 'pca_basis':
        pca_basis(args)