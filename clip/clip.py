import hashlib
import os
import urllib
import warnings
from typing import Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, InterpolationMode
from tqdm import tqdm

import re
try:
    import open_clip as _openclip
    _OPENCLIP_ENABLED = True
except Exception:
    _OPENCLIP_ENABLED = False

print(_OPENCLIP_ENABLED)
from clip.model import build_model
from clip.tokenizer import SimpleTokenizer as _Tokenizer

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def _convert_to_rgb(image):
    return image.convert('RGB')

def _transform(n_px: int, is_train: bool):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])

def _list_openclip_model_names():
    if not _OPENCLIP_ENABLED:
        return []
    try:
        pairs = _openclip.list_pretrained()
    except Exception:
        return []
    names = [f"openclip:{m}:{pt}" for (m, pt) in pairs]
    return sorted(set(names))

def _is_openclip_name(name: str):
    return isinstance(name, str) and name.startswith("openclip:")

def _parse_openclip_name(name: str):
    m = re.match(r"^openclip:([^:]+):([^:]+)$", name)
    if not m:
        print(available_models())
        raise RuntimeError(
            f'Invalid OpenCLIP name "{name}". '
            f'Use format "openclip:<model_name>:<pretrained_tag>", e.g. '
            f'"openclip:ViT-B-32:laion2b_s34b_b79k".'
        )
    return m.group(1), m.group(2)

def available_models() -> List[str]:
    names = list(_MODELS.keys())
    names += _list_openclip_model_names()
    return names


def load(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit: bool = True,
    is_train: bool = False,
    pretrained: bool = True
):
    global _tokenize_impl

    if _is_openclip_name(name):
        if not _OPENCLIP_ENABLED:
            raise RuntimeError(
                'You requested an OpenCLIP model but the "open_clip" package is not installed.\n'
                "Install with: pip install open_clip_torch"
            )
        model_name, pretrained_tag = _parse_openclip_name(name)

        if jit:
            warnings.warn("OpenCLIP models do not use the JIT archive; ignoring jit=True.")
        try:
            model, preprocess_train, preprocess_eval = _openclip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained_tag,
                device=device,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenCLIP model '{model_name}' with pretrained '{pretrained_tag}': {e}")

        model.eval()

        _tokenize_impl = _tokenize_openclip

        return model, preprocess_train, preprocess_eval

    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        if _OPENCLIP_ENABLED and any(name == f"{m}:{pt}" or name == m for (m, pt) in _openclip.list_pretrained()):
            raise RuntimeError(
                f'Model "{name}" looks like an OpenCLIP model. Use the explicit format '
                f'"openclip:{name}" or "openclip:{m}:{pt}". '
                f'See available_models() for options.'
            )
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        try:
            model = build_model(state_dict or model.state_dict()).to(device)
        except KeyError:
            sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
            model = build_model(sd).to(device)

        if str(device) == "cpu":
            model.float()

        _tokenize_impl = _tokenize_openai_clip

        return model, \
               _transform(model.visual.input_resolution, is_train=True), \
               _transform(model.visual.input_resolution, is_train=False)

    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        graphs = [module.graph] if hasattr(module, "graph") else []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)
        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            graphs = [module.graph] if hasattr(module, "graph") else []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)
            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()

    _tokenize_impl = _tokenize_openai_clip

    return model, \
           _transform(model.input_resolution.item(), is_train=True), \
           _transform(model.input_resolution.item(), is_train=False)


def _tokenize_openai_clip(texts, context_length: int = 77, tokenizer=None) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]
    sot_token = _tokenizer.encoder["<start_of_text>"]
    eot_token = _tokenizer.encoder["<end_of_text>"]
    all_tokens = [[sot_token] + _tokenizer.encode(t) + [eot_token] for t in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


def _tokenize_openclip(texts, context_length: int = 77, tokenizer=None) -> torch.LongTensor:
    if not _OPENCLIP_ENABLED:
        raise RuntimeError("open_clip is not installed; cannot tokenize for OpenCLIP.")

    if isinstance(texts, str):
        texts = [texts]

    if tokenizer is not None:
        return tokenizer(texts, context_length=context_length)

    return _openclip.tokenize(texts, context_length=context_length)

_tokenize_impl = _tokenize_openai_clip

def tokenize(texts: Union[str, List[str]], context_length: int = 77, tokenizer=None) -> torch.LongTensor:
    return _tokenize_impl(texts, context_length=context_length, tokenizer=tokenizer)