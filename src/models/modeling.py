import torch
import copy

import clip.clip as clip
from collections import namedtuple
from src.models import utils


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            args.model, args.device, jit=False)
        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)


class ImageEncoderAugmented(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        # Use the model name directly (e.g., "ViT-B/16") matching your _MODELS keys
        name = args.model

        print(f'Loading Augmented Encoder: {name} (jit=False)')

        # Load using your custom load function
        # jit=False is required to allow slicing the transformer layers
        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            name, device=args.device, jit=False
        )

        # --- ViT-B Architecture Slicing (OpenAI CLIP Structure) ---
        if not hasattr(self.model.visual, 'transformer'):
            raise ValueError("ImageEncoderAugmented only supports ViT models.")

        # In non-JIT OpenAI CLIP, resblocks is an nn.Sequential inside the transformer
        resblocks = self.model.visual.transformer.resblocks

        self.conv1 = self.model.visual.conv1
        self.ln_pre = self.model.visual.ln_pre

        # Validate layer count (ViT-B should have 12 layers)
        if len(resblocks) != 12:
            print(
                f"Warning: Model has {len(resblocks)} layers. Slicing logic (0:3, 3:6...) is optimized for 12 layers.")

        self.layer1 = resblocks[0:3]
        self.layer2 = resblocks[3:6]
        self.layer3 = resblocks[6:9]
        self.layer4 = resblocks[9:12]

        self.class_embedding = self.model.visual.class_embedding
        self.positional_embedding = self.model.visual.positional_embedding

        # OpenAI CLIP typically doesn't use patch_dropout, but we handle it if present
        self.patch_dropout = getattr(self.model.visual, 'patch_dropout', torch.nn.Identity())

        self.ln_post = self.model.visual.ln_post
        self.proj = self.model.visual.proj

        if not keep_lang:
            # Remove text encoder components to save memory
            if hasattr(self.model, 'transformer'):
                delattr(self.model, 'transformer')
            if hasattr(self.model, 'token_embedding'):
                delattr(self.model, 'token_embedding')

    def forward(self, images):
        return self.model.encode_image(images)

    def get_features(self, images):
        """Returns intermediate features for LDIFS regularization."""
        # 1. Convolution
        h = self.conv1(images)
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)

        # 2. Add Embeddings (Class + Positional)
        # Replicates OpenAI CLIP logic: concat class token then add positional
        h = torch.cat(
            [self.class_embedding.to(h.dtype) + torch.zeros(h.shape[0], 1, h.shape[-1], dtype=h.dtype, device=h.device),
             h], dim=1)
        h = h + self.positional_embedding.to(h.dtype)

        if not isinstance(self.patch_dropout, torch.nn.Identity):
            h = self.patch_dropout(h)

        # 3. Pre-LayerNorm
        h = self.ln_pre(h)
        h_ln_pre = h

        # 4. Transformer Stack (Sliced)
        h = h.permute(1, 0, 2)  # OpenAI CLIP Transformer expects (Seq, Batch, Dim)

        for r in self.layer1: h = r(h)
        h_layer1 = h.permute(1, 0, 2)

        for r in self.layer2: h = r(h)
        h_layer2 = h.permute(1, 0, 2)

        for r in self.layer3: h = r(h)
        h_layer3 = h.permute(1, 0, 2)

        for r in self.layer4: h = r(h)
        h = h.permute(1, 0, 2)
        h_layer4 = h

        # 5. Post-LayerNorm
        h = self.ln_post(h)

        Outputs = namedtuple("Outputs", ["lnpre", "layer1", "layer2", "layer3", "layer4"])
        return Outputs(h_ln_pre, h_layer1, h_layer2, h_layer3, h_layer4)