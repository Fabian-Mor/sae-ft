# adapted from: https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoderSimple(nn.Module):
    def __init__(self, activation_dim: int, dict_size: int):
        super().__init__()
        self.activation_dim = int(activation_dim)
        self.dict_size = int(dict_size)

        self.encoder_DF = nn.Linear(self.activation_dim, self.dict_size, bias=False)
        self.decoder_DF = nn.Linear(self.dict_size, self.activation_dim, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(self.activation_dim))
        self.b_enc = nn.Parameter(torch.zeros(self.dict_size))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return nn.ReLU()(self.encoder_DF(x - self.b_pre) + self.b_enc)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_FD(x) + self.b_pre

    def forward_pass(self, x: torch.Tensor):
        enc = self.encode(x)
        rec = self.decode(enc)
        return rec, enc

def calculate_loss_simple(autoencoder: SparseAutoencoderSimple, x: torch.Tensor, l1: float) -> torch.Tensor:
    rec, enc = autoencoder.forward_pass(x)
    rec_loss = F.mse_loss(rec, x)
    l1_loss = l1 * enc.sum()
    loss = rec_loss + l1_loss
    return loss

class TopK(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = int(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, indices = torch.topk(x, self.k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(dim=-1, index=indices, value=1.0)
        return x * mask

class BatchTopK(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = int(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, dict_size = x.shape
        total_k = self.k * batch_size

        flat = x.view(-1)

        values, indices = torch.topk(flat, total_k, sorted=False)

        mask = torch.zeros_like(flat)
        mask[indices] = 1.0

        mask = mask.view(batch_size, dict_size)

        return x * mask

class SparseAutoencoderTopK(nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int, batch_top_k=False, normalize=True):
        super().__init__()
        self.activation_dim = int(activation_dim)
        self.dict_size = int(dict_size)
        self.k = int(k)
        self.normalize = normalize
        self.b_pre = nn.Parameter(torch.zeros(self.activation_dim))
        self.encoder = nn.Linear(self.activation_dim, self.dict_size, bias=False)
        self.decoder = nn.Linear(self.dict_size, self.activation_dim, bias=False)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)
        if batch_top_k:
            self.top_k = BatchTopK(self.k)
        else:
            self.top_k = TopK(self.k)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_pre
        pre_activations = self.encoder(x_centered)
        return self.top_k.forward(pre_activations)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.b_pre

    def forward(self, x: torch.Tensor):
        x_target = self.preprocess(x)
        enc = self.encode(x_target)
        rec = self.decode(enc)
        return rec, enc, x_target

def calculate_loss_top_k(autoencoder: SparseAutoencoderTopK, x: torch.Tensor) -> torch.Tensor:
    rec, enc, x_target = autoencoder(x)
    rec_loss = F.mse_loss(rec, x_target)
    return rec_loss