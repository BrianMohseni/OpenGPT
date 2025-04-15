from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


@dataclass
class ModelConfig:
    max_length: int
    num_layers: int
    num_heads: int
    d_model: int
    rate: float
    tokenizer: PreTrainedTokenizer

    def __post_init__(self):
        self.vocab_size = len(self.tokenizer.get_vocab())


class AttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qkv_w = nn.Linear(config.d_model, config.d_model * 3)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.rate)

        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads

        self.register_buffer("mask",
                             torch.tril(torch.ones(config.max_length, config.max_length)).view(1, 1, config.max_length,
                                                                                               config.max_length))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv_w(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # k = (B, num_heads, T, head_dim) -> (B, num_heads, head_dim, T)
        # attn = (B, num_heads, T, T)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        attn = self.dropout(attn)

        res = attn @ v
        res = res.transpose(1, 2).contiguous().view(B, T, C)
        res = self.proj(res)

        return self.dropout(res)


class FastAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qkv_w = nn.Linear(config.d_model, config.d_model * 3)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.rate)

        self.dropout_p = config.rate
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads

    # self.register_buffer("mask", torch.tril(torch.ones(config.max_length, config.max_length)).view(1, 1, config.max_length, config.max_length))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv_w(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p, is_causal=True)

        attn = attn.transpose(1, 2).reshape(B, T, C)
        res = self.proj(attn)

        return self.dropout(res)


class SwiGLU(nn.Module):
    def forward(self, x):
        assert x.shape[-1] % 2 == 0

        x1, x2 = x.chunk(2, dim=-1)

        return x1 * F.silu(x2)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ff1 = nn.Linear(config.d_model, config.d_model * 4)
        self.ff2 = nn.Linear(config.d_model * 4, config.d_model)

        self.act = nn.GELU()

    def forward(self, x):
        x = self.ff1(x)
        x = self.act(x)
        x = self.ff2(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = FastAttentionLayer(config)
        self.mlp = MLP(config)

        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.position_emb = nn.Parameter(torch.zeros(1, config.max_length, config.d_model))
        self.dropout = nn.Dropout(config.rate)

        self.transformerlayers = nn.Sequential(*[DecoderLayer(config) for _ in range(config.num_layers)])
        self.ln = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)

            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, y_true=None):
        B, T = x.size()

        if T > self.config.vocab_size:
            raise ValueError(f"Sequence of length {T} larger than max length of {self.config.max_length}")

        token_emb = self.token_emb(x)
        pos_emb = self.position_emb[:, :T, :]
        x = self.dropout(token_emb + pos_emb)

        x = self.transformerlayers(x)
        x = self.ln(x)
        y_pred = self.lm_head(x)

        loss = None
        if y_true is not None:
            # num_pad_tokens = (x == self.config.tokenizer.pad_token_id).sum().item()
            # num_classes = y_pred.size(-1)
            # weights = torch.ones(num_classes, device=y_pred.device)
            # weights[self.config.tokenizer.pad_token_id] = .1
            loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y_true.reshape(-1))

        return y_pred, loss

    @torch.no_grad()
    def generate(self, prompt, max_length=100, temperature=1.0, top_k=50, device="cpu"):
        input_ids = self.config.tokenizer.encode(prompt, return_tensors="pt").to(device)

        for _ in range(max_length):
            logits, _ = self(input_ids)
            logits = logits[:, -1, :]
            logits = logits / temperature

            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                logits[logits < values[:, -1]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == 32007:
                break

        return self.config.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)

    @torch.no_grad()
    def ChatCompletion(self, prompt, max_length=100, temperature=1.0, top_k=50, device="cpu"):

        chat_prompt = self.config.tokenizer.apply_chat_template(prompt, tokenize=False,
                                                                apply_generation_prompt=True) + "<|assistant|>"
        input_ids = self.config.tokenizer.encode(chat_prompt, truncation=True, max_length=self.config.max_length, return_tensors="pt").to(device)
        # input_ids = self.config.tokenizer.apply_chat_template(prompt, apply_generation_prompt=True, return_tensors="pt").to(device)
        generated = []
        for _ in range(max_length):
            logits, _ = self(input_ids)
            logits = logits[:, -1, :]
            logits = logits / temperature

            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                logits[logits < values[:, -1]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated.append(next_token.item())

            if next_token.item() == self.config.tokenizer.eos_token_id:
                break

            elif input_ids.shape[1] <= self.config.max_length:
                input_ids = input_ids[:, 1:]

        return self.config.tokenizer.decode(generated, skip_special_tokens=True)

