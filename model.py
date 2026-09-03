import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODEL


class RMSNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

        self.eps = eps

    def forward(self, x):

        variance = x.float().pow(2).mean(
            dim=-1,
            keepdim=True
        )

        x = x * torch.rsqrt(
            variance + self.eps
        )

        return self.weight * x.type_as(
            self.weight
        )


def rotate_half(x):

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat(
        (-x2, x1),
        dim=-1
    )


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        theta: float = 10000.0
    ):
        super().__init__()

        inv_freq = 1.0 / (
            theta ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32
                ) / dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False
        )

        self.max_seq_len = (
            max_position_embeddings
        )

        self._build_cache(
            max_position_embeddings
        )

    def _build_cache(self, seq_len):

        device = self.inv_freq.device

        positions = torch.arange(
            seq_len,
            device=device,
            dtype=torch.float32
        )

        freqs = torch.outer(
            positions,
            self.inv_freq
        )

        emb = torch.cat(
            (freqs, freqs),
            dim=-1
        )

        self.register_buffer(
            "cos_cached",
            emb.cos()[None, None, :, :],
            persistent=False
        )

        self.register_buffer(
            "sin_cached",
            emb.sin()[None, None, :, :],
            persistent=False
        )

    def forward(
        self,
        q,
        k,
        position_ids
    ):

        max_position = (
            int(position_ids.max().item()) + 1
        )

        if max_position > self.max_seq_len:

            self._build_cache(
                max_position
            )

            self.max_seq_len = max_position

        cos = self.cos_cached[
            :,
            :,
            position_ids,
            :
        ]

        sin = self.sin_cached[
            :,
            :,
            position_ids,
            :
        ]

        q = (
            q * cos
            + rotate_half(q) * sin
        )

        k = (
            k * cos
            + rotate_half(k) * sin
        )

        return q, k


class SwiGLU(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False
        )

        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False
        )

        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False
        )

    def forward(self, x):

        gate = self.gate_proj(x)
        up = self.up_proj(x)

        return self.down_proj(
            F.silu(gate) * up
        )


class KVCache:

    def __init__(self):
        self.key = None
        self.value = None

    def update(
        self,
        key,
        value
    ):

        if self.key is None:

            self.key = key
            self.value = value

        else:

            self.key = torch.cat(
                [self.key, key],
                dim=-2
            )

            self.value = torch.cat(
                [self.value, value],
                dim=-2
            )

        return self.key, self.value

    def clear(self):
        self.key = None
        self.value = None


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        config
    ):
        super().__init__()

        self.hidden_size = (
            config.hidden_size
        )

        self.num_heads = (
            config.num_heads
        )

        self.num_kv_heads = (
            config.num_kv_heads
        )

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_heads must be divisible "
                "by num_kv_heads"
            )

        self.head_dim = (
            config.hidden_size
            // config.num_heads
        )

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False
        )

        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False
        )

        self.rotary = RotaryEmbedding(
            self.head_dim,
            config.max_seq_len,
            config.rope_theta
        )

    def _repeat_kv(
        self,
        x
    ):

        if self.num_kv_heads == self.num_heads:
            return x

        repeat = (
            self.num_heads
            // self.num_kv_heads
        )

        return x.repeat_interleave(
            repeat,
            dim=1
        )

    def forward(
        self,
        x,
        position_ids=None,
        kv_cache=None,
        use_cache=False
    ):

        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        if position_ids is None:

            start = 0

            if kv_cache is not None and kv_cache.key is not None:
                start = kv_cache.key.shape[-2]

            position_ids = torch.arange(
                start,
                start + seq_len,
                device=x.device
            )

        q, k = self.rotary(
            q,
            k,
            position_ids
        )

        if kv_cache is not None:

            k, v = kv_cache.update(
                k,
                v
            )

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        if hasattr(
            F,
            "scaled_dot_product_attention"
        ):

            is_causal = (
                kv_cache is None
                and seq_len > 1
            )

            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal
            )

        else:

            scale = 1.0 / math.sqrt(
                self.head_dim
            )

            scores = torch.matmul(
                q,
                k.transpose(-2, -1)
            ) * scale

            q_len = q.shape[-2]
            k_len = k.shape[-2]

            if (
                kv_cache is None
                and q_len == k_len
            ):

                mask = torch.triu(
                    torch.ones(
                        q_len,
                        k_len,
                        device=x.device,
                        dtype=torch.bool
                    ),
                    diagonal=1
                )

                scores = scores.masked_fill(
                    mask,
                    torch.finfo(
                        scores.dtype
                    ).min
                )

            weights = F.softmax(
                scores.float(),
                dim=-1
            ).type_as(scores)

            output = torch.matmul(
                weights,
                v
            )

        output = output.transpose(
            1,
            2
        ).contiguous()

        output = output.view(
            batch,
            seq_len,
            self.hidden_size
        )

        output = self.o_proj(output)

        return output


class TransformerBlock(nn.Module):

    def __init__(
        self,
        config
    ):
        super().__init__()

        self.input_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps
        )

        self.attention = CausalSelfAttention(
            config
        )

        self.post_attention_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps
        )

        self.mlp = SwiGLU(
            config.hidden_size,
            config.intermediate_size
        )

    def forward(
        self,
        x,
        position_ids=None,
        kv_cache=None,
        use_cache=False
    ):

        x = x + self.attention(
            self.input_norm(x),
            position_ids=position_ids,
            kv_cache=kv_cache,
            use_cache=use_cache
        )

        x = x + self.mlp(
            self.post_attention_norm(x)
        )

        return x


class NovaLM(nn.Module):

    def __init__(
        self,
        config=None
    ):
        super().__init__()

        if config is None:
            config = MODEL

        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )

        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

        if config.tie_embeddings:

            self.lm_head.weight = (
                self.embedding.weight
            )

        self.apply(
            self._init_weights
        )

    def _init_weights(self, module):

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(
        self,
        input_ids,
        labels=None,
        use_cache=False,
        caches=None
    ):

        batch, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} "
                f"exceeds "
                f"{self.config.max_seq_len}"
            )

        x = self.embedding(
            input_ids
        )

        position_ids = torch.arange(
            seq_len,
            device=input_ids.device
        )

        for i, layer in enumerate(
            self.layers
        ):

            cache = None

            if caches is not None:
                cache = caches[i]

            x = layer(
                x,
                position_ids=position_ids,
                kv_cache=cache,
                use_cache=use_cache
            )

        x = self.norm(x)

        logits = self.lm_head(x)

        loss = None

        if labels is not None:

            shift_logits = (
                logits[:, :-1, :]
                .contiguous()
            )

            shift_labels = (
                labels[:, 1:]
                .contiguous()
            )

            loss = F.cross_entropy(
                shift_logits.view(
                    -1,
                    shift_logits.size(-1)
                ),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {
            "logits": logits,
            "loss": loss
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=128,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        eos_token_id=2
    ):

        self.eval()

        generated = input_ids

        caches = [
            KVCache()
            for _ in self.layers
        ]

        for _ in range(max_new_tokens):

            if generated.shape[1] == 1:
                current = generated
            else:
                current = generated[:, -1:]

            output = self(
                current,
                use_cache=True,
                caches=caches
            )

            logits = output["logits"][:, -1, :]

            if temperature <= 0:
                next_token = logits.argmax(
                    dim=-1,
                    keepdim=True
                )

            else:

                logits = logits / temperature

                if top_k > 0:

                    values, _ = torch.topk(
                        logits,
                        min(
                            top_k,
                            logits.size(-1)
                        )
                    )

                    threshold = values[:, -1].unsqueeze(
                        -1
                    )

                    logits = logits.masked_fill(
                        logits < threshold,
                        float("-inf")
                    )

                if top_p < 1.0:

                    sorted_logits, sorted_indices = torch.sort(
                        logits,
                        descending=True
                    )

                    probabilities = F.softmax(
                        sorted_logits,
                        dim=-1
                    )

                    cumulative = probabilities.cumsum(
                        dim=-1
                    )

                    remove = cumulative > top_p

                    remove[:, 1:] = (
                        remove[:, :-1].clone()
                    )

                    remove[:, 0] = False

                    sorted_logits = sorted_logits.masked_fill(
                        remove,
                        float("-inf")
                    )

                    logits = torch.full_like(
                        logits,
                        float("-inf")
                    )

                    logits.scatter_(
                        -1,
                        sorted_indices,
                        sorted_logits
                    )

                probabilities = F.softmax(
                    logits,
                    dim=-1
                )

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1
                )

            generated = torch.cat(
                [generated, next_token],
                dim=1
            )

            if (
                eos_token_id is not None
                and torch.all(
                    next_token == eos_token_id
                )
            ):
                break

        return generated


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


if __name__ == "__main__":

    from tokenizer import ByteTokenizer

    tokenizer = ByteTokenizer()

    # 設定をtokenizerに合わせる
    MODEL.vocab_size = tokenizer.vocab_size

    model = NovaLM(MODEL)

    params = count_parameters(model)

    print(
        f"Parameters: {params:,}"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    text = "こんにちは"

    tokens = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=False
    )

    input_ids = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device
    )

    output = model(
        input_ids
    )

    print(
        "Input shape:",
        input_ids.shape
    )

    print(
        "Logits shape:",
        output["logits"].shape
    )

    generated = model.generate(
        input_ids,
        max_new_tokens=32
    )

    print(
        "Generated:"
    )

    print(
        tokenizer.decode(
            generated[0].tolist()
        )
    )
