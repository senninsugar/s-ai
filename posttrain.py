from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import MODEL, TRAIN, RUNTIME
from model import NovaLM, count_parameters
from tokenizer import ByteTokenizer


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"[WARN] Invalid JSON "
                    f"{path}:{line_no}: {e}"
                )
                continue

            if isinstance(obj, dict):
                records.append(obj)

    return records


# ============================================================
# Chat formatting
# ============================================================

def normalize_messages(
    record: Dict[str, Any],
) -> List[Dict[str, str]]:

    messages = record.get("messages")

    if isinstance(messages, list):

        result = []

        for message in messages:

            if not isinstance(message, dict):
                continue

            role = str(
                message.get(
                    "role",
                    "user",
                )
            )

            content = message.get(
                "content",
                "",
            )

            if not isinstance(content, str):
                continue

            content = content.strip()

            if not content:
                continue

            result.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return result

    prompt = record.get(
        "prompt",
        record.get(
            "question",
            "",
        ),
    )

    answer = record.get(
        "completion",
        record.get(
            "answer",
            "",
        ),
    )

    if isinstance(prompt, str) and isinstance(answer, str):

        return [
            {
                "role": "user",
                "content": prompt.strip(),
            },
            {
                "role": "assistant",
                "content": answer.strip(),
            },
        ]

    return []


def format_chat(
    messages: List[Dict[str, str]],
) -> str:

    parts = []

    for message in messages:

        role = message["role"]
        content = message["content"]

        parts.append(
            f"<|{role}|>\n"
            f"{content}\n"
            f"<|end|>"
        )

    parts.append(
        "<|assistant|>\n"
    )

    return "\n".join(parts)


# ============================================================
# SFT Dataset
# ============================================================

class SFTDataset(Dataset):

    def __init__(
        self,
        path: str,
        tokenizer: ByteTokenizer,
        max_length: int,
    ):

        self.records = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not self.records:
            raise RuntimeError(
                f"No records found in {path}"
            )

    def __len__(self):
        return len(self.records)

    def _encode(
        self,
        text: str,
    ):

        return self.tokenizer.encode(
            text,
            add_bos=True,
            add_eos=True,
        )

    def __getitem__(self, index):

        record = self.records[index]

        messages = normalize_messages(
            record
        )

        if not messages:
            return self.__getitem__(
                (index + 1) % len(self)
            )

        text = format_chat(
            messages
        )

        tokens = self._encode(text)

        tokens = tokens[
            : self.max_length
        ]

        if len(tokens) < 2:

            tokens = (
                tokens
                + [self.tokenizer.EOS]
            )

        input_ids = tokens[:-1]
        labels = tokens[1:]

        return (
            torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            torch.tensor(
                labels,
                dtype=torch.long,
            ),
        )


def collate_sft(batch):

    max_length = max(
        len(x[0])
        for x in batch
    )

    input_ids = []
    labels = []
    attention_mask = []

    for ids, target in batch:

        padding = (
            max_length
            - ids.shape[0]
        )

        ids = F.pad(
            ids,
            (0, padding),
            value=ByteTokenizer.PAD,
        )

        target = F.pad(
            target,
            (0, padding),
            value=-100,
        )

        mask = torch.ones(
            max_length,
            dtype=torch.long,
        )

        if padding:
            mask[-padding:] = 0

        input_ids.append(ids)
        labels.append(target)
        attention_mask.append(mask)

    return {
        "input_ids": torch.stack(
            input_ids
        ),
        "labels": torch.stack(
            labels
        ),
        "attention_mask": torch.stack(
            attention_mask
        ),
    }


# ============================================================
# SFT loss
# ============================================================

def compute_sft_loss(
    logits,
    labels,
):

    return F.cross_entropy(
        logits.reshape(
            -1,
            logits.size(-1)
        ),
        labels.reshape(-1),
        ignore_index=-100,
    )


# ============================================================
# Optimizer
# ============================================================

def create_optimizer(
    model,
    learning_rate,
):

    decay = []
    no_decay = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)

    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": 0.1,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


# ============================================================
# Learning rate
# ============================================================

def cosine_lr(
    step,
    total_steps,
    warmup_steps,
    max_lr,
    min_lr,
):

    if step < warmup_steps:

        return max_lr * (
            step + 1
        ) / max(
            1,
            warmup_steps,
        )

    progress = (
        step - warmup_steps
    ) / max(
        1,
        total_steps - warmup_steps,
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    value = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )

    return (
        min_lr
        + (
            max_lr - min_lr
        )
        * value
    )


# ============================================================
# Checkpoint
# ============================================================

def save_sft_checkpoint(
    path,
    model,
    optimizer,
    step,
    loss,
):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "loss": loss,
            "model_config": MODEL.__dict__.copy(),
        },
        path,
    )

    print(
        f"[SAVE] {path}"
    )


# ============================================================
# Main SFT
# ============================================================

def train_sft():

    set_seed(
        TRAIN.seed
    )

    device = RUNTIME.device

    tokenizer = ByteTokenizer()

    MODEL.vocab_size = (
        tokenizer.vocab_size
    )

    model = NovaLM(
        MODEL
    )

    model.to(device)

    print(
        "=" * 70
    )

    print(
        "NovaLLM SFT"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Parameters: "
        f"{count_parameters(model):,}"
    )

    dataset_path = (
        Path(TRAIN.data_dir)
        / "sft"
        / "train.jsonl"
    )

    if not dataset_path.exists():

        dataset_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        example = {
            "messages": [
                {
                    "role": "user",
                    "content": "こんにちは",
                },
                {
                    "role": "assistant",
                    "content": "こんにちは！NovaLLMです。",
                },
            ]
        }

        dataset_path.write_text(
            json.dumps(
                example,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "[INFO] Example SFT dataset created:"
        )

        print(
            dataset_path
        )

    dataset = SFTDataset(
        str(dataset_path),
        tokenizer,
        MODEL.max_seq_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=TRAIN.batch_size,
        shuffle=True,
        num_workers=RUNTIME.num_workers,
        pin_memory=(
            device == "cuda"
        ),
        collate_fn=collate_sft,
    )

    optimizer = create_optimizer(
        model,
        learning_rate=5e-5,
    )

    total_steps = min(
        10000,
        max(
            1,
            len(loader) * 10
        ),
    )

    warmup_steps = min(
        500,
        total_steps // 10,
    )

    model.train()

    iterator = iter(loader)

    for step in range(
        total_steps
    ):

        try:

            batch = next(iterator)

        except StopIteration:

            iterator = iter(loader)

            batch = next(iterator)

        input_ids = batch[
            "input_ids"
        ].to(
            device,
            non_blocking=True,
        )

        labels = batch[
            "labels"
        ].to(
            device,
            non_blocking=True,
        )

        lr = cosine_lr(
            step,
            total_steps,
            warmup_steps,
            5e-5,
            5e-6,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(
            set_to_none=True
        )

        use_amp = (
            device == "cuda"
        )

        if use_amp:

            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )

            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
            ):

                output = model(
                    input_ids
                )

                loss = compute_sft_loss(
                    output["logits"],
                    labels,
                )

        else:

            output = model(
                input_ids
            )

            loss = compute_sft_loss(
                output["logits"],
                labels,
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        if (
            step % 10 == 0
            or step == total_steps - 1
        ):

            print(
                f"SFT "
                f"step={step:5d} "
                f"loss={loss.item():.5f} "
                f"lr={lr:.3e}"
            )

        if (
            step > 0
            and step % 500 == 0
        ):

            save_sft_checkpoint(
                "checkpoints/sft_latest.pt",
                model,
                optimizer,
                step,
                loss.item(),
            )

    save_sft_checkpoint(
        "checkpoints/sft_final.pt",
        model,
        optimizer,
        total_steps,
        loss.item(),
    )

    print(
        "SFT completed."
    )


if __name__ == "__main__":
    train_sft()
