from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from config import MODEL, TRAIN, RUNTIME
from data import create_dataloader
from model import NovaLM, count_parameters
from tokenizer import ByteTokenizer


def set_seed(seed: int):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
):

    if step < warmup_steps:

        return max_lr * (
            float(step + 1)
            / float(
                max(1, warmup_steps)
            )
        )

    if step >= max_steps:
        return min_lr

    progress = (
        step - warmup_steps
    ) / float(
        max(
            1,
            max_steps - warmup_steps
        )
    )

    cosine = (
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
        * cosine
    )


def set_optimizer_lr(
    optimizer,
    lr: float,
):

    for group in optimizer.param_groups:
        group["lr"] = lr


def create_optimizer(
    model: nn.Module,
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

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": TRAIN.weight_decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=TRAIN.learning_rate,
        betas=TRAIN.betas,
        eps=TRAIN.eps,
    )

    return optimizer


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    step: int,
    loss: float,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
        "model_config": MODEL.__dict__.copy(),
        "train_config": TRAIN.__dict__.copy(),
    }

    torch.save(
        checkpoint,
        path
    )

    print(
        f"[CHECKPOINT] Saved: {path}"
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    device: str,
):

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    step = checkpoint.get(
        "step",
        0
    )

    loss = checkpoint.get(
        "loss",
        0.0
    )

    print(
        f"[CHECKPOINT] Loaded: {path}"
    )

    print(
        f"Step: {step}"
    )

    print(
        f"Loss: {loss:.6f}"
    )

    return step


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    max_batches=20,
):

    model.eval()

    losses = []

    iterator = iter(
        dataloader
    )

    for _ in range(max_batches):

        try:
            input_ids, labels = next(
                iterator
            )
        except StopIteration:
            break

        input_ids = input_ids.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        output = model(
            input_ids,
            labels=labels
        )

        loss = output["loss"]

        if loss is not None:
            losses.append(
                loss.detach().float().item()
            )

    model.train()

    if not losses:
        return float("nan")

    return sum(losses) / len(losses)


def train():

    set_seed(
        TRAIN.seed
    )

    device = RUNTIME.device

    print("=" * 70)
    print("NovaLLM Pretraining")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    tokenizer = ByteTokenizer()

    MODEL.vocab_size = (
        tokenizer.vocab_size
    )

    model = NovaLM(
        MODEL
    )

    model = model.to(
        device
    )

    parameters = count_parameters(
        model
    )

    print(
        f"Parameters: "
        f"{parameters:,}"
    )

    if device == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    optimizer = create_optimizer(
        model
    )

    train_dir = (
        Path(TRAIN.data_dir)
        / "pretrain"
    )

    if not train_dir.exists():

        train_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        example = (
            "これはNovaLLMの学習用テキストです。\n"
            "ここに大量のTXTまたはJSONLデータを配置してください。\n"
        )

        (train_dir / "example.txt").write_text(
            example,
            encoding="utf-8"
        )

        print(
            f"[INFO] Created example dataset: "
            f"{train_dir / 'example.txt'}"
        )

    train_loader = create_dataloader(
        data_dir=str(train_dir),
        tokenizer=tokenizer,
        seq_len=MODEL.max_seq_len,
        batch_size=TRAIN.batch_size,
        num_workers=RUNTIME.num_workers,
        seed=TRAIN.seed,
    )

    train_iterator = iter(
        train_loader
    )

    scaler = None

    use_amp = (
        device == "cuda"
        and RUNTIME.dtype
        in (
            torch.float16,
            torch.bfloat16,
        )
    )

    if device == "cuda":

        autocast_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            and TRAIN.use_bf16
            else torch.float16
        )

    else:
        autocast_dtype = torch.float32

    checkpoint_path = (
        Path(TRAIN.checkpoint_dir)
        / "latest.pt"
    )

    start_step = 0

    if checkpoint_path.exists():

        try:

            start_step = load_checkpoint(
                str(checkpoint_path),
                model,
                optimizer,
                device,
            )

        except Exception as e:

            print(
                "[WARN] Could not load "
                f"checkpoint: {e}"
            )

            start_step = 0

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0

    running_steps = 0

    start_time = time.time()

    for step in range(
        start_step,
        TRAIN.max_steps
    ):

        lr = get_lr(
            step,
            TRAIN.warmup_steps,
            TRAIN.max_steps,
            TRAIN.learning_rate,
            TRAIN.min_learning_rate,
        )

        set_optimizer_lr(
            optimizer,
            lr
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        accumulation_loss = 0.0

        for micro_step in range(
            TRAIN.grad_accumulation
        ):

            try:

                input_ids, labels = next(
                    train_iterator
                )

            except StopIteration:

                train_iterator = iter(
                    train_loader
                )

                input_ids, labels = next(
                    train_iterator
                )

            input_ids = input_ids.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype,
                ):

                    output = model(
                        input_ids,
                        labels=labels
                    )

                    loss = output["loss"]

            else:

                output = model(
                    input_ids,
                    labels=labels
                )

                loss = output["loss"]

            if loss is None:
                raise RuntimeError(
                    "Model did not return loss"
                )

            accumulation_loss += (
                loss.detach()
                .float()
                .item()
            )

            loss = (
                loss
                / TRAIN.grad_accumulation
            )

            loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            TRAIN.grad_clip
        )

        optimizer.step()

        mean_loss = (
            accumulation_loss
            / TRAIN.grad_accumulation
        )

        running_loss += mean_loss
        running_steps += 1

        if (
            step % 10 == 0
            or step == TRAIN.max_steps - 1
        ):

            elapsed = (
                time.time()
                - start_time
            )

            steps_per_sec = (
                running_steps
                / max(elapsed, 1e-6)
            )

            print(
                f"step={step:7d} "
                f"loss={mean_loss:.5f} "
                f"lr={lr:.3e} "
                f"grad={float(gradient_norm):.3f} "
                f"steps/s={steps_per_sec:.2f}"
            )

        if (
            step > 0
            and step % TRAIN.save_every == 0
        ):

            save_checkpoint(
                str(checkpoint_path),
                model,
                optimizer,
                step,
                mean_loss,
            )

            numbered = (
                Path(
                    TRAIN.checkpoint_dir
                )
                / f"step_{step}.pt"
            )

            save_checkpoint(
                str(numbered),
                model,
                optimizer,
                step,
                mean_loss,
            )

    save_checkpoint(
        str(checkpoint_path),
        model,
        optimizer,
        TRAIN.max_steps,
        mean_loss,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "Training completed."
    )

    print(
        f"Final loss: {mean_loss:.6f}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    train()
