from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import MODEL, TRAIN, RUNTIME
from model import NovaLM, count_parameters
from tokenizer import ByteTokenizer


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

CHECKPOINT_DIR = (
    ROOT_DIR
    / TRAIN.checkpoint_dir
)

PRETRAIN_CHECKPOINT = (
    CHECKPOINT_DIR
    / "pretrain_final.pt"
)

PRETRAIN_LATEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "latest.pt"
)

SFT_LATEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "sft_latest.pt"
)

SFT_FINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "sft_final.pt"
)

DATASET_PATH = (
    ROOT_DIR
    / TRAIN.data_dir
    / "sft"
    / "train.jsonl"
)

TOKENIZER_PATH = (
    ROOT_DIR
    / "tokenizer"
    / "tokenizer.json"
)


# ============================================================
# Constants
# ============================================================

IGNORE_INDEX = -100

SFT_LEARNING_RATE = 5e-5

SFT_MIN_LEARNING_RATE = 5e-6

SFT_EPOCHS = 10

MAX_SFT_STEPS = 10000

SAVE_EVERY = 100

LOG_EVERY = 10


# ============================================================
# Seed
# ============================================================

def set_seed(seed: int):
    """
    乱数を固定する。
    """

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Tokenizer
# ============================================================

def load_tokenizer() -> ByteTokenizer:
    """
    264 token tokenizerを読み込む。
    """

    if not TOKENIZER_PATH.exists():

        raise FileNotFoundError(
            "Tokenizer not found: "
            f"{TOKENIZER_PATH}\n"
            "Run tokenizer.py first."
        )

    tokenizer = ByteTokenizer.load(
        TOKENIZER_PATH
    )

    if tokenizer.vocab_size != 264:

        raise RuntimeError(
            "This SFT pipeline requires "
            "the 264-token tokenizer. "
            f"Got {tokenizer.vocab_size}."
        )

    return tokenizer


# ============================================================
# JSONL
# ============================================================

def load_jsonl(
    path: Path,
) -> List[Dict[str, Any]]:
    """
    JSONL datasetを読み込む。
    """

    if not path.exists():

        raise FileNotFoundError(
            f"SFT dataset not found: {path}"
        )

    records = []

    content = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    for line_number, line in enumerate(
        content.splitlines(),
        start=1,
    ):

        line = line.strip()

        if not line:
            continue

        try:

            obj = json.loads(
                line
            )

        except json.JSONDecodeError as exc:

            print(
                f"[WARN] Invalid JSON "
                f"at line {line_number}: "
                f"{exc}"
            )

            continue

        if isinstance(
            obj,
            dict,
        ):

            records.append(
                obj
            )

    if not records:

        raise RuntimeError(
            f"SFT dataset is empty: {path}"
        )

    return records


# ============================================================
# Message normalization
# ============================================================

def normalize_messages(
    record: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    SFT recordからmessagesを抽出する。
    """

    messages = record.get(
        "messages"
    )

    if isinstance(
        messages,
        list,
    ):

        normalized = []

        for message in messages:

            if not isinstance(
                message,
                dict,
            ):
                continue

            role = str(
                message.get(
                    "role",
                    "",
                )
            ).strip().lower()

            content = message.get(
                "content",
                "",
            )

            if content is None:
                content = ""

            content = str(
                content
            )

            if role not in {
                "system",
                "user",
                "assistant",
            }:

                continue

            if not content.strip():
                continue

            normalized.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return normalized

    # --------------------------------------------------------
    # prompt / completion format
    # --------------------------------------------------------

    prompt = record.get(
        "prompt"
    )

    completion = record.get(
        "completion"
    )

    if (
        isinstance(prompt, str)
        and isinstance(completion, str)
        and prompt.strip()
        and completion.strip()
    ):

        return [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": completion,
            },
        ]

    # --------------------------------------------------------
    # question / answer format
    # --------------------------------------------------------

    question = record.get(
        "question"
    )

    answer = record.get(
        "answer"
    )

    if (
        isinstance(question, str)
        and isinstance(answer, str)
        and question.strip()
        and answer.strip()
    ):

        return [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

    return []


# ============================================================
# Dataset
# ============================================================

class SFTDataset(Dataset):
    """
    NovaLLM SFT Dataset。

    assistant部分だけをloss対象にする。
    """

    def __init__(
        self,
        path: Path,
        tokenizer: ByteTokenizer,
        max_length: int,
    ):

        self.path = Path(
            path
        )

        self.tokenizer = tokenizer

        self.max_length = max(
            2,
            int(max_length),
        )

        records = load_jsonl(
            self.path
        )

        self.samples = []

        skipped = 0

        for record in records:

            messages = normalize_messages(
                record
            )

            if not messages:

                skipped += 1

                continue

            # ------------------------------------------------
            # Assistant回答が存在するか確認
            # ------------------------------------------------

            has_assistant = any(
                message["role"]
                == "assistant"
                and message["content"].strip()
                for message in messages
            )

            if not has_assistant:

                skipped += 1

                continue

            # ------------------------------------------------
            # Chat tokenizer
            # ------------------------------------------------

            tokens = tokenizer.encode_chat(
                messages,
                add_bos=True,
                add_eos=True,
            )

            if len(tokens) < 3:

                skipped += 1

                continue

            # ------------------------------------------------
            # max length
            # ------------------------------------------------

            if len(tokens) > self.max_length:

                tokens = tokens[
                    :self.max_length
                ]

                # 末尾が途中で切れた場合、
                # assistant lossが完全に消える可能性がある。
                #
                # その場合でも既存のassistant領域が
                # 残っていれば利用する。
            # ------------------------------------------------
            # input / labels
            # ------------------------------------------------

            input_ids = tokens[:-1]

            target_tokens = tokens[1:]

            # ------------------------------------------------
            # Assistant-only mask
            #
            # create_sft_labels() は token sequence 全体の
            # maskを作るため、target側に合わせる。
            # ------------------------------------------------

            full_labels = (
                tokenizer.create_sft_labels(
                    tokens,
                    ignore_index=IGNORE_INDEX,
                )
            )

            labels = full_labels[
                1:
            ]

            # ------------------------------------------------
            # 予測対象が存在するか
            # ------------------------------------------------

            valid_labels = [
                value
                for value in labels
                if value != IGNORE_INDEX
            ]

            if not valid_labels:

                skipped += 1

                continue

            if len(input_ids) != len(labels):

                raise RuntimeError(
                    "SFT input/label length mismatch."
                )

            self.samples.append(
                (
                    torch.tensor(
                        input_ids,
                        dtype=torch.long,
                    ),
                    torch.tensor(
                        labels,
                        dtype=torch.long,
                    ),
                )
            )

        if not self.samples:

            raise RuntimeError(
                "No valid SFT samples were created."
            )

        print(
            f"[DATA] Loaded records: "
            f"{len(records):,}"
        )

        print(
            f"[DATA] Valid samples: "
            f"{len(self.samples):,}"
        )

        print(
            f"[DATA] Skipped records: "
            f"{skipped:,}"
        )

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        return self.samples[
            index
        ]


# ============================================================
# Collate
# ============================================================

def collate_sft(
    batch,
):
    """
    padding付きbatchを作る。
    """

    max_length = max(
        ids.shape[0]
        for ids, _ in batch
    )

    input_ids = []

    labels = []

    attention_masks = []

    for ids, target in batch:

        length = ids.shape[0]

        padding = (
            max_length
            - length
        )

        if padding > 0:

            ids = F.pad(
                ids,
                (0, padding),
                value=ByteTokenizer.PAD,
            )

            target = F.pad(
                target,
                (0, padding),
                value=IGNORE_INDEX,
            )

        attention_mask = torch.ones(
            max_length,
            dtype=torch.long,
        )

        if padding > 0:

            attention_mask[
                -padding:
            ] = 0

        input_ids.append(
            ids
        )

        labels.append(
            target
        )

        attention_masks.append(
            attention_mask
        )

    return {
        "input_ids": torch.stack(
            input_ids
        ),
        "labels": torch.stack(
            labels
        ),
        "attention_mask": torch.stack(
            attention_masks
        ),
    }


# ============================================================
# Loss
# ============================================================

def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    SFT loss。

    重要:
    input_idsとlabelsは既に1 tokenずれているため、
    ここで再度shiftしない。

    -100はassistant以外の領域。
    """

    if logits.ndim != 3:

        raise ValueError(
            "Expected logits shape "
            "[batch, sequence, vocab], "
            f"got {tuple(logits.shape)}"
        )

    if labels.ndim != 2:

        raise ValueError(
            "Expected labels shape "
            "[batch, sequence], "
            f"got {tuple(labels.shape)}"
        )

    if (
        logits.shape[0]
        != labels.shape[0]
        or logits.shape[1]
        != labels.shape[1]
    ):

        raise ValueError(
            "Logits and labels dimensions "
            "do not match."
        )

    valid_count = (
        labels != IGNORE_INDEX
    ).sum()

    if int(
        valid_count.item()
    ) == 0:

        raise RuntimeError(
            "No valid assistant labels "
            "exist in this batch."
        )

    return F.cross_entropy(
        logits.reshape(
            -1,
            logits.size(-1),
        ),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


# ============================================================
# Optimizer
# ============================================================

def create_optimizer(
    model: NovaLM,
):
    """
    AdamW optimizer。
    """

    decay = []

    no_decay = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:

            decay.append(
                parameter
            )

        else:

            no_decay.append(
                parameter
            )

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
        lr=SFT_LEARNING_RATE,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


# ============================================================
# Learning rate
# ============================================================

def cosine_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    warmup + cosine decay。
    """

    if step < warmup_steps:

        return (
            max_lr
            * float(step + 1)
            / float(
                max(
                    1,
                    warmup_steps,
                )
            )
        )

    progress = (
        float(
            step - warmup_steps
        )
        /
        float(
            max(
                1,
                total_steps
                - warmup_steps,
            )
        )
    )

    progress = min(
        max(
            progress,
            0.0,
        ),
        1.0,
    )

    cosine_value = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )

    return (
        min_lr
        + (
            max_lr
            - min_lr
        )
        * cosine_value
    )


# ============================================================
# Checkpoint validation
# ============================================================

def validate_checkpoint(
    checkpoint,
    tokenizer: ByteTokenizer,
):
    """
    checkpointとtokenizerの互換性を検証する。
    """

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    vocab_size = checkpoint.get(
        "tokenizer_vocab_size"
    )

    if vocab_size is not None:

        if int(vocab_size) != (
            tokenizer.vocab_size
        ):

            raise RuntimeError(
                "Tokenizer vocabulary mismatch: "
                f"checkpoint={vocab_size}, "
                f"current={tokenizer.vocab_size}"
            )

    special_tokens = checkpoint.get(
        "special_tokens"
    )

    if special_tokens is not None:

        saved = {
            str(key): int(value)
            for key, value
            in special_tokens.items()
        }

        current = {
            str(key): int(value)
            for key, value
            in tokenizer.SPECIAL_TOKENS.items()
        }

        if saved != current:

            raise RuntimeError(
                "Special-token mapping mismatch."
            )


# ============================================================
# Load pretrained
# ============================================================

def load_pretrained(
    model: NovaLM,
    tokenizer: ByteTokenizer,
):
    """
    pretrain_final.ptを読み込む。

    260語彙checkpointは拒否する。
    """

    checkpoint_path = None

    if PRETRAIN_CHECKPOINT.exists():

        checkpoint_path = (
            PRETRAIN_CHECKPOINT
        )

    elif PRETRAIN_LATEST_CHECKPOINT.exists():

        checkpoint_path = (
            PRETRAIN_LATEST_CHECKPOINT
        )

    if checkpoint_path is None:

        raise FileNotFoundError(
            "No pretrained checkpoint found.\n"
            f"Expected:\n"
            f"  {PRETRAIN_CHECKPOINT}\n"
            f"or\n"
            f"  {PRETRAIN_LATEST_CHECKPOINT}\n"
            "Run pretrain.py first."
        )

    print(
        "=" * 70
    )

    print(
        "[LOAD] Pretrained checkpoint:"
    )

    print(
        checkpoint_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    validate_checkpoint(
        checkpoint,
        tokenizer,
    )

    state_dict = checkpoint.get(
        "model"
    )

    if state_dict is None:

        raise RuntimeError(
            "Pretrained checkpoint does not "
            "contain 'model'."
        )

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as exc:

        raise RuntimeError(
            "Pretrained checkpoint is incompatible "
            "with the current model/tokenizer.\n"
            "This usually means the checkpoint was "
            "created with the old 260-token tokenizer."
        ) from exc

    print(
        "[LOAD] Pretrained model loaded successfully."
    )

    print(
        "=" * 70
    )

    return checkpoint_path


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    path: Path,
    model: NovaLM,
    optimizer,
    step: int,
    loss: float,
    tokenizer: ByteTokenizer,
):
    """
    SFT checkpointを保存する。
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "format_version": 2,

        "model": model.state_dict(),

        "optimizer": optimizer.state_dict(),

        "step": int(step),

        "loss": float(loss),

        "model_config": {
            key: value
            for key, value
            in vars(MODEL).items()
        },

        "tokenizer_type": "byte",

        "tokenizer_version": 2,

        "tokenizer_vocab_size": int(
            tokenizer.vocab_size
        ),

        "special_tokens": {
            key: int(value)
            for key, value
            in tokenizer.SPECIAL_TOKENS.items()
        },

        "saved_at": time.time(),
    }

    temporary_path = path.with_suffix(
        ".tmp.pt"
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    temporary_path.replace(
        path
    )

    if not path.exists():

        raise RuntimeError(
            f"Checkpoint save failed: {path}"
        )

    size_mb = (
        path.stat().st_size
        / 1024
        / 1024
    )

    print(
        f"[SAVE] {path}"
    )

    print(
        f"[SAVE] size={size_mb:.2f} MB"
    )


# ============================================================
# AMP
# ============================================================

def get_amp_dtype(
    device: torch.device,
):
    """
    AMP dtypeを決定する。
    """

    if device.type != "cuda":
        return None

    if torch.cuda.is_bf16_supported():

        return torch.bfloat16

    return torch.float16


# ============================================================
# Train
# ============================================================

def train_sft():
    """
    SFT本体。
    """

    set_seed(
        int(TRAIN.seed)
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        RUNTIME.device
    )

    tokenizer = load_tokenizer()

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    MODEL.vocab_size = (
        tokenizer.vocab_size
    )

    if MODEL.vocab_size != 264:

        raise RuntimeError(
            "Expected MODEL.vocab_size=264, "
            f"got {MODEL.vocab_size}"
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print(
        "=" * 70
    )

    print(
        "NovaLLM Supervised Fine-Tuning"
    )

    print(
        "=" * 70
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Vocabulary: "
        f"{MODEL.vocab_size}"
    )

    print(
        f"Sequence length: "
        f"{MODEL.max_seq_len}"
    )

    print(
        f"Dataset: "
        f"{DATASET_PATH}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = SFTDataset(
        path=DATASET_PATH,
        tokenizer=tokenizer,
        max_length=MODEL.max_seq_len,
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=max(
            1,
            int(
                TRAIN.batch_size
            ),
        ),
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        collate_fn=collate_sft,
        drop_last=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = NovaLM(
        MODEL
    )

    model.to(
        device
    )

    print(
        f"[MODEL] Parameters: "
        f"{count_parameters(model):,}"
    )

    # --------------------------------------------------------
    # Pretraining checkpoint
    # --------------------------------------------------------

    loaded_path = load_pretrained(
        model,
        tokenizer,
    )

    print(
        f"[MODEL] Base model: "
        f"{loaded_path}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = create_optimizer(
        model
    )

    # --------------------------------------------------------
    # Training steps
    # --------------------------------------------------------

    steps_per_epoch = max(
        1,
        len(loader),
    )

    total_steps = min(
        MAX_SFT_STEPS,
        max(
            1,
            steps_per_epoch
            * SFT_EPOCHS,
        ),
    )

    warmup_steps = min(
        500,
        max(
            1,
            total_steps // 10,
        ),
    )

    print(
        f"[TRAIN] Samples: "
        f"{len(dataset):,}"
    )

    print(
        f"[TRAIN] Steps/epoch: "
        f"{steps_per_epoch:,}"
    )

    print(
        f"[TRAIN] Epochs: "
        f"{SFT_EPOCHS}"
    )

    print(
        f"[TRAIN] Total steps: "
        f"{total_steps:,}"
    )

    print(
        f"[TRAIN] Warmup steps: "
        f"{warmup_steps:,}"
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    amp_dtype = get_amp_dtype(
        device
    )

    use_amp = (
        amp_dtype is not None
    )

    print(
        f"[TRAIN] AMP: "
        f"{use_amp}"
    )

    if amp_dtype is not None:

        print(
            f"[TRAIN] AMP dtype: "
            f"{amp_dtype}"
        )

    # --------------------------------------------------------
    # Initial checkpoint
    # --------------------------------------------------------

    save_checkpoint(
        SFT_LATEST_CHECKPOINT,
        model,
        optimizer,
        0,
        float("inf"),
        tokenizer,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    iterator = iter(
        loader
    )

    last_loss = float(
        "inf"
    )

    for step in range(
        1,
        total_steps + 1,
    ):

        try:

            batch = next(
                iterator
            )

        except StopIteration:

            iterator = iter(
                loader
            )

            batch = next(
                iterator
            )

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

        attention_mask = batch.get(
            "attention_mask"
        )

        if attention_mask is not None:

            attention_mask = (
                attention_mask.to(
                    device,
                    non_blocking=True,
                )
            )

        # ----------------------------------------------------
        # LR
        # ----------------------------------------------------

        lr = cosine_lr(
            step - 1,
            total_steps,
            warmup_steps,
            SFT_LEARNING_RATE,
            SFT_MIN_LEARNING_RATE,
        )

        for group in (
            optimizer.param_groups
        ):

            group["lr"] = lr

        # ----------------------------------------------------
        # Zero grad
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        try:

            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                ):

                    output = model(
                        input_ids
                    )

                    logits = output[
                        "logits"
                    ]

                    loss = compute_sft_loss(
                        logits,
                        labels,
                    )

            else:

                output = model(
                    input_ids
                )

                logits = output[
                    "logits"
                ]

                loss = compute_sft_loss(
                    logits,
                    labels,
                )

            # ------------------------------------------------
            # NaN / Inf
            # ------------------------------------------------

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite SFT loss: "
                    f"{loss.item()}"
                )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )
            )

            if not torch.isfinite(
                grad_norm
            ):

                raise RuntimeError(
                    "Non-finite gradient norm."
                )

            # ------------------------------------------------
            # Optimizer
            # ------------------------------------------------

            optimizer.step()

            last_loss = float(
                loss.detach().item()
            )

        except torch.cuda.OutOfMemoryError:

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

            print()
            print(
                "[ERROR] CUDA out of memory."
            )

            print(
                "[INFO] Reduce "
                "TRAIN.batch_size or "
                "MODEL.max_seq_len."
            )

            save_checkpoint(
                SFT_LATEST_CHECKPOINT,
                model,
                optimizer,
                step,
                last_loss,
                tokenizer,
            )

            raise

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if (
            step == 1
            or step % LOG_EVERY == 0
            or step == total_steps
        ):

            print(
                f"SFT "
                f"step={step:6d}/"
                f"{total_steps:<6d} "
                f"loss={last_loss:.6f} "
                f"lr={lr:.3e} "
                f"grad={float(grad_norm):.4f}"
            )

        # ----------------------------------------------------
        # Periodic save
        # ----------------------------------------------------

        if (
            step % SAVE_EVERY == 0
            or step == total_steps
        ):

            save_checkpoint(
                SFT_LATEST_CHECKPOINT,
                model,
                optimizer,
                step,
                last_loss,
                tokenizer,
            )

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    print(
        "=" * 70
    )

    print(
        "[TRAIN] SFT finished."
    )

    save_checkpoint(
        SFT_FINAL_CHECKPOINT,
        model,
        optimizer,
        total_steps,
        last_loss,
        tokenizer,
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    if not SFT_FINAL_CHECKPOINT.exists():

        raise RuntimeError(
            "sft_final.pt was not created."
        )

    final_size = (
        SFT_FINAL_CHECKPOINT.stat().st_size
        / 1024
        / 1024
    )

    print(
        "=" * 70
    )

    print(
        "[SUCCESS] sft_final.pt created."
    )

    print(
        f"Path: "
        f"{SFT_FINAL_CHECKPOINT.resolve()}"
    )

    print(
        f"Size: "
        f"{final_size:.2f} MB"
    )

    print(
        f"Final loss: "
        f"{last_loss:.6f}"
    )

    print(
        f"Vocabulary: "
        f"{tokenizer.vocab_size}"
    )

    print(
        "=" * 70
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    try:

        train_sft()

    except KeyboardInterrupt:

        print()
        print(
            "[STOP] SFT interrupted."
        )

        print(
            "[INFO] sft_latest.pt should "
            "contain the latest saved state."
        )

    except Exception as exc:

        print()
        print(
            "=" * 70
        )

        print(
            "[FATAL] SFT failed."
        )

        print(
            f"[FATAL] "
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "=" * 70
        )

        raise
