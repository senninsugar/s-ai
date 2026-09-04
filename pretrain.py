from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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

DATA_DIR = (
    ROOT_DIR
    / TRAIN.data_dir
    / "pretrain"
)

CHECKPOINT_DIR = (
    ROOT_DIR
    / TRAIN.checkpoint_dir
)

LATEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "latest.pt"
)

FINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "pretrain_final.pt"
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

LOG_EVERY = 10

SAVE_EVERY = max(
    1,
    int(TRAIN.save_every),
)

EVAL_EVERY = max(
    1,
    int(TRAIN.eval_every),
)


# ============================================================
# Seed
# ============================================================

def set_seed(seed: int):
    """
    全乱数を固定する。
    """

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Device
# ============================================================

def get_device() -> torch.device:
    """
    RUNTIME.device から torch.device を作る。
    """

    return torch.device(
        RUNTIME.device
    )


# ============================================================
# Tokenizer
# ============================================================

def load_tokenizer() -> ByteTokenizer:
    """
    tokenizer/tokenizer.json を読み込む。

    ファイルが無い場合は新しく作成する。
    """

    if TOKENIZER_PATH.exists():

        tokenizer = ByteTokenizer.load(
            TOKENIZER_PATH
        )

    else:

        print(
            "[INFO] tokenizer.json was not found."
        )

        print(
            "[INFO] Creating a new tokenizer."
        )

        tokenizer = ByteTokenizer()

        tokenizer.save(
            TOKENIZER_PATH
        )

    if tokenizer.vocab_size != 264:

        raise RuntimeError(
            "NovaLLM requires the new 264-token "
            f"tokenizer, but got "
            f"{tokenizer.vocab_size}."
        )

    return tokenizer


# ============================================================
# File loading
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".text",
    ".jsonl",
    ".json",
}


def read_text_file(path: Path) -> str:
    """
    UTF-8 text file を読み込む。
    """

    try:

        return path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )


# ============================================================
# JSON extraction
# ============================================================

def extract_text_from_object(
    obj,
) -> List[str]:
    """
    JSON / JSONL の様々な形式から
    学習用テキストを抽出する。
    """

    result: List[str] = []

    if obj is None:
        return result

    # --------------------------------------------------------
    # String
    # --------------------------------------------------------

    if isinstance(obj, str):

        if obj.strip():
            result.append(
                obj
            )

        return result

    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    if isinstance(obj, list):

        for item in obj:

            result.extend(
                extract_text_from_object(
                    item
                )
            )

        return result

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(obj, dict):

        # 一般的な text field
        preferred_keys = [
            "text",
            "content",
            "document",
            "body",
            "article",
            "description",
        ]

        for key in preferred_keys:

            value = obj.get(key)

            if isinstance(value, str):

                if value.strip():

                    result.append(
                        value
                    )

                    return result

        # ----------------------------------------------------
        # prompt / completion
        # ----------------------------------------------------

        prompt = obj.get(
            "prompt"
        )

        completion = obj.get(
            "completion"
        )

        if (
            isinstance(prompt, str)
            or isinstance(completion, str)
        ):

            parts = []

            if isinstance(prompt, str):
                parts.append(prompt)

            if isinstance(completion, str):
                parts.append(completion)

            combined = "\n".join(
                parts
            ).strip()

            if combined:
                result.append(
                    combined
                )

            return result

        # ----------------------------------------------------
        # question / answer
        # ----------------------------------------------------

        question = obj.get(
            "question"
        )

        answer = obj.get(
            "answer"
        )

        if (
            isinstance(question, str)
            or isinstance(answer, str)
        ):

            parts = []

            if isinstance(question, str):
                parts.append(question)

            if isinstance(answer, str):
                parts.append(answer)

            combined = "\n".join(
                parts
            ).strip()

            if combined:
                result.append(
                    combined
                )

            return result

        # ----------------------------------------------------
        # messages
        # ----------------------------------------------------

        messages = obj.get(
            "messages"
        )

        if isinstance(messages, list):

            parts = []

            for message in messages:

                if not isinstance(
                    message,
                    dict,
                ):
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

                if content is None:
                    continue

                content = str(
                    content
                )

                if not content.strip():
                    continue

                parts.append(
                    f"<|{role}|>\n"
                    f"{content}\n"
                    f"<|end|>"
                )

            combined = "\n".join(
                parts
            ).strip()

            if combined:
                result.append(
                    combined
                )

            return result

        # ----------------------------------------------------
        # Fallback:
        # recursively inspect values
        # ----------------------------------------------------

        for value in obj.values():

            result.extend(
                extract_text_from_object(
                    value
                )
            )

        return result

    return result


# ============================================================
# Dataset text discovery
# ============================================================

def load_training_texts(
    data_dir: Path,
) -> List[str]:
    """
    pretrain データを読み込む。
    """

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = []

    for path in sorted(
        data_dir.rglob("*")
    ):

        if not path.is_file():
            continue

        if path.suffix.lower() in SUPPORTED_EXTENSIONS:

            files.append(path)

    if not files:

        example_path = (
            data_dir
            / "example.txt"
        )

        example_text = (
            "NovaLLMはローカルで動作する"
            "Transformerベースの言語モデルです。\n"
            "機械学習ではデータ、モデル、"
            "最適化、推論が重要です。\n"
            "日本語の文章を扱うためには"
            "UTF-8による正しいtokenizationが必要です。\n"
            "TransformerはAttention機構を利用して"
            "文章中のtoken同士の関係を学習します。\n"
        )

        example_path.write_text(
            example_text,
            encoding="utf-8",
        )

        files.append(
            example_path
        )

        print(
            "[INFO] Created example dataset:"
        )

        print(
            example_path
        )

    texts: List[str] = []

    for path in files:

        try:

            if path.suffix.lower() in {
                ".txt",
                ".text",
            }:

                text = read_text_file(
                    path
                )

                if text.strip():
                    texts.append(
                        text
                    )

            elif path.suffix.lower() == ".jsonl":

                content = read_text_file(
                    path
                )

                for line in content.splitlines():

                    line = line.strip()

                    if not line:
                        continue

                    try:

                        obj = json.loads(
                            line
                        )

                    except json.JSONDecodeError:

                        continue

                    texts.extend(
                        extract_text_from_object(
                            obj
                        )
                    )

            elif path.suffix.lower() == ".json":

                content = read_text_file(
                    path
                )

                try:

                    obj = json.loads(
                        content
                    )

                except json.JSONDecodeError:

                    continue

                texts.extend(
                    extract_text_from_object(
                        obj
                    )
                )

        except Exception as exc:

            print(
                f"[WARN] Failed to read "
                f"{path}: {exc}"
            )

    if not texts:

        raise RuntimeError(
            "No training text was found."
        )

    return texts


# ============================================================
# Token dataset
# ============================================================

class PretrainDataset(Dataset):
    """
    tokenized corpus を固定長 sequence に分割する。

    各 sample:

        input_ids = tokens[:-1]
        labels    = tokens[1:]

    model 側では labels を shift しない。
    """

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: ByteTokenizer,
        max_length: int,
    ):

        self.tokenizer = tokenizer

        self.max_length = max(
            2,
            int(max_length),
        )

        all_tokens: List[int] = []

        # BOSを各documentの先頭に付ける。
        # document境界にはEOSを付ける。
        for text in texts:

            if not isinstance(
                text,
                str,
            ):
                text = str(text)

            if not text.strip():
                continue

            tokens = tokenizer.encode(
                text,
                add_bos=True,
                add_eos=True,
            )

            all_tokens.extend(
                tokens
            )

        if len(all_tokens) < 3:

            raise RuntimeError(
                "Training corpus contains too few tokens."
            )

        self.samples = []

        stride = self.max_length

        # ----------------------------------------------------
        # 固定長 block
        # ----------------------------------------------------

        start = 0

        while (
            start + self.max_length
            <= len(all_tokens)
        ):

            block = all_tokens[
                start:
                start + self.max_length
            ]

            input_ids = block[:-1]

            labels = block[1:]

            if len(input_ids) >= 1:

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

            start += stride

        # ----------------------------------------------------
        # 残り
        # ----------------------------------------------------

        remaining = all_tokens[
            start:
        ]

        if len(remaining) >= 2:

            input_ids = remaining[:-1]

            labels = remaining[1:]

            # 最低限の長さがあれば使用
            if len(input_ids) >= 1:

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
                "No pretraining samples were created."
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

def collate_pretrain(batch):
    """
    batch内のsequenceをpaddingする。
    """

    max_length = max(
        item[0].shape[0]
        for item in batch
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

def compute_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Causal language modeling loss。

    重要:
    labelsは既に1 token右へshift済みなので、
    ここでは再shiftしない。
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
            "Logits and labels sequence "
            "dimensions do not match."
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
    model,
):
    """
    AdamW optimizer。

    2次元以上のparameterだけweight decay。
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
                "weight_decay": TRAIN.weight_decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=TRAIN.lr,
        betas=TRAIN.betas,
        eps=TRAIN.eps,
    )


# ============================================================
# Learning rate
# ============================================================

def get_learning_rate(
    step: int,
) -> float:
    """
    warmup + cosine decay。
    """

    warmup_steps = max(
        1,
        int(TRAIN.warmup),
    )

    max_steps = max(
        1,
        int(TRAIN.max_steps),
    )

    max_lr = float(
        TRAIN.lr
    )

    min_lr = float(
        TRAIN.min_lr
    )

    if step < warmup_steps:

        return (
            max_lr
            * float(step + 1)
            / float(warmup_steps)
        )

    progress = (
        float(
            step - warmup_steps
        )
        /
        float(
            max(
                1,
                max_steps - warmup_steps,
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

    cosine = (
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
        * cosine
    )


# ============================================================
# Checkpoint save
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
    checkpointを安全に保存する。
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "format_version": 2,

        "model": model.state_dict(),

        "optimizer": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),

        "step": int(step),

        "loss": float(loss),

        "model_config": {
            key: value
            for key, value
            in vars(MODEL).items()
        },

        "tokenizer_type": "byte",

        "tokenizer_vocab_size": int(
            tokenizer.vocab_size
        ),

        "tokenizer_version": 2,

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
# Checkpoint validation
# ============================================================

def validate_checkpoint_tokenizer(
    checkpoint,
    tokenizer: ByteTokenizer,
):
    """
    checkpointと現在のtokenizerが一致しているか確認する。
    """

    vocab_size = checkpoint.get(
        "tokenizer_vocab_size"
    )

    if vocab_size is not None:

        if int(vocab_size) != int(
            tokenizer.vocab_size
        ):

            raise RuntimeError(
                "Checkpoint tokenizer vocabulary "
                "does not match current tokenizer: "
                f"checkpoint={vocab_size}, "
                f"current={tokenizer.vocab_size}"
            )

    special_tokens = checkpoint.get(
        "special_tokens"
    )

    if special_tokens is not None:

        current = {
            key: int(value)
            for key, value
            in tokenizer.SPECIAL_TOKENS.items()
        }

        saved = {
            str(key): int(value)
            for key, value
            in special_tokens.items()
        }

        if saved != current:

            raise RuntimeError(
                "Checkpoint special-token mapping "
                "does not match current tokenizer."
            )


# ============================================================
# Checkpoint load
# ============================================================

def load_checkpoint(
    path: Path,
    model: NovaLM,
    optimizer,
    tokenizer: ByteTokenizer,
):
    """
    pretrain checkpointを再開用に読み込む。
    """

    if not path.exists():

        return 0, float("inf")

    print(
        "=" * 70
    )

    print(
        "[LOAD] Pretraining checkpoint"
    )

    print(
        path
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    validate_checkpoint_tokenizer(
        checkpoint,
        tokenizer,
    )

    state_dict = checkpoint.get(
        "model"
    )

    if state_dict is None:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model'."
        )

    try:

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError as exc:

        raise RuntimeError(
            "The checkpoint is incompatible "
            "with the current 264-token model. "
            "Delete old checkpoints and "
            "run pretraining again."
        ) from exc

    optimizer_state = checkpoint.get(
        "optimizer"
    )

    if (
        optimizer is not None
        and optimizer_state is not None
    ):

        optimizer.load_state_dict(
            optimizer_state
        )

    step = int(
        checkpoint.get(
            "step",
            0,
        )
    )

    loss = float(
        checkpoint.get(
            "loss",
            float("inf"),
        )
    )

    print(
        f"[LOAD] step={step}"
    )

    print(
        f"[LOAD] loss={loss}"
    )

    print(
        "[LOAD] Checkpoint loaded successfully."
    )

    print(
        "=" * 70
    )

    return step, loss


# ============================================================
# AMP
# ============================================================

def get_amp_dtype(
    device: torch.device,
):
    """
    CUDAで使用するAMP dtypeを決定する。
    """

    if device.type != "cuda":
        return None

    if (
        getattr(
            TRAIN,
            "use_bf16",
            False,
        )
        and torch.cuda.is_bf16_supported()
    ):

        return torch.bfloat16

    return torch.float16


# ============================================================
# Training
# ============================================================

def train():
    """
    NovaLLM pretraining本体。
    """

    set_seed(
        int(TRAIN.seed)
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    tokenizer = load_tokenizer()

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    MODEL.vocab_size = (
        tokenizer.vocab_size
    )

    if MODEL.vocab_size != 264:

        raise RuntimeError(
            "MODEL.vocab_size must be 264, "
            f"got {MODEL.vocab_size}"
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print(
        "=" * 70
    )

    print(
        "NovaLLM Pretraining"
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
        f"Batch size: "
        f"{TRAIN.batch_size}"
    )

    print(
        f"Gradient accumulation: "
        f"{TRAIN.grad_accumulation}"
    )

    print(
        f"Learning rate: "
        f"{TRAIN.lr}"
    )

    print(
        f"Max steps: "
        f"{TRAIN.max_steps}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    texts = load_training_texts(
        DATA_DIR
    )

    print(
        f"[DATA] Documents: "
        f"{len(texts):,}"
    )

    dataset = PretrainDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=MODEL.max_seq_len,
    )

    print(
        f"[DATA] Samples: "
        f"{len(dataset):,}"
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
        collate_fn=collate_pretrain,
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
    # Optimizer
    # --------------------------------------------------------

    optimizer = create_optimizer(
        model
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    start_step = 0

    last_loss = float(
        "inf"
    )

    # 既存のcheckpointがある場合、
    # 264 vocabなら再開する。
    if LATEST_CHECKPOINT.exists():

        try:

            (
                start_step,
                last_loss,
            ) = load_checkpoint(
                LATEST_CHECKPOINT,
                model,
                optimizer,
                tokenizer,
            )

        except Exception as exc:

            print(
                "[ERROR] Existing checkpoint "
                "cannot be loaded."
            )

            print(
                f"[ERROR] {type(exc).__name__}: {exc}"
            )

            print(
                "[INFO] Remove old checkpoints "
                "if this is a migration from "
                "the previous 260-token tokenizer."
            )

            raise

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
    # Gradient accumulation
    # --------------------------------------------------------

    accumulation_steps = max(
        1,
        int(
            TRAIN.grad_accumulation
        ),
    )

    # --------------------------------------------------------
    # Iterator
    # --------------------------------------------------------

    iterator = iter(
        loader
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    total_steps = max(
        1,
        int(
            TRAIN.max_steps
        ),
    )

    print(
        "=" * 70
    )

    print(
        f"[TRAIN] Starting from step "
        f"{start_step}"
    )

    print(
        f"[TRAIN] Target steps "
        f"{total_steps}"
    )

    print(
        "=" * 70
    )

    accumulated_loss = 0.0

    for step in range(
        start_step + 1,
        total_steps + 1,
    ):

        # ----------------------------------------------------
        # Learning rate
        # ----------------------------------------------------

        lr = get_learning_rate(
            step - 1
        )

        for group in (
            optimizer.param_groups
        ):

            group["lr"] = lr

        # ----------------------------------------------------
        # Accumulation
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        micro_loss = 0.0

        valid_micro_batches = 0

        for micro_step in range(
            accumulation_steps
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

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            try:

                if use_amp:

                    with torch.autocast(
                        device_type="cuda",
                        dtype=amp_dtype,
                    ):

                        output = model(
                            input_ids
                        )

                        loss = compute_lm_loss(
                            output[
                                "logits"
                            ],
                            labels,
                        )

                else:

                    output = model(
                        input_ids
                    )

                    loss = compute_lm_loss(
                        output[
                            "logits"
                        ],
                        labels,
                    )

                if not torch.isfinite(
                    loss
                ):

                    raise RuntimeError(
                        "Non-finite training loss: "
                        f"{loss.item()}"
                    )

                scaled_loss = (
                    loss
                    / accumulation_steps
                )

                scaled_loss.backward()

                micro_loss += float(
                    loss.detach().item()
                )

                valid_micro_batches += 1

            except torch.cuda.OutOfMemoryError:

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

                print()
                print(
                    "[ERROR] CUDA out of memory."
                )

                print(
                    "[INFO] Reduce "
                    "TRAIN.batch_size, "
                    "TRAIN.grad_accumulation, "
                    "or MODEL.max_seq_len."
                )

                save_checkpoint(
                    LATEST_CHECKPOINT,
                    model,
                    optimizer,
                    step,
                    last_loss,
                    tokenizer,
                )

                raise

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(
                    TRAIN.grad_clip
                ),
            )
        )

        if not torch.isfinite(
            grad_norm
        ):

            raise RuntimeError(
                "Non-finite gradient norm."
            )

        # ----------------------------------------------------
        # Optimizer step
        # ----------------------------------------------------

        optimizer.step()

        if valid_micro_batches > 0:

            last_loss = (
                micro_loss
                / valid_micro_batches
            )

        accumulated_loss = last_loss

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if (
            step == start_step + 1
            or step % LOG_EVERY == 0
            or step == total_steps
        ):

            print(
                f"PRETRAIN "
                f"step={step:7d}/"
                f"{total_steps:<7d} "
                f"loss={last_loss:.6f} "
                f"lr={lr:.3e} "
                f"grad={float(grad_norm):.4f}"
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        if (
            step % SAVE_EVERY == 0
            or step == total_steps
        ):

            save_checkpoint(
                LATEST_CHECKPOINT,
                model,
                optimizer,
                step,
                last_loss,
                tokenizer,
            )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "=" * 70
    )

    print(
        "[TRAIN] Pretraining finished."
    )

    save_checkpoint(
        FINAL_CHECKPOINT,
        model,
        optimizer,
        total_steps,
        accumulated_loss,
        tokenizer,
    )

    if not FINAL_CHECKPOINT.exists():

        raise RuntimeError(
            "pretrain_final.pt was not created."
        )

    print(
        "=" * 70
    )

    print(
        "[SUCCESS] Pretraining complete."
    )

    print(
        f"[SUCCESS] Final checkpoint:"
    )

    print(
        FINAL_CHECKPOINT.resolve()
    )

    print(
        f"[SUCCESS] Final loss:"
        f" {accumulated_loss:.6f}"
    )

    print(
        "=" * 70
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    try:

        train()

    except KeyboardInterrupt:

        print()
        print(
            "[STOP] Pretraining interrupted."
        )

        print(
            "[INFO] latest.pt may contain "
            "the latest saved checkpoint."
        )

    except Exception as exc:

        print()
        print(
            "=" * 70
        )

        print(
            "[FATAL] Pretraining failed."
        )

        print(
            f"[FATAL] "
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "=" * 70
        )

        raise
