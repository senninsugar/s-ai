from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader

from tokenizer import ByteTokenizer


class TextRecord:
    def __init__(self, text: str):
        self.text = text


def read_text_file(path: Path) -> Iterator[TextRecord]:
    try:
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    if text.strip():
        yield TextRecord(text)


def read_jsonl_file(path: Path) -> Iterator[TextRecord]:
    try:
        with path.open(
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
                except json.JSONDecodeError:
                    print(
                        f"[WARN] Invalid JSON "
                        f"{path}:{line_no}"
                    )
                    continue

                if isinstance(obj, str):
                    text = obj

                elif isinstance(obj, dict):
                    text = extract_text(obj)

                else:
                    continue

                if text and text.strip():
                    yield TextRecord(text)

    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")


def extract_text(obj: Dict[str, Any]) -> str:
    preferred_keys = [
        "text",
        "content",
        "document",
        "body",
        "prompt",
        "completion",
        "question",
        "answer",
    ]

    parts = []

    for key in preferred_keys:

        value = obj.get(key)

        if isinstance(value, str):
            value = value.strip()

            if value:
                parts.append(value)

    if parts:
        return "\n".join(parts)

    messages = obj.get("messages")

    if isinstance(messages, list):

        result = []

        for message in messages:

            if not isinstance(message, dict):
                continue

            role = message.get("role", "")
            content = message.get("content", "")

            if isinstance(content, str):
                result.append(
                    f"{role}: {content}"
                )

        return "\n".join(result)

    return ""


def discover_files(
    data_dir: str,
) -> List[Path]:

    root = Path(data_dir)

    if not root.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {root}"
        )

    files = []

    for extension in (
        "*.txt",
        "*.text",
        "*.jsonl",
        "*.json",
    ):
        files.extend(root.rglob(extension))

    files = sorted(
        set(files),
        key=lambda p: str(p),
    )

    return files


def iter_records(
    data_dir: str,
) -> Iterator[TextRecord]:

    files = discover_files(data_dir)

    if not files:
        raise RuntimeError(
            f"No training files found in {data_dir}"
        )

    for path in files:

        suffix = path.suffix.lower()

        if suffix in (
            ".txt",
            ".text",
        ):
            yield from read_text_file(path)

        elif suffix == ".jsonl":
            yield from read_jsonl_file(path)

        elif suffix == ".json":

            try:
                data = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception as e:
                print(
                    f"[WARN] Failed to read "
                    f"{path}: {e}"
                )
                continue

            if isinstance(data, list):

                for item in data:

                    if isinstance(item, str):
                        text = item

                    elif isinstance(item, dict):
                        text = extract_text(item)

                    else:
                        continue

                    if text.strip():
                        yield TextRecord(text)

            elif isinstance(data, dict):

                text = extract_text(data)

                if text.strip():
                    yield TextRecord(text)


class PackedTokenDataset(IterableDataset):

    def __init__(
        self,
        data_dir: str,
        tokenizer: ByteTokenizer,
        seq_len: int,
        seed: int = 42,
        shuffle_buffer: int = 1000,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def _token_stream(self):

        for record in iter_records(
            self.data_dir
        ):

            tokens = self.tokenizer.encode(
                record.text,
                add_bos=True,
                add_eos=True,
            )

            for token in tokens:
                yield token

    def _shuffle_stream(
        self,
        stream,
        rng,
    ):

        buffer = []

        for item in stream:

            buffer.append(item)

            if len(buffer) >= self.shuffle_buffer:

                index = rng.randrange(
                    len(buffer)
                )

                yield buffer.pop(index)

        while buffer:

            index = rng.randrange(
                len(buffer)
            )

            yield buffer.pop(index)

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        seed = self.seed

        if worker_info is not None:
            seed += worker_info.id

        rng = random.Random(seed)

        stream = self._token_stream()

        if self.shuffle_buffer > 0:
            stream = self._shuffle_stream(
                stream,
                rng,
            )

        block = []

        for token in stream:

            block.append(token)

            if len(block) == self.seq_len + 1:

                input_ids = block[:-1]
                labels = block[1:]

                yield (
                    torch.tensor(
                        input_ids,
                        dtype=torch.long,
                    ),
                    torch.tensor(
                        labels,
                        dtype=torch.long,
                    ),
                )

                block = []


class FinitePackedDataset(
    PackedTokenDataset
):
    """
    Dataset version that can be restarted
    deterministically for validation.
    """

    pass


def create_dataloader(
    data_dir: str,
    tokenizer: ByteTokenizer,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    shuffle_buffer: int = 1000,
):

    dataset = PackedTokenDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        seq_len=seq_len,
        seed=seed,
        shuffle_buffer=shuffle_buffer,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def estimate_tokens(
    data_dir: str,
    tokenizer: ByteTokenizer,
) -> int:

    total = 0

    for record in iter_records(
        data_dir
    ):
        total += len(
            tokenizer.encode(
                record.text,
                add_bos=True,
                add_eos=True,
            )
        )

    return total


def inspect_dataset(
    data_dir: str,
    tokenizer: ByteTokenizer,
    seq_len: int = 128,
    samples: int = 3,
):

    print("=" * 60)
    print("Dataset inspection")
    print("=" * 60)

    files = discover_files(data_dir)

    print(
        f"Files: {len(files)}"
    )

    for path in files:
        print(
            f"  {path}"
        )

    print()

    total_tokens = estimate_tokens(
        data_dir,
        tokenizer,
    )

    print(
        f"Estimated tokens: "
        f"{total_tokens:,}"
    )

    dataset = PackedTokenDataset(
        data_dir,
        tokenizer,
        seq_len,
        shuffle_buffer=0,
    )

    iterator = iter(dataset)

    for i in range(samples):

        try:
            input_ids, labels = next(
                iterator
            )
        except StopIteration:
            break

        print()
        print(
            f"Sample {i + 1}"
        )

        print(
            "Input shape:",
            tuple(input_ids.shape)
        )

        print(
            "Input text:"
        )

        print(
            tokenizer.decode(
                input_ids.tolist()
            )
        )

        print("-" * 60)


if __name__ == "__main__":

    tokenizer = ByteTokenizer()

    data_dir = "data/pretrain"

    inspect_dataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        seq_len=128,
    )
