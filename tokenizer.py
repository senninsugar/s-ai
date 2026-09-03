import json
from pathlib import Path


class ByteTokenizer:
    """
    UTF-8 byte tokenizer.

    0: PAD
    1: BOS
    2: EOS
    3: UNK
    4-259: byte values 0-255
    """

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    BYTE_OFFSET = 4

    def __init__(self):
        self.vocab_size = 260

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = True,
    ):
        data = text.encode(
            "utf-8",
            errors="replace"
        )

        tokens = []

        if add_bos:
            tokens.append(self.BOS)

        for b in data:
            tokens.append(
                self.BYTE_OFFSET + b
            )

        if add_eos:
            tokens.append(self.EOS)

        return tokens

    def decode(
        self,
        tokens,
        skip_special_tokens=True
    ):
        raw = bytearray()

        for token in tokens:

            if token < self.BYTE_OFFSET:

                if skip_special_tokens:
                    continue

                continue

            value = token - self.BYTE_OFFSET

            if 0 <= value <= 255:
                raw.append(value)

        return raw.decode(
            "utf-8",
            errors="replace"
        )

    def save(self, path):
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = {
            "type": "byte",
            "vocab_size": self.vocab_size,
            "pad": self.PAD,
            "bos": self.BOS,
            "eos": self.EOS,
            "unk": self.UNK,
            "byte_offset": self.BYTE_OFFSET,
        }

        path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {path}"
            )

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        tokenizer = cls()

        if data["vocab_size"] != tokenizer.vocab_size:
            raise ValueError(
                "Tokenizer vocabulary mismatch"
            )

        return tokenizer


def create_tokenizer(path="tokenizer/tokenizer.json"):
    tokenizer = ByteTokenizer()
    tokenizer.save(path)

    print(
        f"Tokenizer created: {path}"
    )

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    return tokenizer


if __name__ == "__main__":

    tokenizer = create_tokenizer()

    text = (
        "こんにちは、NovaLLMです。\n"
        "Hello, world!"
    )

    tokens = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True
    )

    print("Tokens:")
    print(tokens)

    restored = tokenizer.decode(tokens)

    print("\nDecoded:")
    print(restored)
