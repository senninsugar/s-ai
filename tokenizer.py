from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Union


class ByteTokenizer:
    """
    NovaLLM 用のバイトベース tokenizer。

    基本的な文字は UTF-8 のバイト列として扱い、
    チャットで使用する特殊トークンは専用 ID を割り当てる。

    特殊トークン:
        <|pad|>
        <|bos|>
        <|eos|>
        <|unk|>
        <|system|>
        <|user|>
        <|assistant|>
        <|end|>
    """

    # ============================================================
    # Special token IDs
    # ============================================================

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    SYSTEM = 4
    USER = 5
    ASSISTANT = 6
    END = 7

    # 通常の UTF-8 byte token はここから開始
    BYTE_OFFSET = 8

    # byte 0〜255
    BYTE_VOCAB_SIZE = 256

    # 特殊トークン
    SPECIAL_TOKENS = {
        "<|pad|>": PAD,
        "<|bos|>": BOS,
        "<|eos|>": EOS,
        "<|unk|>": UNK,
        "<|system|>": SYSTEM,
        "<|user|>": USER,
        "<|assistant|>": ASSISTANT,
        "<|end|>": END,
    }

    ID_TO_SPECIAL = {
        value: key
        for key, value in SPECIAL_TOKENS.items()
    }

    SPECIAL_TOKEN_LIST = [
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|unk|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
    ]

    # 8 special + 256 bytes
    VOCAB_SIZE = BYTE_OFFSET + BYTE_VOCAB_SIZE

    def __init__(self):
        self.vocab_size = self.VOCAB_SIZE

        self.pad_token_id = self.PAD
        self.bos_token_id = self.BOS
        self.eos_token_id = self.EOS
        self.unk_token_id = self.UNK

        self.system_token_id = self.SYSTEM
        self.user_token_id = self.USER
        self.assistant_token_id = self.ASSISTANT
        self.end_token_id = self.END

    # ============================================================
    # Properties
    # ============================================================

    @property
    def pad_token(self) -> str:
        return "<|pad|>"

    @property
    def bos_token(self) -> str:
        return "<|bos|>"

    @property
    def eos_token(self) -> str:
        return "<|eos|>"

    @property
    def unk_token(self) -> str:
        return "<|unk|>"

    @property
    def system_token(self) -> str:
        return "<|system|>"

    @property
    def user_token(self) -> str:
        return "<|user|>"

    @property
    def assistant_token(self) -> str:
        return "<|assistant|>"

    @property
    def end_token(self) -> str:
        return "<|end|>"

    # ============================================================
    # Token helpers
    # ============================================================

    def is_special_token(self, token_id: int) -> bool:
        """
        token_id が特殊トークンか判定する。
        """

        try:
            token_id = int(token_id)
        except (TypeError, ValueError):
            return False

        return token_id in self.ID_TO_SPECIAL

    def is_byte_token(self, token_id: int) -> bool:
        """
        token_id が通常の byte token か判定する。
        """

        try:
            token_id = int(token_id)
        except (TypeError, ValueError):
            return False

        return (
            self.BYTE_OFFSET
            <= token_id
            < self.BYTE_OFFSET + self.BYTE_VOCAB_SIZE
        )

    def token_to_id(self, token: str) -> int:
        """
        特殊トークンを ID に変換する。

        通常文字そのものはこのメソッドでは扱わない。
        """

        if token in self.SPECIAL_TOKENS:
            return self.SPECIAL_TOKENS[token]

        return self.UNK

    def id_to_token(self, token_id: int) -> str:
        """
        ID を token 文字列に変換する。
        """

        try:
            token_id = int(token_id)
        except (TypeError, ValueError):
            return "<|unk|>"

        if token_id in self.ID_TO_SPECIAL:
            return self.ID_TO_SPECIAL[token_id]

        if self.is_byte_token(token_id):
            value = token_id - self.BYTE_OFFSET

            try:
                return bytes([value]).decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                return "<|unk|>"

        return "<|unk|>"

    # ============================================================
    # Encode
    # ============================================================

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = True,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        テキストを token ID 配列へ変換する。

        特殊トークン:
            <|system|>
            <|user|>
            <|assistant|>
            <|end|>

        は UTF-8 byte に分解せず、専用 token として扱う。

        例:

            <|assistant|>
            こんにちは

        は

            [ASSISTANT, UTF8 byte..., ...]

        になる。
        """

        if not isinstance(text, str):
            text = str(text)

        tokens: List[int] = []

        if add_bos:
            tokens.append(self.BOS)

        if not add_special_tokens:
            data = text.encode(
                "utf-8",
                errors="replace",
            )

            for byte in data:
                tokens.append(
                    self.BYTE_OFFSET + byte
                )

        else:
            tokens.extend(
                self._encode_with_special_tokens(text)
            )

        if add_eos:
            tokens.append(self.EOS)

        return tokens

    def _encode_with_special_tokens(
        self,
        text: str,
    ) -> List[int]:
        """
        特殊トークンを認識して encode する。
        """

        if not text:
            return []

        # 長い特殊トークンを先に処理する。
        special_tokens = sorted(
            self.SPECIAL_TOKEN_LIST,
            key=len,
            reverse=True,
        )

        tokens: List[int] = []

        i = 0
        length = len(text)

        while i < length:

            matched = False

            for special_token in special_tokens:

                if text.startswith(
                    special_token,
                    i,
                ):
                    tokens.append(
                        self.SPECIAL_TOKENS[
                            special_token
                        ]
                    )

                    i += len(special_token)

                    matched = True
                    break

            if matched:
                continue

            # 通常文字を処理する。
            #
            # Python の Unicode 文字列を
            # そのまま byte 化する。
            character = text[i]

            data = character.encode(
                "utf-8",
                errors="replace",
            )

            for byte in data:
                tokens.append(
                    self.BYTE_OFFSET + byte
                )

            i += 1

        return tokens

    # ============================================================
    # Decode
    # ============================================================

    def decode(
        self,
        tokens: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        token ID 配列を文字列に戻す。

        UTF-8 byte token をまとめて復元する。

        特殊トークンは、

            skip_special_tokens=True

        の場合は除外する。

        False の場合は文字列として復元する。
        """

        if tokens is None:
            return ""

        result: List[str] = []

        raw = bytearray()

        def flush_bytes():
            if not raw:
                return

            decoded = self._decode_utf8_safely(
                bytes(raw)
            )

            if decoded:
                result.append(decoded)

            raw.clear()

        for token in tokens:

            try:
                token = int(token)
            except (TypeError, ValueError):
                continue

            # ----------------------------------------------------
            # Special token
            # ----------------------------------------------------

            if token in self.ID_TO_SPECIAL:

                flush_bytes()

                if skip_special_tokens:
                    continue

                result.append(
                    self.ID_TO_SPECIAL[token]
                )

                continue

            # ----------------------------------------------------
            # Byte token
            # ----------------------------------------------------

            if self.is_byte_token(token):

                value = (
                    token
                    - self.BYTE_OFFSET
                )

                raw.append(value)

                continue

            # ----------------------------------------------------
            # Unknown token
            # ----------------------------------------------------

            flush_bytes()

            if not skip_special_tokens:
                result.append(
                    self.ID_TO_SPECIAL[self.UNK]
                )

        flush_bytes()

        return "".join(result)

    # ============================================================
    # UTF-8 decoder
    # ============================================================

    @staticmethod
    def _decode_utf8_safely(
        data: bytes,
    ) -> str:
        """
        壊れた UTF-8 byte 列でも可能な限り復元する。

        不正な byte はスキップする。
        """

        if not data:
            return ""

        result: List[str] = []

        i = 0
        length = len(data)

        while i < length:

            first = data[i]

            # ----------------------------------------------------
            # ASCII
            # ----------------------------------------------------

            if first < 0x80:

                result.append(
                    chr(first)
                )

                i += 1

                continue

            # ----------------------------------------------------
            # 2-byte UTF-8
            # ----------------------------------------------------

            if 0xC2 <= first <= 0xDF:

                needed = 2

            # ----------------------------------------------------
            # 3-byte UTF-8
            # ----------------------------------------------------

            elif 0xE0 <= first <= 0xEF:

                needed = 3

            # ----------------------------------------------------
            # 4-byte UTF-8
            # ----------------------------------------------------

            elif 0xF0 <= first <= 0xF4:

                needed = 4

            else:

                i += 1

                continue

            # ----------------------------------------------------
            # 長さチェック
            # ----------------------------------------------------

            if i + needed > length:
                break

            sequence = data[
                i:i + needed
            ]

            # ----------------------------------------------------
            # continuation byte check
            # ----------------------------------------------------

            valid = True

            for j in range(
                1,
                needed,
            ):

                if not (
                    0x80
                    <= sequence[j]
                    <= 0xBF
                ):
                    valid = False
                    break

            if not valid:

                i += 1

                continue

            # ----------------------------------------------------
            # UTF-8 overlong / surrogate / range checks
            # ----------------------------------------------------

            if needed == 3:

                second = sequence[1]

                # E0 80..9F は overlong
                if first == 0xE0 and second < 0xA0:
                    i += 1
                    continue

                # ED A0..BF は surrogate
                if first == 0xED and second >= 0xA0:
                    i += 1
                    continue

            elif needed == 4:

                second = sequence[1]

                # F0 80..8F は overlong
                if first == 0xF0 and second < 0x90:
                    i += 1
                    continue

                # F4 90..BF は Unicode 範囲外
                if first == 0xF4 and second > 0x8F:
                    i += 1
                    continue

            # ----------------------------------------------------
            # Decode
            # ----------------------------------------------------

            try:

                text = sequence.decode(
                    "utf-8"
                )

            except UnicodeDecodeError:

                i += 1

                continue

            result.append(text)

            i += needed

        return "".join(result)

    # ============================================================
    # Chat helpers
    # ============================================================

    def encode_chat_message(
        self,
        role: str,
        content: str,
        add_eos: bool = False,
    ) -> List[int]:
        """
        1つの chat message を token 化する。

        形式:

            <|role|>
            content
            <|end|>
        """

        role = str(role).lower().strip()

        if role == "system":
            role_id = self.SYSTEM

        elif role == "user":
            role_id = self.USER

        elif role == "assistant":
            role_id = self.ASSISTANT

        else:
            role_id = self.UNK

        tokens = [role_id]

        tokens.extend(
            self.encode(
                content,
                add_bos=False,
                add_eos=False,
                add_special_tokens=True,
            )
        )

        tokens.append(self.END)

        if add_eos:
            tokens.append(self.EOS)

        return tokens

    def encode_chat(
        self,
        messages,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        """
        OpenAI-style messages を token 化する。

        messages:

            [
                {
                    "role": "system",
                    "content": "..."
                },
                {
                    "role": "user",
                    "content": "..."
                },
                {
                    "role": "assistant",
                    "content": "..."
                }
            ]
        """

        tokens: List[int] = []

        if add_bos:
            tokens.append(self.BOS)

        for message in messages:

            if not isinstance(
                message,
                dict,
            ):
                continue

            role = message.get(
                "role",
                "user",
            )

            content = message.get(
                "content",
                "",
            )

            if content is None:
                content = ""

            tokens.extend(
                self.encode_chat_message(
                    role=role,
                    content=str(content),
                    add_eos=False,
                )
            )

        if add_eos:
            tokens.append(self.EOS)

        return tokens

    def encode_assistant_prompt(
        self,
        messages,
        add_bos: bool = True,
    ) -> List[int]:
        """
        チャット推論用。

        既存の messages を token 化し、
        最後に assistant token を追加する。

        例:

            user:
                こんにちは

        →

            <|bos|>
            <|user|>
            こんにちは
            <|end|>
            <|assistant|>
        """

        tokens = self.encode_chat(
            messages,
            add_bos=add_bos,
            add_eos=False,
        )

        tokens.append(
            self.ASSISTANT
        )

        return tokens

    # ============================================================
    # Assistant loss mask
    # ============================================================

    def create_assistant_mask(
        self,
        tokens: List[int],
    ) -> List[int]:
        """
        SFT 用 assistant-only loss mask を作成する。

        assistant の回答部分だけ 1。
        system/user 部分は 0。

        <|assistant|>
        こんにちは
        <|end|>

        の場合、

            assistant token -> 0
            こんにちは      -> 1
            <|end|>          -> 1

        とする。

        CrossEntropyLoss で使う場合は、
        0 の場所を -100 に変換する。
        """

        mask = [0] * len(tokens)

        in_assistant = False

        for i, token in enumerate(tokens):

            token = int(token)

            # assistant 開始
            if token == self.ASSISTANT:

                in_assistant = True

                # assistant 自体は
                # 予測対象にしない
                mask[i] = 0

                continue

            # end
            if token == self.END:

                if in_assistant:
                    mask[i] = 1

                in_assistant = False

                continue

            # EOS
            if token == self.EOS:

                if in_assistant:
                    mask[i] = 1

                continue

            # assistant content
            if in_assistant:

                mask[i] = 1

        return mask

    def create_sft_labels(
        self,
        tokens: List[int],
        ignore_index: int = -100,
    ) -> List[int]:
        """
        SFT 用 labels を作成する。

        assistant 以外は ignore_index。

        重要:
        input_ids と labels は同じ長さで返す。
        """

        if not tokens:
            return []

        mask = self.create_assistant_mask(
            tokens
        )

        labels = []

        for token, enabled in zip(
            tokens,
            mask,
        ):

            if enabled:
                labels.append(
                    int(token)
                )
            else:
                labels.append(
                    ignore_index
                )

        return labels

    # ============================================================
    # Vocabulary
    # ============================================================

    def get_vocab(self):
        """
        全 vocabulary を dict で返す。
        """

        vocab = {}

        for token, token_id in self.SPECIAL_TOKENS.items():
            vocab[token] = token_id

        for byte in range(
            self.BYTE_VOCAB_SIZE
        ):

            token_id = (
                self.BYTE_OFFSET
                + byte
            )

            vocab[
                f"<byte:{byte}>"
            ] = token_id

        return vocab

    def get_special_tokens(self):
        """
        特殊 token 情報を返す。
        """

        return dict(
            self.SPECIAL_TOKENS
        )

    # ============================================================
    # Save
    # ============================================================

    def save(self, path):
        """
        tokenizer 設定を JSON に保存する。
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "type": "byte",
            "version": 2,

            "vocab_size": self.vocab_size,

            "pad": self.PAD,
            "bos": self.BOS,
            "eos": self.EOS,
            "unk": self.UNK,

            "system": self.SYSTEM,
            "user": self.USER,
            "assistant": self.ASSISTANT,
            "end": self.END,

            "byte_offset": self.BYTE_OFFSET,
            "byte_vocab_size": self.BYTE_VOCAB_SIZE,

            "special_tokens": self.SPECIAL_TOKENS,
        }

        path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ============================================================
    # Load
    # ============================================================

    @classmethod
    def load(cls, path):
        """
        tokenizer 設定を読み込む。

        古い tokenizer.json との互換性は意図的に
        厳しくチェックする。

        vocab が違う状態で checkpoint を読み込むと
        embedding が一致しなくなるため。
        """

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(
                f"Tokenizer not found: {path}"
            )

        try:

            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                f"Invalid tokenizer file: {path}"
            ) from exc

        tokenizer_type = data.get(
            "type"
        )

        if tokenizer_type != "byte":

            raise ValueError(
                "Unsupported tokenizer type: "
                f"{tokenizer_type}"
            )

        tokenizer = cls()

        saved_vocab_size = data.get(
            "vocab_size"
        )

        if saved_vocab_size != tokenizer.vocab_size:

            raise ValueError(
                "Tokenizer vocabulary mismatch: "
                f"file={saved_vocab_size}, "
                f"runtime={tokenizer.vocab_size}"
            )

        saved_byte_offset = data.get(
            "byte_offset"
        )

        if saved_byte_offset != tokenizer.BYTE_OFFSET:

            raise ValueError(
                "Tokenizer byte offset mismatch: "
                f"file={saved_byte_offset}, "
                f"runtime={tokenizer.BYTE_OFFSET}"
            )

        saved_special_tokens = data.get(
            "special_tokens"
        )

        if saved_special_tokens is not None:

            normalized_saved = {
                str(key): int(value)
                for key, value
                in saved_special_tokens.items()
            }

            if normalized_saved != tokenizer.SPECIAL_TOKENS:

                raise ValueError(
                    "Tokenizer special-token "
                    "configuration mismatch."
                )

        return tokenizer

    # ============================================================
    # Debug
    # ============================================================

    def describe_tokens(
        self,
        tokens: Iterable[int],
    ):
        """
        token 配列を人間が確認しやすい形式で表示する。
        """

        result = []

        for token in tokens:

            try:
                token = int(token)
            except (TypeError, ValueError):

                result.append(
                    {
                        "id": None,
                        "token": "<invalid>",
                        "type": "invalid",
                    }
                )

                continue

            if token in self.ID_TO_SPECIAL:

                result.append(
                    {
                        "id": token,
                        "token": self.ID_TO_SPECIAL[token],
                        "type": "special",
                    }
                )

            elif self.is_byte_token(token):

                byte_value = (
                    token
                    - self.BYTE_OFFSET
                )

                result.append(
                    {
                        "id": token,
                        "token": f"<byte:{byte_value}>",
                        "type": "byte",
                    }
                )

            else:

                result.append(
                    {
                        "id": token,
                        "token": "<|unk|>",
                        "type": "unknown",
                    }
                )

        return result


# ================================================================
# Factory
# ================================================================

def create_tokenizer(
    path="tokenizer/tokenizer.json",
):
    """
    新しい tokenizer を作成して保存する。
    """

    tokenizer = ByteTokenizer()

    tokenizer.save(path)

    print(
        f"Tokenizer created: {path}"
    )

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    print(
        "Special tokens:"
    )

    for token, token_id in (
        tokenizer.SPECIAL_TOKENS.items()
    ):

        print(
            f"  {token}: {token_id}"
        )

    return tokenizer


# ================================================================
# Test
# ================================================================

def run_tests():
    """
    tokenizer の基本テスト。
    """

    tokenizer = ByteTokenizer()

    print("=" * 60)
    print("NovaLLM Tokenizer Test")
    print("=" * 60)

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    print()

    
    text = (
        "こんにちは、NovaLLMです。\n"
        "Hello, world!"
    )

    tokens = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
    )

    restored = tokenizer.decode(
        tokens,
        skip_special_tokens=True,
    )

    print(
        "Original:"
    )

    print(text)

    print()

    print(
        "Tokens:"
    )

    print(tokens)

    print()

    print(
        "Decoded:"
    )

    print(restored)

    print()

    assert restored == text, (
        "Basic encode/decode test failed"
    )


    chat_text = (
        "<|system|>\n"
        "あなたはNovaLLMです。"
        "<|end|>\n"
        "<|user|>\n"
        "こんにちは！"
        "<|end|>\n"
        "<|assistant|>\n"
        "こんにちは。"
        "<|end|>"
    )

    chat_tokens = tokenizer.encode(
        chat_text,
        add_bos=True,
        add_eos=False,
    )

    print(
        "Chat tokens:"
    )

    print(chat_tokens)

    print()

    print(
        "Token description:"
    )

    for item in tokenizer.describe_tokens(
        chat_tokens
    ):

        print(item)

    print()

    chat_restored = tokenizer.decode(
        chat_tokens,
        skip_special_tokens=False,
    )

    print(
        "Chat decoded:"
    )

    print(chat_restored)

    print()

    assert (
        "<|system|>"
        in chat_restored
    )

    assert (
        "<|user|>"
        in chat_restored
    )

    assert (
        "<|assistant|>"
        in chat_restored
    )

    assert (
        "<|end|>"
        in chat_restored
    )

    
    messages = [
        {
            "role": "system",
            "content": (
                "あなたはNovaLLMです。"
            ),
        },
        {
            "role": "user",
            "content": (
                "こんにちは"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "こんにちは！"
            ),
        },
    ]

    encoded_chat = tokenizer.encode_chat(
        messages,
        add_bos=True,
        add_eos=False,
    )

    print(
        "Encoded chat:"
    )

    print(encoded_chat)

    print()

    assistant_prompt = (
        tokenizer.encode_assistant_prompt(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたはNovaLLMです。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "こんにちは"
                    ),
                },
            ],
            add_bos=True,
        )
    )

    print(
        "Assistant prompt:"
    )

    print(assistant_prompt)

    print()

    decoded_prompt = tokenizer.decode(
        assistant_prompt,
        skip_special_tokens=False,
    )

    print(
        "Decoded assistant prompt:"
    )

    print(decoded_prompt)

    print()

    assert decoded_prompt.endswith(
        "<|assistant|>"
    )


    sft_tokens = tokenizer.encode_chat(
        [
            {
                "role": "user",
                "content": "1+1はいくつ？",
            },
            {
                "role": "assistant",
                "content": "2です。",
            },
        ],
        add_bos=True,
        add_eos=True,
    )

    mask = tokenizer.create_assistant_mask(
        sft_tokens
    )

    labels = tokenizer.create_sft_labels(
        sft_tokens
    )

    print(
        "SFT tokens:"
    )

    print(sft_tokens)

    print()

    print(
        "Assistant mask:"
    )

    print(mask)

    print()

    print(
        "SFT labels:"
    )

    print(labels)

    print()

    assert len(mask) == len(
        sft_tokens
    )

    assert len(labels) == len(
        sft_tokens
    )

    assert 1 in mask

    assert tokenizer.PAD == 0
    assert tokenizer.BOS == 1
    assert tokenizer.EOS == 2
    assert tokenizer.UNK == 3

    assert tokenizer.SYSTEM == 4
    assert tokenizer.USER == 5
    assert tokenizer.ASSISTANT == 6
    assert tokenizer.END == 7

    assert tokenizer.BYTE_OFFSET == 8
    assert tokenizer.vocab_size == 264

    
    print("=" * 60)
    print("ALL TOKENIZER TESTS PASSED")
    print("=" * 60)

    if __name__ == "__main__":

    tokenizer = create_tokenizer()

    print()

    run_tests()
