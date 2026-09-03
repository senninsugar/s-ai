from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import MODEL, RUNTIME
from model import NovaLM
from tokenizer import ByteTokenizer


class NovaInference:

    def __init__(
        self,
        checkpoint: str,
        device: str | None = None,
    ):

        self.device = (
            device
            or RUNTIME.device
        )

        self.tokenizer = (
            ByteTokenizer()
        )

        MODEL.vocab_size = (
            self.tokenizer.vocab_size
        )

        self.model = NovaLM(
            MODEL
        )

        checkpoint_data = torch.load(
            checkpoint,
            map_location=self.device,
            weights_only=False,
        )

        state = checkpoint_data.get(
            "model",
            checkpoint_data,
        )

        self.model.load_state_dict(
            state,
            strict=True,
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            f"[MODEL] Loaded: {checkpoint}"
        )

    def build_prompt(
        self,
        messages,
    ):

        parts = []

        for message in messages:

            role = message[
                "role"
            ]

            content = message[
                "content"
            ]

            parts.append(
                f"<|{role}|>\n"
                f"{content}\n"
                f"<|end|>"
            )

        parts.append(
            "<|assistant|>\n"
        )

        return "\n".join(
            parts
        )

    @torch.no_grad()
    def generate(
        self,
        messages,
        max_new_tokens=256,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    ):

        prompt = self.build_prompt(
            messages
        )

        tokens = self.tokenizer.encode(
            prompt,
            add_bos=True,
            add_eos=False,
        )

        if len(tokens) >= MODEL.max_seq_len:

            tokens = tokens[
                -MODEL.max_seq_len:
            ]

        input_ids = torch.tensor(
            [tokens],
            dtype=torch.long,
            device=self.device,
        )

        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.EOS,
        )

        generated = output[
            0,
            input_ids.shape[1]:,
        ].tolist()

        text = self.tokenizer.decode(
            generated
        )

        if "<|end|>" in text:
            text = text.split(
                "<|end|>",
                1
            )[0]

        return text.strip()


def interactive(
    engine: NovaInference,
):

    messages = []

    print()
    print(
        "=" * 70
    )

    print(
        "NovaLLM interactive mode"
    )

    print(
        "終了: exit / quit"
    )

    print(
        "=" * 70
    )

    while True:

        try:

            user = input(
                "\nYou > "
            )

        except (
            EOFError,
            KeyboardInterrupt,
        ):

            print()
            break

        if user.strip().lower() in (
            "exit",
            "quit",
        ):
            break

        if not user.strip():
            continue

        messages.append(
            {
                "role": "user",
                "content": user,
            }
        )

        answer = engine.generate(
            messages
        )

        print(
            f"\nNova > {answer}"
        )

        messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "sft_final.pt"
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    if not Path(
        args.checkpoint
    ).exists():

        raise FileNotFoundError(
            "Checkpoint not found: "
            + args.checkpoint
        )

    engine = NovaInference(
        checkpoint=args.checkpoint,
        device=args.device,
    )

    interactive(
        engine
    )


if __name__ == "__main__":
    main()
