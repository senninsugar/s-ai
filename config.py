from dataclasses import dataclass
import os
import torch


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 2
    intermediate_size: int = 1408

    max_seq_len: int = 2048

    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    dropout: float = 0.0

    tie_embeddings: bool = True
@dataclass
class TrainConfig:
    seed: int = 42

    batch_size: int = 2
    grad_accumulation: int = 8

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5

    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8

    warmup_steps: int = 500
    max_steps: int = 100000

    grad_clip: float = 1.0

    save_every: int = 1000
    eval_every: int = 500

    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"

    use_bf16: bool = True
    gradient_checkpointing: bool = True


@dataclass
class RuntimeConfig:
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    num_workers: int = min(
        4,
        os.cpu_count() or 1
    )

    compile_model: bool = False

    @property
    def dtype(self):
        if self.device == "cuda":
            if (
                torch.cuda.is_bf16_supported()
                and TRAIN.use_bf16
            ):
                return torch.bfloat16

            return torch.float16

        return torch.float32


MODEL = ModelConfig()
TRAIN = TrainConfig()
RUNTIME = RuntimeConfig()


def print_config():
    print("=" * 60)
    print("NovaLLM configuration")
    print("=" * 60)

    print("\nModel")
    for k, v in MODEL.__dict__.items():
        print(f"  {k}: {v}")

    print("\nTraining")
    for k, v in TRAIN.__dict__.items():
        print(f"  {k}: {v}")

    print("\nRuntime")
    print(f"  device: {RUNTIME.device}")
    print(f"  dtype:   {RUNTIME.dtype}")

    print("=" * 60)


if __name__ == "__main__":
    print_config()
