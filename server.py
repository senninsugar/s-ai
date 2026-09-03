from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import MODEL, RUNTIME
from inference import NovaInference
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


CHECKPOINT = (
    "checkpoints/sft_final.pt"
)

HOST = "0.0.0.0"
PORT = 8000


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="NovaLLM API",
    description="Local OpenAI-compatible LLM API",
    version="1.0.0",
)
app.mount(
    "/ui",
    StaticFiles(directory="ui"),
    name="ui"
)

# ============================================================
# Global model
# ============================================================

engine: Optional[NovaInference] = None

model_lock = asyncio.Lock()


# ============================================================
# Request models
# ============================================================

class ChatMessage(BaseModel):

    role: str

    content: str


class ChatCompletionRequest(BaseModel):

    model: str = "novallm"

    messages: List[ChatMessage]

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
    )

    top_p: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
    )

    top_k: int = Field(
        default=50,
        ge=0,
        le=1000,
    )

    max_tokens: int = Field(
        default=256,
        ge=1,
        le=8192,
    )

    stream: bool = False


class CompletionRequest(BaseModel):

    model: str = "novallm"

    prompt: str

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
    )

    top_p: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
    )

    top_k: int = Field(
        default=50,
        ge=0,
        le=1000,
    )

    max_tokens: int = Field(
        default=256,
        ge=1,
        le=8192,
    )

    stream: bool = False


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
async def startup():

    global engine

    checkpoint = Path(
        CHECKPOINT
    )

    if not checkpoint.exists():

        print(
            "[WARNING] Checkpoint does not exist:"
        )

        print(
            checkpoint
        )

        print(
            "Run pretrain.py and "
            "posttrain.py first."
        )

        return

    print(
        "=" * 70
    )

    print(
        "Loading NovaLLM..."
    )

    print(
        f"Checkpoint: {checkpoint}"
    )

    print(
        f"Device: {RUNTIME.device}"
    )

    engine = NovaInference(
        checkpoint=str(
            checkpoint
        )
    )

    print(
        "NovaLLM ready."
    )

    print(
        "=" * 70
    )


# ============================================================
# Health
# ============================================================

@app.get("/")
async def root():

    return {
        "name": "NovaLLM",
        "version": "1.0.0",
        "status": (
            "ready"
            if engine is not None
            else "model_not_loaded"
        ),
    }


@app.get("/health")
async def health():

    return {
        "status": "ok",
        "model_loaded": (
            engine is not None
        ),
        "device": RUNTIME.device,
        "cuda": torch.cuda.is_available(),
    }


# ============================================================
# Models endpoint
# ============================================================

@app.get("/v1/models")
async def models():

    return {
        "object": "list",
        "data": [
            {
                "id": "novallm",
                "object": "model",
                "created": int(
                    time.time()
                ),
                "owned_by": "local",
            }
        ],
    }


# ============================================================
# Helpers
# ============================================================

def require_engine():

    if engine is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. "
                "Check the checkpoint path."
            ),
        )

    return engine


def convert_messages(
    messages: List[ChatMessage],
):

    return [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in messages
    ]


def create_chat_response(
    model_name: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
):

    created = int(
        time.time()
    )

    return {
        "id": (
            "chatcmpl-"
            + uuid.uuid4().hex
        ),
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                prompt_tokens
                + completion_tokens
            ),
        },
    }


# ============================================================
# Chat Completion
# ============================================================

@app.post(
    "/v1/chat/completions"
)
async def chat_completions(
    request: ChatCompletionRequest,
):

    current_engine = require_engine()

    if not request.messages:

        raise HTTPException(
            status_code=400,
            detail="messages cannot be empty",
        )

    messages = convert_messages(
        request.messages
    )

    if request.stream:

        return StreamingResponse(
            stream_chat(
                current_engine,
                messages,
                request,
            ),
            media_type=(
                "text/event-stream"
            ),
        )

    async with model_lock:

        try:

            prompt = (
                current_engine
                .build_prompt(messages)
            )

            prompt_tokens = len(
                current_engine
                .tokenizer
                .encode(
                    prompt,
                    add_bos=True,
                    add_eos=False,
                )
            )

            content = (
                current_engine.generate(
                    messages,
                    max_new_tokens=(
                        request.max_tokens
                    ),
                    temperature=(
                        request.temperature
                    ),
                    top_k=request.top_k,
                    top_p=request.top_p,
                )
            )

            completion_tokens = len(
                current_engine
                .tokenizer
                .encode(
                    content,
                    add_bos=False,
                    add_eos=False,
                )
            )

            return create_chat_response(
                request.model,
                content,
                prompt_tokens,
                completion_tokens,
            )

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=str(e),
            )


# ============================================================
# Streaming
# ============================================================

async def stream_chat(
    current_engine,
    messages,
    request,
):

    try:

        async with model_lock:

            prompt = (
                current_engine
                .build_prompt(messages)
            )

            prompt_tokens = len(
                current_engine
                .tokenizer
                .encode(
                    prompt,
                    add_bos=True,
                    add_eos=False,
                )
            )

            # 現在の推論エンジンは
            # 一括生成なので、
            # 生成結果を小さな単位に
            # 分割してSSEで返す。
            content = (
                current_engine.generate(
                    messages,
                    max_new_tokens=(
                        request.max_tokens
                    ),
                    temperature=(
                        request.temperature
                    ),
                    top_k=request.top_k,
                    top_p=request.top_p,
                )
            )

            words = content.split()

            response_id = (
                "chatcmpl-"
                + uuid.uuid4().hex
            )

            for index, word in enumerate(
                words
            ):

                payload = {
                    "id": response_id,
                    "object": (
                        "chat.completion.chunk"
                    ),
                    "created": int(
                        time.time()
                    ),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": (
                                    word
                                    + (
                                        " "
                                        if index
                                        < len(words) - 1
                                        else ""
                                    )
                                )
                            },
                            "finish_reason": None,
                        }
                    ],
                }

                yield (
                    "data: "
                    + json.dumps(
                        payload,
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

                await asyncio.sleep(
                    0
                )

            final_payload = {
                "id": response_id,
                "object": (
                    "chat.completion.chunk"
                ),
                "created": int(
                    time.time()
                ),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": (
                        prompt_tokens
                    ),
                    "completion_tokens": len(
                        current_engine
                        .tokenizer
                        .encode(
                            content,
                            add_bos=False,
                            add_eos=False,
                        )
                    ),
                },
            }

            yield (
                "data: "
                + json.dumps(
                    final_payload,
                    ensure_ascii=False,
                )
                + "\n\n"
            )

            yield "data: [DONE]\n\n"

    except Exception as e:

        payload = {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }

        yield (
            "data: "
            + json.dumps(
                payload,
                ensure_ascii=False,
            )
            + "\n\n"
        )


# ============================================================
# Classic Completion API
# ============================================================

@app.post(
    "/v1/completions"
)
async def completions(
    request: CompletionRequest,
):

    current_engine = require_engine()

    messages = [
        {
            "role": "user",
            "content": request.prompt,
        }
    ]

    async with model_lock:

        try:

            content = (
                current_engine.generate(
                    messages,
                    max_new_tokens=(
                        request.max_tokens
                    ),
                    temperature=(
                        request.temperature
                    ),
                    top_k=request.top_k,
                    top_p=request.top_p,
                )
            )

            prompt_tokens = len(
                current_engine
                .tokenizer
                .encode(
                    request.prompt,
                    add_bos=True,
                    add_eos=False,
                )
            )

            completion_tokens = len(
                current_engine
                .tokenizer
                .encode(
                    content,
                    add_bos=False,
                    add_eos=False,
                )
            )

            return {
                "id": (
                    "cmpl-"
                    + uuid.uuid4().hex
                ),
                "object": "text_completion",
                "created": int(
                    time.time()
                ),
                "model": request.model,
                "choices": [
                    {
                        "text": content,
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": (
                        completion_tokens
                    ),
                    "total_tokens": (
                        prompt_tokens
                        + completion_tokens
                    ),
                },
            }

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=str(e),
            )


class GenerateRequest(BaseModel):

    prompt: str

    max_new_tokens: int = 256

    temperature: float = 0.7

    top_k: int = 50

    top_p: float = 0.95

@app.get("/")
async def root():
    return FileResponse("ui/index.html")

@app.post("/generate")
async def generate(
    request: GenerateRequest,
):

    current_engine = require_engine()

    messages = [
        {
            "role": "user",
            "content": request.prompt,
        }
    ]

    async with model_lock:

        result = (
            current_engine.generate(
                messages,
                max_new_tokens=(
                    request.max_new_tokens
                ),
                temperature=(
                    request.temperature
                ),
                top_k=request.top_k,
                top_p=request.top_p,
            )
        )

    return {
        "text": result,
    }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        reload=False,
    )
