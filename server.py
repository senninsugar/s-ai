from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import RUNTIME
from inference import NovaInference

BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"
INDEX_FILE = UI_DIR / "index.html"
CHECKPOINT = BASE_DIR / "checkpoints" / "sft_final.pt"
HOST = "0.0.0.0"
PORT = 8000

engine: Optional[NovaInference] = None
model_lock = asyncio.Lock()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "novallm"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=1000)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    stream: bool = False

class CompletionRequest(BaseModel):
    model: str = "novallm"
    prompt: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=1000)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    stream: bool = False

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0, le=1000)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("=" * 70)
    print("NovaLLM server starting...")
    print(f"Base directory: {BASE_DIR}")
    print(f"UI directory: {UI_DIR}")
    print(f"Index file: {INDEX_FILE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {RUNTIME.device}")
    print("=" * 70)
    if not UI_DIR.exists():
        print(f"[ERROR] UI directory does not exist: {UI_DIR}")
    elif not INDEX_FILE.exists():
        print(f"[ERROR] UI index.html does not exist: {INDEX_FILE}")
    else:
        print("[OK] UI files found.")
    if not CHECKPOINT.exists():
        print("[WARNING] Checkpoint does not exist:")
        print(CHECKPOINT)
        print("Run pretrain.py and posttrain.py first.")
    else:
        try:
            print("Loading NovaLLM...")
            engine = NovaInference(checkpoint=str(CHECKPOINT))
            print("NovaLLM ready.")
        except Exception as e:
            engine = None
            print("[ERROR] Failed to load NovaLLM.")
            print(str(e))
    print("=" * 70)
    yield
    engine = None
    print("NovaLLM server stopped.")

app = FastAPI(
    title="NovaLLM API",
    description="Local OpenAI-compatible LLM API",
    version="1.0.0",
    lifespan=lifespan,
)

if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

@app.get("/", include_in_schema=False)
async def root():
    if not INDEX_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail=f"UI file not found: {INDEX_FILE}",
        )
    return FileResponse(
        path=str(INDEX_FILE),
        media_type="text/html",
    )

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_candidates = [
        UI_DIR / "favicon.ico",
        UI_DIR / "favicon.png",
        UI_DIR / "icon.png",
    ]
    for favicon_file in favicon_candidates:
        if favicon_file.exists():
            media_type = "image/x-icon" if favicon_file.suffix.lower() == ".ico" else "image/png"
            return FileResponse(
                path=str(favicon_file),
                media_type=media_type,
            )
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": engine is not None,
        "device": str(RUNTIME.device),
        "cuda": torch.cuda.is_available(),
        "ui": UI_DIR.exists() and INDEX_FILE.exists(),
    }

@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "novallm",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }

def require_engine() -> NovaInference:
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check checkpoints/sft_final.pt and server startup logs.",
        )
    return engine

def convert_messages(messages: List[ChatMessage]):
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
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
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
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    current_engine = require_engine()
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="messages cannot be empty",
        )
    messages = convert_messages(request.messages)
    if request.stream:
        return StreamingResponse(
            stream_chat(current_engine, messages, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    async with model_lock:
        try:
            prompt = current_engine.build_prompt(messages)
            prompt_tokens = len(
                current_engine.tokenizer.encode(
                    prompt,
                    add_bos=True,
                    add_eos=False,
                )
            )
            content = current_engine.generate(
                messages,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            )
            completion_tokens = len(
                current_engine.tokenizer.encode(
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
            print(f"[ERROR] Chat generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e),
            )

async def stream_chat(
    current_engine: NovaInference,
    messages,
    request: ChatCompletionRequest,
):
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    try:
        async with model_lock:
            prompt = current_engine.build_prompt(messages)
            prompt_tokens = len(
                current_engine.tokenizer.encode(
                    prompt,
                    add_bos=True,
                    add_eos=False,
                )
            )
            content = current_engine.generate(
                messages,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            )
            completion_tokens = len(
                current_engine.tokenizer.encode(
                    content,
                    add_bos=False,
                    add_eos=False,
                )
            )
        if content:
            chunks = []
            current = ""
            for char in content:
                current += char
                if char in " \n。！？、,.!?":
                    chunks.append(current)
                    current = ""
            if current:
                chunks.append(current)
            for chunk in chunks:
                payload = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk,
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
        final_payload = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        print(f"[ERROR] Streaming generation failed: {e}")
        payload = {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    current_engine = require_engine()
    messages = [
        {
            "role": "user",
            "content": request.prompt,
        }
    ]
    if request.stream:
        return StreamingResponse(
            stream_chat(
                current_engine,
                messages,
                ChatCompletionRequest(
                    model=request.model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=request.prompt,
                        )
                    ],
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    max_tokens=request.max_tokens,
                    stream=True,
                ),
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    async with model_lock:
        try:
            content = current_engine.generate(
                messages,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            )
            prompt_tokens = len(
                current_engine.tokenizer.encode(
                    request.prompt,
                    add_bos=True,
                    add_eos=False,
                )
            )
            completion_tokens = len(
                current_engine.tokenizer.encode(
                    content,
                    add_bos=False,
                    add_eos=False,
                )
            )
            return {
                "id": f"cmpl-{uuid.uuid4().hex}",
                "object": "text_completion",
                "created": int(time.time()),
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
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        except Exception as e:
            print(f"[ERROR] Completion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e),
            )

@app.post("/generate")
async def generate(request: GenerateRequest):
    current_engine = require_engine()
    messages = [
        {
            "role": "user",
            "content": request.prompt,
        }
    ]
    async with model_lock:
        try:
            result = current_engine.generate(
                messages,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            )
            return {
                "text": result,
            }
        except Exception as e:
            print(f"[ERROR] Generate failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e),
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=False,
            )
