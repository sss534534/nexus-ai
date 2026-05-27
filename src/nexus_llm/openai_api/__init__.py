"""
OpenAI-Compatible API Module for Nexus-LLM

Provides an OpenAI-compatible API server that wraps the existing InferenceEngine,
enabling existing OpenAI SDK clients to use Nexus-LLM without modification.

Endpoints:
    - POST /v1/chat/completions  - Chat completions (supports SSE streaming)
    - GET  /v1/models            - List available models
    - POST /v1/embeddings        - Text embeddings (placeholder with mock data)
    - GET  /v1/me                - Current user info (API key validation)

Usage::

    from nexus_llm.serving import InferenceEngine, create_inference_engine
    from nexus_llm.openai_api import create_openai_server, run_openai_server

    # Option 1: Create server from an existing engine
    engine = create_inference_engine("checkpoints/best", "tokenizer/nexus_tokenizer.model")
    server = create_openai_server(engine)
    server.run(host="0.0.0.0", port=8000)

    # Option 2: Convenience function
    run_openai_server("checkpoints/best", "tokenizer/nexus_tokenizer.model", "0.0.0.0", 8000)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ..serving import InferenceEngine, InferenceConfig, create_inference_engine
from ..enterprise import SecurityManager, AuditLogger, MetricsCollector, EnterpriseConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI-compatible Pydantic v2 request / response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Matches the OpenAI ``ChatCompletionRequestMessage`` schema.
    """

    role: str = Field(
        ...,
        description="The role of the message author. One of 'system', 'user', 'assistant', or 'tool'.",
    )
    content: Optional[str] = Field(
        None,
        description="The content of the message.",
    )
    name: Optional[str] = Field(
        None,
        description="An optional name for the participant.",
    )


class ChatCompletionRequest(BaseModel):
    """Request body for ``POST /v1/chat/completions``.

    Mirrors the OpenAI Chat Completion request schema so that existing OpenAI
    SDK clients can call this endpoint without any changes.
    """

    model: str = Field(
        "nexus-7b",
        description="ID of the model to use for completion.",
    )
    messages: List[ChatMessage] = Field(
        ...,
        description="A list of messages describing the conversation so far.",
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0-2). Higher values make the output more random.",
    )
    top_p: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter. The model considers tokens with top_p probability mass.",
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum number of tokens to generate in the completion.",
    )
    stream: Optional[bool] = Field(
        False,
        description="If true, the response will be streamed back using Server-Sent Events (SSE).",
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None,
        description="Up to 4 sequences where the API will stop generating further tokens.",
    )
    presence_penalty: Optional[float] = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description="Positive values penalize new tokens based on whether they appear in the text so far.",
    )
    frequency_penalty: Optional[float] = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description="Positive values penalize new tokens based on their existing frequency in the text so far.",
    )
    n: Optional[int] = Field(
        1,
        ge=1,
        description="How many completions to generate for each prompt.",
    )
    user: Optional[str] = Field(
        None,
        description="A unique identifier representing the end-user.",
    )


class CompletionUsage(BaseModel):
    """Token usage statistics for a completion response."""

    prompt_tokens: int = Field(
        ...,
        description="Number of tokens in the prompt.",
    )
    completion_tokens: int = Field(
        ...,
        description="Number of tokens in the generated completion.",
    )
    total_tokens: int = Field(
        ...,
        description="Total number of tokens (prompt + completion).",
    )


class ChatCompletionChoice(BaseModel):
    """A single completion choice in a non-streaming response."""

    index: int = Field(
        ...,
        description="The index of this choice in the list of choices.",
    )
    message: ChatMessage = Field(
        ...,
        description="The generated message.",
    )
    finish_reason: Optional[str] = Field(
        "stop",
        description="The reason the model stopped generating tokens. One of 'stop', 'length', or 'content_filter'.",
    )


class ChatCompletionResponse(BaseModel):
    """Response body for ``POST /v1/chat/completions`` (non-streaming).

    Conforms to the OpenAI Chat Completion response schema.
    """

    id: str = Field(
        ...,
        description="A unique identifier for the completion.",
    )
    object: str = Field(
        "chat.completion",
        description="The object type, always 'chat.completion'.",
    )
    created: int = Field(
        ...,
        description="The Unix timestamp (in seconds) of when the completion was created.",
    )
    model: str = Field(
        ...,
        description="The model used for the completion.",
    )
    choices: List[ChatCompletionChoice] = Field(
        ...,
        description="A list of completion choices.",
    )
    usage: CompletionUsage = Field(
        ...,
        description="Token usage statistics.",
    )


class ChatCompletionDelta(BaseModel):
    """Delta content for a streaming chunk (replaces ``message`` in non-streaming)."""

    role: Optional[str] = Field(
        None,
        description="The role of the author of this message.",
    )
    content: Optional[str] = Field(
        None,
        description="The partial content of the message.",
    )


class ChatCompletionChunkChoice(BaseModel):
    """A single completion choice in a streaming chunk."""

    index: int = Field(
        ...,
        description="The index of this choice.",
    )
    delta: ChatCompletionDelta = Field(
        ...,
        description="The delta content for this chunk.",
    )
    finish_reason: Optional[str] = Field(
        None,
        description="The reason the model stopped generating tokens, if applicable.",
    )


class ChatCompletionChunk(BaseModel):
    """A streaming chunk in the SSE response.

    Conforms to the OpenAI Chat Completion chunk schema.
    """

    id: str = Field(
        ...,
        description="A unique identifier for the completion (shared across all chunks).",
    )
    object: str = Field(
        "chat.completion.chunk",
        description="The object type, always 'chat.completion.chunk'.",
    )
    created: int = Field(
        ...,
        description="The Unix timestamp (in seconds) of when the chunk was created.",
    )
    model: str = Field(
        ...,
        description="The model used for the completion.",
    )
    choices: List[ChatCompletionChunkChoice] = Field(
        ...,
        description="A list of completion choices (typically one).",
    )


# ---------------------------------------------------------------------------
# Model list models
# ---------------------------------------------------------------------------

class ModelObject(BaseModel):
    """Information about a single model."""

    id: str = Field(
        ...,
        description="The model identifier.",
    )
    object: str = Field(
        "model",
        description="The object type, always 'model'.",
    )
    created: int = Field(
        ...,
        description="The Unix timestamp (in seconds) of when the model was created.",
    )
    owned_by: str = Field(
        "nexus-ai",
        description="The organization that owns the model.",
    )
    permission: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="An array of permission objects (OpenAI compatibility).",
    )
    root: Optional[str] = Field(
        None,
        description="The root model identifier (OpenAI compatibility).",
    )
    parent: Optional[str] = Field(
        None,
        description="The parent model identifier (OpenAI compatibility).",
    )


class ModelListResponse(BaseModel):
    """Response body for ``GET /v1/models``."""

    object: str = Field(
        "list",
        description="The object type, always 'list'.",
    )
    data: List[ModelObject] = Field(
        ...,
        description="A list of model objects.",
    )


# ---------------------------------------------------------------------------
# Embedding models
# ---------------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    """Request body for ``POST /v1/embeddings``."""

    model: str = Field(
        "nexus-7b",
        description="ID of the model to use for embedding.",
    )
    input: Union[str, List[str]] = Field(
        ...,
        description="The input text or list of texts to embed.",
    )
    encoding_format: Optional[str] = Field(
        "float",
        description="The format to return the embeddings in. Currently only 'float' is supported.",
    )
    dimensions: Optional[int] = Field(
        None,
        description="The number of dimensions the resulting output embeddings should have.",
    )
    user: Optional[str] = Field(
        None,
        description="A unique identifier representing the end-user.",
    )


class EmbeddingObject(BaseModel):
    """A single embedding object in the response."""

    object: str = Field(
        "embedding",
        description="The object type, always 'embedding'.",
    )
    embedding: List[float] = Field(
        ...,
        description="The embedding vector.",
    )
    index: int = Field(
        ...,
        description="The index of the embedding in the list of embeddings.",
    )


class EmbeddingResponse(BaseModel):
    """Response body for ``POST /v1/embeddings``."""

    object: str = Field(
        "list",
        description="The object type, always 'list'.",
    )
    data: List[EmbeddingObject] = Field(
        ...,
        description="A list of embedding objects.",
    )
    model: str = Field(
        ...,
        description="The model used for embedding.",
    )
    usage: CompletionUsage = Field(
        ...,
        description="Token usage statistics.",
    )


# ---------------------------------------------------------------------------
# User info model
# ---------------------------------------------------------------------------

class UserInfoResponse(BaseModel):
    """Response body for ``GET /v1/me``."""

    object: str = Field(
        "user",
        description="The object type, always 'user'.",
    )
    id: str = Field(
        ...,
        description="The unique identifier for the user.",
    )
    name: Optional[str] = Field(
        None,
        description="The display name of the user.",
    )
    email: Optional[str] = Field(
        None,
        description="The email address of the user.",
    )
    permissions: List[str] = Field(
        default_factory=list,
        description="The permissions associated with the API key.",
    )
    created_at: Optional[str] = Field(
        None,
        description="The ISO 8601 timestamp when the API key was created.",
    )
    expires_at: Optional[str] = Field(
        None,
        description="The ISO 8601 timestamp when the API key expires.",
    )


# ---------------------------------------------------------------------------
# OpenAI-style error response
# ---------------------------------------------------------------------------

class OpenAIErrorDetail(BaseModel):
    """Detailed error information in OpenAI format."""

    message: str = Field(
        ...,
        description="A human-readable error message.",
    )
    type: str = Field(
        "invalid_request_error",
        description="The error type.",
    )
    param: Optional[str] = Field(
        None,
        description="The parameter that caused the error, if applicable.",
    )
    code: Optional[str] = Field(
        None,
        description="A machine-readable error code.",
    )


class OpenAIErrorResponse(BaseModel):
    """Top-level error response matching OpenAI's error format."""

    error: OpenAIErrorDetail = Field(
        ...,
        description="The error detail object.",
    )


# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------

@dataclass
class OpenAIServerConfig:
    """Configuration for the OpenAI-compatible server.

    Attributes:
        model_name: The model name exposed via the API. Defaults to ``"nexus-7b"``.
        model_version: A version string included in metadata.
        default_max_tokens: Default maximum tokens for completions when the
            request does not specify ``max_tokens``.
        embedding_dimension: Dimensionality of mock embedding vectors.
        api_key_required: Whether API key authentication is mandatory.  When
            ``True`` and no ``SecurityManager`` is provided the server will
            still reject requests missing the ``Authorization`` header.
        cors_origins: List of allowed CORS origins (``["*"]`` allows all).
        default_temperature: Default sampling temperature when not specified.
        default_top_p: Default nucleus sampling parameter when not specified.
    """

    model_name: str = "nexus-7b"
    model_version: str = "2.0.0"
    default_max_tokens: int = 512
    embedding_dimension: int = 4096
    api_key_required: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    default_temperature: float = 0.7
    default_top_p: float = 0.95


# ---------------------------------------------------------------------------
# OpenAI-compatible server
# ---------------------------------------------------------------------------

class OpenAICompatibleServer:
    """FastAPI-based server that exposes an OpenAI-compatible chat completions API.

    This server wraps a :class:`~nexus_llm.serving.InferenceEngine` and
    translates OpenAI-format requests into engine calls, then formats the
    responses to match the OpenAI Chat Completions schema exactly.  Existing
    OpenAI SDK clients can point at this server without any code changes.

    Args:
        engine: The Nexus-LLM inference engine used to generate completions.
        config: Optional server configuration.  When ``None`` the defaults in
            :class:`OpenAIServerConfig` are used.
        security: Optional :class:`~nexus_llm.enterprise.SecurityManager`
            instance for API-key authentication and rate-limiting.
        metrics: Optional :class:`~nexus_llm.enterprise.MetricsCollector`
            for recording inference metrics.
        audit: Optional :class:`~nexus_llm.enterprise.AuditLogger` for
            recording audit events.

    Example::

        from nexus_llm.openai_api import OpenAICompatibleServer

        server = OpenAICompatibleServer(engine, security=security_mgr)
        server.run(host="0.0.0.0", port=8000)
    """

    def __init__(
        self,
        engine: InferenceEngine,
        config: Optional[OpenAIServerConfig] = None,
        security: Optional[SecurityManager] = None,
        metrics: Optional[MetricsCollector] = None,
        audit: Optional[AuditLogger] = None,
    ) -> None:
        self.engine = engine
        self.config = config or OpenAIServerConfig()
        self.security = security
        self.metrics = metrics
        self.audit = audit

        # Track token usage across requests for the /v1/me endpoint
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0

        self.app: FastAPI = self._create_app()

    # ------------------------------------------------------------------
    # App construction
    # ------------------------------------------------------------------

    def _create_app(self) -> FastAPI:
        """Build and return the FastAPI application with all routes wired up."""
        app = FastAPI(
            title="Nexus-LLM OpenAI-Compatible API",
            description=(
                "An OpenAI-compatible API server powered by Nexus-LLM. "
                "Existing OpenAI SDK clients can use this endpoint directly."
            ),
            version=self.config.model_version,
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register endpoints
        app.post("/v1/chat/completions", name="chat_completions")(self._chat_completions)
        app.get("/v1/models", name="list_models")(self._list_models)
        app.post("/v1/embeddings", name="create_embeddings")(self._create_embeddings)
        app.get("/v1/me", name="get_me")(self._get_me)

        # Global exception handlers for OpenAI-style error responses
        app.add_exception_handler(HTTPException, self._http_exception_handler)
        app.add_exception_handler(Exception, self._generic_exception_handler)

        return app

    # ------------------------------------------------------------------
    # Authentication helper
    # ------------------------------------------------------------------

    def _authenticate(self, request: Request) -> Optional[Dict[str, Any]]:
        """Validate the ``Authorization: Bearer <key>`` header.

        Returns the key-info dict when authentication succeeds, ``None`` when
        no security manager is configured (open access), or raises
        ``HTTPException(401)`` on invalid / missing keys.
        """
        if self.security is None:
            return None

        authorization: Optional[str] = request.headers.get("authorization")
        if not authorization:
            if self.config.api_key_required:
                raise HTTPException(
                    status_code=401,
                    detail="Missing Authorization header. Provide 'Authorization: Bearer <API_KEY>'.",
                )
            return None

        parts = authorization.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail="Invalid Authorization header format. Expected 'Bearer <API_KEY>'.",
            )

        api_key = parts[1]
        key_info = self.security.validate_api_key(api_key)
        if key_info is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired API key.",
            )

        # Rate-limit check
        client_id = key_info.get("user", "anonymous")
        if not self.security.check_rate_limit(client_id):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please retry later.",
            )

        return key_info

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_completion_id() -> str:
        """Generate an OpenAI-style completion ID (``chatcmpl-<hex>``)."""
        return f"chatcmpl-{secrets.token_hex(24)}"

    # ------------------------------------------------------------------
    # Prompt construction from messages
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(messages: List[ChatMessage]) -> str:
        """Convert a list of chat messages into a single prompt string.

        Uses a simple ``<role>: <content>`` format separated by newlines,
        which works well with most instruction-tuned models.
        """
        parts: List[str] = []
        for msg in messages:
            role = msg.role.upper()
            content = msg.content or ""
            parts.append(f"{role}: {content}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Approximate token counting
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count estimate (~4 characters per token for English)."""
        if not text:
            return 0
        return max(1, int(math.ceil(len(text) / 4.0)))

    # ------------------------------------------------------------------
    # POST /v1/chat/completions
    # ------------------------------------------------------------------

    async def _chat_completions(
        self,
        request: Request,
        body: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, StreamingResponse]:
        """Handle chat completion requests (both streaming and non-streaming).

        Args:
            request: The raw ASGI request (used for auth header inspection).
            body: The parsed request body.

        Returns:
            A :class:`ChatCompletionResponse` for non-streaming requests, or
            a :class:`StreamingResponse` yielding SSE chunks for streaming
            requests.
        """
        # --- Authentication ------------------------------------------------
        key_info = self._authenticate(request)

        # --- Build prompt --------------------------------------------------
        prompt = self._build_prompt(body.messages)
        prompt_tokens = self._estimate_tokens(prompt)

        # --- Resolve generation parameters --------------------------------
        temperature = body.temperature if body.temperature is not None else self.config.default_temperature
        top_p = body.top_p if body.top_p is not None else self.config.default_top_p
        max_tokens = body.max_tokens if body.max_tokens is not None else self.config.default_max_tokens

        # Normalise stop sequences
        stop_sequences: Optional[List[str]] = None
        if body.stop is not None:
            stop_sequences = [body.stop] if isinstance(body.stop, str) else list(body.stop)

        # --- Streaming path -----------------------------------------------
        if body.stream:
            return self._handle_streaming(
                prompt=prompt,
                model=body.model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
                prompt_tokens=prompt_tokens,
                key_info=key_info,
            )

        # --- Non-streaming path -------------------------------------------
        completion_id = self._generate_completion_id()
        created = int(time.time())

        start_time = time.time()
        try:
            result = self.engine.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences,
                return_full_response=True,
            )
        except Exception as exc:
            logger.exception("Inference engine error during chat completion")
            raise HTTPException(
                status_code=500,
                detail=f"Inference engine error: {exc}",
            ) from exc

        latency = time.time() - start_time
        generated_text = result.get("text", "")
        completion_tokens = result.get("tokens_generated", self._estimate_tokens(generated_text))

        # Determine finish reason
        finish_reason = "stop"
        if stop_sequences and any(s in generated_text for s in stop_sequences):
            finish_reason = "stop"
        elif completion_tokens >= max_tokens:
            finish_reason = "length"

        # Update global token counters
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        # --- Metrics & audit ----------------------------------------------
        if self.metrics:
            self.metrics.record_inference_metrics(
                latency=latency,
                tokens=completion_tokens,
                status="success",
            )
        if self.audit:
            self.audit.log_inference_event(
                action="chat_completion",
                model_id=body.model,
                request_id=completion_id,
                tokens_generated=completion_tokens,
                latency_ms=latency * 1000,
            )

        response = ChatCompletionResponse(
            id=completion_id,
            object="chat.completion",
            created=created,
            model=body.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason=finish_reason,
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        return response

    # ------------------------------------------------------------------
    # Streaming helper
    # ------------------------------------------------------------------

    async def _handle_streaming(
        self,
        prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]],
        prompt_tokens: int,
        key_info: Optional[Dict[str, Any]],
    ) -> StreamingResponse:
        """Return a :class:`StreamingResponse` that yields SSE chunks."""

        completion_id = self._generate_completion_id()
        created = int(time.time())

        async def _stream_generator() -> AsyncIterator[str]:
            """Async generator yielding ``data: <json>`` SSE lines."""
            completion_tokens = 0

            # First chunk: role only
            first_chunk = ChatCompletionChunk(
                id=completion_id,
                object="chat.completion.chunk",
                created=created,
                model=model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionDelta(role="assistant", content=None),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {first_chunk.model_dump_json()}\n\n"

            # Subsequent chunks: content tokens
            try:
                async for token_text in self.engine.generate_stream(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ):
                    completion_tokens += 1
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionDelta(content=token_text),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as exc:
                logger.exception("Error during streaming generation")
                error_chunk = ChatCompletionChunk(
                    id=completion_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionDelta(content=f"[ERROR] {exc}"),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"

            # Final chunk: finish_reason
            final_chunk = ChatCompletionChunk(
                id=completion_id,
                object="chat.completion.chunk",
                created=created,
                model=model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionDelta(content=None),
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"

            # Usage chunk (non-standard but commonly expected by clients)
            usage_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            yield f"data: {json.dumps(usage_data)}\n\n"

            # SSE termination signal
            yield "data: [DONE]\n\n"

            # Update global token counters
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens

            # Metrics & audit
            if self.metrics:
                self.metrics.record_inference_metrics(
                    latency=0.0,
                    tokens=completion_tokens,
                    status="success",
                )
            if self.audit:
                self.audit.log_inference_event(
                    action="chat_completion_stream",
                    model_id=model,
                    request_id=completion_id,
                    tokens_generated=completion_tokens,
                    latency_ms=0.0,
                )

        return StreamingResponse(
            _stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # GET /v1/models
    # ------------------------------------------------------------------

    async def _list_models(self, request: Request) -> ModelListResponse:
        """Return the list of available models.

        Currently exposes the configured model name.  In a multi-model
        deployment this would enumerate all loaded models.
        """
        self._authenticate(request)

        now = int(time.time())
        model_obj = ModelObject(
            id=self.config.model_name,
            object="model",
            created=now,
            owned_by="nexus-ai",
            permission=[
                {
                    "id": f"modelperm-{secrets.token_hex(12)}",
                    "object": "model_permission",
                    "created": now,
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": False,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False,
                }
            ],
            root=self.config.model_name,
            parent=None,
        )

        return ModelListResponse(object="list", data=[model_obj])

    # ------------------------------------------------------------------
    # POST /v1/embeddings
    # ------------------------------------------------------------------

    async def _create_embeddings(
        self,
        request: Request,
        body: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Return mock embedding vectors.

        This is a placeholder implementation that returns deterministic
        pseudo-random embeddings based on the input text hash.  A real
        implementation would call a dedicated embedding model.
        """
        self._authenticate(request)

        # Normalise input to a list
        texts: List[str] = [body.input] if isinstance(body.input, str) else list(body.input)

        embedding_objects: List[EmbeddingObject] = []
        total_tokens = 0

        for idx, text in enumerate(texts):
            total_tokens += self._estimate_tokens(text)

            # Deterministic mock embedding based on text hash
            text_hash = hashlib.sha256(text.encode("utf-8")).digest()
            dim = body.dimensions or self.config.embedding_dimension

            # Expand the 32-byte hash into the requested dimensionality
            embedding: List[float] = []
            for i in range(dim):
                byte_idx = i % len(text_hash)
                # Map byte value to float in [-1, 1]
                val = (text_hash[byte_idx] / 127.5) - 1.0
                # Add position-based variation for diversity
                val *= math.sin(i * 0.1 + text_hash[0]) * 0.5 + 0.5
                embedding.append(round(val, 6))

            # L2-normalise the embedding vector
            norm = math.sqrt(sum(v * v for v in embedding))
            if norm > 0:
                embedding = [v / norm for v in embedding]

            embedding_objects.append(
                EmbeddingObject(
                    object="embedding",
                    embedding=embedding,
                    index=idx,
                )
            )

        return EmbeddingResponse(
            object="list",
            data=embedding_objects,
            model=body.model,
            usage=CompletionUsage(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
            ),
        )

    # ------------------------------------------------------------------
    # GET /v1/me
    # ------------------------------------------------------------------

    async def _get_me(self, request: Request) -> UserInfoResponse:
        """Return information about the current API key holder.

        Requires a valid ``Authorization: Bearer <key>`` header.  When no
        ``SecurityManager`` is configured the endpoint returns a default
        anonymous user.
        """
        key_info = self._authenticate(request)

        if key_info is None:
            # No security manager configured -- return anonymous user
            return UserInfoResponse(
                object="user",
                id="anon-default",
                name="Anonymous",
                permissions=["read", "write"],
            )

        return UserInfoResponse(
            object="user",
            id=hashlib.sha256(key_info.get("user", "").encode()).hexdigest()[:16],
            name=key_info.get("user", "unknown"),
            permissions=key_info.get("permissions", []),
            created_at=key_info.get("created_at"),
            expires_at=key_info.get("expires_at"),
        )

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------

    @staticmethod
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Convert FastAPI ``HTTPException`` into an OpenAI-style error JSON."""
        status_code = exc.status_code

        # Map common status codes to OpenAI error types
        error_type_map = {
            400: "invalid_request_error",
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            429: "rate_limit_error",
            500: "server_error",
        }
        error_type = error_type_map.get(status_code, "api_error")

        error_detail = OpenAIErrorDetail(
            message=str(exc.detail),
            type=error_type,
            code=None,
        )
        error_response = OpenAIErrorResponse(error=error_detail)

        return JSONResponse(
            status_code=status_code,
            content=error_response.model_dump(),
        )

    @staticmethod
    async def _generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all handler that returns an OpenAI-style 500 error."""
        logger.exception("Unhandled exception in OpenAI-compatible API")

        error_detail = OpenAIErrorDetail(
            message=f"Internal server error: {exc}",
            type="server_error",
            code="internal_error",
        )
        error_response = OpenAIErrorResponse(error=error_detail)

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(),
        )

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        log_level: str = "info",
    ) -> None:
        """Start the uvicorn server and block until interrupted.

        Args:
            host: Bind address.  Use ``"0.0.0.0"`` to listen on all interfaces.
            port: TCP port to listen on.
            workers: Number of uvicorn worker processes.
            log_level: Uvicorn log level (``"debug"``, ``"info"``, ``"warning"``, ``"error"``).
        """
        try:
            import uvicorn
        except ImportError:
            raise RuntimeError(
                "uvicorn is required to run the OpenAI-compatible server. "
                "Install it with: pip install uvicorn"
            )

        logger.info(
            "Starting Nexus-LLM OpenAI-compatible API server on %s:%d (model=%s)",
            host,
            port,
            self.config.model_name,
        )

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
            log_level=log_level,
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_openai_server(
    engine: InferenceEngine,
    config: Optional[OpenAIServerConfig] = None,
    security: Optional[SecurityManager] = None,
    metrics: Optional[MetricsCollector] = None,
    audit: Optional[AuditLogger] = None,
) -> OpenAICompatibleServer:
    """Create an :class:`OpenAICompatibleServer` from an existing inference engine.

    This is the recommended entry-point when you already have a loaded
    :class:`~nexus_llm.serving.InferenceEngine` instance.

    Args:
        engine: A fully initialised Nexus-LLM inference engine.
        config: Optional server configuration.
        security: Optional security manager for API-key auth.
        metrics: Optional metrics collector.
        audit: Optional audit logger.

    Returns:
        A ready-to-run :class:`OpenAICompatibleServer`.

    Example::

        engine = create_inference_engine("checkpoints/best", "tokenizer/nexus_tokenizer.model")
        server = create_openai_server(engine)
        server.run(port=8000)
    """
    return OpenAICompatibleServer(
        engine=engine,
        config=config,
        security=security,
        metrics=metrics,
        audit=audit,
    )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_openai_server(
    model_path: str,
    tokenizer_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    inference_config: Optional[InferenceConfig] = None,
    server_config: Optional[OpenAIServerConfig] = None,
    enable_security: bool = False,
) -> None:
    """Load a model and immediately start the OpenAI-compatible API server.

    This is a one-shot convenience function that handles engine creation,
    optional security setup, and server startup in a single call.

    Args:
        model_path: Path to the model checkpoint directory.
        tokenizer_path: Path to the tokenizer file.
        host: Bind address (default ``"0.0.0.0"``).
        port: TCP port (default ``8000``).
        inference_config: Optional :class:`~nexus_llm.serving.InferenceConfig`.
            When ``None`` a default config is created using *model_path* and
            *tokenizer_path*.
        server_config: Optional :class:`OpenAIServerConfig`.
        enable_security: If ``True``, create a :class:`~nexus_llm.enterprise.SecurityManager`
            with default :class:`~nexus_llm.enterprise.EnterpriseConfig`.

    Example::

        from nexus_llm.openai_api import run_openai_server

        run_openai_server(
            model_path="checkpoints/best",
            tokenizer_path="tokenizer/nexus_tokenizer.model",
            host="0.0.0.0",
            port=8000,
        )
    """
    # --- Create inference engine -------------------------------------------
    if inference_config is None:
        inference_config = InferenceConfig(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
        )

    logger.info("Loading inference engine (model=%s, tokenizer=%s) ...", model_path, tokenizer_path)
    engine = create_inference_engine(model_path, tokenizer_path, inference_config)
    logger.info("Inference engine loaded successfully.")

    # --- Optional enterprise components -----------------------------------
    security: Optional[SecurityManager] = None
    metrics: Optional[MetricsCollector] = None
    audit: Optional[AuditLogger] = None

    if enable_security:
        enterprise_config = EnterpriseConfig()
        security = SecurityManager(enterprise_config)
        metrics = MetricsCollector(enterprise_config)
        audit = AuditLogger(enterprise_config)
        logger.info("Enterprise security, metrics, and audit enabled.")

    # --- Create and run server --------------------------------------------
    server = OpenAICompatibleServer(
        engine=engine,
        config=server_config,
        security=security,
        metrics=metrics,
        audit=audit,
    )
    server.run(host=host, port=port)


__all__ = [
    # Request / response models
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChoice",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionDelta",
    "CompletionUsage",
    "ModelObject",
    "ModelListResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingObject",
    "UserInfoResponse",
    "OpenAIErrorDetail",
    "OpenAIErrorResponse",
    # Configuration
    "OpenAIServerConfig",
    # Server
    "OpenAICompatibleServer",
    # Factory / convenience
    "create_openai_server",
    "run_openai_server",
]
