"""
Inference Engine and Model Server for Deployment
"""

import os
import json
import time
import uuid
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, AsyncIterator
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.cuda import Stream

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, Depends
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    GPTQ_AVAILABLE = True
except ImportError:
    GPTQ_AVAILABLE = False

from ..model import NexusConfig, NexusForCausalLM
from ..data import NexusTokenizer
from ..enterprise import SecurityManager, AuditLogger, MetricsCollector, EnterpriseConfig


logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    
    # Model settings
    model_path: str = "checkpoints/best"
    tokenizer_path: str = "tokenizer/nexus_tokenizer.model"
    
    # Device settings
    device: str = "cuda"
    device_map: Optional[Dict[str, int]] = None
    
    # Precision settings
    precision: str = "bf16"  # fp16, bf16, fp32, int8, int4
    
    # Quantization settings
    quantization_enabled: bool = False
    quantization_method: str = "bitsandbytes"  # or "gptq"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    
    # Performance settings
    batch_size: int = 1
    max_batch_size: int = 32
    max_seq_length: int = 8192
    
    # KV cache settings
    use_cache: bool = True
    cache_size: int = 1000  # Number of cached sequences
    
    # Streaming settings
    streaming_enabled: bool = True
    stream_batch_size: int = 1
    
    # Thread pool settings
    num_workers: int = 4
    
    # Warmup
    warmup_steps: int = 5


class InferenceEngine:
    """High-performance inference engine for Nexus-7B."""
    
    def __init__(
        self,
        model: NexusForCausalLM,
        tokenizer: NexusTokenizer,
        config: InferenceConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup device
        self._setup_device()
        
        # Setup precision
        self._setup_precision()
        
        # Setup KV cache
        self._setup_cache()
        
        # Warmup model
        self._warmup()
    
    def _setup_device(self):
        """Setup device and device map."""
        if self.config.device_map:
            # Multi-GPU deployment
            self.model = self._apply_device_map(self.config.device_map)
        else:
            self.model = self.model.to(self.config.device)
    
    def _apply_device_map(self, device_map: Dict[str, int]):
        """Apply device map for multi-GPU deployment."""
        # Split model across GPUs
        # This is a simplified implementation
        for name, module in self.model.named_modules():
            for layer_name, gpu_id in device_map.items():
                if layer_name in name:
                    module.to(f"cuda:{gpu_id}")
        
        return self.model
    
    def _setup_precision(self):
        """Setup precision and quantization."""
        if self.config.quantization_enabled:
            self._apply_quantization()
        elif self.config.precision == "fp16":
            self.model = self.model.half()
        elif self.config.precision == "bf16":
            self.model = self.model.to(torch.bfloat16)
        # fp32 is default
    
    def _apply_quantization(self):
        """Apply quantization for memory efficiency."""
        if BITSANDBYTES_AVAILABLE and self.config.quantization_method == "bitsandbytes":
            if self.config.load_in_8bit:
                self.model = self._quantize_8bit()
            elif self.config.load_in_4bit:
                self.model = self._quantize_4bit()
        
        elif GPTQ_AVAILABLE and self.config.quantization_method == "gptq":
            self.model = self._load_gptq_model()
    
    def _quantize_8bit(self) -> NexusForCausalLM:
        """Apply 8-bit quantization."""
        # Replace linear layers with 8-bit quantized versions
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Create 8-bit quantized layer
                quantized = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    has_fp16_weights=False,
                    threshold=6.0,
                )
                
                # Copy weights
                quantized.weight = bnb.nn.Int8Params(
                    module.weight.data.cpu(),
                    requires_grad=False,
                    has_fp16_weights=False,
                )
                
                # Replace module
                setattr(self.model, name.split('.')[-1], quantized)
        
        logger.info("Applied 8-bit quantization")
        return self.model
    
    def _quantize_4bit(self) -> NexusForCausalLM:
        """Apply 4-bit quantization."""
        # Replace linear layers with 4-bit quantized versions
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                quantized = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    quant_type="nf4",
                    compute_dtype=torch.float16,
                )
                
                # Copy weights
                quantized.weight = bnb.nn.Params4bit(
                    module.weight.data.cpu(),
                    requires_grad=False,
                    quant_type="nf4",
                )
                
                if module.bias is not None:
                    quantized.bias = module.bias
                
                # Replace module
                setattr(self.model, name.split('.')[-1], quantized)
        
        logger.info("Applied 4-bit quantization")
        return self.model
    
    def _load_gptq_model(self) -> NexusForCausalLM:
        """Load GPTQ quantized model."""
        # This would load a pre-quantized GPTQ model
        # Placeholder implementation
        logger.info("Loading GPTQ quantized model")
        return self.model
    
    def _setup_cache(self):
        """Setup KV cache for efficient inference."""
        self.kv_cache: Dict[str, Any] = {}
        self.cache_lock = asyncio.Lock()
    
    def _warmup(self):
        """Warmup model for optimal performance."""
        logger.info("Warming up model...")
        
        warmup_prompts = [
            "Hello, how are you?",
            "What is the weather today?",
            "Tell me about artificial intelligence.",
        ]
        
        for prompt in warmup_prompts[:self.config.warmup_steps]:
            self.generate(prompt, max_new_tokens=10)
        
        logger.info("Model warmup completed")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        return_full_response: bool = False,
    ) -> Dict[str, Any]:
        """Generate text from prompt."""
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p
        repetition_penalty = repetition_penalty or self.config.repetition_penalty
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor([input_ids], device=self.config.device)
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            output_ids = self._generate_tokens(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_sequences=stop_sequences,
            )
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Decode output
        generated_text = self.tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True,
        )
        
        # Prepare response
        if return_full_response:
            response = {
                "text": generated_text,
                "prompt": prompt,
                "tokens_generated": len(output_ids[0]) - len(input_ids[0]),
                "latency_ms": latency * 1000,
                "tokens_per_second": (len(output_ids[0]) - len(input_ids[0])) / latency,
            }
        else:
            response = {
                "text": generated_text,
                "tokens_generated": len(output_ids[0]) - len(input_ids[0]),
            }
        
        return response
    
    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        stop_sequences: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Generate token IDs."""
        generated_ids = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.model.model(
                generated_ids,
                past_key_values=past_key_values,
                use_cache=self.config.use_cache,
            )
            
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            past_key_values = outputs.get("past_key_values")
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in generated_ids[0].unique():
                    next_token_logits[0, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[0, indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Check for stop sequences
            if stop_sequences:
                current_text = self.tokenizer.decode(
                    generated_ids[0].tolist(),
                    skip_special_tokens=True,
                )
                
                for stop_seq in stop_sequences:
                    if stop_seq in current_text:
                        return generated_ids
            
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return generated_ids
    
    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Stream generated tokens."""
        # Use config defaults
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p
        repetition_penalty = repetition_penalty or self.config.repetition_penalty
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor([input_ids], device=self.config.device)
        
        generated_ids = input_ids.clone()
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.model.model(
                    generated_ids,
                    past_key_values=past_key_values,
                    use_cache=self.config.use_cache,
                )
                
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                past_key_values = outputs.get("past_key_values")
                
                # Get next token
                next_token_logits = logits[:, -1, :]
                
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                if repetition_penalty != 1.0:
                    for token_id in generated_ids[0].unique():
                        next_token_logits[0, token_id] /= repetition_penalty
                
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[0, indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Decode and yield token
                token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                yield token_text
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate for multiple prompts in batch."""
        # Tokenize all prompts
        batch_input_ids = []
        
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            batch_input_ids.append(input_ids)
        
        # Pad to same length
        max_len = max(len(ids) for ids in batch_input_ids)
        
        padded_ids = []
        attention_mask = []
        
        for ids in batch_input_ids:
            padding_length = max_len - len(ids)
            padded_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)
            attention_mask.append([1] * len(ids) + [0] * padding_length)
        
        input_ids = torch.tensor(padded_ids, device=self.config.device)
        attention_mask = torch.tensor(attention_mask, device=self.config.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self._batch_generate_tokens(
                input_ids,
                attention_mask,
                max_new_tokens or self.config.max_new_tokens,
                **kwargs,
            )
        
        # Decode outputs
        responses = []
        
        for i, output in enumerate(output_ids):
            generated_text = self.tokenizer.decode(
                output.tolist(),
                skip_special_tokens=True,
            )
            
            responses.append({
                "text": generated_text,
                "tokens_generated": len(output) - len(batch_input_ids[i]),
            })
        
        return responses
    
    def _batch_generate_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        **kwargs,
    ) -> torch.Tensor:
        """Generate tokens for batch."""
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            outputs = self.model.model(
                generated_ids,
                attention_mask=attention_mask,
                use_cache=self.config.use_cache,
            )
            
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            
            # Get next tokens for all sequences
            next_token_logits = logits[:, -1, :]
            
            # Apply sampling parameters
            temperature = kwargs.get('temperature', self.config.temperature)
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(attention_mask.shape[0], 1, device=self.config.device),
            ], dim=-1)
        
        return generated_ids


# FastAPI Models for API Server
if FASTAPI_AVAILABLE:
    
    class GenerateRequest(BaseModel):
        """Request model for generation."""
        prompt: str = Field(..., description="Input prompt for generation")
        max_new_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
        temperature: Optional[float] = Field(None, description="Sampling temperature")
        top_k: Optional[int] = Field(None, description="Top-k sampling parameter")
        top_p: Optional[float] = Field(None, description="Top-p sampling parameter")
        repetition_penalty: Optional[float] = Field(None, description="Repetition penalty")
        stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
        stream: Optional[bool] = Field(False, description="Enable streaming response")
    
    class GenerateResponse(BaseModel):
        """Response model for generation."""
        text: str = Field(..., description="Generated text")
        tokens_generated: int = Field(..., description="Number of tokens generated")
        latency_ms: Optional[float] = Field(None, description="Generation latency in ms")
        tokens_per_second: Optional[float] = Field(None, description="Generation speed")
    
    class BatchGenerateRequest(BaseModel):
        """Request model for batch generation."""
        prompts: List[str] = Field(..., description="List of prompts")
        max_new_tokens: Optional[int] = Field(None, description="Maximum tokens per prompt")
        temperature: Optional[float] = Field(None, description="Sampling temperature")
    
    class HealthResponse(BaseModel):
        """Response model for health check."""
        status: str = Field(..., description="Health status")
        model_loaded: bool = Field(..., description="Model loaded status")
        gpu_available: bool = Field(..., description="GPU availability")
        memory_used: Optional[float] = Field(None, description="Memory usage")


class ModelServer:
    """FastAPI-based model server for production deployment."""
    
    def __init__(
        self,
        engine: InferenceEngine,
        security: Optional[SecurityManager] = None,
        audit: Optional[AuditLogger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        self.engine = engine
        self.security = security
        self.audit = audit
        self.metrics = metrics
        
        if FASTAPI_AVAILABLE:
            self.app = self._create_app()
        else:
            self.app = None
            logger.warning("FastAPI not available, server functionality disabled")
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title="Nexus-7B API",
            description="Enterprise-grade API for Nexus-7B model",
            version="1.0.0",
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Routes
        app.post("/generate")(self._generate_endpoint)
        app.post("/generate/stream")(self._generate_stream_endpoint)
        app.post("/generate/batch")(self._batch_generate_endpoint)
        app.get("/health")(self._health_endpoint)
        app.get("/metrics")(self._metrics_endpoint)
        
        return app
    
    async def _generate_endpoint(
        self,
        request: GenerateRequest,
    ) -> GenerateResponse:
        """Generate text endpoint."""
        # Generate
        start_time = time.time()
        
        response = self.engine.generate(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            stop_sequences=request.stop_sequences,
            return_full_response=True,
        )
        
        latency = time.time() - start_time
        
        # Record metrics
        if self.metrics:
            self.metrics.record_inference_metrics(
                latency,
                response['tokens_generated'],
            )
        
        # Audit log
        if self.audit:
            self.audit.log_inference_event(
                action="generate",
                model_id="nexus-7b",
                request_id=str(uuid.uuid4()),
                tokens_generated=response['tokens_generated'],
                latency_ms=latency * 1000,
            )
        
        return GenerateResponse(
            text=response['text'],
            tokens_generated=response['tokens_generated'],
            latency_ms=response['latency_ms'],
            tokens_per_second=response['tokens_per_second'],
        )
    
    async def _generate_stream_endpoint(
        self,
        request: GenerateRequest,
    ):
        """Stream generation endpoint."""
        async def stream_generator():
            for token in await self.engine.generate_stream(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
        )
    
    async def _batch_generate_endpoint(
        self,
        request: BatchGenerateRequest,
    ) -> List[GenerateResponse]:
        """Batch generation endpoint."""
        # Generate batch
        responses = self.engine.batch_generate(
            request.prompts,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        
        return [
            GenerateResponse(
                text=r['text'],
                tokens_generated=r['tokens_generated'],
            )
            for r in responses
        ]
    
    async def _health_endpoint(self) -> HealthResponse:
        """Health check endpoint."""
        gpu_available = torch.cuda.is_available()
        memory_used = None
        
        if gpu_available:
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            gpu_available=gpu_available,
            memory_used=memory_used,
        )
    
    async def _metrics_endpoint(self) -> Dict[str, Any]:
        """Metrics endpoint."""
        if self.metrics:
            return self.metrics.get_metrics_summary()
        
        return {"message": "Metrics collection not enabled"}
    
    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
    ):
        """Run the server."""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
        )


def create_inference_engine(
    model_path: str,
    tokenizer_path: str,
    config: Optional[InferenceConfig] = None,
) -> InferenceEngine:
    """Factory function to create inference engine."""
    
    config = config or InferenceConfig(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    
    # Load model
    model_config = NexusConfig()
    model = NexusForCausalLM.from_pretrained(model_path, model_config)
    
    # Load tokenizer
    from ..data import DataConfig
    data_config = DataConfig(tokenizer_path=tokenizer_path)
    tokenizer = NexusTokenizer(data_config)
    
    # Create engine
    engine = InferenceEngine(model, tokenizer, config)
    
    return engine


def create_model_server(
    model_path: str,
    tokenizer_path: str,
    enterprise_config: Optional[EnterpriseConfig] = None,
    inference_config: Optional[InferenceConfig] = None,
) -> ModelServer:
    """Factory function to create model server."""
    
    # Create inference engine
    engine = create_inference_engine(model_path, tokenizer_path, inference_config)
    
    # Create enterprise components
    security = None
    audit = None
    metrics = None
    
    if enterprise_config:
        security = SecurityManager(enterprise_config)
        audit = AuditLogger(enterprise_config)
        metrics = MetricsCollector(enterprise_config)
    
    # Create server
    server = ModelServer(engine, security, audit, metrics)
    
    return server