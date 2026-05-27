"""
Core Transformer Model Architecture for Nexus-7B
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


@dataclass
class NexusConfig:
    """Configuration for Nexus-7B model architecture."""
    
    # Model dimensions
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 32  # GQA support
    
    # RoPE configuration
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    rope_scaling_type: Optional[str] = "dynamic"
    rope_scaling_factor: Optional[float] = 2.0
    
    # Normalization
    rms_norm_eps: float = 1e-6
    
    # Attention settings
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_bias: bool = False
    use_flash_attention: bool = True
    
    # Activation
    hidden_act: str = "silu"
    
    # Vocabulary
    vocab_size: int = 152064
    tie_word_embeddings: bool = False
    
    # Initialization
    initializer_range: float = 0.02
    
    # Special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) must be <= "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_attention_groups = self.num_attention_heads // self.num_key_value_heads


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation."""
    
    def __init__(self, config: NexusConfig):
        super().__init__()
        self.dim = config.head_dim
        self.base = config.rope_theta
        self.max_position = config.max_position_embeddings
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Scaling for extended positions
        self.scaling_type = config.rope_scaling_type
        self.scaling_factor = config.rope_scaling_factor or 1.0
        
        if self.scaling_type == "linear":
            self._apply_linear_scaling()
        elif self.scaling_type == "dynamic":
            self._apply_dynamic_scaling()
    
    def _apply_linear_scaling(self):
        """Apply linear scaling for extended positions."""
        inv_freq = self.inv_freq / self.scaling_factor
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def _apply_dynamic_scaling(self):
        """Apply dynamic NTK-aware scaling."""
        # Dynamic scaling adjusts base frequency based on position
        # This is computed at runtime for flexibility
        pass
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Compute cosine and sine values for rotary embedding."""
        t = torch.arange(seq_len, device=device, dtype=dtype)
        
        # Dynamic scaling if enabled
        if self.scaling_type == "dynamic" and seq_len > self.max_position:
            scaling_factor = self.scaling_factor
            new_base = self.base * (scaling_factor ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (new_base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        else:
            inv_freq = self.inv_freq
        
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply rotary position embedding to input tensor."""
        seq_len = x.shape[2]
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device)
        
        cos, sin = self._compute_cos_sin(seq_len, x.device, x.dtype)
        
        # Reshape for broadcasting
        cos = cos[position_ids].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin[position_ids].unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        x_rot = self._rotate_half(x)
        return x * cos + x_rot * sin
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims for rotary embedding."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class NexusAttention(nn.Module):
    """Multi-Group Query Attention with RoPE."""
    
    def __init__(self, config: NexusConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_groups = config.num_attention_groups
        
        self.attention_dropout = config.attention_dropout
        self.use_flash_attention = config.use_flash_attention and FLASH_ATTENTION_AVAILABLE
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(
            self.hidden_size, 
            self.num_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias
        )
        
        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias
        )
        
        # Rotary embedding
        self.rotary_emb = RotaryPositionEmbedding(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for attention layer."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        
        # Apply rotary embedding to Q and K
        query_states = self.rotary_emb(query_states.transpose(1, 2), position_ids).transpose(1, 2)
        key_states = self.rotary_emb(key_states.transpose(1, 2), position_ids).transpose(1, 2)
        
        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        if use_cache:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None
        
        # Repeat K and V for group query attention
        key_states = self._repeat_kv(key_states, self.num_groups)
        value_states = self._repeat_kv(value_states, self.num_groups)
        
        # Compute attention
        if self.use_flash_attention:
            attn_output = self._flash_attention(
                query_states, key_states, value_states, attention_mask
            )
            attn_weights = None
        else:
            attn_output, attn_weights = self._standard_attention(
                query_states, key_states, value_states, attention_mask
            )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value states for group query attention."""
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    
    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Flash Attention implementation for efficient computation."""
        # Flash Attention requires specific input format
        qkv = torch.stack([query, key, value], dim=2)
        attn_output = flash_attn_qkvpacked_func(qkv, self.attention_dropout)
        return attn_output
    
    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard scaled dot-product attention."""
        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(attn_weights.shape[-2], attn_weights.shape[-1], 
                      device=attn_weights.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.type(query.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


class NexusMLP(nn.Module):
    """SwiGLU MLP layer."""
    
    def __init__(self, config: NexusConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # SwiGLU requires three projections
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.act_fn = F.silu if config.hidden_act == "silu" else F.gelu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward pass: down(silu(gate(x)) * up(x))"""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class NexusDecoderLayer(nn.Module):
    """Transformer decoder layer."""
    
    def __init__(self, config: NexusConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Self-attention
        self.self_attn = NexusAttention(config, layer_idx)
        
        # MLP
        self.mlp = NexusMLP(config)
        
        # Layer norms (pre-norm architecture)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for decoder layer."""
        residual = hidden_states
        
        # Pre-norm + Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states
        
        # Pre-norm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if use_cache:
            outputs += (attn_outputs[2] if len(attn_outputs) > 2 else None)
        if output_attentions:
            outputs += (attn_outputs[1] if len(attn_outputs) > 1 else None)
        
        return outputs


class NexusModel(nn.Module):
    """Nexus-7B Transformer Model."""
    
    def __init__(self, config: NexusConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            NexusDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Output projection (lm_head)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights with normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        """Forward pass for the model."""
        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states
        use_cache = use_cache or self.config.use_cache
        return_dict = return_dict or self.config.use_return_dict
        
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        batch_size, seq_length = inputs_embeds.shape[:2]
        
        # Generate position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), 
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype
            )
        
        # Initialize past key values
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # Forward through layers
        hidden_states = inputs_embeds
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            # Gradient checkpointing
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
            if output_attentions:
                all_attentions += (layer_outputs[2],)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Compute logits
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        
        # Prepare output
        if not return_dict:
            output = (logits,)
            if use_cache:
                output += (next_decoder_cache,)
            if output_hidden_states:
                output += (all_hidden_states,)
            if output_attentions:
                output += (all_attentions,)
            return output
        
        return {
            "logits": logits,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
    
    def _gradient_checkpointing_func(self, fn, *args):
        """Apply gradient checkpointing to a function."""
        def create_custom_forward(*inputs):
            return fn(*inputs)
        
        return torch.utils.checkpoint.checkpoint(create_custom_forward, *args)
    
    @classmethod
    def from_pretrained(cls, pretrained_path: str, config: Optional[NexusConfig] = None):
        """Load model from pretrained weights."""
        import safetensors
        
        if config is None:
            config = NexusConfig()
        
        model = cls(config)
        
        # Load weights from safetensors
        weights = safetensors.load_file(pretrained_path)
        model.load_state_dict(weights)
        
        return model
    
    def save_pretrained(self, save_directory: str):
        """Save model weights to directory."""
        import safetensors
        import os
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save weights
        safetensors.save_file(
            self.state_dict(),
            os.path.join(save_directory, "model.safetensors")
        )
        
        # Save config
        import json
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f)


class NexusForCausalLM(nn.Module):
    """Nexus model for causal language modeling."""
    
    def __init__(self, config: NexusConfig):
        super().__init__()
        self.model = NexusModel(config)
        self.config = config
        
        # Loss function
        self.loss_fn = CrossEntropyLoss(reduction='none')
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        **kwargs,
    ) -> Union[Tuple, dict]:
        """Forward pass with optional loss computation."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Compute loss
            loss = self.loss_fn(shift_logits, shift_labels)
            
            # Mask padding tokens
            if attention_mask is not None:
                valid_mask = shift_labels != self.config.pad_token_id
                loss = loss[valid_mask].mean()
            else:
                loss = loss.mean()
        
        if isinstance(outputs, dict):
            outputs["loss"] = loss
            return outputs
        else:
            return (loss,) + outputs if loss is not None else outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.model(generated_ids, use_cache=True)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                
                # Get last token logits
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
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[0, indices_to_remove] = float('-inf')
                
                # Sample or greedy decode
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Check for EOS
                if next_token.item() == self.config.eos_token_id:
                    break
        
        return generated_ids
    
    @classmethod
    def from_pretrained(cls, pretrained_path: str, config: Optional[NexusConfig] = None):
        """Load causal LM from pretrained weights."""
        model = NexusModel.from_pretrained(pretrained_path, config)
        causal_lm = cls(model.config)
        causal_lm.model = model
        return causal_lm
    
    def save_pretrained(self, save_directory: str):
        """Save causal LM to directory."""
        self.model.save_pretrained(save_directory)