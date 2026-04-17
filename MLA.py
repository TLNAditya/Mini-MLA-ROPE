import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# -----------------------------------------------------------------------------
# Rotary Position Embedding (RoPE) utilities
# -----------------------------------------------------------------------------

def precompute_freqs_complex_matrix(dim, max_seq_len, base = 10000.0) -> torch.Tensor:
    """
    Precompute the complex frequency tensor for RoPE.
    Returns a tensor of shape [max_seq_len, dim//2] with complex values.
    """
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len)
    freqs = torch.outer(positions, theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)   # complex rotation
    return freqs_cis

def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to a tensor.
    Args:
        x: [batch, seq_len, n_heads, head_dim]
        freqs_cis: [max_seq_len, head_dim//2] (complex)
    Returns:
        Tensor of same shape as x.
    """
    # Reshape to complex: split last dim into pairs
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Slice freqs to match sequence length and broadcast
    freqs_cis = freqs_cis[:x.shape[1]].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
    # Rotate and convert back
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(3)  # combine real/imag pairs back to last dim
    return x_out.type_as(x)

# -----------------------------------------------------------------------------
# Multi‑head Latent Attention (MLA) – Training Mode (with RoPE)
# -----------------------------------------------------------------------------

class MultiHeadLatentAttention(nn.Module):
    """
    MLA as described in DeepSeek‑V2 paper, Section 2.1 and Appendix C.
    
    Args:
        d_model:        Model hidden dimension.
        n_heads:        Number of attention heads.
        d_latent:       Dimension of the compressed latent for KV (c_KV). Paper uses 4*d_head.
        d_latent_q:     Dimension of the compressed latent for Q (c_Q). Paper uses 1536 for 5120 dim.
        d_head_rope:    Dimension of the decoupled RoPE part per head. Paper uses 64.
        max_seq_len:    Maximum sequence length (for RoPE precomputation).
        rope_base:      Base for RoPE (default 10000).
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_latent: int,
        d_latent_q: int,
        d_head_rope: int,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads          # dimension per head for compressed part (d_h)
        self.d_head_rope = d_head_rope
        self.d_latent = d_latent
        self.d_latent_q = d_latent_q
        
        # ---- Query low‑rank compression ----
        self.W_dq = nn.Linear(d_model, d_latent_q, bias=False)     # down‑projection
        self.W_uq = nn.Linear(d_latent_q, d_model, bias=False)     # up‑projection for compressed Q
        
        # ---- Key‑Value joint low‑rank compression ----
        self.W_dkv = nn.Linear(d_model, d_latent, bias=False)      # down‑projection for KV
        self.W_uk = nn.Linear(d_latent, d_model, bias=False)       # up‑projection for K
        self.W_uv = nn.Linear(d_latent, d_model, bias=False)       # up‑projection for V
        
        # ---- Decoupled RoPE components ----
        # RoPE queries: derived from compressed query latent
        self.W_qr = nn.Linear(d_latent_q, n_heads * d_head_rope, bias=False)
        # RoPE key: derived directly from hidden state (not compressed)
        self.W_kr = nn.Linear(d_model, d_head_rope, bias=False)    # shared key for all heads
        
        # ---- Output projection ----
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Precompute RoPE frequencies for the decoupled dimensions
        self.register_buffer(
            "freqs_cis_rope",
            precompute_freqs_cis(d_head_rope, max_seq_len, rope_base),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_kv_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: Optional mask to add to attention scores (e.g., causal mask).
            return_kv_cache: If True, returns the latent KV cache (c_KV and k_rope) for inference.
        
        Returns:
            output: [batch, seq_len, d_model]
            kv_cache: Tuple (c_kv, k_rope) if return_kv_cache else None
        """
        B, T, _ = hidden_states.shape
        
        # -----------------------------------------------------------------
        # 1. Query compression and RoPE
        # -----------------------------------------------------------------
        c_q = self.W_dq(hidden_states)                               # [B, T, d_latent_q]
        q_compressed = self.W_uq(c_q)                                # [B, T, d_model]
        # Split into heads for compressed part
        q_c = q_compressed.view(B, T, self.n_heads, self.d_head)     # [B, T, n_heads, d_head]
        
        # RoPE query from compressed latent
        q_rope = self.W_qr(c_q)                                      # [B, T, n_heads * d_head_rope]
        q_rope = q_rope.view(B, T, self.n_heads, self.d_head_rope)   # [B, T, n_heads, d_head_rope]
        q_rope = apply_rope(q_rope, self.freqs_cis_rope)
        
        # Concatenate compressed and RoPE parts
        q = torch.cat([q_c, q_rope], dim=-1)                         # [B, T, n_heads, d_head + d_head_rope]
        
        # -----------------------------------------------------------------
        # 2. Key‑Value compression and RoPE
        # -----------------------------------------------------------------
        c_kv = self.W_dkv(hidden_states)                             # [B, T, d_latent]
        k_compressed = self.W_uk(c_kv)                               # [B, T, d_model]
        v_compressed = self.W_uv(c_kv)                               # [B, T, d_model]
        
        # Split into heads for compressed part
        k_c = k_compressed.view(B, T, self.n_heads, self.d_head)     # [B, T, n_heads, d_head]
        v_c = v_compressed.view(B, T, self.n_heads, self.d_head)     # [B, T, n_heads, d_head]
        
        # RoPE key from original hidden states (shared across heads)
        k_rope = self.W_kr(hidden_states)                            # [B, T, d_head_rope]
        k_rope = apply_rope(k_rope.unsqueeze(2), self.freqs_cis_rope)  # [B, T, 1, d_head_rope]
        k_rope = k_rope.expand(-1, -1, self.n_heads, -1)             # [B, T, n_heads, d_head_rope]
        
        # Concatenate compressed and RoPE parts
        k = torch.cat([k_c, k_rope], dim=-1)                         # [B, T, n_heads, d_head + d_head_rope]
        
        # Values remain compressed only (RoPE not applied to values)
        v = v_c                                                      # [B, T, n_heads, d_head]
        
        # -----------------------------------------------------------------
        # 3. Attention (multi‑head scaled dot‑product)
        # -----------------------------------------------------------------
        # Transpose to [B, n_heads, T, d_head_total]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        d_head_total = self.d_head + self.d_head_rope
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head_total)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)                  # [B, n_heads, T, d_head]
        
        # -----------------------------------------------------------------
        # 4. Output projection
        # -----------------------------------------------------------------
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.W_o(attn_output)
        
        # Optionally return KV cache for inference
        kv_cache = (c_kv, k_rope[:, :, 0, :]) if return_kv_cache else None
        return output, kv_cache

# -----------------------------------------------------------------------------
# Inference‑Optimised MLA (with absorption)
# -----------------------------------------------------------------------------

class MultiHeadLatentAttentionInference(nn.Module):
    """
    Inference‑only version of MLA that uses the absorption trick to avoid
    materialising full keys and values. Only caches c_kv and k_rope.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_latent: int,
        d_latent_q: int,
        d_head_rope: int,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_head_rope = d_head_rope
        self.d_latent = d_latent
        self.d_latent_q = d_latent_q
        
        # Query compression (same as training)
        self.W_dq = nn.Linear(d_model, d_latent_q, bias=False)
        # After absorption, this matrix will contain W_uk * W_uq
        self.W_uq_eff = nn.Linear(d_latent_q, d_model, bias=False)
        self.W_qr = nn.Linear(d_latent_q, n_heads * d_head_rope, bias=False)
        
        # KV compression
        self.W_dkv = nn.Linear(d_model, d_latent, bias=False)
        self.W_kr = nn.Linear(d_model, d_head_rope, bias=False)
        
        # After absorption, this will contain W_o * W_uv
        self.W_o_eff = nn.Linear(d_model, d_model, bias=False)
        
        self.register_buffer(
            "freqs_cis_rope",
            precompute_freqs_cis(d_head_rope, max_seq_len, rope_base),
            persistent=False,
        )

    def absorb_matrices(self, W_uk: torch.Tensor, W_uv: torch.Tensor, W_uq: torch.Tensor, W_o: torch.Tensor):
        """
        Absorb the up‑projection matrices to enable latent‑space attention.
        Must be called after training with the trained weights.
        Args:
            W_uk: weight of W_uk (d_model, d_latent)
            W_uv: weight of W_uv (d_latent, d_model)
            W_uq: weight of W_uq (d_model, d_latent_q)
            W_o:  weight of W_o  (d_model, d_model)
        """
        with torch.no_grad():
            # Absorb W_uk into W_uq: new effective up‑projection = W_uk @ W_uq
            self.W_uq_eff.weight.data = W_uk @ W_uq
            # Absorb W_uv into W_o: new output projection = W_o @ W_uv
            self.W_o_eff.weight.data = W_o @ W_uv

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden_states: [batch, seq_len, d_model]
            past_kv: Tuple of (c_kv_past, k_rope_past) from previous steps.
            attention_mask: causal mask.
        Returns:
            output: [batch, seq_len, d_model]
            new_kv: Tuple (c_kv_new, k_rope_new) to be used in next step.
        """
        B, T, _ = hidden_states.shape
        
        # 1. Compute compressed KV and RoPE key
        c_kv = self.W_dkv(hidden_states)                     # [B, T, d_latent]
        k_rope = self.W_kr(hidden_states)                    # [B, T, d_head_rope]
        k_rope = apply_rope(k_rope.unsqueeze(2), self.freqs_cis_rope).squeeze(2)  # [B, T, d_head_rope]
        
        # Concatenate with past cache if provided
        if past_kv is not None:
            c_kv_past, k_rope_past = past_kv
            c_kv = torch.cat([c_kv_past, c_kv], dim=1)
            k_rope = torch.cat([k_rope_past, k_rope], dim=1)
        
        # 2. Compressed query
        c_q = self.W_dq(hidden_states)                       # [B, T, d_latent_q]
        # Effective query in compressed space (after absorption)
        q_compressed = self.W_uq_eff(c_q)                    # [B, T, d_model]
        q_c = q_compressed.view(B, T, self.n_heads, self.d_head)
        
        # RoPE query
        q_rope = self.W_qr(c_q)                              # [B, T, n_heads * d_head_rope]
        q_rope = q_rope.view(B, T, self.n_heads, self.d_head_rope)
        q_rope = apply_rope(q_rope, self.freqs_cis_rope)
        
        # Concatenate for final query
        q = torch.cat([q_c, q_rope], dim=-1)                 # [B, T, n_heads, d_head + d_head_rope]
        q = q.transpose(1, 2)                                # [B, n_heads, T, d_head_total]
        
        # 3. Keys in latent space: we have c_kv (d_latent) and k_rope (d_head_rope)
        # The compressed key part is not explicitly up‑projected; we use c_kv directly.
        # For multi‑head attention, we need to project c_kv to per‑head dimension.
        # In a full implementation, this is done by splitting c_kv into heads.
        # Here we assume d_latent is a multiple of n_heads (e.g., 4*d_head * n_heads).
        # For brevity, we treat c_kv as [B, T, n_heads, d_latent // n_heads].
        head_dim_latent = self.d_latent // self.n_heads
        c_kv_heads = c_kv.view(B, -1, self.n_heads, head_dim_latent).transpose(1, 2)  # [B, n_heads, T, head_dim_latent]
        
        # k_rope is shared across heads: [B, T, d_head_rope] -> [B, n_heads, T, d_head_rope]
        k_rope_heads = k_rope.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        
        # Concatenate to form full key in latent space
        k_latent = torch.cat([c_kv_heads, k_rope_heads], dim=-1)   # [B, n_heads, T, head_dim_latent + d_head_rope]
        
        # 4. Attention scores
        d_head_total = self.d_head + self.d_head_rope
        scores = torch.matmul(q, k_latent.transpose(-2, -1)) / math.sqrt(d_head_total)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_weights = F.softmax(scores, dim=-1)
        
        # 5. Values: after absorption, values are just c_kv (or c_kv up‑projected to d_model).
        # But since we absorbed W_uv into W_o, we can directly use c_kv as the value input
        # and then apply W_o_eff.
        # c_kv shape: [B, T, d_latent] -> we treat as [B, T, n_heads, head_dim_latent]
        attn_output = torch.matmul(attn_weights, c_kv_heads)        # [B, n_heads, T, head_dim_latent]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_latent)
        
        # 6. Output projection (with absorbed W_uv)
        output = self.W_o_eff(attn_output)                          # [B, T, d_model]
        
        new_kv = (c_kv, k_rope)
        return output, new_kv