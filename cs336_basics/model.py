
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# === Transformer 模型定义 ===
# 这是模型运行所必需的全部结构定义。
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model, self.num_heads, self.head_dim = d_model, num_heads, d_model // num_heads
        self.qkv_proj, self.out_proj = nn.Linear(d_model, 3 * d_model), nn.Linear(d_model, d_model)
        self.dropout, self.scale = nn.Dropout(dropout), math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale
        if mask is not None: attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = (self.dropout(attn_probs) @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1, self.linear2, self.dropout = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
    def forward(self, x): return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention, self.feed_forward = MultiHeadAttention(d_model, num_heads, dropout), FeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2, self.dropout = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

class TinyStoriesTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout, pad_token_id=0):
        super().__init__()
        self.vocab_size, self.d_model, self.max_seq_len, self.pad_token_id = vocab_size, d_model, max_seq_len, pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm, self.lm_head = nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)): torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None: torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm): torch.nn.init.zeros_(module.bias); torch.nn.init.ones_(module.weight)
    
    def _create_causal_mask(self, seq_len: int, device): return torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)
    
    def forward(self, input_ids, targets=None):
        B, L = input_ids.shape
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(0, L, device=input_ids.device).unsqueeze(0))
        x = self.dropout(tok_emb + pos_emb)
        mask = self._create_causal_mask(L, input_ids.device)
        for block in self.blocks: x = block(x, mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None: loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=self.pad_token_id)
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature, top_k, top_p, eos_token_id):
        self.eval()
        for _ in range(max_new_tokens):
            input_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            logits, _ = self(input_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all(): break
        return input_ids
