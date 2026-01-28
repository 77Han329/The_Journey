import torch
import torch.nn as nn 
import function as F
from einops import rearrange

class Linear(nn.Module):
    def __init__(self, in_feature:int, out_feature:int, device=None, dtype=None):
        super().__init__()
        
        self.weight = nn.Parameter(
            torch.empty(out_feature,in_feature,device,dtype)
        )
        
        std = (2 / in_feature + out_feature) **0.5
        nn.init.trunc_normal_(self.weight,mean=0,std=std, a=-3*std, b=3*std)
        
    def forward(self,x):
        
        output = torch.einsum(
            "...i,oi->...o",
            x,self.weight
        )
        
        return output
    
    
class TextEmbedding(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int, device=None, dtype=None):
        super().__init__()
        
        self.weight = nn.Parameter(
            torch.empty(vocab_size,embedding_dim,device,dtype)
        )
        
        nn.init.trunc_normal_(self.weight)
        
    def forward(self, x):
        
        
        return self.weight[x]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(
            torch.ones(d_model,device=device,dtype=dtype)
        )
        
    def calculate_rms(self,x:torch.Tensor, eps:float)->torch.Tensor:
        
        res = torch.sqrt(
            torch.mean(x*x, dim=-1, keepdim=True) + eps
        )
        
        return res
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input shape of x (b,s,d)
        output shape (b,s,d)
        """
        
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = self.calculate_rms(x,self.eps)
        
        result = (x / rms) * self.gain.to(x.dtype)
        
        return result.to(in_dtype)


class FFN(nn.Module):
    def __init__(self,d_model:int,d_ff:int,device=None, dtype=None):
        super().__init__()
        
        self.w1 = Linear(in_feature=d_model,out_feature=d_ff,device=device,dtype=dtype)
        self.w3 = Linear(in_feature=d_model,out_feature=d_ff,device=device,dtype=dtype)
        
        self.w2 = Linear(in_feature=d_ff,out_feature=d_model,device=device,dtype=dtype)
        
    def forward(self,x):
        
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
    
class RoPE(nn.Module):
    def __init__(self, seq_len: int, theta: float, d_k: int, device=None):
        super().__init__()

        power = torch.arange(0, d_k, 2, device=device).float() / d_k  # (d_k/2,)
        freq = 1.0 / (theta ** power)                                 # (d_k/2,)

        seq = torch.arange(seq_len, device=device)                     # (S,)
        angle_mat = torch.outer(seq, freq)                             # (S, d_k/2)

        self.register_buffer("cos_cache", angle_mat.cos(), persistent=False)
        self.register_buffer("sin_cache", angle_mat.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_position: torch.Tensor) -> torch.Tensor:
        # x: (B,H,S,d_k)
        cos = self.cos_cache[token_position].unsqueeze(0).unsqueeze(0).to(x.dtype)  # (1,1,S,d/2)
        sin = self.sin_cache[token_position].unsqueeze(0).unsqueeze(0).to(x.dtype)

        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_odd * cos + x_even * sin
        return out


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, seq_len: int, theta: float=10000,
                 device=None, dtype=None, use_rope=False):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_head = num_heads
        self.seq_len = seq_len
        self.use_rope = use_rope

        self.proj_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.proj_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.proj_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.proj_out = Linear(d_model, d_model, device=device, dtype=dtype)

        if use_rope:
            d_k = d_model // num_heads
            self.rope = RoPE(seq_len=seq_len, theta=theta, d_k=d_k, device=device)

        self.register_buffer("mask",torch.tril(
            torch.ones(seq_len,seq_len,device=device,dtype=torch.bool)),persistent=False)
        
    def forward(self, x):
        b, s, _ = x.shape

        # mask = torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        

        Q = rearrange(self.proj_q(x), "b s (h d) -> b h s d", h=self.num_head)
        K = rearrange(self.proj_k(x), "b s (h d) -> b h s d", h=self.num_head)
        V = rearrange(self.proj_v(x), "b s (h d) -> b h s d", h=self.num_head)

        if self.use_rope:
            token_position = torch.arange(s, device=x.device)
            Q = self.rope(Q, token_position)
            K = self.rope(K, token_position)

        mask = self.mask[:s,:s]
        mask = mask.unsqueeze(-1).unsqueeze(-1)
        attn = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        attn = rearrange(attn, "b h s d -> b s (h d)")
        return self.proj_out(attn)
    
    
class Transformer_block(nn.Module):
    def __init__(self, d_model:int, d_k:int, seq_len:int, d_ff:int, num_head:int, theta: float = 10000.0, eps: float = 1e-5, device=None, dtype=None, 
                 use_rope=False, norm='prenorm'):
        super().__init__()
        self.norm = norm
        self.mha = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_head,
            seq_len=seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
            use_rope=use_rope
        )
        
        self.attn_norm = RMSNorm(
            d_model=d_model,eps=eps,device=device, dtype=dtype
        )
        
        self.ffn = FFN(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )
        
        self.ffn_norm = RMSNorm(
            d_model=d_model,eps=eps,device=device, dtype=dtype
        )
        
    def forward(self,x):
        if self.norm == 'prenorm':
            x = x + self.mha(self.attn_norm(x))
            
            x = x + self.ffn(self.ffn_norm(x))
        
        else:
            raise ValueError("need to have prenorm")
        
        return x 
    
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size:int, seq_len:int, num_layers:int, d_model:int, d_k:int, d_ff:int, num_head:int, theta: float = 10000.0, eps: float = 1e-5, device=None, dtype=None, 
                 use_rope=False, norm='prenorm'):
        super().__init__()
        
        
        
        self.token_embedding = TextEmbedding(
            vocab_size=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )
        
        self.transformer_blocks = nn.ModuleList([
            Transformer_block(
                d_model=d_model,
                d_k=d_k,
                seq_len=seq_len,
                d_ff=d_ff,
                num_head=num_head,
                theta=theta,
                eps=eps,
                device=device,
                dtype=dtype,
                use_rope=use_rope,
                norm=norm
            ) 
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype
        )
        
        self.linear_out = Linear(in_feature=d_model,out_feature=vocab_size,device=device,dtype=dtype)
        
    def forward(self, x):
        
        x = self.token_embedding(x)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
            
        x = self.norm(x)
        
        logtis = self.linear_out(x)
        
        return logtis
        