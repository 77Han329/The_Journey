import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from nn_utils import scaled_dot_product_attention
import nn_utils


class Embedding(nn.Module):
    """
    Input shape: (B, S)
    Output shape: (B, S, embd_size)
    """
    def __init__(self, vocabsize: int, embd_size: int, device=None, dtype=None):
        super().__init__()
        
        factory_kwargs = {'device': device,
                         'dtype': dtype}
        
        self.weight = nn.Parameter(
            (torch.empty(vocabsize, embd_size, **factory_kwargs))
        )
        
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        
    def forward(self,x):
        """
        Input shape: (B,S)
        Output shape: (B, S, D)
        B, S = x.shape
        D = self.weight.shape[1]

        out = torch.empty(B, S, D, device=x.device, dtype=self.weight.dtype) 这里吧最后一个维度先补齐

        for b in range(B):
            for s in range(S):
                token_id = x[b, s]             
                out[b, s] = self.weight[token_id] 然后根据weight 里面的索引, 类似查表的形式给每一个s 里面的token 找到对应的embeding

        return out  
        """
        return self.weight[x]
    
    
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        
        factory_kwargs = {'device':device, 'dtype':dtype}
        
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features,**factory_kwargs)
        )
        
        std = (2 / (in_features+out_features)) ** 0.5
        
        nn.init.trunc_normal_(self.weight, mean=0, std = std, a= -3*std, b= 3*std)
        
    def forward(self,x):
        
        """
        
        perform output = XW^T
        shape of X (B,S,in) shape of W^t (in,out)
        """
        """
        等价于
        return x @ self.weight.T
    
        return torch.matmul(x,self.weight.T)
        """

        return torch.einsum("...i,oi->...o", x, self.weight)
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int, num_head:int, device=None, dtype=None):
        super().__init__()
        
        assert d_model % num_head ==0
        
        self.d_k = d_model // num_head
        self.num_head = num_head
        self.d_model = d_model
        
        self.proj_q = Linear(d_model,d_model,device=device,dtype=dtype)
        self.proj_k = Linear(d_model,d_model,device=device,dtype=dtype)
        self.proj_v = Linear(d_model,d_model,device=device,dtype=dtype)
        
        self.proj_out = Linear(d_model,d_model,device=device,dtype=dtype)
        
        
    def forward(self,x):
        
        """
        shape of X (B,S,d_model)
        shape of w_q (d_model,d_model)
        """
        
        batch_size, seq_len, _ = x.shape
        mask = torch.tril(torch.ones(seq_len,seq_len,device=x.device, dtype=torch.bool))
        
        
        # 用rerange 的写法（CS336 标准）
        q = rearrange(self.proj_q(x),"b s (h d)->b h s d",h=self.num_head)
        k = rearrange(self.proj_k(x),"b s (h d)->b h s d",h=self.num_head)
        v = rearrange(self.proj_v(x),"b s (h d)->b h s d",h=self.num_head)
        
        attention_score = scaled_dot_product_attention(q,k,v,mask)
        
        
        attention_score = rearrange(
            attention_score,
            "b h s d->b s (h d)"
        )
        
        
        ## 不用rerange 的写法
        # batch_size, seq_len, _ = x.shape
        # q = self.proj_q(x)
        # k = self.proj_k(x)
        # v = self.proj_v(x)
        # k = k.T
        # q = q.view(batch_size, seq_len, self.num_head, self.d_k).transpose(1, 2)
        # k = k.view(batch_size, seq_len, self.num_head, self.d_k).transpose(1, 2)
        # v = v.view(batch_size, seq_len, self.num_head, self.d_k).transpose(1, 2)
        
        # attention_score = scaled_dot_product_attention(q,k,v,mask)
        
        # attention_score = attention_score.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)        
        
        output = self.proj_out(attention_score)
        
        return output
    
    
class RoPE(nn.Module):
    def __init__(self, seq_len: int, d_k: int, theta=10000, device=None):
        super().__init__()
        
        power = torch.arange(0,d_k,2,device=device).float() / d_k
        
        freq = 1.0 / theta**power
        
        token_position = torch.arange(seq_len,device=device)
        
        angle_mat = torch.outer(token_position,freq)
        
        self.register_buffer("cos_cache",angle_mat.cos(),persistent=False)
        self.register_buffer("sin_cache",angle_mat.sin(),persistent=False)
        
    def forward(self,x, token_position):
        
        cos = self.cos_cache[token_position].to(x.dtype)
        
        sin = self.sin_cache[token_position].to(x.dtype)
        
        if x.ndim > cos.ndim and cos.ndim >= 3:
            
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            
            
        output = torch.empty_like(x)
        
        x_even = x[...,0::2]
        x_odd = x[...,1::2]
        
        output[...,0::2] = x_even * cos - x_odd * sin
        output[...,1::2] = x_odd * cos + x_even * sin
        
        return output
    

class SwiGlu(nn.Module):
    def __init__(self, d_model: int, d_ff:int, device=None, dtype=None):
        super().__init__()
        
        self.d_ff = d_ff
        self.d_model = d_model
        
        
        self.w1 = Linear(out_features=d_ff, in_features=d_model,device=device,dtype=dtype)
        self.w2 = Linear(out_features=d_model, in_features=d_ff,device=device,dtype=dtype)
        self.w3 = Linear(out_features=d_ff, in_features=d_model,device=device,dtype=dtype)
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        
        gate = nn_utils.silu(self.w1(x)) 
        signal = self.w3(x)
        
        output = self.w2(gate * signal)
        
        return output
        
        
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
        
    def rms(self,x):
        
        return torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        
    def forward(self,x):
        
        rms = self.rms(x)
        
        output = x / rms * self.gamma
        
        return output
    
    
class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
        self.shift = nn.Parameter(torch.zeros(d_model,device=device,dtype=dtype))
        
        
    def forward(self,x):
        
        mean = torch.mean(x,dim=-1,keepdim=True)
        
        var = torch.mean((x - mean) **2, dim=-1, keepdim=True)
        
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        output = norm_x * self.scale + self.shift 
        
        return output