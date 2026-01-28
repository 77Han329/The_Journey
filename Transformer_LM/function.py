import torch
import math

def silu(x:torch.Tensor):
    return x * torch.sigmoid(x)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    
    m = torch.max(x,dim=dim,keepdim=True).values
    
    shift_logits = x - m
    
    out = torch.exp(shift_logits) / torch.sum(torch.exp(shift_logits),dim=dim,keepdim=True)
    
    return out
    
def scaled_dot_product_attention(q:torch.Tensor,k:torch.Tensor, v:torch.Tensor,mask=None)-> torch.Tensor:
    
    d_k = q.shape[-1]
    
    dot_product = torch.einsum("...nk,...mk->...nm",q,k)/math.sqrt(d_k)
    
    
    
    if mask is not None:
        
        dot_product = dot_product.masked_fill(
            mask ==0,
            float('-inf')
        )
        
    prob = softmax(dot_product,dim=-1)
    
    
    
    output = torch.einsum(
        "...nm,...md->...nd", prob,v
    )
    
    return output


def cross_entropy(logits:torch.Tensor, label:torch.Tensor) -> torch.Tensor:
    
    """
    1. we need max of logits to make softmax calculation stable 
    2. we need do log sum exp to make calculation stable
    3. we need target_logits: (b,s,1) means the logits for each token
    """
    
    m = torch.max(logits,dim=-1,keepdim=True).values
    
    target_logits = torch.gather(
        input=logits,
        dim=-1,
        index=label.unsqueeze(-1)
    )
    
    log_sum_exp = torch.log(
        torch.sum(torch.exp(logits-m),dim=-1,keepdim=True)
    ) + m
    
    loss = - target_logits + log_sum_exp
    
    return torch.mean(loss)
