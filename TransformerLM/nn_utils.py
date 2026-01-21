import math 
import torch 






def scaled_dot_product_attention(Q:torch.Tensor,
                                 K:torch.Tensor,
                                 V:torch.Tensor,
                                 mask:torch.Tensor=None,) -> torch.Tensor:
    
    """
    Q, K, V have same seq_len at training stage but at inference it's usall that the len of Q does not match the len of K
    Q shape :(batch_size, num_heads, seq_len of Query, d_k)
    K, V shape :(batch_size, num_heads, seq_len of Key, d_k)
    
    
    mask shape: (batch_size, 1 or num_heads, seq_len, seq_len)
    """
    
    d_k = Q.shape[-1]
    
    dot_product = torch.einsum("...nk,...mk->...nm",Q,K)
    
    # dot_product = torch.matmul(Q,K.transpose(-1,-2))
    
    
    if mask is not None:
        dot_product = dot_product.masked_fill(
            mask == 0,
            float('-inf')
        )
        
        
    prob = torch.nn.functional.softmax(dot_product / math.sqrt(d_k),dim=-1)
    
    
    
    
    res = torch.einsum("...nm,...mk->...nk",prob, V)
    
    return res