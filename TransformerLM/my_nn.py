import torch.nn as nn
import torch.nn.functional as F
import torch


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
        