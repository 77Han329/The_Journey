import torch
import math
import torch.nn as nn 


class MySelfAttention_V1(nn.Module):
    def __init__(self, hidden_dim=728):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim,hidden_dim)
        self.k_proj = nn.Linear(hidden_dim,hidden_dim)
        self.v_proj = nn.Linear(hidden_dim,hidden_dim)
        
    def forward(self, x):
        
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        K_T = K.transpose(-1,-2)
        
        attn_value = torch.matmul(Q,K_T)
        
        attn_weight = torch.softmax(attn_value / math.sqrt(self.hidden_dim),dim=-1)
        
        attn_score = torch.matmul(attn_weight,V)
        
        return attn_score
    
    
    
# 加入dropout
# 加入aatention_mask
# output mat proj 
    
class MySelfAttention_V2(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()

        self.hiddend_dim = hidden_dim
        
        self.qkv_proj = nn.Linear(hidden_dim, 3*hidden_dim)
        self.dropout = nn.Dropout(self,dropout_rate)
        
        self.output_proj = nn.Linear(hidden_dim,hidden_dim)        
        
        def forward(self, x, attention_mask = None):
            QKV = self.qkv_proj(x)
            Q,K,V = torch.split(QKV, hidden_dim ,-1)
            K_t = K.transpose(-1,-2)
            
            attn_weight = torch.matmul(Q,K_t) / math.sqrt(hidden_dim)
            
            if attention_mask is not None:
                attn_weight = attn_weight.masked_fill(
                    attn_weight == 0,
                    float("1e20")
                )
            attn_weight = self.dropout(dropout_rate)
            attn_weight = nn.Softmax(attn_weight)
            
            attn_score = attn_weight @ V
            
            output = self.output_proj(attn_score)
            
            return output
            
            
    
class MyMHA(nn.Module):
    def __init__(self, head_num, hidden_dim, head_dim, drop_out=0.0):
        super().__init__()
        
        self.head_num = head_num
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        
        assert hidden_dim == head_num * head_dim

        # Q, K, V 投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # dropout
        self.attn_dropout = nn.Dropout(drop_out)

        # 输出 projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 1. Q K V 线性映射
        Q = self.q_proj(x)       # (b, s, h)
        K = self.k_proj(x)       # (b, s, h)
        V = self.v_proj(x)       # (b, s, h)
        
        # 2. reshape 为多头格式 (b, head_num, s, head_dim)
        Q = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        
        # 3. 注意力权重 (b, head_num, s, s)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # 4. mask（如有）
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-1e20'))

        # 5. softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 6. 加权求和
        attn_output = torch.matmul(attn_weights, V)   # (b, head_num, s, head_dim)

        # 7. 把头拼回去 (b, s, hidden_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # 8. 输出 projection
        output = self.output_proj(attn_output)

        return output