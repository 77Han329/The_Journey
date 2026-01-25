import math

def get_cos_scheduler(
    it:int,
    lr_max:float,
    lr_min:float,
    warmup_it:int,
    cos_it:int
)->float:
    
    if it < warmup_it:
        return lr_max * it / warmup_it
    
    elif it >= cos_it:
        return lr_min
    
    else:
    ## decay 
        decay = (it - warmup_it) / (cos_it - warmup_it)
        
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
        
        return lr_min + coeff * (lr_max - lr_min)
    
    