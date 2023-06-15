import torch


__all__ = ['get_pad_mask', 'get_subsequent_mask']

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' 
    For masking out the subsequent info. 
    Inputs:
        seq: []
    Outputs:
        subsequent_mask: (1, token_num, token_num)
    '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask