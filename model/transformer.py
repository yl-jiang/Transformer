from torch import nn
import torch
from utils.common import *
from utils.sublayers import *

__all__ = ['Transformer']

class Transformer(nn.Module):
    ''' 
    A sequence to sequence model with attention mechanism. 
    '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):
        """
        Inputs:
            n_src_vocab: 输入词库中token总个数
            n_trg_vocab: 输出词库中token总个数
            src_pad_idx: 输入词库中用于padding的token index
            trg_pad_idx: 输出词库中用于padding的token index
            d_word_vec: 每个token的embedding dimension
            d_model: 等于d_word_vec
            d_inner: feedforward layer中hidden dimension
            n_layers: attention layer层数
            n_head: 注意力头个数
            d_k: 
            d_v:

        """

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):
        """
        Inputs:
            src_seq: list of token index that in each input sentence / (sentence_num, enc_token_num)
            trg_seq: list of token index that in each output sentence / (sentence_num, dec_token_num)
        Outputs:
            tensor with shape (sentence_num * dec_token_num, n_trg_vocab)
        """
        # src_mask: (sentence_num, enc_token_num, 1)
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        # trg_mask: (sentence_num, dec_token_num, 1) & (1, dec_token_num, dec_token_num) -> (sentence_num, dec_token_num, dec_token_num)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        # enc_output: (sentence_num, enc_token_num, embedding_size)
        enc_output, *_ = self.encoder(src_seq, src_mask)
        # dec_output: (sentence_num, dec_token_num, embedding_size)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        # seq_logit: (sentence_num, dec_token_num, n_trg_vocab)
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))

