import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import *

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        Inputs:
            q: (sentence_num, n_head, enc_token_num, d_k) or (sentence_num, n_head, dec_token_num, d_k) or (sentence_num, n_head, dec_token_num, d_k)
            k: (sentence_num, n_head, enc_token_num, d_k) or (sentence_num, n_head, dec_token_num, d_k) or (sentence_num, n_head, enc_token_num, d_k)
            v: (sentence_num, n_head, enc_token_num, d_k) or (sentence_num, n_head, dec_token_num, d_k) or (sentence_num, n_head, enc_token_num, d_k)
            mask: (sentence_num, 1, enc_token_num, 1) or (sentence_num, 1, dec_token_num, dec_token_num) or ()
        Ouputs:
            output: (sentence_num, n_head, enc_token_num, d_v) or (sentence_num, n_head, dec_token_num, d_v)
            attn: (sentence_num, n_head, enc_token_num, enc_token_num) or (sentence_num, n_head, dec_token_num, dec_token_num) or (sentence_num, n_head, dec_token_num, enc_token_num)
        """
        #    (sentence_num, n_head, enc_token_num, d_k) & (sentence_num, n_head, d_k, enc_token_num) -> (sentence_num, n_head, enc_token_num, enc_token_num)
        # or (sentence_num, n_head, dec_token_num, d_k) & (sentence_num, n_head, d_k, dec_token_num) -> (sentence_num, n_head, dec_token_num, dec_token_num)
        # or (sentence_num, n_head, dec_token_num, d_k) & (sentence_num, n_head, d_k, enc_token_num) -> (sentence_num, n_head, dec_token_num, enc_token_num)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        #    (sentence_num, n_head, enc_token_num, enc_token_num) & (sentence_num, n_head, enc_token_num, d_v) -> (sentence_num, n_head, enc_token_num, d_v)
        # or (sentence_num, n_head, dec_token_num, dec_token_num) & (sentence_num, n_head, dec_token_num, d_v) -> (sentence_num, n_head, dec_token_num, d_v)
        # or (sentence_num, n_head, dec_token_num, enc_token_num) & (sentence_num, n_head, enc_token_num, d_v) -> (sentence_num, n_head, dec_token_num, d_v)
        output = torch.matmul(attn, v)

        return output, attn
    

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """
        Inputs:
            n_head: 注意力头的个数
            d_model: 等于embedding_size
            d_k: key laten dimension
            d_v: value laten dimension
        """
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        """
        Inputs:
            q: (sentence_num, enc_token_num, embedding_size) or (sentence_num, dec_token_num, embedding_size) 
            k: (sentence_num, enc_token_num, embedding_size) or (sentence_num, dec_token_num, embedding_size) 
            v: (sentence_num, enc_token_num, embedding_size) or (sentence_num, dec_token_num, embedding_size)  
            mask: Encoder-> (sentence_num, enc_token_num, 1) or Decoder-> (sentence_num, dec_token_num, dec_token_num)
        Outputs:
            q: (sentence_num, enc_token_num, embedding_size) or (sentence_num, dec_token_num, embedding_size)
            attn: (sentence_num, n_head, enc_token_num, enc_token_num) 
                or (sentence_num, n_head, dec_token_num, dec_token_num) 
                or (sentence_num, n_head, dec_token_num, enc_token_num)
        """

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # q: (sentence_num, enc_token_num, embedding_size) -> (sentence_num, enc_token_num, n_head * d_k) -> (sentence_num, enc_token_num, n_head, d_k)
        # or (sentence_num, dec_token_num, embedding_size) -> (sentence_num, dec_token_num, n_head * d_k) -> (sentence_num, dec_token_num, n_head, d_k)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        # k: (sentence_num, enc_token_num, embedding_size) -> (sentence_num, enc_token_num, n_head * d_k) -> (sentence_num, enc_token_num, n_head, d_k)
        # or (sentence_num, dec_token_num, embedding_size) -> (sentence_num, dec_token_num, n_head * d_k) -> (sentence_num, dec_token_num, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k) 
        # v: (sentence_num, enc_token_num, embedding_size) -> (sentence_num, enc_token_num, n_head * d_v) -> (sentence_num, enc_token_num, n_head, d_v)
        # or (sentence_num, dec_token_num, embedding_size) -> (sentence_num, dec_token_num, n_head * d_v) -> (sentence_num, dec_token_num, n_head, d_v)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v) 

        # Transpose for attention dot product: b x n x lq x dv
        # q: (sentence_num, n_head, enc_token_num, d_k) or (sentence_num, n_head, dec_token_num, d_k)
        # k: (sentence_num, n_head, enc_token_num, d_k) or (sentence_num, n_head, dec_token_num, d_k)
        # v: (sentence_num, n_head, enc_token_num, d_v) or (sentence_num, n_head, dec_token_num, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            # Encoder: (sentence_num, enc_token_num, 1) -> (sentence_num, 1, enc_token_num, 1)
            # or 
            # Decoder: (sentence_num, dec_token_num, dec_token_num) -> (sentence_num, 1, dec_token_num, dec_token_num)
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # q:    (sentence_num, n_head, enc_token_num, d_v) or (sentence_num, n_head, dec_token_num, d_v)
        # attn: (sentence_num, n_head, enc_token_num, enc_token_num) or (sentence_num, n_head, dec_token_num, enc_token_num)
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # q: (sentence_num, n_head, enc_token_num, d_v) -> (sentence_num, enc_token_num, n_head, d_v) -> (sentence_num, enc_token_num, n_head*d_v)
        # or q: (sentence_num, n_head, dec_token_num, d_v) -> (sentence_num, dec_token_num, n_head, d_v) -> (sentence_num, dec_token_num, n_head*d_v)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q: (sentence_num, token_num, n_head*d_v) -> (sentence_num, token_num, embedding_size)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        """
        Inputs:
            d_in: embedding_size
            d_hid: hidden feature's dimension
        """
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        two feedforward layers with residual connection
        """

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x



class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        """
        Inputs:
            d_model: 等于embedding_size
            d_inner:
            n_head:
            d_k:
            d_v:
        """
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        """
        Inputs:
            enc_input: (sentence_num, token_num, embedding_size)
        Outputs:
            enc_output: (sentence_num, token_num, embedding_size)
            enc_slf_attn: (sentence_num, n_head, token_num, token_num)
        """
        # enc_output: (sentence_num, token_num, embedding_size)
        # enc_slf_attn: (sentence_num, n_head, token_num, token_num)
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # enc_output: (sentence_num, token_num, embedding_size)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        """
        Inputs:
            d_model: 等于embedding_size
            d_inner:
            n_head: attention layer中注意力头的个数
            d_k:
            d_v:
        """
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        """
        Inputs:
            dec_input: (sentence_num, dec_token_num, embedding_size)
            enc_input: (sentence_num, enc_token_num, embedding_size)
            slf_attn_mask: bool tensor, the element is false if it is pad token / (sentence_num, enc_token_num, 1)
            dec_enc_attn_mask: bool tensor / (sentence_num, dec_token_num, dec_token_num)
        Outputs:
            dec_output: (sentence_num, dec_token_num, embedding_size)
            dec_slf_attn: (sentence_num, n_head, dec_token_num, dec_token_num)
            dec_enc_attn: (sentence_num, n_head, dec_token_num, enc_token_num)
        """
        # dec_output: (sentence_num, dec_token_num, embedding_size)
        # dec_slf_attn: (sentence_num, n_head, dec_token_num, dec_token_num)
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        # dec_output: (sentence_num, dec_token_num, embedding_size)
        # dec_enc_attn: (sentence_num, n_head, dec_token_num, enc_token_num)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

    

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        """
        Inputs:
            d_hid: token的embedding维度
            n_position: 需要大于等于token_num的个数
        """
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' 
        Sinusoid position encoding table 
        Inputs:
            n_position: 位置编码的个数(需要小于等于每个sentence中的token个数)
            d_hid: token的embedding维度
        Outputs:
            sinusoid_table: (1, n_position, d_hid)
        '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            """
            Inputs:
                position: a scalar
            Outputs:
                list of float scalar, length is d_hid
            """
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        # sinusoid_table: (n_position, d_hid)
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """
        Inputs:
            x: (sentence_num, token_num, embedding_size)
        Outputs:
            feature with position embedding: (sentence_num, token_num, embedding_size)
        """
        # (sentence_num, token_num, embedding_size) & (1, token_num, embedding_size)
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
        """
        Inputs:
            n_src_vocab: token总数
            d_word_vec: 每个token的embedding长度
            n_layers: encoder layer数量
            n_head: 自注意力头个数
            d_k: key的dimension
            d_v: value的dimension
            d_model: 等于d_word_vec
            d_inner: fc层的hidden dimension
            pad_idx: 填充id, 比如, 输入长度为100, 但是每次的句子长度并不一样, 后面就需要用统一的数字填充, 而这里就是指定这个数字, 这样, 网络在遇到填充id时, 就不会计算其与其它符号的相关性。(该idx对应的embedding全部初始化为0, 且不参与后续的参数更新)
        """

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        """
        Inputs:
            src_seq: list of idx (sentence_num, token_num)
            src_mask: bool tensor with shape of (sentence_num, token_num)
            return_attns:
        Outputs:
            enc_output: (sentence_num, token_num, embedding_size)
            enc_slf_attn: (sentence_num, n_head, token_num, token_num)
        """

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)  # (sentence_num, token_num, embedding_size)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))  # (sentence_num, token_num, embedding_size)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            # enc_output: (sentence_num, token_num, embedding_size)
            # enc_slf_attn: (sentence_num, n_head, token_num, token_num)
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):
        """
        Inputs:
            n_trg_vocab: 输出词库中token总个数
            d_word_vec: 每个token的embedding dimension
            n_layers: transformer的层数
            n_head: 注意力头个数
            d_k: key的dimension
            d_v: value的dimension
            d_model:
            d_inner: fc层的hidden dimension
            pad_idx: padding token在vocabulary里的index
        """

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        """
        Inputs:
            trg_seq: 
            trg_mask: (sentence_num, dec_token_num, dec_token_num)
            enc_output: (sentence_num, enc_token_num, embedding_size)
            src_mask: (sentence_num, enc_token_num, 1)
        Outputs:
            dec_outputs: (sentence_num, dec_token_num, embedding_size)
        """

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)  # (sentence_num, dec_token_num, embedding_size)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))  # (sentence_num, dec_token_num, embedding_size)
        dec_output = self.layer_norm(dec_output)  # (sentence_num, dec_token_num, embedding_size)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


