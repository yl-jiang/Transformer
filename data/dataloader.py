import pickle
from config import Constants
import torch
from torchtext.data.utils  import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import io
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from pathlib import Path


__all__ = ['build_wmt_dataloader', 'Collator']


class Collator:
    def __init__(self, 
                 src_spacy_model_name='zh_core_web_md', 
                 tgt_spacy_model_name='en_core_web_md', 
                 src_txt_filepath='', 
                 tgt_txt_filepath='', 
                 **kwargs) -> None:
        
        self.src_spacy_model_name = src_spacy_model_name
        self.tgt_spacy_model_name = tgt_spacy_model_name
        self.src_txt_filepath = src_txt_filepath
        self.tgt_txt_filepath = tgt_txt_filepath

        self.SRC_LANGUAGE = src_spacy_model_name.split('_')[0]
        self.TGT_LANGUAGE = tgt_spacy_model_name.split('_')[0]
        
        # define speical symbols and indices
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        self.special_symbols = [Constants.UNK_WORD, Constants.PAD_WORD, Constants.BOS_WORD, Constants.EOS_WORD]

        self.token_transform = self.build_token_transform()
        self.vocab_transform = self.build_vocab_transform()
        self.text_transform = self.build_text_transform()

    def vocab_size(self):
        return {'src': len(self.vocab_transform[self.SRC_LANGUAGE]), 'tgt': len(self.vocab_transform[self.TGT_LANGUAGE])}
    
    def special_token_index(self, token_name):
        token2id = dict([(token, idx) for token, idx in zip(self.special_symbols, [self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX])])
        return token2id[token_name]

    def build_token_transform(self):
        token_transform = {}
        token_transform[self.SRC_LANGUAGE]= get_tokenizer('spacy', language=self.src_spacy_model_name)
        token_transform[self.TGT_LANGUAGE]= get_tokenizer('spacy', language=self.tgt_spacy_model_name) 
        return token_transform
    
    def build_vocab_transform(self):
        vocab_transform = {}
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            train_iter = self.yield_source_pair()
            vocab_transform[ln] = build_vocab_from_iterator(self.yield_tokens(train_iter, ln), min_freq=1, specials=self.special_symbols, special_first=True)
        return vocab_transform

    def yield_source_pair(self):
        raw_src_iter = io.open(self.src_txt_filepath, encoding='utf8').readlines()
        raw_tgt_iter = io.open(self.tgt_txt_filepath, encoding='utf8').readlines()

        for (raw_src, raw_tgt) in zip(raw_src_iter, raw_tgt_iter):
            yield (raw_src.strip(), raw_tgt.strip())

    def yield_tokens(self, data_iter:Iterable, language:str) -> List[str]:
        language_index = {self.SRC_LANGUAGE:0, self.TGT_LANGUAGE: 1}
        for data_sample in data_iter:
            yield self.token_transform[language](data_sample[language_index[language]])

    def sequentail_transforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([self.BOS_IDX]), torch.tensor(token_ids), torch.tensor([self.EOS_IDX])))

    def build_text_transform(self):
        text_transform = {}
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            text_transform[ln] = self.sequentail_transforms(self.token_transform[ln], self.vocab_transform[ln], self.tensor_transform)
        return text_transform

    def __call__(self, batch):
        """
        Inputs:
            batch: [
                ('加利福尼亚州水务工程的新问题', 'New Questions Over California Water Project'), 
                ('在加利福尼亚州一个主要水务管理区披露州长杰瑞·布朗领导的行政当局将提供政府资金以完成两条巨型输水隧道的规划之后，有一些评论家和一位州议员表示，他们想进一步了解由谁来为州长所支持的拟耗资160亿美元的水务工程承担费用。', "Critics and a state lawmaker say they want more explanations on who's paying for a proposed $16 billion water project backed by Gov. Jerry Brown, after a leading California water district said Brown's administration was offering government funding to finish the planning for the two giant water tunnels."), 
                ('评论家表示，洛杉矶的MWD周四所提及的政府资金可能有悖于该州长期以来的承诺，该承诺说，为了实现布朗所设想的挖掘两条35英里长隧道，以便于把水从萨克拉门托河向南输送，并主要供应给加利福尼亚州中部和南部地区的愿景，各地方水管区（而非加利福尼亚州自身）将承担费用。', "Critics said the government funding described by the Los Angeles-based Metropolitan Water District on Thursday could run counter to long-standing state assurances that various local water districts, not California itself, would pay for Brown's vision of digging twin 35-mile-long tunnels to carry water from the Sacramento River south, mainly for Central and Southern California."), 
                ('这两条隧道的2.48亿美元初期费用支出尚未获得监管部门批准便已经成为一项正在进行的联邦审计的主题。', 'The $248 million in preliminary spending for the tunnels, which have yet to win regulatory approval, already is the topic of an ongoing federal audit.'), 
                ('一些州议员也于周三责令对隧道费用支出情况进行一次州内审计。', 'On Wednesday, state lawmakers ordered a state audit of the tunnels-spending as well.'), 
                ('该州的发言人南希·沃格尔周四表示，尽管该账户属于洛杉矶市政水管区，却不会动用该州的任何普通资金来完成这两条隧道当前规划阶段的工程。', "On Thursday, state spokeswoman Nancy Vogel said that despite the account of the Los Angeles-based Metropolitan Water District, no money from the state's general fund would be used finishing the current planning phase of the twin tunnels."), 
                ('不过，该隧道项目的反对者和一个纳税人团体周四提出批评，而且本周的审计命令幕后的州立法者之一州议会女议员苏珊·艾格曼则于周四要求该州政府澄清相关情况。', "However, opponents of the tunnels and a taxpayer group were critical Thursday, and Assemblywoman Susan Eggman, one of the state lawmakers behind this week's audit order, asked the state Thursday for clarification."), 
                ('霍华德贾维斯纳税人协会立法主任大卫·沃尔夫表示：“这是一场骗局。”', '"It\'s a shell game," said David Wolfe, the Howard Jarvis Taxpayers Association\'s legislative director.'), 
                ('我认为就昨天的审计（请求）来说：其中的问题远比答案多。', 'I think it comes back to the audit (request) yesterday: There are way more questions here than there are answers.'), 
                ('该隧道项目受到布朗以及加利福尼亚州中部和南部一些具有政治影响力的供水管理区和用水客户的支持。', 'The tunnels project is endorsed by Brown and by some politically influential water districts and water customers in Central and Southern California.')
                ]
        Outputs:
            src_batch: (batch_size, src_token_num)
            tgt_batch: (batch_size, tgt_token_num)
        """
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transform[self.SRC_LANGUAGE](src_sample))
            tgt_batch.append(self.text_transform[self.TGT_LANGUAGE](tgt_sample))
        
        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD_IDX)
        return {'src': src_batch.permute(1, 0), 'trg': tgt_batch.permute(1, 0)}
    

def build_wmt_dataloader(src_spacy_model_name='zh_core_web_md', 
                         tgt_spacy_model_name='en_core_web_md', 
                         src_txt_filepath='', 
                         tgt_txt_filepath='', 
                         batch_size=2048, 
                         num_workers=2, 
                         pin_memory=True, 
                         drop_last=True, 
                         ):
    assert Path(src_txt_filepath).exists(), f"{Path(src_txt_filepath).resolve()} is not exist!"
    assert Path(tgt_txt_filepath).exists(), f"{Path(tgt_txt_filepath).resolve()} is not exist!"
    from .datasets import WMTIterDataset
    collate_fn = Collator(src_spacy_model_name, tgt_spacy_model_name, src_txt_filepath, tgt_txt_filepath)
    train_dataset = WMTIterDataset(src_txt_filepath, tgt_txt_filepath)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last)
    train_loader.vocab_size = collate_fn.vocab_size()
    train_loader.special_token_index = collate_fn.special_token_index
    return train_dataset, train_loader
