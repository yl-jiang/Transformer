from typing import Iterator
import io
from torch.utils.data import IterableDataset

__all__ = ['WMTIterDataset']


def yield_source_pair(src_filepath, tgt_filepath):
    raw_src_iter = io.open(src_filepath, encoding='utf8').readlines()
    raw_tgt_iter = io.open(tgt_filepath, encoding='utf8').readlines()

    for (raw_src, raw_tgt) in zip(raw_src_iter, raw_tgt_iter):
        yield (raw_src.strip(), raw_tgt.strip())

class WMTIterDataset(IterableDataset):  # 可迭代对象, 要实现__iter__方法(返回一个迭代器), 不能实现__next__方法

    def __init__(self, src_filepath, tgt_filepath):
        self.src_filepath = src_filepath
        self.tgt_filepath = tgt_filepath

    def __len__(self):
        raw_src_iter = io.open(self.src_filepath, encoding='utf8').readlines()
        raw_tgt_iter = io.open(self.tgt_filepath, encoding='utf8').readlines()
        assert len(raw_src_iter) == len(raw_tgt_iter)
        return len(raw_src_iter)
    
    def __iter__(self) -> Iterator:
        return yield_source_pair(self.src_filepath, self.tgt_filepath)