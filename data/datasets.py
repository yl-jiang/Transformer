from typing import Iterator
import io
from torch.utils.data import IterableDataset
from pathlib import Path

__all__ = ['WMTIterDataset']


def yield_source_pair(src_filepath, tgt_filepath):
    with open(src_filepath, encoding='utf8', mode='r') as f:
        raw_src_iter = list(f.readlines())

    with open(tgt_filepath, encoding='utf8', mode='r') as f:
        raw_tgt_iter = list(f.readlines())
    
    for (raw_src, raw_tgt) in zip(raw_src_iter, raw_tgt_iter):
        yield (raw_src.strip(), raw_tgt.strip())

class WMTIterDataset(IterableDataset):  # 可迭代对象, 要实现__iter__方法(返回一个迭代器), 不能实现__next__方法

    def __init__(self, src_filepath, tgt_filepath):
        assert Path(src_filepath).exists(), f"{Path(src_filepath).resolve()} is not exist!"
        assert Path(tgt_filepath).exists(), f"{Path(tgt_filepath).resolve()} is not exist!"
        self.src_filepath = src_filepath
        self.tgt_filepath = tgt_filepath

    def _iter_count(self, filepath):
        from itertools import (takewhile, repeat)
        buffer = 1024 * 1024
        with open(filepath, errors='ignore', encoding='utf8', mode='r') as f:
            buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
            return sum(buf.count('\n') for buf in buf_gen)

    def __len__(self):
        src_line_count = self._iter_count(self.src_filepath)
        tgt_line_count = self._iter_count(self.tgt_filepath)
        assert src_line_count == tgt_line_count

        return src_line_count
    
    def __iter__(self) -> Iterator:
        return yield_source_pair(self.src_filepath, self.tgt_filepath)