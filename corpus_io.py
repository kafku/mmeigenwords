# coding: utf-8
import os
import contextlib
from collections import Counter
from itertools import chain
import functools
from dask import delayed
from dask.diagnostics import ProgressBar
from dask_utils import tree_agg
import numpy as np

def split_corpus(path, max_len=10000000, join=True, convert=None, overlap=0,
                 tokenizer=lambda x: x.split()):
    """
    Args:
        path:
        max_len:
    """
    with open(path, "r") as corpus_file:
        head_char = None
        last_char = None
        last_token = None
        while True:
            text = corpus_file.read(max_len)
            if len(text) == 0:
                break

            head_char = text[0]
            tokenized_text = tokenizer(text)
            if head_char == ' ' or last_char == ' ':
                if last_token is not None:
                    tokenized_text = last_token + tokenized_text
            else:
                tokenized_text.insert(1, last_token[-1] + tokenized_text[0])
                tokenized_text = last_token[:-1] + tokenized_text[1:]

            last_char = text[-1]
            #last_token = tokenized_text[-1]
            last_token = tokenized_text[len(tokenized_text) - overlap - 1:]

            if join:
                yield ' '.join(tokenized_text[:-1])
            else:
                if convert is not None:
                    yield convert(tokenized_text[:-1])
                else:
                    yield tokenized_text[:-1]

def corpus_segment(path, max_len=10000000, join=True, convert=None,
                   seek_pos=None, tokenizer=lambda x: x.split()):
    with open(path, "r") as corpus_file:
        if seek_pos is not None:
            corpus_file.seek(seek_pos)

        text = corpus_file.read(max_len)
        if len(text) == 0:
            return None

        if join:
            return text
        else:
            tokenized_text = tokenizer(text)[1:-1]
            if convert is not None:
                return convert(tokenized_text)
            else:
                return tokenized_text

def read_corpus(path, max_len=10000000, convert=None):
    return chain.from_iterable(split_corpus(path, max_len, join=False, convert=convert))

def count_word_dask(path, vocab_size, n_oversample=1000,
                    n_workers=-1, n_partition=200, verbose=False):
    if n_workers == -1:
        n_workers = os.cpu_count()

    corpus_info = os.stat(path)
    read_size = corpus_info.st_size // n_partition
    count_words_topk = lambda tokens: Counter(
        dict(Counter(tokens).most_common(vocab_size + n_oversample)))
    add_counter = lambda x, y: Counter(dict((x + y).most_common(vocab_size + n_oversample)))

    load = functools.partial(corpus_segment, path, max_len=read_size,
                             convert=None, join=False)
    load_delayed = [delayed(load)(seek_pos=pos) for pos in np.arange(n_partition) * read_size]
    count_delayed = [delayed(count_words_topk)(tokens) for tokens in load_delayed]
    count = tree_agg(count_delayed, aggregate=add_counter)

    context_mgr = ProgressBar() if verbose else contextlib.suppress()
    with context_mgr:
        return count.compute(scheduler='processes')
