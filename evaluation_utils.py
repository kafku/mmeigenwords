# coding: utf-8

import json
import subprocess
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

def get_commit_ID():
    commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    return commit_id.decode('ascii').replace('\n', '')

# NOTE: see https://cs.stanford.edu/people/karpathy/deepimagesent/
def load_split(path, columns=('id', 'split')):
    with open(path) as f:
        data = json.load(f)

    return pd.DataFrame([(entry['cocoid'], entry['split']) for entry in data['images']],
                        columns=columns)

def eval_retrieval(query_ids, caption_table, search_func,
                   query_col='image_id', target_col='id',
                   rank=None,
                   *args, **kwargs):
    if rank is None:
        rank = [1, 5, 10, 20]

    id_arr_converter = lambda x: x.astype(caption_table[target_col].dtype)
    id_converter = lambda x: caption_table[query_col].dtype.type(x)

    recall = []
    for query_id in tqdm(query_ids):
        # top-matches
        topmatch = search_func(query_id, *args, **kwargs)
        caption_ids, score = zip(*topmatch)
        caption_ids = id_arr_converter(np.array(caption_ids))

        # ground truth
        ground_truth = set(caption_table[caption_table[query_col] == id_converter(query_id)][target_col].values)
        y_true = caption_table[target_col].isin(ground_truth)

        # recall@k
        r_at_k = []
        for k in rank:
            y_pred = caption_table[target_col].isin(caption_ids[:k])
            r_at_k.append(recall_score(y_true, y_pred, average='binary'))
        recall.append(r_at_k)
        # TODO: median rank of the first ground truth

    return np.array(recall)
