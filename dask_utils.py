# coding: utf-8

from operator import add
from dask import delayed

def tree_agg(src_nodes, aggregate=add):
    aggregate_delayed = list(src_nodes)
    while True:
        aggregated = []
        if len(aggregate_delayed) == 1:
            return aggregate_delayed[0]

        if len(aggregate_delayed) % 2 == 1:
            aggregated.append(aggregate_delayed.pop(0))

        for idx in range(0, len(aggregate_delayed), 2):
            aggregated.append(delayed(aggregate)(aggregate_delayed[idx],
                                                 aggregate_delayed[idx + 1]))

        aggregate_delayed = list(aggregated)
