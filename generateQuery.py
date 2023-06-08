import pandas as pd
import numpy as np
import time

def SampleTupleThenRandom(all_cols,
                          num_filters,
                          rng,
                          data,
                          return_col_idx=False):
    s = data.iloc[rng.randint(0, 10)]
    vals = s.values
    
    # Giant hack for DMV.
    #vals[6] = vals[6].to_datetime64()

    idxs = rng.choice(len(all_cols), replace=False, size=num_filters)
    cols = np.take(all_cols, idxs)
    cols.tolist()

    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    ops = rng.choice(['<=', '>='], size=num_filters)
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [True for i in range(num_filters)]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)
    ops = ops.tolist()
    
    if num_filters == len(all_cols):
        ranges = [[None, None] for i in range(num_filters)]
        for i in range(num_filters):
            if ops[i] == '>=':
                ranges[i][0] = vals[i]
                ranges[i][1] = data[all_cols[i]].max()
            if ops[i] == '<=':
                ranges[i][1] = vals[i]
                ranges[i][0] = data[all_cols[i]].min()
        if return_col_idx:
            return np.arange(len(all_cols)), ops, vals, ranges
        return all_cols, ops, vals, ranges


    vals = vals[idxs]
    # print("vals:", vals)
    ranges = [[None, None] for i in range(num_filters)]
    for i in range(num_filters):
        if ops[i] == '>=':
            ranges[i][0] = vals[i]
            ranges[i][1] = data[cols[i]].max()
        if ops[i] == '<=':
            ranges[i][1] = vals[i]
            ranges[i][0] = data[cols[i]].min()
    if return_col_idx:
        return idxs, ops, vals, ranges

    return cols, ops, vals, ranges
