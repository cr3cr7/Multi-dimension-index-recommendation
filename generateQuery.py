import pandas as pd
import numpy as np
import time
import common

def SampleTupleThenRandom(all_cols,
                          num_filters,
                          rng,
                          data,
                          return_col_idx=False):
    # s = data.loc[rng.randint(0, 10), all_cols]
    s = data.iloc[rng.randint(0, len(data))]
    vals = s.values
    
    # Giant hack for DMV.
    #vals[6] = vals[6].to_datetime64()

    idxs = rng.choice(len(all_cols), replace=False, size=num_filters)
    cols = np.take(all_cols, idxs)
    cols = cols.tolist()

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
            if str(type(ranges[i][0])) == "<class 'pandas._libs.tslibs.timestamps.Timestamp'>":
                ranges[i][1] = ranges[i][1].strftime('%Y-%m-%d %H:%M:%S')
                ranges[i][0] = ranges[i][0].strftime('%Y-%m-%d %H:%M:%S')
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
        if str(type(ranges[i][0])) == "<class 'pandas._libs.tslibs.timestamps.Timestamp'>":
                ranges[i][1] = ranges[i][1].strftime('%Y-%m-%d %H:%M:%S')
                ranges[i][0] = ranges[i][0].strftime('%Y-%m-%d %H:%M:%S')
    if return_col_idx:
        return idxs, ops, vals, ranges

    return cols, ops, vals, ranges


def SampleGaussianQueries(all_cols,
                          num_filters,
                          rng,
                          data,
                        #   table: common.CsvTable,
                          return_col_idx=False):
    # aspect_ratio = [1/4, 1/2, 3/4]
    # Generate window query with aspect ratio 1/4, 1, 4
    idxs = rng.choice(len(all_cols), replace=False, size=num_filters)
    cols = np.take(all_cols, idxs)
    cols = cols.tolist()
    
    ops = rng.choice(['<=', '>='], size=num_filters)
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [True for i in range(num_filters)]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)
    ops = ops.tolist()
    
    # selectivity = rng.uniform(0.1, 0.9)
    selectivity = 0.1
    one_col_selectivity = selectivity ** (1 / num_filters)
    
    ranges = [[None, None] for i in range(num_filters)]
    for i in range(num_filters):
        # center = rng.uniform(data[cols[i]].min(), data[cols[i]].max())
        gaussian_center = rng.normal(data[cols[i]].mean(), data[cols[i]].std() / 5)
        
        min_val = gaussian_center - (gaussian_center - data[cols[i]].min()) * one_col_selectivity / 2
        max_val = gaussian_center + (data[cols[i]].max() - gaussian_center) * one_col_selectivity / 2
        ranges[i][0] = min_val
        ranges[i][1] = max_val
        if str(type(ranges[i][0])) == "<class 'pandas._libs.tslibs.timestamps.Timestamp'>":
            ranges[i][1] = ranges[i][1].strftime('%Y-%m-%d %H:%M:%S')
            ranges[i][0] = ranges[i][0].strftime('%Y-%m-%d %H:%M:%S')
        if return_col_idx:
            return idxs, ops, vals, ranges

    return cols, ops, None, ranges

def generate_distribution_query(query_amount, num_filters, domains, maximum_range, all_cols, cluster_center_amount=10, sigma_percent=0.2):
    '''
    generate clusters of queries
    '''
    # first, generate cluster centers
    assert len(domains) == len(all_cols)
    
    import random
    centers = []
    for i in range(cluster_center_amount):
        center = [] # [D1, D2,..., Dk]
        for k in range(len(domains)):
            ck = random.uniform(domains[k][0], domains[k][1])
            center.append(ck)
        centers.append(center)

    # second, generate expected range for each dimension for each center
    centers_ranges = []
    for i in range(cluster_center_amount):
        ranges = [] # the range in all dimensions for a given center
        for k in range(len(domains)):
            ran = random.uniform(0, maximum_range[k])
            ranges.append(ran)
        centers_ranges.append(ranges)

    # third, generate sigma for each dimension for each center
    centers_sigmas = []
    for i in range(cluster_center_amount):
        sigmas = []
        for k in range(len(domains)):
            sigma = random.uniform(0, maximum_range[k] * sigma_percent)
            sigmas.append(sigma)
        centers_sigmas.append(sigmas)

    # fourth, generate queries
    scan_conds = []
    for i in range(query_amount):
        rng = np.random.RandomState()
        ranges = []
        
        # choose a center
        center_index = random.randint(0, cluster_center_amount-1) # this is inclusive            
        query_lower, query_upper = [], []
        
        idxs = rng.choice(len(all_cols), replace=False, size=num_filters)
        cols = np.take(all_cols, idxs)
        cols = cols.tolist()
        
        for k in range(num_filters):
            # consider whether or not to use this dimension
            k = idxs[k]
            
            L, U = None, None
            
            center = centers[center_index]
            query_range = centers_ranges[center_index][k]
            L = center[k] - query_range/2
            U = center[k] + query_range/2
            L = random.gauss(L, centers_sigmas[center_index][k])
            U = random.gauss(U, centers_sigmas[center_index][k])
            if L <= domains[k][0]:
                L = domains[k][0]
            if U >= domains[k][1]:
                U = domains[k][1]
            if L > U:
                L, U = U, L
            
            ranges.append([L, U])
        # distribution_query.append(ranges)
        # qcols.append(cols)
        scan_conds.append([cols, ranges])
    return scan_conds