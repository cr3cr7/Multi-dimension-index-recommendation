import math

import datasets
import numpy as np
import generateQuery
import common
import random

# Dataset Index后的数据 -> [Indexed Dataset]
# Trainning Query Generation -> [Training set], [统计一下每个查询扫描block数 Natural Order]
# 生成查询
def QueryGeneration(nums, train_data, cols):
    Queries = []
    scan_conditions = []

    for i in range(nums):
        rng = np.random.RandomState()
        query_cols_nums = random.randint(1, len(cols))
        qcols, qops, qvals, qranges = generateQuery.SampleTupleThenRandom(cols, query_cols_nums, rng, train_data.table.data)
        conditions = []
        for i in range(len(qcols)):
            if str(type(qvals[i])) == "<class 'str'>":
                qvals[i] = "'" + qvals[i] + "'"
            if str(type(qvals[i])) == "<class 'numpy.int64'>" or str(type(qvals[i])) == "<class 'numpy.float64'>":
                qvals[i] = str(qvals[i])
            if str(type(qvals[i])) == "<class 'pandas._libs.tslibs.timestamps.Timestamp'>":
                qvals[i] = "'" + str(qvals[i]) + "'"
        for qcol, qop, qval in zip(qcols, qops, qvals):
            conditions.append(f"(self.table.data['{qcol}'] {qop} {qval})")
        #print(conditions)
        predicate = " & ".join(conditions)
        Queries.append(predicate)
        
        scan_conditions.append([qcols, qranges])
    return Queries, scan_conditions

# Random Block Generation (Block Size = 20) -> [Indexed Dataset]
def RandomBlockGeneration(table, block_size):
    shuffle_train_data = common.TableDataset(table)
    arr = np.arange(table.data.shape[0]) // block_size
    random.shuffle(arr)
    shuffle_train_data.table.data["id"] = arr
    return shuffle_train_data

# Block Selection Number (Query) -> [Block Scan Number]
# 统计该查询需要扫描的数量
def BlocksScanNumber(qcols, qranges, train_data, block_size):
    blocks_num = math.ceil(table.data.shape[0] / block_size)
    scan_blocks = 0
    for id in range(blocks_num):
        Block = common.block(train_data.table, block_size, qcols, id)
        #print(Block._get_data())
        if Block._is_scan(qcols, qranges):
            scan_blocks += 1
    return scan_blocks

# Sample (Query) -> [Sampled Data according to Query (Indexed Dataset)]（分桶，保持原始数据distinct值不变）
def Sample(train_data, query):
    df, Indexed_df = train_data._predicate_data(query)
    return df, Indexed_df

if __name__ == "__main__":
    rng = np.random.RandomState()

    cols = [
    'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
    'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
    'Suspension Indicator', 'Revocation Indicator'
    ]
    

    nums = len(cols)

    table = datasets.LoadDmv('dmv-tiny-sort.csv')
    
    ### Genenate block id
    block_size = 20
    table.data["id"] = np.arange(table.data.shape[0]) // block_size

    
    train_data = common.TableDataset(table)

    
    
    queries = ["(self.table.data['Color'] >= 'WH   ')"]
    scan_conds = [[['Color'], [['WH   ', 'WH   ']]]]
    
    # Generate Query
    queries, scan_conds = QueryGeneration(1, train_data, cols)
    
    print("Scan Condition:", scan_conds[0][0], scan_conds[0][1])
    for i in range(len(queries)):
        # Get Predeicate Data
        predicate_data, Indexed_predicate_data = Sample(train_data, queries[i])
        print(predicate_data)
        # Scan Blocks and Get Blocks Numbers
        original_scans = BlocksScanNumber(scan_conds[i][0], scan_conds[i][1], train_data, block_size)
        print(original_scans)

    # Random Block Generation
    shuffle_train_data = RandomBlockGeneration(table, block_size)

    for i in range(len(queries)):
        # Get Predeicate Data
        predicate_data, Indexed_predicate_data = Sample(shuffle_train_data, queries[i])
        print(predicate_data)
        #
        shuffle_scans = BlocksScanNumber(scan_conds[i][0], scan_conds[i][1], shuffle_train_data, block_size)
        print(shuffle_scans)


    

