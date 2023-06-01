import math

import datasets
import numpy as np
import generateQuery
import common


# Dataset Index后的数据 -> [Indexed Dataset]
# Trainning Query Generation -> [Training set], [统计一下每个查询扫描block数 Natural Order]



# Random Block Generation (Block Size = 20) -> [Indexed Dataset]



# Block Selection Number (Query) -> [Block Scan Number]



# Sample (Query) -> [Sampled Data according to Query (Indexed Dataset)]（分桶，保持原始数据distinct值不变）


if __name__ == "__main__":
    rng = np.random.RandomState()

    cols = [
    'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
    'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
    'Suspension Indicator', 'Revocation Indicator'
    ]
    

    nums = len(cols)

    table = datasets.LoadDmv('dmv-tiny-sort.csv')
    table = table.data
    print(table.columns)
    qcols, qops, qranges = generateQuery.SampleTupleThenRandom(cols, 1, rng, table)
    qcols = ['Color']
    qops = ['<=']
    qranges = [['DK BR', 'WH   ']]
    print(qcols, qops, qranges)
    #print(qcols, qranges)

    ### Genenate block id
    size = 20
    table["id"] = np.arange(table.shape[0]) // size

    blocks_num = math.ceil(table.shape[0] / size)
    for id in range(blocks_num): 
        Block = common.block(table, 20, qcols, id)
        df = Block._get_data()
        print(df)
        isScan = Block._is_scan(qcols, qranges)
        print(isScan)