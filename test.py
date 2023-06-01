import math

import datasets
import numpy as np
import generateQuery
import common

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


    size = 20
    table["id"] = np.arange(table.shape[0]) // size

    blocks_num = math.ceil(table.shape[0] / size)
    for id in range(blocks_num): 
        Block = common.block(table, 20, qcols, id)
        df = Block._get_data()
        print(df)
        isScan = Block._is_scan(qcols, qranges)
        print(isScan)