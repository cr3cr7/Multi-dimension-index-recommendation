import math

import datasets
import numpy as np
import pandas as pd
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

    qcols, qops, qranges = generateQuery.SampleTupleThenRandom(cols, 1, rng, table)
    qcols = ['Color']
    qops = ['<=']
    qranges = [['DK BR', 'WH   ']]
    print(qcols, qops, qranges)
    #print(qcols, qranges)

    ### Genenate block id
    size = 20
    table['id'] = np.arange(table.shape[0]) // size

    blocks = math.ceil(table.shape[0] / size)
    for id in range(blocks): 
        Blocks = common.block(table, 20, qcols, id)
        df = Blocks._get_data()
        print(df)
        isScan = Blocks._is_scan(qcols, qranges)
        print(isScan)