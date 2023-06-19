from util.Block import RandomBlockGeneration, BlockDataset, QueryGeneration
import datasets
import torch
import generateQuery
import numpy as np
table = datasets.LoadDmv('dmv-tiny.csv')
print(table.data.dtypes)
rng = np.random.RandomState()
#qcols, qops, qvals, qranges = generateQuery.SampleTupleThenRandom(table.ColumnNames(), 16, rng, table.data)

""" cols = table.ColumnNames()
block = BlockDataset(table, 20, cols, 99, True)
cnt = 0
for i in range(5):
    b,q ,flag = block[i]
    print(q)
print(cnt) """
cols = ['Body Type', 'Fuel Type', 'Reg Valid Date', 'Reg Expiration Date']
query, scan_conds = QueryGeneration(1, table.data, cols)