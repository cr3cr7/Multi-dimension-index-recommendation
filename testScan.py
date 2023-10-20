from util.NewBlock import QueryGeneration
import datasets
import torch
import generateQuery
import numpy as np
import pandas as pd
import pickle

from zorder import ZorderBlock

def get_query(file_path):
    with open(file_path, 'rb') as f:
        query = pickle.load(f)
    Queries = []
    for one_query in query:
        query_col, query_range = one_query
        Predicates = []
        for one_predicate in zip(query_col, query_range):
            col, col_range = one_predicate
            colname = str(col)
            
            Predicates.append(f"(table['{colname}'] >= {col_range[0]})")
            Predicates.append(f"(table['{colname}'] <= {col_range[1]})")

        Predicates = " & ".join(Predicates)
        Queries.append(Predicates)
    return Queries

def Selectivity(table, query):
    now = len(table.loc[eval(query)])
    return now / len(table), now

block_size = 10000
rowNum = 1e6
colsNum = 100
dist = "UNI"

filename = f'GAUData-{int(rowNum/1000)}K-{colsNum}Col_{dist}.csv'
csv_file = './datasets/{}'.format(filename)
table = pd.read_csv(csv_file)


save_path = f"./datasets/scan_condation_{filename.split('.')[0]}_3.pkl"

Queries = get_query(save_path)


for i in Queries:
    print(i)
    selec, card = Selectivity(table, i)
    print(f"Selectivity: {selec * 100}%")
    print("Cardinality: ", card)
    print('='*20)  
