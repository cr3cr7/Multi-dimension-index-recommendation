import pandas as pd
import interleavebits
from util import NewBlock
import math
import copy
from util.NewBlock import QueryGeneration
import numpy as np
import common
import os 
import pickle
import datasets

class ALLBlock:
    
    def __init__(self, table, block_size, cols):
        self.table = copy.deepcopy(table)
        self.block_size = block_size
        self.cols = cols
        self.block_nums = math.ceil(self.table.shape[0] / self.block_size)

        self.cols_min = {}
        self.cols_max = {}
        
    def zero_except_id(self, id):
        indexed_df = self.table[self.table["block_id"] == id]
        self.collect_cols_min_max(indexed_df)
        
    def collect_cols_min_max(self, df):
        for i in self.cols:
            self.cols_min[i] = df[i].min()
            self.cols_max[i] = df[i].max()
            
    def _is_scan(self, qcols, qranges):
        for idx, i in enumerate(qcols):
            if self.cols_min.get(i, False):
                if i == 'Reg Valid Date' or i == 'Reg Expiration Date':
                    qranges[idx][0] = pd.to_datetime(qranges[idx][0])
                    qranges[idx][1] = pd.to_datetime(qranges[idx][1])
                if self.cols_min[i] > qranges[idx][1] or self.cols_max[i] < qranges[idx][0]:
                    return False
        return True

def Zorder(df, cols, original_df, name):
    vals = [[] for i in range(df.shape[0])]
    i = 0
    for index, row in df.iterrows():
        for idx, c in enumerate(cols):
            vals[i].append(row[c])
        zvalue = interleavebits.interleavem(*vals[i])
        df.loc[index, "zvalue"] = zvalue
        i += 1
    original_df["zvalue"] = df["zvalue"]
    original_df.sort_values(by="zvalue").to_csv(f"./datasets/{name}-1000-zorder.csv", index=False)
    # df.sort_values(by="zvalue").to_csv("./datasets/dmv-tiny-zorder.csv", index=False)




def ZorderBlock(df, cols, block_size):
    cols_num = len(cols)
    save_path = f"./datasets/scan_condation_{cols_num}Cols.pkl"
    if not os.path.exists(save_path):
        Queries, scan_conds = QueryGeneration(100, df.loc[:, cols], cols)
        pickle.dump(Queries, open(f"./datasets/Queries_{cols_num}Cols.pkl", "wb"))
        pickle.dump(scan_conds, open(save_path, "wb"))
    else:
        print("Load scan condations")
        scan_conds = pickle.load(open(save_path, "rb"))
    
    df = df.sort_values(by="zvalue")
    num_of_blocks = math.ceil(df.shape[0] / block_size)
    df['block_id'] = [i // block_size for i in range(0, df.shape[0])]
    Blocks = ALLBlock(df, block_size, cols)
    scan = 0
    for i in range(0, num_of_blocks):
        Blocks.zero_except_id(i)
        for j in range(0, len(scan_conds)):
            cur_scan_conds = scan_conds[j]
            scan += int(Blocks._is_scan(cur_scan_conds[0], cur_scan_conds[1]))
    print("scan", scan)


def LoadDmv(filename='dmv-clean.csv', 
            cols=['VIN','City','Zip','County','Reg Valid Date','Reg Expiration Date', 'zvalue']):
    csv_file = './datasets/{}'.format(filename)
    # cols = ['VIN','City','Zip','County','Reg Valid Date','Reg Expiration Date', 'zvalue']
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64, 'Reg Expiration Date': np.datetime64}
    return common.CsvTable('DMV', csv_file, cols, type_casts)

def LoadLineitem(filename, cols):
    csv_file = './datasets/{}'.format(filename)
    return common.CsvTable('Lineitem', csv_file, cols, sep=',')



if __name__ == "__main__":
    # df = pd.read_csv("./datasets/DMV_tuples_df.csv")
    # original_df = pd.read_csv("./datasets/dmv-tiny.csv")
    # cols = original_df.columns.tolist()
    # print(cols)
    # cols = ['VIN','City','Zip','County','Reg Valid Date','Reg Expiration Date']
    # cols = ['Color', 'State']
    # cols = ['l_orderkey', 'l_partkey']
    # all_cols = cols
    cols = ['0', '1', '2']
    all_cols = list(map(str, range(100)))
    # Zorder(df, cols, original_df=original_df)
    
    block_size = 200
    # table = LoadDmv("dmv-tiny-zorder.csv", cols=cols)
    name = 'RandomWalk_10000'
    # name = 'dmv-tiny'
    # name  = 'linitem_1000'
    # table = LoadLineitem(f"{name}.csv", cols=cols)
    table = datasets.LoadRandomWalk(100, 10000)
    df = common.TableDataset(table).tuples_df
    Zorder(df, cols, original_df=table.data, name=name)
    # print(table.data)

    ZorderBlock(table.data, all_cols, block_size)


    