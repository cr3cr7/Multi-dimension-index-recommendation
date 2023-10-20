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
import re
import wfdb

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
    if "zvalue" in df.columns:
        print('Find zvalue column')
        return 
    else:
        print("Do not have zvalue column, re-generating ...")
        import time
        t1 = time.time()  
        vals = [[] for i in range(df.shape[0])]
        i = 0
        for index, row in df.iterrows():
            for idx, c in enumerate(cols):
                vals[i].append(row[c])
            zvalue = interleavebits.interleavem(*vals[i])
            df.loc[index, "zvalue"] = zvalue
            i += 1
        original_df["zvalue"] = df["zvalue"]
        print("Z Value Generation time cost: ", time.time() - t1)
        original_df.to_csv(f"./datasets/{name}-zorder.csv", index=False)
    # df.sort_values(by="zvalue").to_csv("./datasets/dmv-tiny-zorder.csv", index=False)




def ZorderBlock(df, cols, block_size, dist):
    cols_num = len(cols)
    save_path = f"./datasets/scan_condation_{name}.pkl"
    # save_path = f"./datasets/scan_condation_{cols_num}Cols.pkl"
    if not os.path.exists(save_path):
        # assert 0
        print(f'Dont find scan condations {save_path}, generating ...')
        if 'zvalue' in cols:
            # remove
            cols.remove('zvalue')
        Queries, scan_conds = QueryGeneration(100, df.loc[:, cols], cols, dist=dist, query_cols_nums=8)
        # pickle.dump(Queries, open(f"./datasets/Queries_{name}.pkl", "wb"))
        # pickle.dump(Queries, open(f"./datasets/Queries_{cols_num}Cols.pkl", "wb"))
        pickle.dump(scan_conds, open(save_path, "wb"))
    else:
        print(f"Load scan condations: {save_path}")
        scan_conds = pickle.load(open(save_path, "rb"))
    
    get_selectivity(df, scan_conds)
    
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
            cols=['VIN','City','Zip','County','Reg Valid Date','Reg Expiration Date', 'zvalue'], zvalue=True):
    csv_file = './datasets/{}'.format(filename)
    if zvalue:
        cols.append('zvalue')
        csv_file = csv_file.split('.csv')[0] + "-zorder.csv"
    # cols = ['VIN','City','Zip','County','Reg Valid Date','Reg Expiration Date', 'zvalue']
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64, 'Reg Expiration Date': np.datetime64}
    return common.CsvTable('dmv-clean', csv_file, cols, type_casts, nrows=1000000)

def LoadLineitem(filename, cols):
    csv_file = './datasets/{}'.format(filename)
    return common.CsvTable('Lineitem', csv_file, cols, sep=',')


def get_selectivity(table: pd.DataFrame, scan_cond):
    print("Start Sample Data From Query ...")
    SampledData = []
    SampledIdx = []
    Selectivity = []
    for one_batch_index in range(len(scan_cond)):
        one_batch_query_cols, one_batch_val_ranges = scan_cond[one_batch_index]
        preds = []
        for col, (min_, max_) in zip(one_batch_query_cols, one_batch_val_ranges):
            if col in ['Reg Valid Date', 'Reg Expiration Date', 'VIN', 'County', 'City'] \
                + ['Record Type','Registration Class','State','County','Body Type','Fuel Type','Color','Scofflaw Indicator','Suspension Indicator','Revocation Indicator']:
                preds.append(f"(table['{col}'] >= '{min_}')")
                preds.append(f"(table['{col}'] <= '{max_}')")
            else:
                preds.append(f"(table['{col}'] >= {min_})")
                preds.append(f"(table['{col}'] <= {max_})")
        query = " & ".join(preds)
        
        predicate_data = table.loc[eval(query)]
        selected_indices = predicate_data.index

        Selectivity.append(predicate_data.shape[0] / table.shape[0])
        SampledIdx.extend(selected_indices)
    print("Sampling Finished!")
    print("Sampling Data Length: ", len(SampledIdx))
    print("Average Selectivity: ", np.mean(Selectivity))

def A_heuristic(table, scan_cond, block_size):
    scan_queue = []
    SampledIdx = []
    for one_batch_index in range(len(scan_cond)):
        one_batch_query_cols, one_batch_val_ranges = scan_cond[one_batch_index]
        preds = []
        for col, (min_, max_) in zip(one_batch_query_cols, one_batch_val_ranges):
            if col in ['Reg Valid Date', 'Reg Expiration Date', 'VIN', 'County', 'City'] \
                + ['Record Type','Registration Class','State','County','Body Type','Fuel Type','Color','Scofflaw Indicator','Suspension Indicator','Revocation Indicator']:
                preds.append(f"(table['{col}'] >= '{min_}')")
                preds.append(f"(table['{col}'] <= '{max_}')")
            else:
                preds.append(f"(table['{col}'] >= {min_})")
                preds.append(f"(table['{col}'] <= {max_})")
        query = " & ".join(preds)
        
        predicate_data = table.loc[eval(query)]
        selected_indices = predicate_data.index
        scan_queue.extend(selected_indices)

    table["sort"] = 100
    table.loc[scan_queue, "sort"] = 1
    
    for i in range(0, table.shape[0], block_size):
        table.loc[i:i+block_size, "sort"]
    
    table = table.sort_values(by="sort")
    pass
        

def process_ecg(input_dir):
    with open(f'{input_dir}/RECORDS', 'r') as f:
        for idx, p in enumerate(f):
            p = p.strip()
        pattern = r"patient(\d+)"
        match = re.search(pattern, p)
        if match:
            patient_id = int(match.group(1))
        else:
            raise ValueError("Patient ID couldn't have been extracted.")
        path = os.path.join(f"{input_dir}", p)
        data = wfdb.rdsamp(path)
        print(data)
        assert 0

def process_ecg_tiny(file_path):
    data = pd.read_csv(file_path)
    print(data)
    assert 0


        
if __name__ == "__main__":
    # process_ecg_tiny("./datasets/ptbdb_normal.csv")
    # process_ecg("./datasets/physionet.org/files/ptbdb/1.0.0/")
    
    # df = pd.read_csv("./datasets/DMV_tuples_df.csv")
    # original_df = pd.read_csv("./datasets/dmv-tiny.csv")
    # cols = original_df.columns.tolist()
    # print(cols)
    # cols = ['VIN','City','Zip','County','Reg Valid Date','Reg Expiration Date']
    cols = ['Record Type','Registration Class','State']
    # cols = ['Color', 'State']
    # cols = ['l_orderkey', 'l_partkey']
    all_cols = ['Record Type','Registration Class','State','County','Body Type','Fuel Type','Reg Valid Date','Color','Scofflaw Indicator','Suspension Indicator','Revocation Indicator']
    # Zorder(df, cols, original_df=original_df)
    
    block_size = 10000
    # block_size = 200
    # dist = 'GAU'
    dist = "UNI"
    

    rowNum = int(1e6)
    colsNum = 100
    cols = ['0', '1', '2']
    all_cols = list(map(str, range(colsNum)))
    # name = f'RandomWalk-{int(rowNum/1000)}K-{colsNum}Col.csv'.split(".")[0]
    name = f'GAUData-{int(rowNum/1000)}K-{colsNum}Col_{dist}'
    # name = 'dmv-tiny'
    # name = 'dmv-clean'
    # name  = 'linitem_1000'
    # table = LoadLineitem(f"{name}.csv", cols=cols)
    # table = LoadDmv("dmv-clean.csv", cols=all_cols)
    # table = datasets.LoadRandomWalk(colsNum, rowNum, dist=dist, zvalue=True)
    table = datasets.LoadGAUDataset(colsNum, rowNum, dist, zvalue=True)
    # table = datasets.LoadRandomWalk(100, int(1e6))
    df = common.TableDataset(table).tuples_df
    Zorder(df, cols, original_df=table.data, name=table.name)
    # print(table.data)
    # all_cols = cols
    ZorderBlock(table.data, all_cols, block_size, dist=dist)


    