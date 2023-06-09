import math
import copy
import time

import datasets
import numpy as np
import generateQuery
import common
import random
from torch.utils import data
import pandas as pd
import torch
import torch.nn.functional as F

# Dataset Index后的数据 -> [Indexed Dataset]
# Trainning Query Generation -> [Training set], [统计一下每个查询扫描block数 Natural Order]
# 生成查询
def QueryGeneration(nums, train_data: pd.DataFrame, cols):
    Queries = []
    scan_conditions = []

    for i in range(nums):
        rng = np.random.RandomState()
        query_cols_nums = random.randint(1, len(cols))
        qcols, qops, qvals, qranges = generateQuery.SampleTupleThenRandom(cols, query_cols_nums, rng, train_data)
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


class BlockDataset(data.Dataset):
    """Wrap a Block and yield one block as Pytorch Dataset element."""
    
    def __init__(self, table: common.CsvTable, 
                       block_size: int, 
                       qcols: list):
        self.table = copy.deepcopy(table)
        self.block_size = block_size
        self.qcols = qcols
        
        
        s = time.time()
        # [cardianlity, num cols].
        self.orig_tuples_np = np.stack(
            [self.Discretize(c) for c in self.table.Columns()], axis=1)
        self.orig_tuples = torch.as_tensor(
            self.orig_tuples_np.astype(np.float32, copy=False))
        self.tuples_df = pd.DataFrame(self.orig_tuples_np)
        self.tuples_df.columns = self.table.ColumnNames()
        print('done, took {:.1f}s'.format(time.time() - s))
        
        self.cols_min = {}
        self.cols_max = {}
        
    def Discretize(self, col):
        """Discretize values into its Column's bins.

        Args:
          col: the Column.
        Returns:
          col_data: discretized version; an np.ndarray of type np.int32.
        """
        return common.Discretize(col)
    
        
    def zero_except_id(self, tensor, df, id):
        # tensor: PyTorch tensor
        # id: value to match in first column
        tensor[tensor[:, 0] != id] = 0
        df = df[df["id"] == id]
        self.collect_cols_min_max(df)
        return tensor[:, 1:]

    def collect_cols_min_max(self, tuples_df):
        for i in self.qcols:
            self.cols_min[i] = tuples_df[i].min()
            self.cols_max[i] = tuples_df[i].max()
    
    
    def RandomBlockGeneration(self):
        arr = np.arange(self.table.data.shape[0]) // self.block_size
        random.shuffle(arr)
        # new tuple = (card, cols + 1)
        new_tuple = torch.cat([torch.as_tensor(arr.reshape(-1, 1)), self.orig_tuples], dim=1)
        new_tuple_df = self.table.data.copy(deep=True)
        new_tuple_df["id"] = arr
        return new_tuple, new_tuple_df
    
    def __len__(self):
        return 64
    
    def __getitem__(self, idx):
        # idx: index of the block
            
        # 1. Get the block
        new_block, new_tuple_df = self.RandomBlockGeneration()
        new_idx = random.randint(0, (new_block.shape[0] // self.block_size) - 1) 
        new_block = self.zero_except_id(new_block, new_tuple_df, new_idx)
        
        # 2. Get the query
        Queries, scan_conds = QueryGeneration(1, self.table.data, self.qcols)
    
        result = self._is_scan(scan_conds[0][0], scan_conds[0][1])
       
        # 3. Get the sampled Query data
        query_sample_data = self.Sample(self.table, Queries[0])
        
        # Define the desired size of the padded tensor
        desired_size = (99, len(self.qcols))

        # Get the current size of the tensor
        current_size = query_sample_data.size()
    
        # Compute the amount of padding needed for each dimension
        pad_amounts = [desired_size[i] - current_size[i] for i in range(len(desired_size))]

        # Pad the tensor with 0s
        query_sample_data = F.pad(query_sample_data, (0, pad_amounts[1], 0, pad_amounts[0]), mode='constant', value=0)
        
        # block(batch, card, cols)  query(batch, card, cols)  result(batch, 1)
        return new_block.to(torch.int), query_sample_data.to(torch.int), torch.tensor([result], dtype=torch.float32)
        
         
    def _is_scan(self, qcols, qranges):
        for idx, i in enumerate(qcols):
            if self.cols_min[i] > qranges[idx][1] or self.cols_max[i] < qranges[idx][0]:
                return False
        return True
    
    def Sample(self, train_data, query):
        # df, Indexed_df = train_data._predicate_data(query)
        
        predicate_data = train_data.data.loc[eval(query)]
        selected_indices = predicate_data.index
        # return predicate_data, self.tuples_df.loc[selected_indices]
        return torch.tensor(self.tuples_df.loc[selected_indices].values, dtype=torch.float32)