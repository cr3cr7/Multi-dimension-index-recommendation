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
from torch import nn

# Dataset Index后的数据 -> [Indexed Dataset]
# Trainning Query Generation -> [Training set], [统计一下每个查询扫描block数 Natural Order]
# 生成查询
def QueryGeneration(nums, train_data: pd.DataFrame, cols):
    Queries = []
    scan_conditions = []

    for i in range(nums):
        rng = np.random.RandomState()
        # Choose Number of Columns 
        # query_cols_nums = random.randint(1, len(cols))
        query_cols_nums = train_data.shape[1]
        
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
                       cols: list,
                       pad_size: int,
                       rand: bool = False
                       ):
        self.table = copy.deepcopy(table)
        self.block_size = block_size
        self.cols = cols
        self.block_nums = math.ceil(self.table.data.shape[0] / self.block_size)
        self.rand = rand
        self.pad_size = pad_size

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
        
        # Generate test Queries
        self.testQuery, self.testScanConds =  QueryGeneration(100, self.table.data, self.cols)
        self.sample_data = []
        for query in self.testQuery:
            self.sample_data.append(self.Sample(self.table, query))


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
        #tensor[tensor[:, 0] != id] = 0
        tensor = tensor[tensor[:, 0] == id] 
        indexed_df = df[df["id"] == id]
        self.collect_cols_min_max(indexed_df)
        return tensor[:, 1:]

    def collect_cols_min_max(self, df):
        for i in self.cols:
            self.cols_min[i] = df[i].min()
            self.cols_max[i] = df[i].max()
    
    def RandomBlockGeneration(self):
        arr = np.arange(self.table.data.shape[0]) // self.block_size
        random.shuffle(arr)
        # new tuple = (card, cols + 1)
        new_tuple = torch.cat([torch.as_tensor(arr.reshape(-1, 1)), self.orig_tuples], dim=1)
        new_tuple_df = self.table.data.copy(deep=True)
        new_tuple_df["id"] = arr
        return new_tuple, new_tuple_df
    
    def DataToBlock(self):
        # (batch_size, BinNum)
        p1s = []
        p2s = []
        count = torch.zeros((1, self.block_nums))
        for one_batch in self.orig_tuples:
            mask = self.update_mask(count)
            logits = self.MLP(one_batch) 
            # mask invalid bins
            logits = logits + (1 - mask) * -1e9
            # Sample soft categorical using reparametrization trick:
            p1 = F.gumbel_softmax(logits, tau=1, hard=False)
            # Sample hard categorical using "Straight-through" trick:
            p2 = F.gumbel_softmax(logits, tau=1, hard=True)
            p1s.append(p1)
            p2s.append(p2)
            
            count = count + p2
            
        return torch.cat(p2s, dim=0)

    def update_mask(self, count):
        mask = torch.where(count >= self.block_size, torch.zeros_like(count), torch.ones_like(count))
        return mask
    
    def LoadBlockWithID(self, id):
        one_hot = self.data_distribution.clone()
        one_hot[:, id] = 0
        selected_block = self.data_distribution - one_hot
        selected_block = selected_block[:, id]
        indices = selected_block.nonzero()
        return indices, torch.index_select(self.orig_tuples, 0, indices)
    
    def getQuery(self, rand: bool):
        if rand:
            Queries, scan_conds = QueryGeneration(1, self.table.data, self.cols)
            return Queries, scan_conds
        else:
            return self.testQuery, self.testScanConds

    def __len__(self):
        return len(self.sample_data)
    
    def __getitem__(self, idx):     
        # 2. Get the query
        #print(self.table.data)
        Queries, scan_conds = self.getQuery(rand=self.rand)
        # print("qcols: ", scan_conds[0][0], "qrange: ", scan_conds[0][1])
    
        
        # 3. Get the sampled Query data
        query_sample_data = self.sample_data[idx]
        
        # Define the desired size of the padded tensor
        desired_size = (self.pad_size, len(self.cols))

        # Get the current size of the tensor
        current_size = query_sample_data.size()
        
        if current_size[0] < desired_size[0]:
            # Compute the amount of padding needed for each dimension
            pad_amounts = [desired_size[i] - current_size[i] for i in range(len(desired_size))]

            # Pad the tensor with 0s
            query_sample_data = F.pad(query_sample_data, (0, pad_amounts[1], 0, pad_amounts[0]), mode='constant', value=0)
        elif current_size[0] > desired_size[0]:
            start_idx = (current_size[0] - desired_size[0]) // 2
            end_idx = start_idx + desired_size[0]
            
            query_sample_data = query_sample_data.narrow(0, start_idx, desired_size[0])
        
        # FIXME: Padding Columns Number
        # block(batch, card, cols)  query(batch, card, cols)  result(batch, 1)
        item = {'table': self.orig_tuples.to(torch.float), \
                'query': query_sample_data.to(torch.int), \
                'col': scan_conds[idx][0],\
                'range': scan_conds[idx][1]}
        # return self.orig_tuples.to(torch.float), \
        #         query_sample_data.to(torch.int), \
        #         scan_conds[idx][0], \
        #         scan_conds[idx][1]
        return item
        
         
    def _is_scan(self, qcols, qranges):
        for idx, i in enumerate(qcols):
            if self.cols_min.get(i, False):
                if i == 'Reg Valid Date' or i == 'Reg Expiration Date':
                    qranges[idx][0] = pd.to_datetime(qranges[idx][0])
                    qranges[idx][1] = pd.to_datetime(qranges[idx][1])
                #print("qcol: ", i)
                if self.cols_min[i] > qranges[idx][1] or self.cols_max[i] < qranges[idx][0]:
                    return False
        return True
    
    def Sample(self, train_data, query):
        # df, Indexed_df = train_data._predicate_data(query)
        
        predicate_data = train_data.data.loc[eval(query)]
        selected_indices = predicate_data.index
        # return predicate_data, self.tuples_df.loc[selected_indices]
        return torch.tensor(self.tuples_df.loc[selected_indices].values, dtype=torch.float32)
    
    # Block Selection Number (Query) -> [Block Scan Number]
    # 统计该查询需要扫描的数量
    def BlocksScanNumber(self, qcols, qranges, train_data, block_size):
        scan_blocks = 0
        for id in range(self.block_nums):
            Block = common.block(train_data.table, block_size, qcols, id)
            #print(Block._get_data())
            if Block._is_scan(qcols, qranges):
                scan_blocks += 1
        return scan_blocks