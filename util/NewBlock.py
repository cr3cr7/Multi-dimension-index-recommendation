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
import os
import pickle

def get_domains(dataset):
    min_values = np.amin(dataset, axis=0)
    max_values = np.amax(dataset, axis=0)
    domains = np.array(list(zip(min_values, max_values)))
    return domains

# Dataset Index后的数据 -> [Indexed Dataset]
# Trainning Query Generation -> [Training set], [统计一下每个查询扫描block数 Natural Order]
# 生成查询
def QueryGeneration(nums, train_data: pd.DataFrame, cols, table: common.CsvTable=None, query_cols_nums=10, dist="UNI"):
    Queries = []
    scan_conditions = []

    if dist == "GAU":
        domains = get_domains(train_data)
        maximum_range_percent = 0.1
        maximum_range = [(domains[i,1] - domains[i,0]) * maximum_range_percent for i in range(len(domains))]
        return None, generateQuery.generate_distribution_query(nums, 
                                                               query_cols_nums, 
                                                               domains, 
                                                               maximum_range, 
                                                               cols, 
                                                               cluster_center_amount=10, 
                                                               sigma_percent=0.2)
    
    for i in range(nums):
        rng = np.random.RandomState()
        # Choose Number of Columns 
        # query_cols_nums = random.randint(1, len(cols))
        # query_cols_nums = train_data.shape[1]
        # query_cols_nums = 10
        if dist == "UNI":
            qcols, qops, qvals, qranges = generateQuery.SampleTupleThenRandom(cols, query_cols_nums, rng, train_data)
        elif dist == "GAU":
            qcols, qops, qvals, qranges = generateQuery.SampleGaussianQueries(cols, query_cols_nums, rng, train_data)
        else:
            raise NotImplementedError
        # conditions = []
        # for i in range(len(qcols)):
        #     if str(type(qvals[i])) == "<class 'str'>":
        #         qvals[i] = "'" + qvals[i] + "'"
        #     if str(type(qvals[i])) == "<class 'numpy.int64'>" or str(type(qvals[i])) == "<class 'numpy.float64'>":
        #         qvals[i] = str(qvals[i])
        #     if str(type(qvals[i])) == "<class 'pandas._libs.tslibs.timestamps.Timestamp'>":
        #         qvals[i] = "'" + str(qvals[i]) + "'"
        # for qcol, qop, qval in zip(qcols, qops, qvals):
        #     conditions.append(f"(self.table.data['{qcol}'] {qop} {qval})")
        #     # conditions.append(f"(self.table['{qcol}'] {qop} {qval})")
        # #print(conditions)
        # predicate = " & ".join(conditions)
        # Queries.append(predicate)
        
        scan_conditions.append([qcols, qranges])
    return None, scan_conditions

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
        # self.tuples_df.to_csv(f"datasets/{table.name}_tuples_df.csv", index=False)
        print('done, took {:.1f}s'.format(time.time() - s))
        
        self.cols_min = {}
        self.cols_max = {}
        
        # Generate test Queries
        # save_path = "./datasets/scan_condation_1000.pkl"
        cols_num = len(cols)
        # save_path = f"./datasets/scan_condation_{cols_num}Cols.pkl"
        save_path = f'/datasets/scan_condation_{table.name}_{cols_num}Cols.pkl'
        if not os.path.exists(save_path):
            self.testQuery, self.testScanConds = QueryGeneration(100, self.table.data.loc[:, cols], self.cols)
            # pickle.dump(scan_conds, open(save_path, "wb"))
        else:
            print("*"*50 + "Load scan_conds from file! " + "*"*50)
            self.testScanConds = pickle.load(open(save_path, "rb"))
            self.testQuery = pickle.load(open(f"./datasets/Queries_{cols_num}Cols.pkl", "rb"))
            
        # self.testQuery, self.testScanConds =  QueryGeneration(100, self.table.data, self.cols)
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
        return torch.tensor(self.tuples_df.loc[selected_indices].values, dtype=torch.float32), 
    
    def SampleBasedOnQuery(self, Queries, scan_conds):
        # reserve pad_size dataframe memory
        new_df = pd.DataFrame(index=range(self.pad_size), columns=self.cols)
        query_cols = scan_conds[0][0]
        query_ranges = scan_conds[0][1]
        
        new_query_ranges = []
        for col in self.cols:
            if col not in query_cols:
                new_query_ranges.append([self.table.data[col].min(),
                                         self.table.data[col].max()])
            else:
                idx = query_cols.index(col)
                new_query_ranges.append(query_ranges[idx])
        query_ranges = np.asarray(new_query_ranges)
        # print(query_ranges)
        for i in range(self.pad_size):
            # Generate a random bit string to determine which values to use
            bits = np.random.randint(2, size=len(query_cols))
            values = np.where(bits, query_ranges[:, 0], query_ranges[:, 1])
            new_df.loc[i] = values
   
        new_df = new_df.drop_duplicates()
        
        # Discretize
        new_np = np.stack(
            [common.Discretize(c, new_df[c.name]) for c in self.table.Columns()], axis=1)
        
        new_discretized_df = pd.DataFrame(new_np, columns=self.cols)
        
        # padding from sample
        if new_discretized_df.shape[0] < self.pad_size:
            target = self.pad_size - new_discretized_df.shape[0] 
            query_sample_data = self.Sample(self.table, Queries[0])
            new_discretized_df.append(query_sample_data.sample(n= target if target < query_sample_data.shape[0] else query_sample_data.shape[0]))
        return torch.tensor(new_discretized_df.values, dtype=torch.float32) 
    
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
    
    
class BlockDataset_V2(data.Dataset):
    """Wrap a Block and yield one block as Pytorch Dataset element."""
    
    def __init__(self, table: common.CsvTable, 
                       block_size: int, 
                       cols: list,
                       pad_size: int,
                       rand: bool = False,
                       dist: str = "UNI"
                       ):
        self.table = copy.deepcopy(table)
        self.block_size = block_size
        self.cols = cols
        self.rand = rand
        self.pad_size = pad_size
        assert self.pad_size > 0 and self.pad_size <= len(self.table.data)
        
        s = time.time()
        # [cardianlity, num cols].
        self.orig_tuples_np = np.stack(
            [self.Discretize(c) for c in self.table.Columns()], axis=1)
        # TODO: Original Data or Discrete data?
        # self.orig_tuples_np = table.data.to_numpy()
        self.orig_tuples = torch.as_tensor(
            self.orig_tuples_np.astype(np.float32, copy=False))
        self.tuples_df = pd.DataFrame(self.orig_tuples_np)
        self.tuples_df.columns = self.table.ColumnNames()
        # self.tuples_df.to_csv(f"datasets/{table.name}_tuples_df.csv", index=False)
        print('done, took {:.1f}s'.format(time.time() - s))
        
        self.cols_min = {}
        self.cols_max = {}
        
        # Generate test Queries
        # save_path = "./datasets/scan_condation_1000.pkl"
        cols_num = len(cols)
        if "dmv" in table.name or "Lineitem" in table.name:
            save_path = f"./datasets/scan_condation_{cols_num}Cols.pkl"
            if 'dmv-clean' in table.name:
                save_path = f'./datasets/scan_condation_{table.name}.pkl'
        elif 'UniData' in table.name:
            save_path = f"./datasets/scan_condation_UniData-1000K-2Col_Skew.pkl"
        else:
            save_path = f'./datasets/scan_condation_{table.name}.pkl'
        if not os.path.exists(save_path):
            raise ValueError(f"Scan condation file {save_path} not exists!")
            self.testQuery, self.testScanConds = QueryGeneration(100, self.table.data.loc[:, cols], self.cols)
            # pickle.dump(scan_conds, open(save_path, "wb"))
        else:
            
            if table.name == "RandomWalk-10K-100Col":
                save_path = './datasets/scan_conds.json'
                print("*"*50 + f"Load scan_conds:{save_path} from file! " + "*"*50)
                import json
                self.testScanConds = json.load(open('./datasets/scan_conds.json', 'r'))
                self.testQuery = None
                
                # padding testScanConds to same length
                max_len = 10
                col_padding = "111"
                range_padding = [0.00, 0.00]
                for scan_cond in self.testScanConds:
                    if len(scan_cond[0]) < max_len:
                        # pad scan_cond until max_len
                        # print(scan_cond[0])
                        # print(scan_cond[1])
                        # assert 0
                        scan_cond[0].extend([col_padding] * (max_len - len(scan_cond[0])))
                        scan_cond[1].extend([range_padding] * (max_len - len(scan_cond[1])))        
                    assert len(scan_cond[0]) == len(scan_cond[1]) == max_len
            else:
                print("*"*50 + f"Load scan_conds:{save_path} from file! " + "*"*50)
                self.testScanConds = pickle.load(open(save_path, "rb"))[:100]
                # TODO: Deprecated
                self.testQuery = pickle.load(open(f"./datasets/Queries_3Cols.pkl", "rb"))[:100]
        
        # Start Sample Data From Query
        print("Start Sample Data From Query ...")
        self.SampledData = []
        self.SampledIdx = []
        self.Selectivity = []
        if 'UniData' in table.name:
            self.testScanConds = self.testScanConds[:100]
        for one_batch_index in range(len(self.testScanConds)):
            one_batch_query_cols, one_batch_val_ranges = self.testScanConds[one_batch_index]
            preds = []
            for col, (min_, max_) in zip(one_batch_query_cols, one_batch_val_ranges):
                if col in ['Reg Valid Date', 'Reg Expiration Date', 'VIN', 'County', 'City'] \
                    + ['Record Type','Registration Class','State','County','Body Type','Fuel Type','Color','Scofflaw Indicator','Suspension Indicator','Revocation Indicator']:
                    preds.append(f"(self.table.data['{col}'] >= '{min_}')")
                    preds.append(f"(self.table.data['{col}'] <= '{max_}')")
                else:
                    preds.append(f"(self.table.data['{col}'] >= {min_})")
                    preds.append(f"(self.table.data['{col}'] <= {max_})")
            query = " & ".join(preds)
            data, idx = self.Sample(self.table, query)
            # print(idx)
            self.Selectivity.append(data.shape[0] / self.table.data.shape[0])
            self.SampledIdx.extend(idx)
        print("Sampling Finished!")
        print("Sampling Data Length: ", len(self.SampledIdx))
        print("Average Selectivity: ", np.mean(self.Selectivity))
        
        
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
        return len(self.testScanConds)
    
    def __getitem__(self, idx):     
        # 2. Get the query
        #print(self.table.data)
        _, scan_conds = self.getQuery(rand=self.rand)
        # print("qcols: ", scan_conds[0][0], "qrange: ", scan_conds[0][1])
        
        # Define the desired size of the padded tensor
        desired_size = (self.pad_size, len(self.cols))

        # Sample the Fixed-size Table
        # if (self.orig_tuples.shape[0] - self.pad_size) != 0:
        #     ix = torch.randint(0, self.orig_tuples.shape[0] - self.pad_size, (1,))
        #     self.train_tuples = self.orig_tuples[ix:ix+self.pad_size]
        # else:
        #     ix = 0
        #     self.train_tuples = self.orig_tuples
         
        # Sample data that satisfy the query
        sample_size = int(self.pad_size / 3) if int(self.pad_size / 3) < len(self.SampledIdx) else len(self.SampledIdx)
        ix_1 = torch.Tensor(random.sample(self.SampledIdx, sample_size)).long()
        
        # No Sample
        # sample_size = 0
        # ix_1 = torch.Tensor([]).long()
        
        # Sample random data
        ix_2 = torch.randint(0, self.orig_tuples.shape[0], (self.pad_size - sample_size,))
        ix = torch.cat([ix_1, ix_2], dim=0)
        train_tuples = self.orig_tuples[ix]
        
        
        
        # FIXME: Padding Columns Number
        # block(batch, card, cols)  query(batch, card, cols)  result(batch, 1)
        item = {'table': train_tuples, \
                'table_idx': ix, \
                'col': scan_conds[idx][0],\
                'range': scan_conds[idx][1]}
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
        return self.orig_tuples[selected_indices], selected_indices
        # return torch.tensor(self.tuples_df.loc[selected_indices].values, dtype=torch.float32)
    
    def SampleBasedOnQuery(self, Queries, scan_conds):
        # reserve pad_size dataframe memory
        new_df = pd.DataFrame(index=range(self.pad_size), columns=self.cols)
        query_cols = scan_conds[0][0]
        query_ranges = scan_conds[0][1]
        
        new_query_ranges = []
        for col in self.cols:
            if col not in query_cols:
                new_query_ranges.append([self.table.data[col].min(),
                                         self.table.data[col].max()])
            else:
                idx = query_cols.index(col)
                new_query_ranges.append(query_ranges[idx])
        query_ranges = np.asarray(new_query_ranges)
        # print(query_ranges)
        for i in range(self.pad_size):
            # Generate a random bit string to determine which values to use
            bits = np.random.randint(2, size=len(query_cols))
            values = np.where(bits, query_ranges[:, 0], query_ranges[:, 1])
            new_df.loc[i] = values
   
        new_df = new_df.drop_duplicates()
        
        # Discretize
        new_np = np.stack(
            [common.Discretize(c, new_df[c.name]) for c in self.table.Columns()], axis=1)
        
        new_discretized_df = pd.DataFrame(new_np, columns=self.cols)
        
        # padding from sample
        if new_discretized_df.shape[0] < self.pad_size:
            target = self.pad_size - new_discretized_df.shape[0] 
            query_sample_data = self.Sample(self.table, Queries[0])
            new_discretized_df.append(query_sample_data.sample(n= target if target < query_sample_data.shape[0] else query_sample_data.shape[0]))
        return torch.tensor(new_discretized_df.values, dtype=torch.float32) 
    
class BlockDataset_Eval(data.Dataset):
    """Wrap a Block and yield one Row as Pytorch Dataset element."""
    
    def __init__(self, table: common.CsvTable,
                       block_size: int,
                       cols: list,
                       pad_size: int,
                       orig_tuples_np: np.ndarray,
                       orig_tuples: torch.tensor,
                       tuples_df: pd.DataFrame,
                       testQuery: list,
                       testScanConds: list,
                       rand: bool = False,
                       **kwargs
                       ):
        self.table = table
        self.orig_tuples_np = orig_tuples_np
        self.orig_tuples = orig_tuples
        self.tuples_df = tuples_df
        
        self.testQuery = testQuery
        self.testScanConds = testScanConds
        
    def __getitem__(self, idx):
        return self.orig_tuples[idx], idx
    
    def __len__(self):
        return len(self.orig_tuples)    