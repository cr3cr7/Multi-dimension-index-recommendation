import pandas as pd
import numpy as np
from datasets import LoadRandomWalk
import interleavebits
import time

from model.generation import RankingModel_v2, RankingModel_v4
from util.NewBlock import BlockDataset_Eval
from torch.utils import data
import torch


class EvalDataset(data.Dataset):
    """Wrap a Block and yield one Row as Pytorch Dataset element."""
    
    def __init__(self, orig_tuples):
        self.orig_tuples = orig_tuples
        
    def __getitem__(self, idx):
        return self.orig_tuples[idx], idx
    
    def __len__(self):
        return len(self.orig_tuples) 

def Zorder(df, cols):        
    vals = [[] for i in range(df.shape[0])]
    i = 0
    for index, row in df.iterrows():
        for idx, c in enumerate(cols):
            vals[i].append(row[c])
        zvalue = interleavebits.interleavem(*vals[i])
        df.loc[index, "zvalue"] = zvalue
        i += 1


def create_data(colsNum, rowNum):
    n_series = colsNum
    n_steps = rowNum
    cols = list(map(str, range(colsNum)))
    steps = np.random.normal(0, 1, size=(n_series, n_steps))
    position = np.cumsum(steps, axis=1).T
    df = pd.DataFrame(position, columns=cols)
    return df
 
def Discretize(col, data=None):
    # distinct values
    dvs = col.unique()
    if data is None:
        data = col
    bin_ids = pd.Categorical(data, categories=dvs, ordered=True).codes
    assert len(bin_ids) == len(data), (len(bin_ids), len(data))
    #add 1 to everybody for padding, 0 is reserved for padding
    bin_ids = bin_ids + 1
    bin_ids = bin_ids.astype(np.int32, copy=False)
    return bin_ids

def test_curve_construction_time(type, cols, rows):
    df = create_data(cols, rows)
    discrete_data = np.stack([Discretize(df[c]) for c in df.columns], axis=1)
    discrete_df = pd.DataFrame(discrete_data, columns=df.columns)
    print('*'*20)
    print("Start to test {} curve construction time ...".format(type))
    print("Data shape: ", discrete_data.shape)
    if type == 'zorder':
        t1 = time.time()
        Zorder(discrete_df, discrete_df.columns)
        print("Z Value Generation time cost: ", time.time() - t1)
        print('*'*20)
    elif type == 'ML':
        torch_data = torch.from_numpy(discrete_data.astype(np.float32, copy=False))
        dataset = EvalDataset(torch_data)
        data_loader = data.DataLoader(dataset, batch_size=256, num_workers=0, shuffle=False)
        # ranking_model = RankingModel_v2(0, 
        #                                 0, 
        #                                 len(df.columns), 
        #                                 0,
        #                                 input_bins=[],
        #                                 sparse=False,
        #                                 if_pretraining=False)
        ranking_model = RankingModel_v4(len(df.columns), 2, 
                                             input_bins=[len(dataset) for c in df.columns])
        ranking_model.eval()
        results = []
        t1 = time.time()
        for i, (table, idx) in enumerate(data_loader):
            # print(table.shape)
            with torch.no_grad():
                results.append(ranking_model(table,
                                            0, i, None).numpy())
        results = np.concatenate(results)
        print("ML Model Generation time cost: ", time.time() - t1)
        print('*'*20)
    else:
        raise ValueError("type should be zorder or ML")
    
    
 
if __name__ == "__main__":
    # test settings
    # print("Col Scale up test ...")
    # cols = [2, 4, 8, 16] + [2, 4, 8, 16]
    # rows = [int(1e6)] * 4 + [int(1e6)] * 4
    # types = ['zorder'] * 4 + ['ML'] * 4
    
    # for cur_type, col, row in zip(types, cols, rows):
    #     test_curve_construction_time(cur_type, col, row)
        

    # print("Row Scale up test ...")
    # cols = [3, 3, 3, 3] + [3, 3, 3, 3]
    # rows = [int(1e4), int(1e5), int(1e6), int(1e7)] * 2
    # types = ['zorder'] * 4 + ['ML'] * 4
    
    # cols = [3, 3, 3, 3]
    # rows = [int(1e4), 
    # int(1e5), int(1e6), int(1e7)] 
    # types = ['ML'] * 4
    
    # cols = [4, 3, 5]
    # rows = [int(1e4)] * 3
    # types = ['ML'] * 3
    
    
    # for cur_type, col, row in zip(types, cols, rows):
    #     test_curve_construction_time(cur_type, col, row)
    
    print("Col Scale up test ...")
    cols = [2, 4, 8, 16]
    rows = [int(1e6)] * 4
    types = ['ML'] * 4
    
    for cur_type, col, row in zip(types, cols, rows):
        test_curve_construction_time(cur_type, col, row)
        
    print("Row Scale up test ...")
    cols = [3, 3, 3, 3]
    rows = [int(1e4), int(1e5), int(1e6), int(1e7)]
    types = ['ML'] * 4

    for cur_type, col, row in zip(types, cols, rows):
        test_curve_construction_time(cur_type, col, row)
    