from typing import Any
import torch
from torch import nn
import pytorch_lightning as pl
from common import TableDataset
from torch.utils.data import DataLoader
#from data import datasets
import datasets
from util.Block import RandomBlockGeneration, BlockDataset
from model.model_interface import ReportModel
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import math

class RankingModel(nn.Module):
    def __init__(self, BlockSize, BlockNum, col_num):
        super(RankingModel, self).__init__()
        self.capacity = BlockSize
        self.BlockNum = BlockNum
        self.MLP = nn.Sequential(
            nn.Linear(col_num, 32),
            nn.ReLU(),
            nn.Linear(32, self.BlockNum),
            nn.ReLU()
        )
        
        
    
    def forward(self, table):
        # (batch_size, RowNum, BlockNum)
        p1ss = []
        p2ss = []
        for one_batch in table:
            count = torch.zeros((1, self.BlockNum), requires_grad=True)
            count = count.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            p1s = []
            p2s = []
            for row in one_batch:
                mask = self.update_mask(count)
                logits = self.MLP(row.to(torch.float)) 
                # mask invalid bins
                logits = logits + (1 - mask) * -1e9
                # Sample soft categorical using reparametrization trick:
                p1 = F.gumbel_softmax(logits, tau=1, hard=False)
                # Sample hard categorical using "Straight-through" trick:
                p2 = F.gumbel_softmax(logits, tau=1, hard=True)
                p1s.append(p1)
                p2s.append(p2)
                
                count = count + p2
            p1ss.append(torch.cat(p1s, dim=0))
            p2ss.append(torch.cat(p2s, dim=0))
        return torch.stack(p2ss, dim=0)
        #return torch.cat(p1s, dim=0), torch.cat(p2s, dim=0)

    def update_mask(self, count):
        mask = torch.where(count >= self.capacity, torch.zeros_like(count), torch.ones_like(count))
        mask = mask.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return mask
    
class FilterModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, one_hot, id):
        indices = []
        for one_batch in one_hot:
            table = one_batch
            table_cp = table.clone()
            table_cp[:,id] = 0
            selected_block = table - table_cp
            selected_block = selected_block[:,id].nonzero().reshape(-1)
            indices.append(selected_block)
        return torch.stack(indices)

class GenerationTrainer(pl.LightningModule):
    def __init__(self, num_workers=8, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_workers = num_workers
        self.lr = kargs['lr']

    
    def forward(self):
        indexed_data = self.ranking_model(self.table.tuples)
        for id in self.block_nums:
            selected_block = self.filter_model(indexed_data, id)

        return selected_block

    def setup(self, stage=None):
        dataset = self.hparams.dataset.lower()
        if dataset == 'tpch':
            table = datasets.LoadTPCH()
        elif dataset == 'dmv-tiny':
            table = datasets.LoadDmv('dmv-tiny.csv')
        elif dataset == 'lineitem':
            table = datasets.LoadDmv('lineitem.csv')
        else:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{dataset}')
        print(table.data.info())
        # self.data_module = TableDataset(table)
        # Assign train/val datasets for use in dataloaders
        self.cols = table.ColumnNames()
        self.block_nums = math.ceil(table.data.shape[0] / self.hparams.block_size)


        self.load_model()

    def load_model(self):
        self.ranking_model = RankingModel(self.hparams.block_size, self.block_nums, len(self.cols))

        self.filter_model = FilterModel()
        
if __name__ == "__main__":
    table = datasets.LoadDmv('dmv-tiny.csv')
    train_data = TableDataset(table)
    print(train_data.tuples.size())
    cols = table.ColumnNames()
    

    block_nums = math.ceil(train_data.table.data.shape[0] / 20)

    ranking_model = RankingModel(20, block_nums, len(cols))

    indexed_block = ranking_model.forward(train_data.tuples)
    print(indexed_block)

    filter_model = FilterModel()
    id = 1
    select_block = filter_model.forward(indexed_block, id)
    print(select_block.size())

