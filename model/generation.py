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
import torchsort

class RankingModel(nn.Module):
    def __init__(self, BlockSize, BlockNum, col_num, dmodel):
        super(RankingModel, self).__init__()
        self.capacity = BlockSize
        self.BlockNum = BlockNum
        self.col_num = col_num
        self.dmodel = dmodel
        self.MLP = nn.Sequential(
            nn.Linear(col_num * dmodel, 32),
            nn.ReLU(),
            nn.Linear(32, self.BlockNum),
            nn.ReLU()
        )
        
        
    
    def forward(self, table):
        # (batch_size, RowNum, colNum, dmodel)
        RowNum = table.shape[1]
        p1ss = []
        p2ss = []
        for one_batch in table:
            count = torch.zeros((1, self.BlockNum), requires_grad=True)
            # count = count.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            p1s = []
            p2s = []
            for row in one_batch:
                mask = self.update_mask(count)
                logits = self.MLP(row.reshape(-1))
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
        # mask = mask.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return mask
    
class RankingModel_v2(nn.Module):
    def __init__(self, BlockSize, BlockNum, col_num, dmodel):
        super(RankingModel_v2, self).__init__()
        self.capacity = BlockSize
        self.BlockNum = BlockNum
        self.col_num = col_num
        self.dmodel = dmodel
        self.MLP = nn.Sequential(
            nn.Linear(col_num * dmodel, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Sigmoid()
            # nn.ReLU()
        )
        # self.ln1 = nn.LayerNorm(100)
        self.apply(self._init_weights)
    
    
    def forward(self, table):
        rows = table.shape[1]
        table = table.reshape(-1, rows, self.col_num * self.dmodel)
        scores = self.MLP(table).reshape(-1, rows)
        # scores = nn.functional.softmax(scores, dim=1)
        
        # TODO: SoftRank很容易受到Score相对大小的影响，需要进行归一化
        # TODO: MinMax相同做特殊处理, 会有大量block id相同，应该怎么处理？
        # TODO: 排序不均匀，导致Block Size不同
        min_vals = torch.min(scores, dim=1, keepdim=True)[0]
        max_vals = torch.max(scores, dim=1, keepdim=True)[0]
        scaled_scores = ((scores - min_vals) / (max_vals - min_vals + 0.1)) * 100
        # print(scaled_scores[0])
        ranks = torchsort.soft_rank(scaled_scores, regularization="kl")
        # print(ranks[0])
        # ranks_1 = []
        # for score in scores:
        #     scaled_tensor = score.reshape(-1, rows)
        #     scaled_tensor = (scaled_tensor - scaled_tensor.min()) / (scaled_tensor.max() - scaled_tensor.min()) * 100
        #     rank = torchsort.soft_rank(scaled_tensor, regularization="kl")
        #     print(rank)
        #     print(ranks[0])
        #     assert rank == ranks[0]

        # Generate Block ID
        other = ranks.detach() % self.capacity
        ranks = (ranks - other) / self.capacity
        # print("RANKS")
        # print(ranks[0])
        # Block 0 for padding
        ranks = ranks + 1
        return ranks.reshape(-1, rows, 1)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
class FilterModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, one_hot, id):
        block_id = []
        indices = []
        for one_batch in one_hot:
            table_cp = one_batch.clone().detach()
            table_cp[:,id] = 0
            table_diff = one_batch - table_cp
            
            selected_block = torch.sum(table_diff, dim=1, keepdim=True)
            
            
            rows = one_batch[:, id].nonzero().reshape(-1)
            
            block_id.append(selected_block)
            indices.append(rows)
        return torch.stack(block_id).unsqueeze(-1), torch.stack(indices)
        """ table = one_hot.clone().detach()
        table[:,:,id] = 0
        table_diff = one_hot - table
        selected_block = torch.sum(table_diff, dim=2, keepdim=True)
        return selected_block """

class FilterModel_v2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, block_id, target_id):
        """
        block_id: [batch_size, rows, 1]
        target_id: int
        """
        # Zero is for padding
        target_id += 1
        block_id = block_id.squeeze(-1)  # eliminate last dimension
        block_id_cp = block_id.clone().detach()
        block_id_cp[block_id_cp == target_id] = 0
        block_diff = (block_id - block_id_cp) / target_id
        indices = []
        # TODO: 这里nonzero是否准确，是否优化？
        for i in range(block_diff.shape[0]):
            rows = block_diff[i].nonzero().reshape(-1)
            indices.append(rows)
            # rows = rows.reshape(100, -1)
        rows = torch.stack(indices)
        # TODO:Padding rows to (100, 20), Block Size < 20做特殊处理
        if rows.shape[1] < 20:
            rows = torch.cat([rows, torch.zeros(100, 20 - rows.shape[1], dtype=torch.long)], dim=1)
        return block_diff.unsqueeze(-1).unsqueeze(-1), rows

        

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

