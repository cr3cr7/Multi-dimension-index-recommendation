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

from sklearn.cluster import estimate_bandwidth, AgglomerativeClustering, KMeans

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


class RankingModel_v3(nn.Module):
    def __init__(self, BlockSize, BlockNum, col_num, dmodel):
        super(RankingModel_v3, self).__init__()
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
        rows = table.shape[1]
        table = table.reshape(-1, rows, self.col_num * self.dmodel)
        
        # table = self.MLP(table)
        scores = self.clustering(table)
    
        min_vals = torch.min(scores, dim=1, keepdim=True)[0]
        max_vals = torch.max(scores, dim=1, keepdim=True)[0]
        scaled_scores = ((scores - min_vals) / (max_vals - min_vals)) * len(table)

        original_ranks = torchsort.soft_rank(scaled_scores, regularization="l2", regularization_strength=0.1)
        # original_ranks = torchsort.soft_rank(scaled_scores, regularization="kl", regularization_strength=1)
        # original_ranks = torchsort.soft_rank(scores, regularization="l2", regularization_strength=0.01)
        # print(original_ranks[0].reshape(-1))
        
        sorted_indices = torch.argsort(scores, dim=1)
        rank_indices = torch.zeros_like(sorted_indices)
        for batch in range(sorted_indices.shape[0]):
            rank_indices[batch, sorted_indices[batch]] = torch.arange(sorted_indices.shape[1])
        # rank_indices[:, sorted_indices] = torch.arange(sorted_indices.shape[1])
        rank_indices = rank_indices // self.capacity
        # Block 0 for padding
        rank_indices = rank_indices + 1
        return original_ranks.reshape(-1, rows, 1), rank_indices.reshape(-1, rows, 1), scores.reshape(-1, rows, 1)
        
        
    def clustering(self, table):
        # FIXME: 这里每个Batch table不一样
        array = table[0].detach().numpy()
    
        bandwidth = estimate_bandwidth(array, quantile=1)
        clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=bandwidth).fit(array)
        # clusters = KMeans(n_clusters=self.cluster_num).fit(array)
        labels = clusters.labels_
        # For every cluster, calculate the distance between each point and the cluster center
        cluster_centers = torch.stack([torch.mean(table[:, labels == i, :], dim=1) for i in range(clusters.n_clusters_)])
        # cluster_centers = torch.stack([torch.mean(table[:, labels == i, :], dim=1) for i in range(self.cluster_num)])
        # Gather according to labels
        center_data = torch.gather(cluster_centers.permute(1, 0, 2), 1, torch.from_numpy(labels).reshape(1, -1, 1).expand(table.shape[0], -1, array.shape[1]))
    
        distances = torch.mean(torch.square(table - center_data), dim=-1)
        min_vals = torch.min(distances, dim=1, keepdim=True)[0]
        max_vals = torch.max(distances, dim=1, keepdim=True)[0]
        distances = ((distances - min_vals) / (max_vals - min_vals)) * 1
        scores = distances + torch.from_numpy(labels).reshape(1, -1)
        return scores
        
        
        
class RankingModel_v2(nn.Module):
    def __init__(self, BlockSize, BlockNum, col_num, dmodel):
        super(RankingModel_v2, self).__init__()
        self.capacity = BlockSize
        self.BlockNum = BlockNum
        self.col_num = col_num
        self.dmodel = dmodel
        self.MLP = nn.Sequential(
            nn.LayerNorm(col_num * dmodel),
            nn.Linear(col_num * dmodel, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 1)
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
        
        min_vals = torch.min(scores, dim=1, keepdim=True)[0]
        max_vals = torch.max(scores, dim=1, keepdim=True)[0]
        scaled_scores = ((scores - min_vals) / (max_vals - min_vals)) * 100
        # print(scaled_scores[0].reshape(-1))
        original_ranks = torchsort.soft_rank(scaled_scores, regularization="l2", regularization_strength=0.01)
        # print(original_ranks[0].reshape(-1))

        # Generate Block ID
        # other = original_ranks.detach() % self.capacity
        # rank_indices = (original_ranks - other) / self.capacity
        
        # Get Rank Index of scaled_scores
        sorted_indices = torch.argsort(scores, dim=1)
        rank_indices = torch.zeros_like(sorted_indices)
        rank_indices[:, sorted_indices] = torch.arange(sorted_indices.shape[1])
        rank_indices = rank_indices // self.capacity
        # Block 0 for padding
        rank_indices = rank_indices + 1
        # print(rank_indices[0].reshape(-1))
        return original_ranks.reshape(-1, rows, 1), rank_indices.reshape(-1, rows, 1)
        # return rank_indices.reshape(-1, rows, 1)
        
        # block_id = []
        # idx = 0
        # sorted_indices = torch.argsort(ranks)
        # ranks_cp= ranks.clone().detach()
        # ranks_diff = ranks - ranks_cp
        # for num in range(self.BlockNum):
        #     for batch in range(ranks_cp.shape[0]):
        #         ranks_cp[batch, sorted_indices[batch, idx:idx+self.capacity]] = num + 1
        #     idx = idx + self.capacity
        # ranks_add = ranks_diff + ranks_cp
        # return ranks_add.reshape(-1, rows, 1)
    
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

