# Unit test for ScanCoster.py, have to add the current path to sys.path
import sys
sys.path.append('..')

import torch
from torch import nn
import pytorch_lightning as pl
from common import TableDataset
from torch.utils.data import DataLoader
#from data import datasets
import datasets
from util.NewBlock import RandomBlockGeneration, BlockDataset
from model.model_interface import ReportModel
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from model.generation import RankingModel, FilterModel, RankingModel_v2, FilterModel_v2
import math
from model.summarization import FeedFoward, SummarizationModel, Classifier, Embedding, SummaryTrainer, SummarizationModel2, Classifier_v1
import pandas as pd
import numpy as np
import generateQuery

class RankingCostTrainer(pl.LightningModule):
    def __init__(self, 
                 num_workers=8,
                 **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_workers = num_workers
        self.lr = kargs['lr']
        self.rand = kargs['rand']
        self.configure_loss()
        
        
        # self.SumModel = SummaryTrainer(**kargs)
        # self.SumModel.setup(stage='fit')
        # for param_tensor in self.SumModel.state_dict():
        #     print(param_tensor, "\t", self.SumModel.state_dict()[param_tensor].size())
        
        # print(torch.load(self.PATH)['state_dict'].keys())
            
        # self.SumModel.load_from_checkpoint(self.PATH)

    def forward(self, table, query_cols, val_range):
        # 大小应为 batch_size ，此处手动设置
        
        # proces Query and Range
        # (batch_size * query_cols)
        query_cols = list(zip(*query_cols))
        for i in range(len(val_range)):
            val_range[i] = list(zip(*val_range[i]))

        val_range = list(zip(*val_range))

        table = self.embedding_model(table.to(torch.int))
        data2id, _ = self.ranking_model(table)
        # cost = data2id.clone().detach()
        # total_cost = data2id - cost
        total_cost = torch.zeros_like(data2id)
        

        for one_batch_index in range(len(val_range)):
            # cost[:, :] = 0    
            for id in range(self.block_nums):
                block_id, indices = self.filter_model(data2id, id)
                indices = indices.cpu().numpy()

                one_batch_query_cols = query_cols[one_batch_index]
                one_batch_val_ranges = val_range[one_batch_index]
                
                one_batch_block_df = self.table_dataset.table.data.loc[indices[one_batch_index],:]
                is_scan = True                
                for idx, q in enumerate(one_batch_query_cols):
                    if q == 'Reg Valid Date' or q == 'Reg Expiration Date':
                        min_range = pd.to_datetime(one_batch_val_ranges[idx][0])
                        max_range = pd.to_datetime(one_batch_val_ranges[idx][1])
                        if one_batch_block_df[q].min() > max_range or one_batch_block_df[q].max() < min_range:
                            is_scan = False
                            break
                        continue
                    if one_batch_block_df[q].min() > one_batch_val_ranges[idx][1] or one_batch_block_df[q].max() < one_batch_val_ranges[idx][0]:
                        is_scan = False
                        break
                if is_scan:
                    total_cost[one_batch_index, indices[one_batch_index]] = 1
                    # cost[one_batch_index, indices[one_batch_index]] = 1
            # total_cost = total_cost + cost
        #return scan
        total_scan = data2id * total_cost
        
        return total_scan

    def training_step(self, batch, batch_idx):
        table = batch['table']
    
        query_cols = batch['col']
        val_range = batch['range']
        # table, query_sample_data, target = batch 
        scan = self(table, query_cols, val_range)
        
        scan = scan.sum() / 20
   
        self.log('scan', scan, on_step=True, on_epoch=True, prog_bar=True)
        return scan

    def validation_step(self, batch, batch_idx):
        table = batch['table']
        # (query_cols * batch_size)
        query_cols = batch['col']
        # print(col)
        val_range = batch['range']
    
        # proces Query and Range
        # (batch_size * query_cols)
        query_cols = list(zip(*query_cols))
        for i in range(len(val_range)):
            val_range[i] = list(zip(*val_range[i]))

        val_range = list(zip(*val_range))
        
        # table, query_sample_data, target = batch
        
        table = self.embedding_model(table.to(torch.int))
        data2id, _ = self.ranking_model(table)
        scan = 0
        for id in range(self.block_nums):
            block_id, indices = self.filter_model(data2id, id)

            # batch_size * block_size
            indices = indices.cpu().numpy()
            for one_batch_index in range(indices.shape[0]):
                one_batch_query_cols = query_cols[one_batch_index]
                one_batch_val_ranges = val_range[one_batch_index]
                
                one_batch_block_df = self.table_dataset.table.data.loc[indices[one_batch_index],:]
                is_scan = True
                for idx, q in enumerate(one_batch_query_cols):
                    if q == 'Reg Valid Date' or q == 'Reg Expiration Date':
                        min_range = pd.to_datetime(one_batch_val_ranges[idx][0])
                        max_range = pd.to_datetime(one_batch_val_ranges[idx][1])
                        if one_batch_block_df[q].min() > max_range or one_batch_block_df[q].max() < min_range:
                            is_scan = False
                            break
                        continue
                    if one_batch_block_df[q].min() > one_batch_val_ranges[idx][1] or one_batch_block_df[q].max() < one_batch_val_ranges[idx][0]:
                        is_scan = False
                        break
                if is_scan:
                    scan += 1
        #self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict({'val_scan': scan}, on_step=True, on_epoch=True, prog_bar=True)
        return scan

    def test_step(self, batch, batch_idx):
        table, query_sample_data, target = batch 
        scan = self(table, query_sample_data)
        scan = scan.sum()
        self.log('scan', scan, on_step=True, on_epoch=True, prog_bar=True)
        return scan

    def setup(self, stage=None):
        dataset = self.hparams.dataset.lower()
        if dataset == 'tpch':
            table = datasets.LoadTPCH()
        elif dataset == 'dmv-tiny':
            table = datasets.LoadDmv('dmv-tiny.csv')
        elif dataset == 'lineitem':
            table = datasets.LoadLineitem('lineitem.csv')
        else:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{dataset}')
        print(table.data.info())
        # self.data_module = TableDataset(table)
        # Assign train/val datasets for use in dataloaders
        self.cols = table.ColumnNames()
        self.block_nums = math.ceil(table.data.shape[0] / self.hparams.block_size)
        if stage == 'fit' or stage is None:
            self.trainset = BlockDataset(table, self.hparams.block_size, self.cols, self.hparams.pad_size, rand=self.rand)
            self.valset = self.trainset

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = BlockDataset(table, self.hparams.block_size, self.cols, self.hparams.pad_size, rand=self.rand)

        self.load_model(table.Columns())
        ReportModel(self.ranking_model)
        ReportModel(self.embedding_model)
        self.table_dataset = TableDataset(table)

    def load_model(self, columns):
        # self.ranking_model = RankingModel(self.hparams.block_size, self.block_nums, len(self.cols), self.hparams.dmodel)
        self.ranking_model = RankingModel_v2(self.hparams.block_size, self.block_nums, len(self.cols), self.hparams.dmodel)

        # self.filter_model = FilterModel()
        self.filter_model = FilterModel_v2()

        self.embedding_model = Embedding(d_model=self.hparams.dmodel, 
                                        nin=len(columns), 
                                        input_bins=[c.DistributionSize() for c in columns])
        

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.hparams.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.hparams.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.hparams.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'ce':
            self.loss_function = F.cross_entropy
        else:
            raise ValueError("Invalid Loss Type!")
        
    def configure_optimizers(self):
        # if hasattr(self.hparams, 'weight_decay'):
        #     weight_decay = self.hparams.weight_decay
        # else:
        #     weight_decay = 0
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]
        
        
if __name__ == '__main__':
    print(1)
    dmodel = 2
    block_size = 3
    block_nums = 4 
    input_bins = [2, 3, 4]
    total_card = block_size * block_nums
    ranking_model = RankingModel(block_size, block_nums, len(input_bins)).to(device='cuda')

    filter_model = FilterModel().to(device='cuda')

    query_model = SummarizationModel(d_model=dmodel, 
                                nin=len(input_bins), 
                                input_bins=input_bins,
                                pad_size=total_card).to(device='cuda')
    block_model = SummarizationModel(d_model=dmodel, 
                                nin=len(input_bins), 
                                input_bins=input_bins,
                                pad_size=block_size).to(device='cuda')
    classifier = Classifier(dmodel).to(device='cuda')
    
    query = torch.randint(low=0, high=2, size=(64, total_card, 3)).to(device='cuda')
    table = torch.randint(low=0, high=2, size=(64, total_card, 3)).to(device='cuda')
    
    total_scan = 0.0
    query_embed = query_model(query)
    data2id = ranking_model(table.to(torch.long))
    table = table.to(torch.int)
    for id in range(block_nums):
        indices = filter_model(data2id, id)
        
        block = torch.gather(table, 1, indices.unsqueeze(-1).expand(-1, -1, table.size(-1)))
        
        block_embed = block_model(block)
        scan = classifier(block_embed, query_embed)
        # scan = (scan > 0.5).float()
        total_scan = scan + total_scan

    total_scan = total_scan.sum()
    
    print(next(query_model.parameters()).grad)
    print(next(ranking_model.parameters()).grad)
    
    total_scan.backward()
    
    print(next(query_model.parameters()).grad)
    print(next(ranking_model.parameters()).grad)
    
    