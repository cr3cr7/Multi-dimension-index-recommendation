# Unit test for ScanCoster.py, have to add the current path to sys.path
import sys
sys.path.append('..')

import torch
from torch import nn
import pytorch_lightning as pl
from common import TableDataset
from torch.utils.data import DataLoader
#from data import datasets
import torch.optim.lr_scheduler as lrs
import datasets
from util.NewBlock import RandomBlockGeneration, BlockDataset, BlockDataset_V2, BlockDataset_Eval
from model.model_interface import ReportModel
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from model.generation import RankingModel, FilterModel, RankingModel_v2, FilterModel_v2, RankingModel_v3, RankingModel_v4
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
        self.validation_step_outputs = []
        
        # self.baseline = pd.read_csv('./datasets/linitem_1000-zorder.csv')
        # self.baseline = pd.read_csv('./datasets/RandomWalk-10K-100Col-zorder.csv')
        self.baseline = pd.read_csv('./datasets/RandomWalk-1000K-100Col-zorder.csv')
        # self.baseline = pd.read_csv('./datasets/dmv-tiny-zorder.csv')
        print(self.baseline)
        
        
    def forward(self, table, batch_table_idx, query_cols, val_range):
        # 大小应为 batch_size ，此处手动设置
        
        # proces Query and Range
        # (batch_size * query_cols)
        
        # table = self.embedding_model(table.to(torch.int))
        if isinstance(self.ranking_model, RankingModel_v4) or isinstance(self.ranking_model, RankingModel_v2):
            original_rank, data2id, distance, recont_loss = self.ranking_model(table, self.hparams.train_block_size, self.current_epoch)
        else:
            original_rank, data2id, distance = self.ranking_model(table, self.hparams.train_block_size, self.current_epoch)
        # data2id = self.ranking_model(table)
        # cost = data2id.clone().detach()
        # total_cost = data2id - cost
        total_cost = torch.zeros_like(data2id)
        
        # TODO:这一段需要做加速
        for one_batch_index in range(len(val_range)):
            # start_idx = int(batch_table_idx[one_batch_index])
            # batch_table = self.table_dataset.table.data[start_idx: self.hparams.pad_size + start_idx].reset_index(drop=True)
            batch_idx = batch_table_idx[one_batch_index]
            batch_table = self.trainset.table.data.iloc[batch_idx.cpu().numpy()].reset_index(drop=True)
            
            # batch_table = self.table_dataset.table.data
            one_batch_query_cols = query_cols[one_batch_index]
            one_batch_val_ranges = val_range[one_batch_index]
    
            for id in range(self.train_block_nums):
                
                # TODO: 这里需要加速
                block_id, indices = self.filter_model(data2id, id)
                indices = indices.cpu().numpy()  

                one_batch_block_df = batch_table.loc[indices[one_batch_index],:]

                if id + 1 < self.train_block_nums:
                    assert one_batch_block_df.shape[0] == self.hparams.train_block_size, 'Block Size Error'
                
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
        
        # TODO: 是否要对Loss改进？让收敛更好
        all_loss = 0
        all_regular_loss = 0
        for cur_batch, one_batch in enumerate(original_rank * total_cost):
            if torch.is_tensor(distance):
                cur_dis = distance[cur_batch]
            loss = 0
            regular_loss = 0
            one_batch = one_batch.view(-1)
            indices = one_batch.nonzero().reshape(-1)
            sorted_indices = indices[torch.argsort(one_batch[indices], descending=True)]
            for i in range(len(sorted_indices) - 1):
                if (i % self.hparams.train_block_size == (self.hparams.train_block_size - 1)):
                    continue
                # print(one_batch[sorted_indices[i]], one_batch[sorted_indices[i + 1]], one_batch[sorted_indices[i]] - one_batch[sorted_indices[i + 1]])
                loss += torch.abs(one_batch[sorted_indices[i]] - one_batch[sorted_indices[i + 1]])
                if torch.is_tensor(distance):
                    regular_loss += torch.abs(cur_dis[sorted_indices[i]] - cur_dis[sorted_indices[i + 1]])
                # loss += F.l1_loss(one_batch[sorted_indices[i]], one_batch[sorted_indices[i + 1]])
            
            loss += len(sorted_indices) / self.hparams.train_block_size

            # assert 0       
            # loss = one_batch[sorted_indices[-1]]
            loss = loss / self.hparams.train_block_size
            if torch.is_tensor(distance):
                regular_loss = regular_loss / self.hparams.train_block_size
            all_loss = all_loss + loss
            all_regular_loss = all_regular_loss + regular_loss
            assert len(sorted_indices) % self.hparams.train_block_size == 0

        # Add Flip the total_cost
        # flipped = torch.ones_like(total_cost) - total_cost
        # for one_batch in original_rank * flipped:
        #     loss = 0
        #     one_batch = one_batch.view(-1)
        #     indices = one_batch.nonzero().reshape(-1)
        #     sorted_indices = indices[torch.argsort(one_batch[indices], descending=True)]
        #     for i in range(len(sorted_indices) - 1):
        #         if (i % self.hparams.block_size == (self.hparams.block_size - 1)):
        #             continue
        #         loss += torch.abs(one_batch[sorted_indices[i]] - one_batch[sorted_indices[i + 1]])
        #     loss += len(sorted_indices) / 20
        #     loss = loss / self.hparams.block_size
        #     all_loss = all_loss - loss
        #     assert len(sorted_indices) % 20 == 0
        # print(regular_loss)
        # print(all_loss)
        # assert 0
        # return all_loss
        if isinstance(self.ranking_model, RankingModel_v4) or isinstance(self.ranking_model, RankingModel_v2):
            recont_loss = recont_loss / (100)
            # recont_loss = recont_loss
            self.log_dict({'cluster_loss': all_loss, 'recont_loss': recont_loss}, on_step=True, on_epoch=True, prog_bar=True)
            # return all_loss + recont_loss
            # print("Loss")
            # print(all_loss, recont_loss, regular_loss)
            return all_loss + recont_loss
            # return all_loss
        
        return all_loss - 100 * regular_loss
        # return total_scan.sum() / total_cost.sum()

    def training_step(self, batch, batch_idx):
        # Testing
        if self.current_epoch < 120:
            # Ours
            assert len(self.baseline) == len(self.trainset.table.data)
            table = batch['table']
            batch_table_idx = batch['table_idx']
            original_rank, data2id, distance, recont_loss = self.ranking_model(table, self.hparams.train_block_size, self.current_epoch)
            our_score = distance[0].reshape(-1, 1)
            
            # Baseline
            current_baseline = self.baseline.iloc[batch_table_idx[0].cpu().numpy()]
            z_value = torch.from_numpy(current_baseline['zvalue'].values)
            sorted_indices = torch.argsort(z_value)
            rank_indices = torch.zeros_like(sorted_indices)
            rank_indices[sorted_indices] = torch.arange(len(sorted_indices))
            
            import random
            def pairwise_data_loss(rank_indices, our_score):
                batch_num = 1000
                bce_loss = nn.BCELoss()
                total_loss = 0.0
                for i in range(batch_num):
                    # randomly select two samples
                    idx1, idx2 = random.sample(range(len(our_score)), 2)
                    # calculate BCE loss
                    if idx1 == idx2:
                        true_label = 0.5
                    else:
                        true_label = 1 if rank_indices[idx1] > rank_indices[idx2] else 0
                    loss = bce_loss(torch.sigmoid(our_score[idx1] - our_score[idx2]), torch.tensor([true_label], device=self.device, dtype=torch.float32))
                    total_loss += loss
                return total_loss / batch_num
                
            
            def listnet_loss(y_i, z_i):
                """
                y_i: (n_i, 1)
                z_i: (n_i, 1)
                """

                P_y_i = F.softmax(y_i, dim=0)
                P_z_i = F.softmax(z_i, dim=0)
                return - torch.sum(P_y_i * torch.log(P_z_i))
            # loss = listnet_loss(rank_indices.float(), our_score)
            loss = pairwise_data_loss(rank_indices, our_score)
            return loss
        
        table = batch['table']
        batch_table_idx = batch['table_idx']
        query_cols = batch['col']
        val_range = batch['range']
        
        query_cols = list(zip(*query_cols))
        for i in range(len(val_range)):
            val_range[i] = list(zip(*val_range[i]))
        val_range = list(zip(*val_range))
        
        # table, query_sample_data, target = batch 
        scan = self(table, batch_table_idx, query_cols, val_range)
        
        # scan = scan.sum() / self.hparams.batch_size
        
        self.log('scan', scan, on_step=True, on_epoch=True, prog_bar=True)
        return scan

    def on_validation_epoch_end(self):   
        # proces Query and Range
        scan_cond = self.valset.testScanConds
        scores = torch.cat(self.validation_step_outputs, dim=0)
        # Get Rank Index of scaled_scores
        sorted_indices = torch.argsort(scores)
        rank_indices = torch.zeros_like(sorted_indices, device=scores.device)
        rank_indices[sorted_indices] = torch.arange(len(scores), device=scores.device)
        # rank_indices = rank_indices // self.hparams.test_block_size
        rank_indices = torch.div(rank_indices, self.hparams.test_block_size, rounding_mode='floor')
        # Block 0 for padding
        data2id = rank_indices + 1
        
        scan = 0
        for id in range(self.test_block_nums):
            target_id = id + 1
            indices = torch.where(data2id == target_id)[0].cpu().numpy()
    
            one_batch_block_df = self.valset.table.data.loc[indices,:]
            
            # FIXME: Have problem in DDP training
            if target_id < self.test_block_nums:
                # print(target_id, self.test_block_nums)
                assert one_batch_block_df.shape[0] == self.hparams.test_block_size, one_batch_block_df.shape[0]
            
            for idx, query in enumerate(scan_cond):
                one_batch_query_cols = query[0]
                one_batch_val_ranges = query[1]
                
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
                
        
        # Free Memory
        self.validation_step_outputs.clear()
        self.log_dict({'val_scan': scan}, on_epoch=True, prog_bar=True)
        
        
    def validation_step(self, batch, batch_idx):
        batch = batch.unsqueeze(0)
        # table = self.embedding_model(table.to(torch.int))
        scores = self.ranking_model(batch, self.hparams.test_block_size, self.current_epoch)
        self.validation_step_outputs.append(scores)


    def test_scan(self, batch_table, block_nums, block_size, query_cols, val_range, data2id) -> int:
        scan = 0
        for one_batch_index in range(len(val_range)):
            # batch_table = self.table_dataset.table.data
            for id in range(block_nums):
                
                block_id, indices = self.filter_model(data2id, id)
                indices = indices.cpu().numpy()

                one_batch_query_cols = query_cols[one_batch_index]
                one_batch_val_ranges = val_range[one_batch_index]
                
                one_batch_block_df = batch_table.loc[indices[one_batch_index],:]

                assert one_batch_block_df.shape[0] == block_size, 'Block Size Error'
                
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
        elif dataset == 'dmv':
            table = datasets.LoadDmv('DMV_1000.csv')
        elif dataset == 'lineitem':
            table = datasets.LoadLineitem('linitem_1000.csv', cols=['l_orderkey', 'l_partkey'])
        elif dataset == 'randomwalk':
            # table = datasets.LoadRandomWalk(100, 10000)
            table = datasets.LoadRandomWalk(100, int(1e6))
        else:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{dataset}')
        print("Table Len:", len(table.data))
        print(table.data.info())
        # self.data_module = TableDataset(table)
        # Assign train/val datasets for use in dataloaders
        self.cols = table.ColumnNames()
        self.train_block_nums = math.ceil(self.hparams.pad_size / self.hparams.test_block_size)
        self.test_block_nums = math.ceil(table.data.shape[0] / self.hparams.test_block_size)
        if stage == 'fit' or stage is None:
            self.trainset = BlockDataset_V2(table, self.hparams.train_block_size, self.cols, self.hparams.pad_size, rand=self.rand)
            self.valset = BlockDataset_Eval(**vars(self.trainset))

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = BlockDataset_V2(table, self.hparams.test_block_size, self.cols, self.hparams.pad_size, rand=self.rand)

        self.load_model(table.Columns())
        ReportModel(self.ranking_model)
        # ReportModel(self.embedding_model)
        # self.table_dataset = TableDataset(table)

    def load_model(self, columns):
        # state_dict = torch.load(self.PATH)
        # print(state_dict)

        
        # self.ranking_model = RankingModel(self.hparams.block_size, self.block_nums, len(self.cols), self.hparams.dmodel)
        self.ranking_model = RankingModel_v2(self.hparams.block_size, 
                                             self.train_block_nums, 
                                             len(self.cols), 
                                             self.hparams.dmodel,
                                             input_bins=[c.DistributionSize()+1 for c in columns])
        # self.ranking_model = RankingModel_v3(len(self.cols), self.hparams.dmodel)
        # self.ranking_model = RankingModel_v4(len(self.cols), self.hparams.dmodel, 
        #                                      input_bins=[c.DistributionSize()+1 for c in columns])

        # self.filter_model = FilterModel()
        self.filter_model = FilterModel_v2()
        # self.embedding_model = Embedding(d_model=self.hparams.dmodel, 
        #                                 nin=len(columns), 
        #                                 input_bins=[c.DistributionSize() for c in columns])
        # self.embedding_model.load_state_dict(state_dict, strict=False)

    def train_dataloader(self):
        pin_memory = True if "cuda" in str(self.device) else False
        return DataLoader(self.trainset, batch_size=self.hparams.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=pin_memory)
    def val_dataloader(self):
        pin_memory = True if "cuda" in str(self.device) else False
        return DataLoader(self.valset, batch_size=self.hparams.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=pin_memory)

    def test_dataloader(self):
        pin_memory = True if "cuda" in self.device else False
        return DataLoader(self.testset, batch_size=self.hparams.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=pin_memory)
    
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
            elif self.hparams.lr_scheduler == 'onecycler':
                scheduler = lrs.OneCycleLR(optimizer, 
                                           max_lr=self.hparams.lr, 
                                           epochs=self.hparams.max_epochs,
                                           steps_per_epoch=len(self.train_dataloader()))
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
    
    