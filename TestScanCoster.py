import sys
sys.path.append('..')
print(sys.path)

import torch
from torch import nn
import pytorch_lightning as pl
# from common import TableDataset
from torch.utils.data import DataLoader
#from data import datasets
import datasets
from util.NewBlock import RandomBlockGeneration, BlockDataset
from model.model_interface import ReportModel
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from model.generation import RankingModel, FilterModel
import math
from model.summarization import FeedFoward, SummarizationModel, Classifier, Embedding



class ScanCostTrainer(pl.LightningModule):
    def __init__(self, 
                 num_workers=8,
                 **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_workers = num_workers
        self.lr = kargs['lr']
        self.configure_loss()

    def forward(self, table, query):
        # 大小应为 batch_size ，此处手动设置
        # total_scan = torch.zeros(64, requires_grad=True).reshape(64, 1)
        total_scan = 0
        #total_scan = total_scan.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        query_embed = self.query_model(query)
        data2id = self.ranking_model(table)
        table = table.to(torch.int)
        for id in range(self.block_nums):
            block_id, indices = self.filter_model(data2id, id)
            
            block = table * block_id

            selected_block = torch.gather(block, 1, indices.unsqueeze(-1).expand(-1, -1, block.size(-1)))
            

            block_embed = self.block_model(selected_block)
            scan = self.classifier(block_embed, query_embed)
            # scan = (scan > 0.5).float()
            total_scan = scan + total_scan
        #return scan
        return total_scan

    def training_step(self, batch, batch_idx):
        table, query_sample_data, target = batch 
        scan = self(table, query_sample_data)
        """ table_size = torch.tensor(table.size()[0], dtype=float)
        target = (table_size / query_size).ceil() """
        loss = self.loss_function(scan, target)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        table, query_sample_data, target = batch
        scan = self(table, query_sample_data)
        """ table_size = torch.tensor(table.size()[1], dtype=float)
        target = (table_size / query_size).ceil() """
        loss = self.loss_function(scan, target)

        for name, param in self.ranking_model_parameters():
            if param.grad is not None:
                print(f'Parameter: {name}, Gradient: {param.grad}')
            else:
                print(f'Parameter: {name}, Gradient: None')
        #self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict({'val_loss': loss}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        table, query_sample_data, target = batch 
        scan = self(table, query_sample_data)
        """ table_size = torch.tensor(table.size()[0], dtype=float)
        target = (table_size / query_size).ceil() """
        loss = self.loss_function(scan, target)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

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
        if stage == 'fit' or stage is None:
            self.trainset = BlockDataset(table, self.hparams.block_size, self.cols)
            self.valset = BlockDataset(table, self.hparams.block_size, self.cols)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = BlockDataset(table, self.hparams.block_size, self.cols)

        self.load_model(table.Columns())
        ReportModel(self.ranking_model)
        ReportModel(self.ranking_model)
        ReportModel(self.query_model)
        ReportModel(self.block_model)
        ReportModel(self.classifier)


    def load_model(self, columns):
        self.ranking_model = RankingModel(self.hparams.block_size, self.block_nums, len(self.cols))

        self.filter_model = FilterModel()

        self.query_model = SummarizationModel(d_model=self.hparams.dmodel, 
                                        nin=len(columns), 
                                        input_bins=[c.DistributionSize() for c in columns],
                                        pad_size=50)
        self.block_model = SummarizationModel(d_model=self.hparams.dmodel, 
                                        nin=len(columns), 
                                        input_bins=[c.DistributionSize() for c in columns],
                                        pad_size=self.hparams.block_size)
        self.classifier = Classifier(self.hparams.dmodel)

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
            self.parameters(), lr=self.hparams.lr)

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
                                pad_size=total_card).to(device='cuda')
    block_model = SummarizationModel(d_model=dmodel, 
                                nin=len(input_bins), 
                                pad_size=block_size).to(device='cuda')
    classifier = Classifier(dmodel).to(device='cuda')
    
    embedding_model = Embedding(d_model=dmodel, 
                                nin=len(input_bins), 
                                input_bins=input_bins).to(device='cuda')
    
    query = torch.randint(low=0, high=2, size=(64, total_card, 3)).to(device='cuda')
    table = torch.randint(low=0, high=2, size=(64, total_card, 3)).to(device='cuda')
    
    total_scan = 0.0
    
    em_query = embedding_model(query)
    
    query_embed = query_model(em_query)
    
    """ query_embed = query_embed.sum()
    query_embed.backward()
    print(next(query_model.parameters()).grad)
    print(next(embedding_model.parameters()).grad)
    print(em_query.grad) """

    data2id = ranking_model(table.to(torch.float))
    
    table = embedding_model(table)
    
    """ block_id, indices = filter_model(data2id, 1)
    block_id.retain_grad()
    
    block = block_id.unsqueeze(-1) * table
    block.retain_grad()
    

    block_embed = block_model(block) """
    
    
    for id in range(block_nums):
        block_id, indices = filter_model(data2id, id)

        block = table * block_id.unsqueeze(-1)

        selected_block = torch.gather(block, 1, indices.unsqueeze(-1).expand(-1, -1, block.size(2)).unsqueeze(-1).expand(-1, -1, -1, block.size(-1)))
            

        block_embed = block_model(selected_block)
        scan = classifier(block_embed, query_embed)
        # scan = (scan > 0.5).float()
        total_scan = scan + total_scan
        
    total_scan = total_scan.sum()
    print(next(query_model.parameters()).grad)
    print(next(ranking_model.parameters()).grad)
    print(next(classifier.parameters()).grad)
    total_scan.backward()
    
    print(next(query_model.parameters()).grad)
    print(next(block_model.parameters()).grad)
    print(next(ranking_model.parameters()).grad)
    