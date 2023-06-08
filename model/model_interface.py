# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import torch
import importlib
from model import made, transformer
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
#from data import datasets
import datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from common import TableDataset
import pytorch_lightning as pl
import numpy as np

def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb


class MInterface(pl.LightningModule):
    def __init__(self, 
                 model_name, 
                 loss, 
                 lr, 
                 num_workers=8,
                 dataset='TPCH', 
                 batchsize=512, 
                 **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_workers = num_workers
        self.dataset = dataset
        self.batch_size = batchsize
        self.configure_loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss_function(out, labels)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, labels, filename = batch
        out = self(img)
        loss = self.loss_function(out, labels)
        label_digit = labels.argmax(axis=1)
        out_digit = out.argmax(axis=1)

        correct_num = sum(label_digit == out_digit).cpu().item()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', correct_num/len(out_digit),
                 on_step=False, on_epoch=True, prog_bar=True)

        return (correct_num, len(out_digit))

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

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

    def load_model(self, columns, fixed_ordering=None):
        model_name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        if model_name == 'made':
            self.model = self.MakeMade(
                scale=self.hparams.fc_hiddens,
                cols_to_train=columns,
                seed=self.hparams.seed,
                fixed_ordering=fixed_ordering,
            )
        elif model_name == 'transformer':
            fixed_ordering = None
            self.model = self.MakeTransformer(columns, fixed_ordering)
            print("="*100)
            print(self.model.name())
            print("="*100)
        else:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {model_name}!')
            
    def MakeTransformer(self,cols_to_train, fixed_ordering, seed=None):
        return transformer.Transformer(
            num_blocks=self.hparams.blocks,
            d_model=self.hparams.dmodel,
            d_ff=self.hparams.dff,
            num_heads=self.hparams.heads,
            nin=len(cols_to_train),
            input_bins=[c.DistributionSize() for c in cols_to_train],
            use_positional_embs=True,
            activation=self.hparams.transformer_act,
            fixed_ordering=fixed_ordering,
            column_masking=self.hparams.column_masking,
            seed=seed,
        )
    
    def MakeMade(self, scale, cols_to_train, seed, fixed_ordering=None):
        if self.hparams.inv_order:
            print('Inverting order!')
            fixed_ordering = InvertOrder(fixed_ordering)

        model = made.MADE(
            nin=len(cols_to_train),
            hidden_sizes=[scale] * self.hparams.layers if self.hparams.layers > 0 else [512, 256, 512, 128, 1024],
            nout=sum([c.DistributionSize() for c in cols_to_train]),
            input_bins=[c.DistributionSize() for c in cols_to_train],
            input_encoding=self.hparams.input_encoding,
            output_encoding=self.hparams.output_encoding,
            embed_size=32,
            seed=seed,
            do_direct_io_connections=self.hparams.direct_io,
            natural_ordering=False if seed is not None and seed != 0 else True,
            residual_connections=self.hparams.residual,
            fixed_ordering=fixed_ordering,
            column_masking=self.hparams.column_masking,
        )

        return model

    def setup(self, stage=None):
        if self.dataset == 'TPCH':
            table = datasets.LoadTPCH()
        elif self.dataset == 'DMV-tiny':
            table = datasets.LoadDmv('dmv-tiny-sort.csv')
        else:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{self.dataset}')
        print(table.data.info())
        self.data_module = TableDataset(table)
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.data_module
            self.valset = self.data_module

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.data_module

        self.load_model(table.columns)
        ReportModel(self.model)

        # # If you need to balance your data using Pytorch Sampler,
        # # please uncomment the following lines.
    
        # with open('./data/ref/samples_weight.pkl', 'rb') as f:
        #     self.sample_weight = pkl.load(f)


    # def train_dataloader(self):
    #     sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset)*20)
    #     return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, sampler = sampler)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)



