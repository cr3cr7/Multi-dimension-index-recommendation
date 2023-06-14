import torch
from torch import nn
import pytorch_lightning as pl
#from common import TableDataset
from torch.utils.data import DataLoader
#from data import datasets
import datasets
from util.Block import RandomBlockGeneration, BlockDataset
from model.model_interface import ReportModel
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

class Embedding(nn.Module):
    def __init__(self, d_model, nin, input_bins):
        super().__init__()
        self.nin = nin
        self.input_bins = input_bins
        self.embeddings = nn.ModuleList()
        for i in range(nin):
            # +1 for padding
            self.embeddings.append(nn.Embedding(self.input_bins[i] + 1, d_model)) 
    
    def forward(self, x):
        y_embed = []
        for nat_idx in range(self.nin):
            y_embed.append(self.embeddings[nat_idx](x[:, :, nat_idx]))
        inp = torch.stack(y_embed, 2)
        return inp

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
    
class SummarizationModel(nn.Module):
    def __init__(self, d_model, nin, pad_size):
        super().__init__() 
        # better init
        self.apply(self._init_weights)
        
        self.summarization = nn.Sequential(
            FeedFoward(pad_size * d_model * nin),
            nn.Linear(pad_size * d_model * nin, d_model),
            nn.ReLU()
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, x):
        # x: [batch_size, nin]
        # inp: [batch_size, nin, d_model]
        """ y_embed = []
        for nat_idx in range(self.nin):
            y_embed.append(self.embeddings[nat_idx](x[:, :, nat_idx]))
        inp = torch.stack(y_embed, 2) """
        # inp,  torch.Size([4, 99, 11, 32])
        #inp = inp.reshape(inp.shape[0], -1)
        inp = x.reshape(x.shape[0], -1)
        inp = self.summarization(inp)
        return inp

class Classifier(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, block_embedding, query_embedding):  
        """Element-wise multiplication followed by a linear layer and a sigmoid"""
        # block_embedding: [batch_size, block_num, d_model]
        # query_embedding: [batch_size, query_num, d_model]
        # output: [batch_size, block_num, query_num]
        
        # element-wise multiplication
        elementwise_product = block_embedding * query_embedding
        # apply linear layer and sigmoid
        output = self.classifier(elementwise_product)
        return output.view(output.shape[0], -1)
        

class SummaryTrainer(pl.LightningModule):
    def __init__(self, 
                 num_workers=8,
                 **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_workers = num_workers
        self.lr = kargs['lr']
        self.rand = kargs['rand']
        self.configure_loss()

    def forward(self, query, block):
        em_query = self.embedding_model(query)
        query_embed = self.model(em_query)

        em_block = self.embedding_model(block)
        block_embed = self.model(em_block)
        
        scan = self.classifier(block_embed, query_embed)
        return scan
        
    
    def training_step(self, batch, batch_idx):
        new_block, query_sample_data, result = batch
        scan = self(query_sample_data, new_block)
        loss = self.loss_function(scan, result)
  
        scan = scan > 0.5
        acc_metric = accuracy_score(result.cpu(), scan.cpu())
        TruePerc = result.sum() / len(result)
        self.log_dict({'loss': loss, 'train_acc': acc_metric, 'train_TruePerc': TruePerc}, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
        
    def validation_step(self, batch, batch_idx):
        new_block, query_sample_data, result = batch
        scan = self(query_sample_data, new_block)
        loss = self.loss_function(scan, result)
        # Measure Accuracy
        scan = scan > 0.5
        acc_metric = accuracy_score(result.cpu(), scan.cpu())
        f1 = f1_score(result.cpu(), scan.cpu())
        TruePerc = result.sum() / len(result)
    
        self.log_dict({'val_loss': loss, "f1": f1, 'val_acc': acc_metric, 'TruePerc': TruePerc}, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
        
    def test_step(self, batch, batch_idx):
        new_block, query_sample_data, result = batch
        scan = self(query_sample_data, new_block)
        loss = self.loss_function(scan, result)
        
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        

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
        if stage == 'fit' or stage is None:
            self.trainset = BlockDataset(table, self.hparams.block_size, self.cols, self.hparams.pad_size, rand=self.rand)
            self.valset = BlockDataset(table, self.hparams.block_size, self.cols, self.hparams.pad_size, rand=self.rand)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = BlockDataset(table, self.hparams.block_size, self.cols, self.hparams.pad_size, rand=self.rand)

        self.load_model(table.columns)
        ReportModel(self.model)
        ReportModel(self.classifier)

    def load_model(self, columns):
        self.model = SummarizationModel(d_model=self.hparams.dmodel, 
                                        nin=len(columns), 
                                        pad_size=self.hparams.pad_size)
    
        self.classifier = Classifier(self.hparams.dmodel)
        
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
    model = SummarizationModel(32, 3, [10, 10, 10])
    x = torch.randint(0, 10, (32, 1, 3))
    print(model(x).shape)
    
    # test classifier
    # Define the input tensors
    block_embedding = torch.randn(2, 3, 4)  # [batch_size, block_num, d_model]
    query_embedding = torch.randn(2, 3, 4)  # [batch_size, query_num, d_model]

    # Initialize the classifier module
    classifier = Classifier(d_model=4)

    # Compute the output of the classifier module
    output = classifier(block_embedding, query_embedding)

    # Check that the output has the expected shape
    assert output.shape == (2, 3), output.shape

    # Check that the output values are between 0 and 1
    assert (output >= 0).all() and (output <= 1).all()

    # Check that the output values are different for different input tensors
    block_embedding2 = torch.randn(2, 3, 4)
    query_embedding2 = torch.randn(2, 3, 4)
    output2 = classifier(block_embedding2, query_embedding2)
    assert not torch.allclose(output, output2)