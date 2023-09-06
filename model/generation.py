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
from model.transformer import Block

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

class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean

class VAE(nn.Module):
    def __init__(self, col_num, dmodel):
        super(VAE, self).__init__()
        self.col_num = col_num
        self.dmodel = dmodel
        self.encoder = nn.Sequential(
            nn.Linear(col_num , 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU()
        )
        dmodel = 128
        d_ff = 128
        num_heads = 2
        num_blocks = 2
        activation = "gelu"
        self.encoder_block = nn.Sequential(*[
            Block(dmodel,
                  d_ff,
                  num_heads,
                  activation,
                  do_residual=True)
            for i in range(num_blocks)
        ])
        self.decoder_block = nn.Sequential(*[
            Block(64,
                  d_ff,
                  num_heads,
                  activation,
                  do_residual=True)
            for i in range(num_blocks)
        ])
        # self.latent_dim = int((dmodel * col_num) / 4) 
        self.latent_dim = 64
        self.reparameterize = Lambda(128, self.latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, col_num),
            nn.ReLU()
        )
    
    def forward(self, table):
        encode_table = self.encoder(table)
        encode_table = self.encoder_block(encode_table)
        z = self.reparameterize(encode_table)
        z = self.decoder_block(z)
        recon_table = self.decoder(z)
        
        
        mu = self.reparameterize.latent_mean
        logvar = self.reparameterize.latent_logvar
        loss = self.loss_function(recon_table, table, mu, logvar)
        return loss, z

    
    def loss_function(self, recon_x, x, mu, logvar):
        # recon_loss = F.mse_loss(recon_x, x, size_average=False)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return (recon_loss + KLD) / (x.shape[0] * x.shape[1])


class RankingModel_v4(nn.Module):
    def __init__(self, col_num, dmodel, input_bins):
        super(RankingModel_v4, self).__init__()
        self.col_num = col_num
        self.dmodel = dmodel
        self.model = VAE(col_num, dmodel)
        self.input_bins = input_bins
        
        self.apply(self._init_weights)
        
    def forward(self, table, BlockSize, current_epoch):
        rows = table.shape[1]
        # table = table.reshape(-1, rows, self.col_num * self.dmodel)
        table = table.reshape(-1, rows, self.col_num)
        
        table = table / torch.tensor(self.input_bins)
        loss, z = self.model(table)
        # print(loss)
        # print(z.shape)
    
        scores = self.clustering(z)
    
        min_vals = torch.min(scores, dim=1, keepdim=True)[0]
        max_vals = torch.max(scores, dim=1, keepdim=True)[0]
        scaled_scores = ((scores - min_vals) / (max_vals - min_vals)) * len(table)

        # original_ranks = torchsort.soft_rank(scaled_scores, regularization="l2", regularization_strength=0.01)
        regularization_strength = (0.995)**current_epoch
        # regularization_strength = 1
        original_ranks = torchsort.soft_rank(scaled_scores, regularization="l2", regularization_strength=regularization_strength)
        # original_ranks = torchsort.soft_rank(scores, regularization="l2", regularization_strength=0.01)
        # print(original_ranks[0].reshape(-1))
        
        sorted_indices = torch.argsort(scores, dim=1)
        rank_indices = torch.zeros_like(sorted_indices)
        for batch in range(sorted_indices.shape[0]):
            rank_indices[batch, sorted_indices[batch]] = torch.arange(sorted_indices.shape[1])
        # rank_indices[:, sorted_indices] = torch.arange(sorted_indices.shape[1])
        rank_indices = rank_indices // BlockSize
        # Block 0 for padding
        rank_indices = rank_indices + 1
        # print(rank_indices[0].reshape(-1))
        return original_ranks.reshape(-1, rows, 1), rank_indices.reshape(-1, rows, 1), scores.reshape(-1, rows, 1), loss
    
    def clustering(self, table):
        # FIXME: Hack for train and validation
        if table.shape[1] == 1000:
            all_scores = []
            for idx, array in enumerate(table.detach().numpy()):
                bandwidth = estimate_bandwidth(array, quantile=1)
                clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=bandwidth).fit(array)
                labels = clusters.labels_
                torch_array = table[idx]
                cluster_centers = torch.stack([torch.mean(torch_array[labels == i, :], dim=0) for i in range(clusters.n_clusters_)])
                center_data = torch.gather(cluster_centers, 0, torch.from_numpy(labels).reshape(-1, 1).expand(-1, array.shape[1]))
                
                distances = torch.mean(torch.square(torch_array - center_data), dim=-1)
                min_vals = torch.min(distances, dim=0, keepdim=True)[0]
                max_vals = torch.max(distances, dim=0, keepdim=True)[0]
                distances = ((distances - min_vals) / (max_vals - min_vals)) * 1
                scores = distances + torch.from_numpy(labels)
                all_scores.append(scores)
            return torch.stack(all_scores, dim=0)
        array = table[0].detach().numpy()
        # print(array)
        bandwidth = estimate_bandwidth(array, quantile=1)
        clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=bandwidth).fit(array)
        # clusters = KMeans(n_clusters=self.cluster_num).fit(array)
        labels = clusters.labels_
        # print(labels)
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                # nn.init.xavier_uniform_(module.bias)
                # nn.init.kaiming_uniform_(module.bias, mode='fan_in', nonlinearity='relu')
        elif isinstance(module, nn.Embedding):
            # torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            nn.init.xavier_uniform_(module.weight)

class RankingModel_v3(nn.Module):
    def __init__(self, col_num, dmodel):
        super(RankingModel_v3, self).__init__()
        self.col_num = col_num
        self.dmodel = dmodel
        self.MLP = nn.Sequential(
            nn.Linear(col_num , 32),
            nn.ReLU(),
            nn.Linear(32, dmodel),
            nn.ReLU()
        )
        self.cluster_num = 5
    
    def forward(self, table, BlockSize, current_epoch):
        rows = table.shape[1]
        # table = table.reshape(-1, rows, self.col_num * self.dmodel)
        table = table.reshape(-1, rows, self.col_num)
        
        table = self.MLP(table)
        scores = self.clustering(table)
    
        min_vals = torch.min(scores, dim=1, keepdim=True)[0]
        max_vals = torch.max(scores, dim=1, keepdim=True)[0]
        scaled_scores = ((scores - min_vals) / (max_vals - min_vals)) * len(table)

        # original_ranks = torchsort.soft_rank(scaled_scores, regularization="l2", regularization_strength=0.01)
        regularization_strength = (0.995)**current_epoch
        original_ranks = torchsort.soft_rank(scaled_scores, regularization="l2", regularization_strength=0.001)
        # original_ranks = torchsort.soft_rank(scores, regularization="l2", regularization_strength=0.01)
        # print(original_ranks[0].reshape(-1))
        
        sorted_indices = torch.argsort(scores, dim=1)
        rank_indices = torch.zeros_like(sorted_indices)
        for batch in range(sorted_indices.shape[0]):
            rank_indices[batch, sorted_indices[batch]] = torch.arange(sorted_indices.shape[1])
        # rank_indices[:, sorted_indices] = torch.arange(sorted_indices.shape[1])
        rank_indices = rank_indices // BlockSize
        # Block 0 for padding
        rank_indices = rank_indices + 1
        # print(rank_indices[0].reshape(-1))
        return original_ranks.reshape(-1, rows, 1), rank_indices.reshape(-1, rows, 1), scores.reshape(-1, rows, 1)
        
        
    def clustering(self, table):
        # FIXME: Hack for train and validation
        if table.shape[1] == 1000:
            all_scores = []
            for idx, array in enumerate(table.detach().numpy()):
                bandwidth = estimate_bandwidth(array, quantile=1)
                clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=bandwidth).fit(array)
                labels = clusters.labels_
                torch_array = table[idx]
                cluster_centers = torch.stack([torch.mean(torch_array[labels == i, :], dim=0) for i in range(clusters.n_clusters_)])
                center_data = torch.gather(cluster_centers, 0, torch.from_numpy(labels).reshape(-1, 1).expand(-1, array.shape[1]))
                
                distances = torch.mean(torch.square(torch_array - center_data), dim=-1)
                min_vals = torch.min(distances, dim=0, keepdim=True)[0]
                max_vals = torch.max(distances, dim=0, keepdim=True)[0]
                distances = ((distances - min_vals) / (max_vals - min_vals)) * 1
                scores = distances + torch.from_numpy(labels)
                all_scores.append(scores)
            return torch.stack(all_scores, dim=0)
        array = table[0].detach().numpy()
        # print(array)
        bandwidth = estimate_bandwidth(array, quantile=1)
        clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=bandwidth).fit(array)
        # clusters = KMeans(n_clusters=self.cluster_num).fit(array)
        labels = clusters.labels_
        # print(labels)
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
    def __init__(self, BlockSize, BlockNum, col_num, dmodel, input_bins):
        super(RankingModel_v2, self).__init__()
        self.capacity = BlockSize
        self.BlockNum = BlockNum
        self.col_num = col_num
        self.dmodel = dmodel
        self.input_bins = input_bins
        
        self.model = VAE(col_num, dmodel)
        self.latent_size = self.model.latent_dim
        # self.latent_size = 32
        
        self.MLP = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 1)
            # nn.Sigmoid()
            # nn.ReLU() 
        )
        
        d_model = self.latent_size
        # d_ff = 256
        # num_heads = 4
        # num_blocks = 4
        d_ff = 128
        num_heads = 2
        num_blocks = 2
        activation = "gelu"
        self.blocks = nn.Sequential(*[
            Block(d_model,
                  d_ff,
                  num_heads,
                  activation,
                  do_residual=True)
            for i in range(num_blocks)
        ])
        
        # self.ln1 = nn.LayerNorm(100)
        self.apply(self._init_weights)
    
    
    def forward(self, table, BlockSize, current_epoch):
        rows = table.shape[1]
        # table = table.reshape(-1, rows, self.col_num * self.dmodel)
        table = table.reshape(-1, rows, self.col_num)
        # FIXME:让输入在每个维度上归一化, which one is better? (torch.tensor(self.input_bins) and Max(self.input_bins) )
        # table = table / torch.tensor(self.input_bins)
        # table = table / max(self.input_bins)
        # table = table / 10
        # FIMXE: Hack For larg tbale --Log transform
        if max(self.input_bins) > 1200:
            table = torch.log(table + 1)

        loss, z = self.model(table)
        zz = self.blocks(z)
        # print(zz.shape)
        scores = self.MLP(zz).reshape(-1, rows)
        # scores = nn.functional.softmax(scores, dim=1)
        
        # For Fast Inference
        if not self.training:
            return scores.reshape(-1)
            
        min_vals = torch.min(scores, dim=1, keepdim=True)[0]
        max_vals = torch.max(scores, dim=1, keepdim=True)[0]
        scaled_scores = ((scores - min_vals) / (max_vals - min_vals)) * len(table)
        # regularization_strength = (0.995)**current_epoch
        # regularization_strength = 0.5 * (0.995)**current_epoch
        # if regularization_strength < 0.01:
        #     regularization_strength = 0.01
        regularization_strength = 0.01
        original_ranks = torchsort.soft_rank(scaled_scores, regularization="l2", regularization_strength=regularization_strength)
        # original_ranks = torchsort.soft_rank(scaled_scores, regularization="l2", regularization_strength=0.01)


        # Generate Block ID
        # other = original_ranks.detach() % self.capacity
        # rank_indices = (original_ranks - other) / self.capacity
        
        # Get Rank Index of scaled_scores
        sorted_indices = torch.argsort(scores, dim=1)
        rank_indices = torch.zeros_like(sorted_indices, device=scores.device)
        for batch in range(sorted_indices.shape[0]):
            rank_indices[batch, sorted_indices[batch]] = torch.arange(sorted_indices.shape[1], device=scores.device)
        # rank_indices[:, sorted_indices] = torch.arange(sorted_indices.shape[1])
        # rank_indices = rank_indices // BlockSize
        rank_indices = torch.div(rank_indices, BlockSize, rounding_mode='floor')
      
        # Block 0 for padding
        rank_indices = rank_indices + 1
        # print(rank_indices[0].reshape(-1))
        return original_ranks.reshape(-1, rows, 1), rank_indices.reshape(-1, rows, 1), scores.reshape(-1, rows, 1), loss
        
        # Old version 1
        # sorted_indices = torch.argsort(scores, dim=1)
        # rank_indices = torch.zeros_like(sorted_indices)
        # rank_indices[:, sorted_indices] = torch.arange(sorted_indices.shape[1])
        # rank_indices = rank_indices // BlockSize
        # # Block 0 for padding
        # rank_indices = rank_indices + 1
        # # print(rank_indices[0].reshape(-1))
        # return original_ranks.reshape(-1, rows, 1), rank_indices.reshape(-1, rows, 1), 0
        
        # Old version 0
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
            # nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                # nn.init.xavier_uniform_(module.bias)
                # nn.init.kaiming_uniform_(module.bias, mode='fan_in', nonlinearity='relu')
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.xavier_uniform_(module.weight)

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

