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
from model.transformer import Block, LayerNorm

from sklearn.cluster import estimate_bandwidth, AgglomerativeClustering, KMeans


class AddBlock(nn.Module):
    """A Add ATT block.

    Args:
      d_model: last dim of input and output of this module.
      d_ff: the hidden dim inside the FF net.
      num_heads: number of parallel heads.
    """

    def __init__(self,
                 d_model,
                 d_ff,
                 num_heads,
                 activation='relu',
                 do_residual=False):
        super(Block, self).__init__()

        self.mlp = nn.Sequential(
            Conv1d(d_model, d_ff),
            nn.ReLU() if activation == 'relu' else GeLU(),
            Conv1d(d_ff, d_model),
        )
        self.norm1 = LayerNorm(features=d_model)
        self.norm2 = LayerNorm(features=d_model)
        self.attn = InputDimAttention(input_dim=d_model)
        self.do_residual = do_residual

    def set_attn_mask(self, mask):
        self.attn.attn_mask = mask

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        if self.do_residual:
            x += residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        if self.do_residual:
            x += residual

        return x

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        dropout = 0.2
        self.net = nn.Sequential(
            nn.Linear(n_embd, 32 * n_embd),
            nn.ReLU(),
            nn.Linear(32 * n_embd, n_embd),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class AdaCurveBlock(nn.Module):
    """Transformer like block with adaptive curve attention."""
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 activation='relu', 
                 do_residual=True):
        super(AdaCurveBlock, self).__init__()

        self.mlp = FeedFoward(input_dim)
        self.norm1 = LayerNorm(features=input_dim)
        self.norm2 = LayerNorm(features=input_dim)
        self.attn = InputDimAttention(input_dim=input_dim)
        self.do_residual = do_residual
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x)
        if self.do_residual:
            x += residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        if self.do_residual:
            x += residual

        return x
    


class InputDimAttention(nn.Module):
    def __init__(self, input_dim):
        super(InputDimAttention, self).__init__()
        # Define a small feed-forward network to compute attention weights
        # self.attention_network = nn.Sequential(
        #     nn.Linear(input_dim, input_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(input_dim // 2, input_dim),
        # )
        
        # self.attention_network = Attention
        
        # TODO: Visualize attention weights
        hidden_size = 32
        # dropout_rate = 0.1
        self.attention_network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            # nn.LeakyReLU(),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size * 2),
            # nn.LeakyReLU(),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            # nn.LeakyReLU(),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, input_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # Apply attention network to compute scores
        attention_scores = self.attention_network(x)
        
        # Compute attention weights using softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights
        out = x * attention_weights
        
        return out, attention_weights


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation):
        assert in_features == out_features, [in_features, out_features]
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, out_features, bias=True))
        self.layers.append(nn.Linear(in_features, out_features, bias=True))
        self.activation = activation

    def forward(self, input):
        out = input
        out = self.activation(out)
        out = self.layers[0](out)
        out = self.activation(out)
        out = self.layers[1](out)
        return input + out


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

        if False and self.training:
        # if self.training:    
            # add more randomness to latent vector during training for more diversity
            temperature = 1
            std = torch.exp(0.5 * self.latent_logvar)
            eps = temperature * torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean

class AutoEncoder(nn.Module):
    def __init__(self, col_num, dmodel):
        super(AutoEncoder, self).__init__()
        self.col_num = col_num
        self.dmodel = dmodel
    
        
        hs = [128, 64, col_num]
        self.net = []
        activation=nn.ReLU
        residual_connections = True
        for h0, h1 in zip(hs, hs[1:]):
            if residual_connections:
                if h0 == h1:
                    self.net.extend([
                        ResidualBlock(
                            h0, h1, activation=activation(inplace=False))
                    ])
                else:
                    self.net.extend([
                        nn.Linear(h0, h1),
                    ])
            else:
                self.net.extend([
                    nn.Linear(h0, h1),
                    activation(inplace=True),
                ])
        self.decoder = nn.Sequential(*self.net)
        
    def forward(self, embedding, table):
        recon_table = self.decoder(embedding)
        loss = self.loss_function_AE(recon_table, table)
        return loss, embedding

    def loss_function_AE(self, recon_x, x):
        """For autoencoder"""
        return F.mse_loss(recon_x, x)

class VAE(nn.Module):
    def __init__(self, col_num, dmodel):
        super(VAE, self).__init__()
        self.col_num = col_num
        self.dmodel = dmodel
        
        # TODO: Embedding
        dmodel = 64
        self.encoder = nn.Sequential(
            nn.Linear(col_num , 32),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(32, dmodel),
            nn.ReLU()
            # nn.Dropout(0.1)
        )
        
        # d_ff = 32
        # num_heads = 8
        # num_blocks = 4
        # d_ff = 128
        # num_heads = 16
        # num_blocks = 6
        # activation = "gelu"
        # self.encoder_block = nn.Sequential(*[
        #     Block(dmodel,
        #           d_ff,
        #           num_heads,
        #           activation,
        #           do_residual=True)
        #     for i in range(num_blocks)
        # ])
         # self.latent_dim = int((dmodel * col_num) / 4) 
        self.latent_dim = 16
        # self.decoder_block = nn.Sequential(*[
        #     Block(self.latent_dim,
        #           d_ff,
        #           num_heads,
        #           activation,
        #           do_residual=True)
        #     for i in range(num_blocks)
        # ])

        self.reparameterize = Lambda(dmodel, self.latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, col_num),
            nn.ReLU()
        )
    
    def forward(self, table):
        encode_table = self.encoder(table)
        # encode_table = self.encoder_block(encode_table)
        z = self.reparameterize(encode_table)
        # deocde_table = self.decoder_block(z)
        recon_table = self.decoder(z)
        
        
        mu = self.reparameterize.latent_mean
        logvar = self.reparameterize.latent_logvar
        # loss = self.loss_function(recon_table, table, mu, logvar)
        loss = self.loss_function_AE(recon_table, table)
        return loss, z

    def loss_function_AE(self, recon_x, x):
        """For autoencoder"""
        return F.mse_loss(recon_x, x)
    
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
    def forward(self, table, BlockSize, current_epoch, baseline):
        if len(table.shape) == 2:
            table = table.unsqueeze(0)
        rows = table.shape[1]
        # table = table.reshape(-1, rows, self.col_num * self.dmodel)
        table = table.reshape(-1, rows, self.col_num)
        
        # table = table / torch.tensor(self.input_bins)
        loss, z = self.model(table)
        # print(loss)
        # print(z.shape)
    
        scores = self.clustering(z)
        # For Fast Inference
        if not self.training:
            return scores.reshape(-1)
        
        min_vals = torch.min(scores, dim=1, keepdim=True)[0]
        max_vals = torch.max(scores, dim=1, keepdim=True)[0]
        scaled_scores = ((scores - min_vals) / (max_vals - min_vals)) * len(table)

        # original_ranks = torchsort.soft_rank(scaled_scores, regularization="l2", regularization_strength=0.01)
        # regularization_strength = (0.995)**current_epoch
        regularization_strength = 0.01
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
        return original_ranks.reshape(-1, rows, 1), rank_indices.reshape(-1, rows, 1), scores.reshape(-1, rows, 1), loss, 0
    
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
        array = table[0].detach().cpu().numpy()
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
    
    def __init__(self, BlockSize, BlockNum, col_num, dmodel, input_bins, sparse, if_pretraining, feature_stats):
        super(RankingModel_v2, self).__init__()
        self.capacity = BlockSize
        self.BlockNum = BlockNum
        self.col_num = col_num
        self.dmodel = dmodel
        self.input_bins = input_bins
        self.sparse = sparse
        self.if_pretraining = if_pretraining
        self.feature_stats = feature_stats

        self.normalization = True
        if self.normalization:
            self.mins, self.maxs = self.feature_stats
        
        if self.sparse:
            self.SparseLayer = InputDimAttention(input_dim=col_num)
        
        # Make Embedding table for each columns
        self.embed_size = 2
        NoEmbed = False
        if NoEmbed:
            self.embed_size = 0
        self.embeddings = nn.ModuleList()
        for i, dist_size in enumerate(self.input_bins):
            if dist_size <= self.embed_size or NoEmbed:
                embed = None
            else:
                embed = nn.Embedding(dist_size, self.embed_size)
            self.embeddings.append(embed)

        activation=nn.ReLU
        residual_connections = True
        
        # self.net1 = self.build_net([self.col_num] + [128, 256, 256], residual_connections, activation)
        # self.net2 = self.build_net([256, 128, 32, 1], residual_connections, activation)
        self.d_model = 4
        # self.net1 = self.build_net([self.col_num] + [128, 256, 256, 256, 256], residual_connections, activation)
        self.net2 = self.build_net([128, 128, 32, 1], residual_connections, activation)
        # self.net2 = self.build_net([col_num, 32, 128, 32, 1], residual_connections, activation)
        # self.net_k =self.build_net([1, self.d_model, self.d_model], residual_connections, activation)
        
        
        # self.MLP_1 = nn.Linear(1, d_model)
        # d_ff = 256
        # num_heads = 8
        # num_blocks = 4
        # d_ff = 64
        # num_heads = 2
        # num_blocks = 2
        # d_ff = 256
        # num_heads = 16
        # num_blocks = 6
        # d_ff = 128
        # num_heads = 2
        # num_blocks = 2
        # activation = "gelu"
        # self.blocks = nn.Sequential(*[
        #     Block(self.d_model,
        #           d_ff,
        #           num_heads,
        #           activation,
        #           do_residual=True)
        #     for i in range(num_blocks)
        # ])
        
        # Sparse Layers Block
        layer_1 = 32
        self.SparseLayer_1 = InputDimAttention(input_dim=col_num * (self.embed_size + 1))
        self.MLP_v1 = self.build_net([col_num * (self.embed_size + 1), 16, layer_1], residual_connections, activation)
        
        layer_2 = 256
        # layer_2 = 128
        self.SparseLayer_2 = InputDimAttention(input_dim=layer_1)
        self.MLP_v2 = self.build_net([layer_1, 128, layer_2], residual_connections, activation)
    
        self.SparseLayer_3 = InputDimAttention(input_dim=layer_2)
        self.MLP_v3 = self.build_net([layer_2, 256, 128, 256], residual_connections, activation)
        # self.MLP_v3 = self.build_net([layer_2, 128, 256], residual_connections, activation)
        
        self.SparseLayer_4 = InputDimAttention(input_dim=256)
        self.MLP_v4 = self.build_net([layer_2, 256, 128], residual_connections, activation)
        
        # self.blocks = nn.Sequential(*[
        #     AdaCurveBlock(col_num, col_num, activation, do_residual=True)
        #     for i in range(6)
        # ])
        
        self.AE = AutoEncoder(col_num, dmodel)
        self.ln1 = nn.LayerNorm(col_num * (self.embed_size + 1))
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(256)
        self.ln4 = nn.LayerNorm(256)
        # from diffsort import DiffSortNet
        # self.sorter = DiffSortNet('odd_even', size=100)
        
        self.apply(self._init_weights)

    def forward(self, table, BlockSize, current_epoch, baseline):
        if len(table.shape) == 2:
            table = table.unsqueeze(0)
        rows = table.shape[1]
        # table = table.reshape(-1, rows, self.col_num * self.dmodel)
        # table = table.reshape(-1, rows, self.col_num)
        
        # if max(self.input_bins) > 1200:
        #     table = torch.log(table + 1)
       
        # if self.sparse:
        #     table = self.SparseLayer(table)
        
        # table = self.net1(table)
        # scores = self.net2(z).reshape(-1, rows)
    
        # For transformer
        # table = table.reshape(-1, rows, 1)
        # table = self.net_k(table)
        # z = self.blocks(table).reshape(-1, rows, self.col_num * self.d_model)
        table_embed = self.GetEmbedding(table)
        # for i in range(self.col_num):
        #     table[:, :, i] = (table[:, :, i] - self.mins[i]) / self.maxs[i]
        table_embed = self.ln1(table_embed)
        z, att_1 = self.SparseLayer_1(table_embed)
        z = self.MLP_v1(z)
        z, att_2 = self.SparseLayer_2(z)
        z = self.MLP_v2(z)
        z = self.ln2(z)
        z, att_3 = self.SparseLayer_3(z)
        z = self.ln3(z)
        z = self.MLP_v3(z)
        z, att_4 = self.SparseLayer_4(z)
        z = self.ln4(z)
        z = self.MLP_v4(z)
        
        # z = self.blocks(table)
        # att_1 = 0
        
        # scores = self.MLP_2(z).reshape(-1, rows)
        scores = self.net2(z).reshape(-1, rows)
        
        # scores = nn.Tanh()(scores)
        if self.normalization:
            for i in range(self.col_num):
                table[:, :, i] = (table[:, :, i] - self.mins[i]) / self.maxs[i]
        loss, _ = self.AE(z, table)
        # loss = 0.0
        
        # scores = self.net(table).reshape(-1, rows)
        if self.if_pretraining:
            baseline = torch.log(baseline + 1)
            scores = scores * baseline
        # scores = nn.functional.softmax(scores, dim=1)
        
        # For Fast Inference
        if not self.training:
            return scores.reshape(-1)
            
        min_vals = torch.min(scores, dim=1, keepdim=True)[0]
        max_vals = torch.max(scores, dim=1, keepdim=True)[0]
        scaled_scores = ((scores - min_vals) / (max_vals - min_vals) + 1) * len(table)
        # scaled_scores = scores
        
        # regularization_strength = (0.995)**current_epoch
        # regularization_strength = 0.5 * (0.995)**current_epoch
        # if regularization_strength < 0.01:
        #     regularization_strength = 0.01
        regularization_strength = 0.01
        original_ranks = torchsort.soft_rank(scaled_scores, regularization="l2", regularization_strength=regularization_strength)
       
        # original_ranks, _ = self.sorter(scaled_scores)
        # print(original_ranks[0].reshape(-1))
        
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
        return original_ranks.reshape(-1, rows, 1), rank_indices.reshape(-1, rows, 1), scores.reshape(-1, rows, 1), loss, att_1
        
    def build_net(self, hs, residual_connections=True, activation=nn.ReLU):
        net = []
        for h0, h1 in zip(hs, hs[1:]):
            if residual_connections:
                if h0 == h1:
                    net.extend([
                        ResidualBlock(
                            h0, h1, activation=activation(inplace=False))
                    ])
                else:
                    net.extend([
                        nn.Linear(h0, h1),
                    ])
            else:
                net.extend([
                    nn.Linear(h0, h1),
                    nn.BatchNorm1d(h1),
                    activation(inplace=True)
                ])
        return nn.Sequential(*net)

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
    
    def GetEmbedding(self, table):
        # table shape: (batch_size, RowNum, colNum)
        # let table get embedding and concat with table
        batch_size, RowNum, colNum = table.shape 
        processed_cols = []
        for i in range(colNum):
            col = table[:, :, i]  # Shape: [batch_size, RowNum]
            if self.embeddings[i] is not None:
                # Embed the column (assuming categorical data)
                col_embed = self.embeddings[i](col.long())  # Shape: [batch_size, RowNum, embed_size]
                col = col.unsqueeze(-1)  # Add dimension for concatenation: [batch_size, RowNum, 1]
                # Concatenate original column (expanded) with its embedding
                # Transform col feature with min max
                if self.normalization:
                    col = (col - self.mins[i]) / self.maxs[i]
                col = torch.cat([col, col_embed], dim=2)  # Shape: [batch_size, RowNum, 1 + embed_size]
            else:
                # For columns without embedding, simply add a dimension to keep consistent shape
                col = col.unsqueeze(-1)  # Shape: [batch_size, RowNum, 1]
            processed_cols.append(col)# Concatenate all processed columns along the last dimension
        table_processed = torch.cat(processed_cols, dim=2)  # Shape: [batch_size, RowNum, colNum * (1 + embed_size)]
        return table_processed
        
class RankingModel_v2_(nn.Module):
    def __init__(self, BlockSize, BlockNum, col_num, dmodel, input_bins, sparse, if_pretraining):
        super(RankingModel_v2_, self).__init__()
        self.capacity = BlockSize
        self.BlockNum = BlockNum
        self.col_num = col_num
        self.dmodel = dmodel
        self.input_bins = input_bins
        self.sparse = sparse
        self.if_pretraining = if_pretraining
        
        if self.sparse:
            self.SparseLayer = InputDimAttention(input_dim=col_num)
        
        
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
        # num_heads = 8
        # num_blocks = 4
        d_ff = 128
        num_heads = 8
        num_blocks = 4
        # d_ff = 256
        # num_heads = 16
        # num_blocks = 6
        # d_ff = 128
        # num_heads = 2
        # num_blocks = 2
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
    
    
    def forward(self, table, BlockSize, current_epoch, baseline):
        rows = table.shape[1]
        # table = table.reshape(-1, rows, self.col_num * self.dmodel)
        table = table.reshape(-1, rows, self.col_num)
        # FIXME:让输入在每个维度上归一化, which one is better? (torch.tensor(self.input_bins) and Max(self.input_bins) )
        # table = table / torch.tensor(self.input_bins, device=table.device)
        # table = table / max(self.input_bins)
        # table = table / 10
        # FIMXE: Hack For larg tbale --Log transform
        if max(self.input_bins) > 1200:
            table = torch.log(table + 1)

        if self.sparse:
            table = self.SparseLayer(table)
        
        loss, z = self.model(table)
        zz = self.blocks(z)
        # print(zz.shape)
        scores = self.MLP(zz).reshape(-1, rows)
        if self.if_pretraining:
            baseline = torch.log(baseline + 1)
            scores = scores * baseline
        # scores = nn.functional.softmax(scores, dim=1)
        
        # For Fast Inference
        if not self.training:
            return scores.reshape(-1)
            
        min_vals = torch.min(scores, dim=1, keepdim=True)[0]
        max_vals = torch.max(scores, dim=1, keepdim=True)[0]
        scaled_scores = ((scores - min_vals) / (max_vals - min_vals)) * len(table)
        # scaled_scores = scores
        
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
        
        # if rows.shape[1] < 20:
        #     rows = torch.cat([rows, torch.zeros(100, 20 - rows.shape[1], dtype=torch.long)], dim=1)
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

