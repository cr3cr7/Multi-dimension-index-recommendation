import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchsort

# bin_boundaries = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
# scores = torch.randn(10, requires_grad=True)
# print(scores)
# bin_indices = torch.bucketize(scores, bin_boundaries)
# print(bin_indices)
# x = sum(scores + (bin_indices - scores).detach())
# print(x)
# x.backward()


# # Define the input tensor
# x = torch.tensor([1., 2., 3., 2., 1.], requires_grad=True)

# # Compute the unique values using torch.unique()
# unique_values = torch.unique(x)

# # Compute the sorted values using torch.sort()
# sorted_values, indices = torch.sort(x)

# # Compute the one-hot tensor corresponding to the sorted values
# one_hot = torch.zeros(len(x), len(unique_values), dtype=torch.float32)
# one_hot.scatter_(1, (sorted_values == unique_values.unsqueeze(0)).nonzero()[:, 1:], 1)

# # Compute the STE gradient
# grad = one_hot[indices]

# # Print the resulting gradient
# print(grad)


# class LearnedSortOrder(nn.Module):
#     def __init__(self, n):
#         super(LearnedSortOrder, self).__init__()
#         self.n = n
#         self.perm = nn.Parameter(torch.randperm(n), requires_grad=False)

#     def forward(self, x):
#         return x[:, self.perm]

#     def inverse(self, x):
#         return x[:, self.perm.inverse()]

#     def get_perm(self):
#         return self.perm

#     def set_perm(self, perm):
#         self.perm = nn.Parameter(perm, requires_grad=False)

#     def get_perm_inv(self):
#         return self.perm.inverse()

#     def set_perm_inv(self, perm):
#         self.perm = nn.Parameter(perm.inverse(), requires_grad=False)

# x = self.linear1(x)        
# min, max = int(x.min()), int(x.max())        
# bins = torch.linspace(min, max+1, 16)
# x_buckets = torch.bucketize(x.detach(), bins) # forced to detach here
# x = x + (x - x_buckets).detach() # Reintroduce gradients here. <------------
# x = self.linear2(x)

class LearnedSortOrder(nn.Module):
    def __init__(self, BinNum):
        super(LearnedSortOrder, self).__init__()
        self.BinNum = BinNum
        self.MLP = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, self.BinNum),
            nn.ReLU()
        )
        self.capcity = 3
    
    def forward(self, x):
        # (batch_size, BinNum)
        p1s = []
        p2s = []
        count = torch.zeros((1, self.BinNum))
        for one_batch in x:
            mask = self.update_mask(count)
            logits = self.MLP(one_batch) 
            # mask invalid bins
            logits = logits + (1 - mask) * -1e9
            # Sample soft categorical using reparametrization trick:
            p1 = F.gumbel_softmax(logits, tau=1, hard=False)
            # Sample hard categorical using "Straight-through" trick:
            p2 = F.gumbel_softmax(logits, tau=1, hard=True)
            p1s.append(p1)
            p2s.append(p2)
            
            count = count + p2
            
            
        return torch.cat(p1s, dim=0), torch.cat(p2s, dim=0)

    def update_mask(self, count):
        mask = torch.where(count >= self.capcity, torch.zeros_like(count), torch.ones_like(count))
        return mask
    
class LearnedSortOrder_v2(nn.Module):
    def __init__(self, BinNum):
        super(LearnedSortOrder_v2, self).__init__()
        self.BinNum = BinNum
        self.MLP = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )
        self.capcity = 3
    
    def forward(self, x):
        # (batch_size, 1)
        scores = self.MLP(x) 
        scores_min, scores_max = int(scores.min()), int(scores.max())  
        bin_boundaries = torch.linspace(scores_min, scores_max+1, self.BinNum)
        bin_indices = torch.bucketize(scores, bin_boundaries)
        import torch.nn.functional as F

        # bin_indices: (batch_size, BinNum)
        one_hot = F.one_hot(bin_indices, num_classes=self.BinNum).float()
        # x = sum(scores + (bin_indices - scores).detach())
        # print(x)
        return one_hot, one_hot
        
class LearnedSortOrder_v3(nn.Module):
    def __init__(self, BinNum):
        super(LearnedSortOrder_v3, self).__init__()
        self.BinNum = BinNum
        self.MLP = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.capcity = 3
        
    def forward(self, x):
        scores = self.MLP(x)
        # print(scores)
        ranks = torchsort.soft_rank(scores.reshape(1, -1), regularization="kl")
        print(scores.reshape(1, -1))
        print(ranks)
        assert 0 
        # print(ranks % self.capcity)
        # print(((ranks) - (ranks % self.capcity))/ self.capcity)
        other = ranks.detach() % self.capcity
        ranks = (ranks - other) / self.capcity
        ranks = ranks + 1
        return None, ranks.reshape(-1, 1)

    
def SumRudBins(y, mask):
    """
    y: (batch_size, BinNum)
    mask: (batch_size, 1)
    """
    results = []
    for cur_mask in mask:
        print(y)
        result = y * cur_mask.reshape(y.shape[0], 1)
        result = torch.sum(result, dim=0)
        print(result)
        devided = torch.where(result == 0, torch.ones_like(result), result)
        print(devided)
        
        result = result / devided
        
        print(result)
        assert 0 
        results.append(result)
    return sum(results)

def SumRudBins_v2(y, mask):
    results = []
    for cur_mask in mask:
        loss = 0
        result = y * cur_mask.reshape(y.shape[0], 1)
        # print(result)
        # result = torch.tensor([1,3,1])
        loss = sum(result)
        # for i in range(1, 5):
        #     loss = loss + find_unique(result, i)
        # print(loss)
        
        # devided = torch.where(result == 0, torch.ones_like(result), result)
        # print(devided)
        # result = result / devided
        # print(result)
        results.append(loss)
    return sum(results)

def find_unique(rank, unique_id):
    print(unique_id)
    print(rank)
    filter = torch.where(rank == unique_id, torch.ones_like(rank), torch.zeros_like(rank))
    print(filter)
    assert 0
    if torch.any(filter):
        rank = rank * filter
        return sum(rank) / sum(rank)
    else:
        return 0

def training_step(model, train_data, array):
    """
    model: LearnedSortOrder
    train_data: (batch_size, input_size)
    mask: (batch_size, 1)
    """
    # array = GenRandomArray(10).reshape(-1, 1)
    p1, p2 = model(array)
    loss = sum(SumRudBins_v2(p2, train_data))
    loss.backward()
    # print(next(model.parameters()).grad)
    # assert 0
    return loss

def training_loop(model, train_data, array, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        loss = training_step(model, train_data, array)
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 10 == 0:
            print('Epoch {}, Loss {}'.format(epoch, loss.item()))
    return loss


def GenRandomArray(length):
    """
    Generate a random array of length n with values in [0, n)
    """
    return torch.randperm(length, dtype=torch.float)

def GenTrainData(input_size, num_hot, batch_size):
    """
    Generate training data, which is a batch of random one-hot arrays
    """
    # Generate random indices for each multi-hot vector
    indices = torch.randint(low=0, high=input_size, size=(batch_size, num_hot.max()))

    # Create a tensor of zeros with shape (batch_size, input_size)
    x = torch.zeros((batch_size, input_size))

    # Set the values at the random indices to 1
    for i in range(batch_size):
        x[i, indices[i, :num_hot[i]]] = 1

    # Print the resulting tensor
    return x



def seed_everything(seed=42):
    """
    Seed everything for reproducibility
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    seed_everything()
    
    num_hot = np.random.randint(1, 10, 100)
    train_data = GenTrainData(10, num_hot, 100)
    print(train_data)
    
    array = GenRandomArray(10).reshape(-1, 1)
    print(array)
    
    model = LearnedSortOrder_v3(4)
    print(model(array))
    
    # [0., 0., 0., 1.],  
    # [0., 0., 1., 0.],
    # [0., 1., 0., 0.],
    # [1., 0., 0., 0.],
    # [1., 0., 0., 0.],
    # [0., 0., 1., 0.],
    # [0., 0., 0., 1.],
    # [0., 1., 0., 0.],
    # [0., 0., 1., 0.]
    
    # [1., 0., 0., 0.],
    # [0., 0., 0., 1.],
    # [0., 0., 0., 1.],
    # [1., 0., 0., 0.],
    # [0., 0., 0., 1.],
    # [0., 1., 0., 0.],
    # [1., 0., 0., 0.],
    # [0., 0., 1., 0.],
    # [0., 0., 1., 0.]
    
    training_loop(model, train_data, array, epochs=800, lr=1e-3)
    
    print(model(array))
    
    # [1., 0., 0., 0.],
    # [1., 0., 0., 0.],
    # [1., 0., 0., 0.],
    # [1., 0., 0., 0.],
    # [1., 0., 0., 0.],
    # [1., 0., 0., 0.],
    # [1., 0., 0., 0.],
    # [1., 0., 0., 0.],
    # [1., 0., 0., 0.]
    
    # [0., 0., 0., 1.],
    # [0., 0., 0., 1.],
    # [1., 0., 0., 0.],
    # [1., 0., 0., 0.],
    # [1., 0., 0., 0.],
    # [0., 0., 1., 0.],
    # [0., 0., 1., 0.],
    # [0., 0., 1., 0.],
    # [0., 1., 0., 0.]]
    