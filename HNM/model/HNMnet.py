import torch.nn as nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter


class HNM(nn.Module):
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5):
        super(HNM, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes,
                               (3, 3, 3), padding=(1, 0, 0))
        self.conv3 = nn.Conv3d( 2 * n_planes, n_classes, (3, 1, 1), padding=(1, 0, 0))
        self.features_size = self._get_final_flattened_size()
        self.gcn = Scalable_gcn(input_dim=self.features_size, N_class=n_classes)
        # self.fc = nn.Linear(self.features_size, n_classes)
        
        self.apply(self.weight_init)
        
    def _get_final_flattened_size(self): 
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.features_size)
        x = self.gcn(x)
        return x

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, weight):

        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = weight

    def forward(self, adjacency, input_feature):

        support = torch.mm(input_feature, self.weight)
        output = torch.mm(adjacency, support)
        
        return output


class Scalable_gcn(nn.Module):
    def __init__(self, input_dim, N_class):
        super().__init__()

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(N_class) 
        self.weight1 = Parameter(torch.FloatTensor(input_dim, 128))
        self.weight2 = Parameter(torch.FloatTensor(128, N_class))
        self.gcn1 = GraphConvolution(input_dim, 128,self.weight1)
        self.gcn2 = GraphConvolution(128, N_class,self.weight2)
        self.reset_parameters()
        self.N_class = N_class

    def forward(self,features):
        
        A = self.AdjacencyCompute(features).cuda()
        x = self.gcn1(A, features)
        x = F.tanh(self.bn1(x))
        x = self.gcn2(A, x)
        x = F.tanh(self.bn2(x)) 

        return x

    def reset_parameters(self):

        nn.init.xavier_normal_(self.weight1)
        nn.init.xavier_normal_(self.weight2)

    def AdjacencyCompute(self,features):  

        N = features.size(0) 
        sigma = 10
        temp = torch.norm(features.repeat(N, 1) - features.repeat(1, N).view(N * N, -1), dim=1) 
        adjacency_e = torch.exp(-temp.pow(2) / (sigma)).view(N, N) # [100,100]
        _, position = torch.topk(adjacency_e, round(N / (self.N_class)), dim=1, sorted=False, out=None)
        adjacency0 = torch.zeros(N, N).cuda()

        for num in range(N):     
            
            adjacency0[num, position[num,:]] = 0.9
            adjacency0[num,num] = 0.5

        adjacency_e = torch.mul(adjacency0,adjacency_e) 
        adjacency = torch.eye(N).cuda() + adjacency_e
        d = torch.sum(adjacency,dim=1)
        d = d + 1
        d = torch.sqrt(d) 
        D = torch.diag(d) 
        inv_D = torch.inverse(D)
        adjacency = torch.mm(torch.mm(inv_D, adjacency),inv_D) 
        
        return adjacency