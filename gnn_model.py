import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class SimpleGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6):
        super(SimpleGNN, self).__init__()
        self.num_layers = num_layers
        
        # Dynamically create conv layers
        self.conv_layers = nn.ModuleList([
            GCNConv(in_channels if i == 0 else hidden_channels, hidden_channels)
            for i in range(num_layers)
        ])
        
        # Dynamically create batch norm layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        
        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply conv layers with residual connections
        for i in range(self.num_layers):
            identity = x
            x = self.conv_layers[i](x, edge_index)
            x = F.relu(x)
            if i > 0:  # Add residual connection after first layer
                x = x + identity
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x