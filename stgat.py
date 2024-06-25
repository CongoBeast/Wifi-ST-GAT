import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn.conv import GATConv



class BasicGAT(nn.Module):
    def __init__(self, in_features, hidden_channels, num_layers, out_features, dropout=0.6):
        super(BasicGAT, self).__init__()
        self.convs = nn.ModuleList()
        
        self.dropout = torch.nn.Dropout(dropout)

        # First layer (no previous layer output)
        self.convs.append(GATConv(in_features, hidden_channels, heads=1, dropout=dropout))

        self.temporal_conv = nn.Conv1d(in_features, hidden_channels, kernel_size=3, padding=1)

        # Additional layers (use output from previous layer)
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout))
        
        self.layer_norm_out = nn.LayerNorm(hidden_channels)

        # Output layer
        self.conv_out = GATConv(hidden_channels, out_features, heads=1, dropout=dropout)
    
    def forward(self, x, edge_index):
        # Initial temporal convolution
        x_temp = x.squeeze(1).permute(0, 2, 1)  # Reshape input for 
        residual = x_temp
        # x_temp = self.temporal_conv(x_temp)
        # x_temp = torch.relu(x_temp)
        # x_temp = self.dropout(x_temp)
        x_temp = torch.relu(self.temporal_conv(x_temp) + residual)
        
        
        # Extract node features after the initial temporal convolution
        # Reshape back to (batch_size, nodes, features)
        x_gat = x_temp.permute(0, 2, 1) 
        
        
        # Flatten the batch and nodes dimensions
        x_gat = x_gat.contiguous().view(-1, x_gat.shape[-1]) 
        
        
        # Apply GAT layers
        for conv in self.convs:
            x_gat = conv(x_gat, edge_index)
            # Apply ReLU activation after each GAT layer
            x_gat = F.relu(x_gat)  
        
        # Reshape back to (batch_size, nodes, features)
        x_gat = x_gat.view(x_temp.shape[0], -1, x_gat.shape[-1])
        
        # Reshape to (batch_size, features, nodes) for final temporal convolution
        x_gat = x_gat.permute(0, 2, 1) 
        
        # Final temporal convolution
        x_final = self.temporal_conv(x_gat)
        x_final = x_final[:, :, 0]
        x_final = self.layer_norm_out(x_final)
        
        return torch.relu(x_final)
      
      
        # def forward(self, x, edge_index):

    #     x_temp = x.squeeze(1).permute(0, 2, 1)
    #     x_temp = self.temporal_conv(x_temp)

    #     x_gat = x[:, 0, 0, :]

    #     # Apply temporal convolution
    #     # x = x.unsqueeze(1)  # Add channel dimension for Conv1d
    #     # x = self.temporal_conv(x)
    #     # x = x.squeeze(1)  # Remove channel dimension after Conv1d
    #     x = F.relu(x_gat)  # Apply ReLU activation

    #     for conv in self.convs:
    #         x = conv(x, edge_index)
    #         x = F.relu(x)  # Apply ReLU activation after each GAT layer

    #     # Output layer
    #     x = self.conv_out(x, edge_index)
    #     return x