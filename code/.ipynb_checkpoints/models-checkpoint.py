import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GAE, VGAE, GATConv, AGNNConv
from torch.nn import functional as F

class GCNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNet, self).__init__()

        self.conv1 = GCNConv(in_channels, 2 * in_channels)  # , cached=True)
        self.conv2 = GCNConv(2 * in_channels, out_channels)  # data.num_classes)#, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index))  # , edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)  # , edge_weight)
        return F.log_softmax(x, dim=1)


class ChebyNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChebyNet, self).__init__()

        #         self.conv1 = GCNConv(in_channels, 64)#, cached=True)
        #         self.conv2 = GCNConv(64, out_channels=num_classes)#data.num_classes)#, cached=True)
        self.conv1 = ChebConv(in_channels, 64, K=2)
        self.conv2 = ChebConv(64, out_channels, K=2)

    def forward(self, data, use_edge_weight=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        if use_edge_weight:
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)



class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, 8 , heads=8, dropout=0.6)
        self.conv2 = GATConv( 8 * 8, out_channels, heads=1, concat=True, dropout=0.6)

    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


class AGNNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AGNNet, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, 64)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(64, out_channels)

    def forward(self, data):
        x = F.dropout(data.x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, data.edge_index)
        x = self.prop2(x, data.edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class GAEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
#         data = self.split_edges(data)
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index))#, edge_weight))
        return self.conv2(x, edge_index)#, edge_weight)

class baseline_GAEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAEncoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, 2 * in_channels, cached=True)
        self.fc2 = nn.Linear(2 * out_channels, in_channels, cached=True)

    def forward(self, x, edge_index):
#         data = self.split_edges(data)
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.fc1(x))#, edge_weight))
        return self.fc2(x)#, edge_weight)



class VGAEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGAEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logvar = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        mu, var = self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)
        return mu, var


class baseline_VGAEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(baseline_VGAEncoder, self).__init__()
        self.lin = nn.Linear(in_channels, 2 * in_channels)
        self.lin_mu = nn.Linear(2 * in_channels, out_channels)
        self.lin_logvar = nn.Linear(2 * in_channels, out_channels)
#         self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
#         self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
#         self.conv_logvar = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x):
        x = F.relu(self.lin(x))
        mu, var = self.lin_mu(x), self.lin_logvar(x, edge_index)
        return self.lin2(x)


class linear_baseline(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(linear_baseline, self).__init__()

        self.linear1 = nn.Linear(in_channel, 64)
        self.linear2 = nn.Linear(64, out_channel)

    def forward(self, data):
        x = data.x
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x)