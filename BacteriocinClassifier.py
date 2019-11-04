import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F


class ConvLayer(nn.Module):

    def __init__(self, in_channels, conv_filters, kernel_size, dropout, pool_size):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=conv_filters,
                              kernel_size=kernel_size)

        self.convdrop = nn.Dropout(dropout)
        self.max_pooling = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)

        x = self.convdrop(x)
        x = self.max_pooling(x)

        return x


class BacteriocinClassifier(nn.Module):
    """https://github.com/emzodls/neuripp/blob/master/models.py"""

    def __init__(self, conv_filters1=75, conv_filters2=300, dense_units=120, dropout1=0.0,
                 dropout2=0.0, kernel_sizes=(6, 8, 10), maps_per_kernel=2, pool_size=3):
        super(BacteriocinClassifier, self).__init__()
        self.dropout2 = dropout2

        in_channels = 1024  # Number of ELMo dimensions per residue
        in_features_dense1 = 69600  # Final dimensionality after convolutions, concatenation and
        # flattening

        self.convs_lower = [ConvLayer(in_channels=in_channels, conv_filters=conv_filters1,
                                      kernel_size=kernel_size, dropout=dropout1,
                                      pool_size=pool_size)
                            for kernel_size in kernel_sizes for _ in range(maps_per_kernel)]

        # Unpack to include in print statement
        (self.convs_lower11, self.convs_lower12, self.convs_lower21, self.convs_lower22,
         self.convs_lower31, self.convs_lower32) = self.convs_lower

        self.conv_upper = ConvLayer(in_channels=conv_filters1,
                                    conv_filters=conv_filters2, kernel_size=kernel_sizes[0],
                                    dropout=0, pool_size=3)

        self.dense1 = nn.Linear(in_features=in_features_dense1, out_features=dense_units)
        self.dense2 = nn.Linear(in_features=dense_units, out_features=2)

    def forward(self, x):
        x = [conv(x) for conv in self.convs_lower]
        x = torch.cat(x, dim=2)

        x = self.conv_upper(x)

        # Tensorflow has channels in last dimension, therefore we need to swap axes to use
        # tensorflow weights
        x = x.permute(0, 2, 1)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.dense1(x))

        x = F.dropout(x, p=self.dropout2)

        x = torch.sigmoid(self.dense2(x))

        return x
