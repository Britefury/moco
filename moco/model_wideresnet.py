import torch, torch.nn as nn, torch.nn.functional as F

class WideResNetBlock (nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0, activate_before_residual=False):
        super(WideResNetBlock, self).__init__()

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.activate_before_residual = activate_before_residual

        self.channels = out_channels

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)
        if dropout_rate > 0.0:
            self.drop = nn.Dropout(dropout_rate)
        else:
            self.drop = None
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels or stride != 1:
            self.proj_conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0)
        else:
            self.proj_conv = None

    def forward(self, x):
        y = x
        residual = x

        # Conv 1
        y = self.lrelu(self.bn1(y))
        if self.activate_before_residual:
            residual = y
        y = self.conv1(y)

        # Conv 2
        if self.drop is not None:
            y = self.conv2(self.drop(self.lrelu(self.bn2(y))))
        else:
            y = self.conv2(self.lrelu(self.bn2(y)))

        if self.proj_conv is not None:
            residual = self.proj_conv(residual)

        return y + residual


class WideResNetGroup (nn.Module):
    def __init__(self, in_channels, out_channels, blocks_per_group, stride=1, dropout_rate=0.0,
                 activate_before_residual=False):
        super(WideResNetGroup, self).__init__()

        blocks = []
        blocks.append(WideResNetBlock(in_channels, out_channels, stride=stride,
                                      dropout_rate=dropout_rate, activate_before_residual=activate_before_residual))
        for i in range(1, blocks_per_group):
            blocks.append(WideResNetBlock(out_channels, out_channels, dropout_rate=dropout_rate))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class WideResNet (nn.Module):
    def __init__(self, num_outputs, blocks_per_group, channel_multiplier, dropout_rate=0.0):
        super(WideResNet, self).__init__()

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)

        self.group1 = WideResNetGroup(16, 16 * channel_multiplier,
                                      blocks_per_group, dropout_rate=dropout_rate, activate_before_residual=True)
        self.group2 = WideResNetGroup(16 * channel_multiplier, 32 * channel_multiplier,
                                      blocks_per_group, stride=2, dropout_rate=dropout_rate)
        self.group3 = WideResNetGroup(32 * channel_multiplier, 64 * channel_multiplier,
                                      blocks_per_group, stride=2, dropout_rate=dropout_rate)
        self.end_bn = nn.BatchNorm2d(64 * channel_multiplier)
        self.fc = nn.Linear(64 * channel_multiplier, num_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.lrelu(self.end_bn(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(len(x), -1)
        return self.fc(x)


def wrn28_2(num_classes):
    return WideResNet(num_classes, 4, 2)

