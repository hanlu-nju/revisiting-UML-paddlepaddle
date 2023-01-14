# from x2paddle import torch2paddle
import paddle.nn as nn
import paddle
import paddle.nn.functional as F
from model.networks.dropblock import DropBlock
from model import init


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2D(planes)
        self.maxpool = nn.MaxPool2D(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.shape[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * self.
                                num_batches_tracked, 1.0 - self.drop_rate)
                gamma = (1 - keep_rate
                         ) / self.block_size ** 2 * feat_size ** 2 / (feat_size -
                                                                      self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class ResNet(nn.Layer):

    def __init__(self, block=BasicBlock, keep_prob=1.0, avg_pool=True,
                 drop_rate=0.1, dropblock_size=5, drop_block=True, **kwargs):
        self.inplanes = 3
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate
                                       )
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate= \
            drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate= \
            drop_rate, drop_block=drop_block, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate= \
            drop_rate, drop_block=drop_block, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob)
        self.drop_rate = drop_rate

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0,
                    drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2D(self.inplanes, planes *
                                                 block.expansion, kernel_size=1, stride=1, bias_attr=False),
                                       nn.BatchNorm2D(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))
        return x


def Res12(keep_prob=1.0, avg_pool=True, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs
                   )
    return model
