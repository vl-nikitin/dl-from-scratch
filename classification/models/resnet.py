from collections import OrderedDict
from torch.nn import Module, Conv2d, BatchNorm2d
from torch.nn.functional import relu

class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        needs_reduction = in_channels != out_channels
        if needs_reduction:
            print(in_channels, out_channels)
            assert 2 * in_channels == out_channels
        stride = 1 if not needs_reduction else 2
        
        self.layer1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        self.layer2 = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(out_channels)

        self.sc = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        
    def forward(self, x):
        output = self.layer1(x)
        output = self.bn1(output)
        output = relu(output)
        output = self.layer2(output)
        skip_connection = self.sc(x)
        output += skip_connection
        output = self.bn2(output)
        return relu(output)

class ResidualBottleneckBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBottleneckBlock, self).__init__()

        bottleneck_channels = out_channels // 2
        needs_reduction = in_channels != bottleneck_channels
        if needs_reduction:
            print(in_channels, bottleneck_channels)
            assert 2 * in_channels == bottleneck_channels
        stride = 1 if not needs_reduction else 2

        self.layer1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.bn1 = BatchNorm2d(bottleneck_channels)
        self.layer2 = Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = BatchNorm2d(bottleneck_channels)
        self.layer3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1)
        self.bn3 = BatchNorm2d(out_channels)

        self.sc = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        
    def forward(self, x):
        output = self.layer1(x)
        output = self.bn1(output)
        output = relu(output)
        output = self.layer2(output)
        output = self.bn2(output)
        output = relu(output)
        output = self.layer3(output)
        skip_connection = self.sc(x)
        output += skip_connection
        output = self.bn3(output)

        return output
        
class ResNet(ClassificationModel):
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)

        self.convolutions = None
        self.pool = None
        self.classifier = None

    @staticmethod
    def build_convolutions(in_channels, first_layer_out_channels, blocksets_params):
        blocksets = [
            ('conv1', Conv2d(in_channels, first_layer_out_channels, kernel_size=7, stride=2, padding=3)),
            ('pool1', torch.nn.MaxPool2d(kernel_size=2))
        ]
        for blockset_name in blocksets_params:
            blockset = ResNet.__build_blockset(blocksets_params[blockset_name])
            blocksets.append((blockset_name, blockset))
        return torch.nn.Sequential(OrderedDict(blocksets))

    @staticmethod
    def build_classifier(in_params, out_params):
        return torch.nn.Linear(in_params, out_params)

    @staticmethod
    def __build_blockset(params):
        blocks = []
        num_blocks, in_channels, out_channels, is_bottleneck = params
        for i in range(num_blocks):
            block = None
            block_in_channels = in_channels if i == 0 else out_channels
            if not is_bottleneck:
                block = ResidualBlock(in_channels=block_in_channels, out_channels=out_channels)
            else:
                block = ResidualBottleneckBlock(in_channels=block_in_channels, out_channels=out_channels)
            blocks.append(block)

        return torch.nn.Sequential(*blocks)
        
    def forward(self, x):
        x = self.convolutions(x)
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = torch.nn.Flatten()(x)
        x = self.classifier(x)

        return x

class ResNet18(ResNet):
    def __init__(self, *args, in_channels=3, num_classes=10, features=64, **kwargs):
        super(ResNet18, self).__init__(*args, **kwargs)

        self.convolutions = self.build_convolutions(in_channels, features, {
            'conv2': [2, features, features, False],
            'conv3': [2, features, 2 * features, False],
            'conv4': [2, 2 * features, 4 * features, False],
            'conv5': [2, 4 * features, 8 * features, False],
        })

        self.pool = torch.nn.AvgPool2d(kernel_size=7)

        self.classifier = self.build_classifier(8 * features, num_classes)