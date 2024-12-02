from classification.models.classification import ClassificationModel
from collections import OrderedDict
from torch.nn import Module, Conv2d, BatchNorm2d, Sequential, Linear, MaxPool2d, AvgPool2d, Flatten, Identity
from torch.nn.functional import relu
from enum import Enum

class SkipConnectionType(Enum):
    A = 0
    B = 1
    C = 2

class ResidualBlockBase(Module):
    def __init__(self):
        super(ResidualBlockBase, self).__init__()

    @staticmethod
    def build_skip_connection(in_channels, out_channels, stride, skip_connection_type):
        assert isinstance(skip_connection_type, SkipConnectionType)

        match skip_connection_type:
            case SkipConnectionType.A:
                raise NotImplementedError()
            case SkipConnectionType.B:
                return Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else Identity()
            case SkipConnectionType.C:
                return Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            case _:
                raise RuntimeError("Skip connection type must be one of 'A', 'B' or 'C'")


class ResidualBlock(ResidualBlockBase):
    def __init__(self, in_channels, out_channels, skip_connection_type=SkipConnectionType.B):
        super(ResidualBlock, self).__init__()

        needs_reduction = in_channels != out_channels
        if needs_reduction:
            assert 2 * in_channels == out_channels

        stride = 2 if needs_reduction else 1
        
        self.layer1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        self.layer2 = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(out_channels)

        self.sc = self.build_skip_connection(in_channels, out_channels, stride, skip_connection_type)

    def forward(self, x):
        output = self.layer1(x)
        output = self.bn1(output)
        output = relu(output)
        output = self.layer2(output)
        skip_connection = self.sc(x)
        output += skip_connection
        output = self.bn2(output)
        return relu(output)

class ResidualBottleneckBlock(ResidualBlockBase):
    def __init__(self, in_channels, out_channels, skip_connection_type=SkipConnectionType.B):
        super(ResidualBottleneckBlock, self).__init__()

        bottleneck_channels = out_channels // 4
        needs_reduction = in_channels == 2 * bottleneck_channels
        if needs_reduction:
            assert 2 * in_channels == out_channels

        stride = 2 if needs_reduction else 1

        self.layer1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.bn1 = BatchNorm2d(bottleneck_channels)
        self.layer2 = Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = BatchNorm2d(bottleneck_channels)
        self.layer3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1)
        self.bn3 = BatchNorm2d(out_channels)

        self.sc = self.build_skip_connection(in_channels, out_channels, stride, skip_connection_type)
        
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
            ('pool1', MaxPool2d(kernel_size=2))
        ]
        for blockset_name in blocksets_params:
            blockset = ResNet.__build_blockset(**blocksets_params[blockset_name])
            blocksets.append((blockset_name, blockset))
        return Sequential(OrderedDict(blocksets))

    @staticmethod
    def build_classifier(in_params, out_params):
        return Linear(in_params, out_params)

    @staticmethod
    def __build_blockset(num_blocks=2, in_channels=64, out_channels=64, is_bottleneck=False):
        blocks = []
        for i in range(num_blocks):
            block = None
            block_in_channels = in_channels if i == 0 else out_channels
            if not is_bottleneck:
                block = ResidualBlock(in_channels=block_in_channels, out_channels=out_channels)
            else:
                block = ResidualBottleneckBlock(in_channels=block_in_channels, out_channels=out_channels)
            blocks.append(block)

        return Sequential(*blocks)
        
    def forward(self, x):
        x = self.convolutions(x)
        x = self.pool(x)
        x = Flatten()(x)
        x = self.classifier(x)

        return x

class ResNet18(ResNet):
    def __init__(self, *args, in_channels=3, num_classes=10, features=64, **kwargs):
        super(ResNet18, self).__init__(*args, **kwargs)

        self.convolutions = self.build_convolutions(in_channels, features, {
            'conv2': {'num_blocks': 2, 'in_channels': features, 'out_channels': features, 'is_bottleneck': False},
            'conv3': {'num_blocks': 2, 'in_channels': features, 'out_channels': 2 * features, 'is_bottleneck': False},
            'conv4': {'num_blocks': 2, 'in_channels': 2 * features, 'out_channels': 4 * features, 'is_bottleneck': False},
            'conv5': {'num_blocks': 2, 'in_channels': 4 * features, 'out_channels': 8 * features, 'is_bottleneck': False},
        })

        self.pool = AvgPool2d(kernel_size=7)

        self.classifier = self.build_classifier(8 * features, num_classes)

class ResNet34(ResNet):
    def __init__(self, *args, in_channels=3, num_classes=10, features=64, **kwargs):
        super(ResNet34, self).__init__(*args, **kwargs)

        self.convolutions = self.build_convolutions(in_channels, features, {
            'conv2': {'num_blocks': 3, 'in_channels': features, 'out_channels': features, 'is_bottleneck': False},
            'conv3': {'num_blocks': 4, 'in_channels': features, 'out_channels': 2 * features, 'is_bottleneck': False},
            'conv4': {'num_blocks': 6, 'in_channels': 2 * features, 'out_channels': 4 * features, 'is_bottleneck': False},
            'conv5': {'num_blocks': 3, 'in_channels': 4 * features, 'out_channels': 8 * features, 'is_bottleneck': False},
        })

        self.pool = AvgPool2d(kernel_size=7)

        self.classifier = self.build_classifier(8 * features, num_classes)

class ResNet50(ResNet):
    def __init__(self, *args, in_channels=3, num_classes=10, features=64, **kwargs):
        super(ResNet50, self).__init__(*args, **kwargs)

        self.convolutions = self.build_convolutions(in_channels, features, {
            'conv2': {'num_blocks': 3, 'in_channels': features, 'out_channels': 4 * features, 'is_bottleneck': True},
            'conv3': {'num_blocks': 4, 'in_channels': 4 * features, 'out_channels': 8 * features, 'is_bottleneck': True},
            'conv4': {'num_blocks': 6, 'in_channels': 8 * features, 'out_channels': 16 * features, 'is_bottleneck': True},
            'conv5': {'num_blocks': 3, 'in_channels': 16 * features, 'out_channels': 32 * features, 'is_bottleneck': True},
        })

        self.pool = AvgPool2d(kernel_size=7)

        self.classifier = self.build_classifier(32 * features, num_classes)

class ResNet101(ResNet):
    def __init__(self, *args, in_channels=3, num_classes=10, features=64, **kwargs):
        super(ResNet101, self).__init__(*args, **kwargs)

        self.convolutions = self.build_convolutions(in_channels, features, {
            'conv2': {'num_blocks': 3, 'in_channels': features, 'out_channels': 4 * features, 'is_bottleneck': True},
            'conv3': {'num_blocks': 4, 'in_channels': 4 * features, 'out_channels': 8 * features, 'is_bottleneck': True},
            'conv4': {'num_blocks': 23, 'in_channels': 8 * features, 'out_channels': 16 * features, 'is_bottleneck': True},
            'conv5': {'num_blocks': 3, 'in_channels': 16 * features, 'out_channels': 32 * features, 'is_bottleneck': True},
        })

        self.pool = AvgPool2d(kernel_size=7)

        self.classifier = self.build_classifier(32 * features, num_classes)

class ResNet152(ResNet):
    def __init__(self, *args, in_channels=3, num_classes=10, features=64, **kwargs):
        super(ResNet152, self).__init__(*args, **kwargs)

        self.convolutions = self.build_convolutions(in_channels, features, {
            'conv2': {'num_blocks': 3, 'in_channels': features, 'out_channels': 4 * features, 'is_bottleneck': True},
            'conv3': {'num_blocks': 8, 'in_channels': 4 * features, 'out_channels': 8 * features, 'is_bottleneck': True},
            'conv4': {'num_blocks': 36, 'in_channels': 8 * features, 'out_channels': 16 * features, 'is_bottleneck': True},
            'conv5': {'num_blocks': 3, 'in_channels': 16 * features, 'out_channels': 32 * features, 'is_bottleneck': True},
        })

        self.pool = AvgPool2d(kernel_size=7)

        self.classifier = self.build_classifier(32 * features, num_classes)