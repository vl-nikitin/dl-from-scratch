from .classification import ClassificationModel
import torch.nn

class VGG(ClassificationModel):
    def __init__(self, *args, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.batch_norm = batch_norm

        self.convolutions = None
        self.classifier = None

    @staticmethod
    def conv_block(in_channels, out_channels, depth=1, batch_norm=False, last_shallow=False):
        layers = []
        for i in range(depth):
            kernel_size = 1 if last_shallow and i == depth - 1 else 3
            padding = 0 if last_shallow and i == depth - 1 else 1
            layers.append(torch.nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding))

            if batch_norm:
                layers.append(torch.nn.BatchNorm2d(out_channels))

            layers.append(torch.nn.ReLU(inplace=True))

        layers.append(torch.nn.MaxPool2d((2, 2)))

        return torch.nn.Sequential(*layers)

    @staticmethod
    def build_convolutions(blocks_params, batch_norm=False):
        blocks = []
        for params in blocks_params.values():
            in_channels, out_channels, depth, last_shallow = params
            blocks.append(VGG.conv_block(in_channels, out_channels, depth=depth, batch_norm=batch_norm))
        return torch.nn.Sequential(*blocks)

    @staticmethod
    def build_classifier(in_params, out_params):
        return torch.nn.Sequential(
            torch.nn.Linear(in_params, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, out_params)
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = torch.nn.Flatten()(x)
        x = self.classifier(x)

        return x

class VGG11(VGG):
    def __init__(self, *args, in_channels=3, num_classes=10, features=64, batch_norm=False, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)

        self.convolutions = self.build_convolutions({
            'block1': [in_channels, features, 1, False],
            'block2': [features, 2 * features, 1, False],
            'block3': [2 * features, 4 * features, 2, False],
            'block4': [4 * features, 8 * features, 2, False],
            'block5': [8 * features, 8 * features, 2, False],
        }, batch_norm=batch_norm)

        self.classifier = self.build_classifier(512 * 7 * 7, num_classes)

class VGG13(VGG):
    def __init__(self, *args, in_channels=3, num_classes=10, features=64, batch_norm=False, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)

        self.convolutions = self.build_convolutions({
            'block1': [in_channels, features, 2, False],
            'block2': [features, 2 * features, 2, False],
            'block3': [2 * features, 4 * features, 2, False],
            'block4': [4 * features, 8 * features, 2, False],
            'block5': [8 * features, 8 * features, 2, False],
        })

        self.classifier = self.build_classifier(512 * 7 * 7, num_classes)

class VGG16Light(VGG):
    def __init__(self, *args, in_channels=3, num_classes=10, features=64, batch_norm=False, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)

        self.convolutions = self.build_convolutions({
            'block1': [in_channels, features, 2, False],
            'block2': [features, 2 * features, 2, False],
            'block3': [2 * features, 4 * features, 3, True],
            'block4': [4 * features, 8 * features, 3, True],
            'block5': [8 * features, 8 * features, 3, True],
        })

        self.classifier = self.build_classifier(512 * 7 * 7, num_classes)

class VGG16(VGG):
    def __init__(self, *args, in_channels=3, num_classes=10, features=64, batch_norm=False, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)

        self.convolutions = self.build_convolutions({
            'block1': [in_channels, features, 2, False],
            'block2': [features, 2 * features, 2, False],
            'block3': [2 * features, 4 * features, 3, False],
            'block4': [4 * features, 8 * features, 3, False],
            'block5': [8 * features, 8 * features, 3, False],
        })

        self.classifier = self.build_classifier(512 * 7 * 7, num_classes)

class VGG19(VGG):
    def __init__(self, *args, in_channels=3, num_classes=10, features=64, batch_norm=False, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)

        self.convolutions = self.build_convolutions({
            'block1': [in_channels, features, 2, False],
            'block2': [features, 2 * features, 2, False],
            'block3': [2 * features, 4 * features, 4, False],
            'block4': [4 * features, 8 * features, 4, False],
            'block5': [8 * features, 8 * features, 4, False],
        })

        self.classifier = self.build_classifier(512 * 7 * 7, num_classes)