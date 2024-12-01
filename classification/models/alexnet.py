from .classification import ClassificationModel
import torch.nn

class AlexNet(ClassificationModel):
    def __init__(self, criterion, *args, in_channels=3, num_classes=10, **kwargs):
        super(AlexNet, self).__init__(criterion, *args, **kwargs)
        
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
        
        self.conv1 = torch.nn.Conv2d(in_channels, 96, kernel_size=(11, 11), stride=4, padding=2)
        self.lrn1 = torch.nn.LocalResponseNorm(5, k=2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        

        self.conv2 = torch.nn.Conv2d(96, 256, kernel_size=(5, 5), groups=2, padding=2)
        self.lrn2 = torch.nn.LocalResponseNorm(5, k=2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = torch.nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
        self.conv4 = torch.nn.Conv2d(384, 384, kernel_size=(3, 3), groups=2, padding=1)
        
        self.conv5 = torch.nn.Conv2d(384, 256, kernel_size=(3, 3), groups=2, padding=1)
        self.pool5 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.flatten = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc3 = torch.nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.lrn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.lrn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool5(x)

        x = self.flatten(x)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)

        return self.fc3(x)