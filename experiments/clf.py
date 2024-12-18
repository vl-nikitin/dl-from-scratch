import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
# sys.path.append(path.abspath(path.pardir))
print(sys.path)

from classification.models.classification import get_dataloaders, train
from classification.models.alexnet import AlexNet
from classification.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from torch.nn.functional import cross_entropy
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2 import RandomCrop, RandomHorizontalFlip
from torchinfo import summary

def alexnet_cifar10():
    normalization = [[0.491, 0.482 ,0.446], [0.247, 0.243, 0.261]]
    train_augmentations = [
        RandomCrop(224, padding=28),
        RandomHorizontalFlip(p=0.5),
    ]
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(CIFAR10, normalization=normalization, train_augmentations=train_augmentations, batch_size=32)
    alexNet = AlexNet(cross_entropy, in_channels=3, num_classes=10, learning_rate=5 * 1e-5, scheduler_gamma=0.95)
    trainer, callbacks = train(alexNet, train_dataloader, val_dataloader, max_epochs=2, name='AlexNet_CIFAR10', version='NewEnvTest')
    trainer.test(dataloaders=test_dataloader)

def resnet():
    # resNet = ResNet18(cross_entropy)
    # resNet = ResNet34(cross_entropy)
    # resNet = ResNet50(cross_entropy)
    # resNet = ResNet101(cross_entropy)
    resNet = ResNet152(cross_entropy)
    print(summary(resNet, (2, 3, 224, 224)))


def main():
    # alexnet_cifar10()
    resnet()

if __name__ == "__main__":
   main() 