from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import os

normalizer = transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalizer])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalizer])

test_transforms = valid_transforms

transforms = {
    'train': train_transforms,
    'valid': valid_transforms,
    'test': test_transforms,
}


def get_loader(loader_type, data_directory):
    dataset = datasets.ImageFolder(os.path.join(data_directory, loader_type),
                                   transform=transforms[loader_type])
    return DataLoader(dataset, batch_size=32, shuffle=loader_type == 'train')
