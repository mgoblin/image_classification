import torch
import torchvision
from PIL import Image
from torchvision import transforms


def check_image(path):
    try:
        Image.open(path)
        return True
    except IOError:
        return False


class Data:

    def __init__(self, path='./', batch_size=256):
        # Transform input image
        # - first all images resized to 64x64
        # - convert to tensor
        # - normalize. RGB channels have values [0 .. 1].
        #   Normalization make values new_value = (value - mean)/std.
        #   [-2.5 ... 2.5]
        self.img_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_data = torchvision.datasets.ImageFolder(
            root=path + '/train/',
            transform=self.img_transforms,
            is_valid_file=check_image
        )
        val_data = torchvision.datasets.ImageFolder(
            root=path + '/val/',
            transform=self.img_transforms,
            is_valid_file=check_image
        )
        test_data = torchvision.datasets.ImageFolder(
            root=path + '/test/',
            transform=self.img_transforms,
            is_valid_file=check_image
        )
        self.train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
        self.val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
        self.test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
