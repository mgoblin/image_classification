import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image


class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def check_image(path):
    try:
        Image.open(path)
        return True
    except:
        return False


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print(
            'Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.
                format(epoch, training_loss,
                       valid_loss,
                       num_correct / num_examples))


img_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_data_path = "./train/"
val_data_path = "./val/"
test_data_path = "./test/"

train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=img_transforms, is_valid_file=check_image)
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=img_transforms, is_valid_file=check_image)
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=img_transforms, is_valid_file=check_image)

batch_size = 128
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

simplenet = SimpleNet()

optimizer = optim.Adam(simplenet.parameters(), lr=0.001)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

simplenet.to(device)

train(simplenet, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader, epochs=6, device=device)

labels = ['cat', 'fish']

img1 = Image.open("./input/animals-fish-66401.jpg")
img2 = Image.open("./input/DDTzb7oWAAAiuFh.jpg")
img3 = Image.open("./input/zhivotnye-koty-454639.jpg")

img1 = img_transforms(img1).to(device)
img2 = img_transforms(img2).to(device)
img3 = img_transforms(img3).to(device)

prediction1 = F.softmax(simplenet(img1))
prediction1 = prediction1.argmax()

prediction2 = F.softmax(simplenet(img2))
prediction2 = prediction2.argmax()

prediction3 = F.softmax(simplenet(img3))
prediction3 = prediction3.argmax()


print(labels[prediction1])
print(labels[prediction2])
print(labels[prediction3])

torch.save(simplenet, "./trained_model/simplenet.trained")
