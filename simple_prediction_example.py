import os

import torch
import torch.nn.functional as f
import torch.optim as optim
import torch.utils.data
from PIL import Image

import simple_net as net
import data


def train(model, optimizer, loss_fn, data_holder, epochs=8, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in data_holder.train_data_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(data.train_data_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in data_holder.val_data_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(f.softmax(output), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(data.val_data_loader.dataset)

        print(
            'Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.
                format(epoch, training_loss,
                       valid_loss,
                       num_correct / num_examples))


if __name__ == '__main__':

    simple_net = net.SimpleNet()
    data = data.Data('./')
    if not os.path.exists("./trained_model/simplenet.trained"):
        optimizer = optim.Adam(simple_net.parameters(), lr=0.001)
        train(simple_net, optimizer, torch.nn.CrossEntropyLoss(), data)
        torch.save(simple_net, "./trained_model/simplenet.trained")
    else:
        torch.load("./trained_model/simplenet.trained")

    labels = ['cat', 'fish']
    img1 = Image.open("./input/lyubopitnii-polosatii-kotenok.orig.jpg")
    img1 = data.img_transforms(img1).to(torch.device("cpu"))

    prediction1_tensor = simple_net(img1)
    prediction1 = f.softmax(prediction1_tensor, dim=1)
    prediction1 = prediction1.argmax()

    print(labels[prediction1])
