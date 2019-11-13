import torch.hub
import torch.utils.data
from PIL import Image
from torchvision import transforms
from torchvision import models

if __name__ == '__main__':
    print(dir(models))

    cnnnet = torch.hub.load('pytorch/vision:v0.4.2', 'mobilenet_v2', pretrained=True)
    cnnnet.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img1 = Image.open("./input/Suit_Jacket_Grey_A_1024x1024.jpg")

    with open('./imagenet1000_clsidx_to_labels.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    input_tensor = preprocess(img1)
    input_batch = input_tensor.unsqueeze(0)

    output = cnnnet(input_batch)

    _, index = torch.max(output, 1)
    percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100

    print(classes[index[0].item()], percentage[index[0]].item())