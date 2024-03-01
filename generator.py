import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()  
    return model

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(requests.get(img_path, stream=True).raw)
    image = transform(image).unsqueeze(0)
    return image

def generate_adversarial_example(model, image, target_class, epsilon=0.1):
    image_var = Variable(image, requires_grad=True)
    model.zero_grad()
    output = model(image_var)
    loss = -output[0, target_class]  # Negative sign for gradient ascent
    loss.backward()
    image_data_grad = image_var.grad.data
    perturbed_image = image_var.data + epsilon * image_data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  
    return perturbed_image

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def display_images(original_image, perturbed_image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    imshow(original_image.squeeze().cpu(), title='Original Image')
    plt.subplot(1, 2, 2)
    imshow(perturbed_image.squeeze().cpu(), title='Perturbed Image')
    plt.show()



if __name__ == "__main__":
    model = load_model()

    # URL to an example image
    img_path = ''  
    image = preprocess_image(img_path)

    # Example target class (based on target)
    target_class = 123  # Example index

    perturbed_image = generate_adversarial_example(model, image, target_class)

    display_images(image, perturbed_image)

    # _, predicted = model(perturbed_image).max(1)
    # print(predicted)
