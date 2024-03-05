# Adversarial Example Generator

Target a pretrained ResNet-50 model to misclassify images by applying carefully crafted perturbations. The core idea is to slightly modify an input image so that it is still recognizable to humans but gets misclassified by the model.

## How It Works

The script performs the following steps:

- **Model Loading**: Loads a pretrained ResNet-50 model from PyTorch's model zoo.
- **Image Preprocessing**: Downloads and preprocesses the image to match the input format expected by ResNet-50.
- **Adversarial Example Generation**: Applies gradient ascent to the image to maximize the loss for a chosen target class, effectively perturbing the image. (Fast Gradient Sign Method)
- **Visualization**: Displays the original and perturbed images, highlighting the effectiveness of the perturbation.
- **Classification**: Classifies the perturbed image using the model to verify if the adversarial attack was successful.
