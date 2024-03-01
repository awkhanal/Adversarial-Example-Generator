# Adversarial Example Generator

This project demonstrates how to generate adversarial examples using PyTorch. It targets a pretrained ResNet-50 model to misclassify images by applying carefully crafted perturbations. The core idea is to slightly modify an input image so that it is still recognizable to humans but gets misclassified by the model.

## Usage

To use this project, follow these steps:

- **Prepare Your Image**: Place the image you want to test in a known directory or have the URL ready if you wish to download it programmatically.
- **Modify the Script**: Open the main script and replace the `img_path` variable with the path or URL to your target image. If necessary, adjust the `target_class` to the class you want the model to misclassify the image as.
- **Run the Script**: Execute the main script to generate the adversarial example and see the results.

The script will display the original and perturbed images side by side, and print the model's prediction for the perturbed image.

## How It Works

The script performs the following steps:

- **Model Loading**: Loads a pretrained ResNet-50 model from PyTorch's model zoo.
- **Image Preprocessing**: Downloads and preprocesses the image to match the input format expected by ResNet-50.
- **Adversarial Example Generation**: Applies gradient ascent to the image to maximize the loss for a chosen target class, effectively perturbing the image.
- **Visualization**: Displays the original and perturbed images, highlighting the effectiveness of the perturbation.
- **Classification**: Classifies the perturbed image using the model to verify if the adversarial attack was successful.
