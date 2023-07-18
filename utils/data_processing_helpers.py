import numpy as np
import cv2
import torch


# Process the image as input for the model
def handle_input_model(image, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (96, 96)) / 255
    image = np.transpose(image, (2, 0, 1))
    image = image[None, :, :, :]
    image = torch.from_numpy(image).float()
    image = image.to(device)

    return image


# Convert relative image coordinates to absolute coordinates
def cvt_to_absolute_coordinate(org_image, relative_image):
    padding_width = relative_image.width / 10
    padding_height = relative_image.height / 10

    org_height, org_width, _ = org_image.shape
    x = int((relative_image.xmin - padding_width) * org_width)
    y = int((relative_image.ymin - padding_height) * org_height)
    w = int((relative_image.width + padding_width * 2) * org_width)
    h = int((relative_image.height + padding_height * 2) * org_height)

    return abs(x), abs(y), abs(w), abs(h)
