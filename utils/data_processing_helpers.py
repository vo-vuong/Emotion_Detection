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
    org_height, org_width, _ = org_image.shape
    x = int(relative_image.xmin * org_width)
    y = int(relative_image.ymin * org_height)
    w = int(relative_image.width * org_width)
    h = int(relative_image.height * org_height)

    return abs(x), abs(y), abs(w), abs(h)
