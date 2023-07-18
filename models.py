import torch
import torch.nn as nn


class EmotionCNN(nn.Module):
    # Defines the convolution, normalization, pooling layers and relu activation used in the body model
    def _make_block(self, in_chanels, out_chanels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_chanels,
                out_channels=out_chanels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_chanels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_chanels,
                out_channels=out_chanels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_chanels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = self._make_block(in_chanels=3, out_chanels=16)
        self.conv2 = self._make_block(in_chanels=16, out_chanels=32)
        self.conv3 = self._make_block(in_chanels=32, out_chanels=64)
        self.conv4 = self._make_block(in_chanels=64, out_chanels=128)

        self.linear_1 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(5 * 5 * 128, 512), nn.ReLU()
        )
        self.linear_2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.shape[0], -1)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x


if __name__ == "__main__":
    random_image = torch.rand(4, 3, 96, 96)  # [B, C, H, W]
    model = EmotionCNN(num_classes=8)
    predictions = model(random_image)  # Forward pass
    print("predictions result shape {}".format(predictions.shape))
    print(predictions)
