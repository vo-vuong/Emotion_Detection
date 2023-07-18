import os
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class EmotionDataset(Dataset):
    def __init__(self, root="data", train=True, ratio=0.8, transform=None):
        self.image_folder = os.path.join(root, "images")

        # Read labels file
        with open(os.path.join(root, "images", "labels.csv"), "r") as label_file:
            next(label_file)
            gts = label_file.readlines()
            gts = [gt.strip().split(",") for gt in gts]
            labels = [gt[2] for gt in gts]

        gts_train, gts_test, _, _ = train_test_split(
            gts, labels, train_size=ratio, random_state=10, stratify=labels
        )
        if train:
            self.gts = gts_train
        else:
            self.gts = gts_test

        self.transform = transform
        self.classes = [
            "surprise",
            "anger",
            "disgust",
            "fear",
            "sad",
            "contempt",
            "neutral",
            "happy",
        ]

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, item):
        _, img_path, label, _ = self.gts[item]
        image = cv2.imread(os.path.join(self.image_folder, img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
        label = self.classes.index(label)

        return image, label


if __name__ == "__main__":
    train_set = EmotionDataset(root="data")
    test_set = EmotionDataset(root="data", train=False)

    image, label = train_set.__getitem__(10)
    print(train_set.classes[label])
    print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("image", image)
    cv2.waitKey(0)
