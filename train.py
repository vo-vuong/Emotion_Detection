import os
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ColorJitter
from utils.matrix_helpers import plot_confusion_matrix
from dataset import EmotionDataset
from models import EmotionCNN


def get_args():
    parser = argparse.ArgumentParser(description="Train a EmotionCNN model")
    parser.add_argument("--data_path", "-d", type=str, default="data")
    parser.add_argument("--num_workers", "-n", type=str, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--log_path", "-l", type=str, default="tensorboard")
    parser.add_argument("--save_path", "-s", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-sc", type=str, default=None)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    train_transform = Compose(
        [
            ToTensor(),
            Resize((96, 96), antialias=False),
            ColorJitter(brightness=0.125, contrast=0.1, saturation=0.1, hue=0.05),
            Normalize(mean=[0.5384, 0.4309, 0.3805], std=[0.2563, 0.2310, 0.2234]),
        ]
    )

    test_transform = Compose(
        [
            ToTensor(),
            Resize((96, 96), antialias=False),
            Normalize(mean=[0.5384, 0.4309, 0.3805], std=[0.2563, 0.2310, 0.2234]),
        ]
    )

    train_set = EmotionDataset(
        root=args.data_path, train=True, transform=train_transform
    )
    test_set = EmotionDataset(
        root=args.data_path, train=False, transform=test_transform
    )
    classes_data = train_set.classes

    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=False,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = EmotionCNN(num_classes=8).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Used to store the lowest loss value for Early stopping
    best_loss = 100000

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        print("Continue training at the epoch: ", start_epoch + 1)
    else:
        start_epoch = 0

    # Create folder save model
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    # Create folder save log tensorboard
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.mkdir(args.log_path)
    writer = SummaryWriter(args.log_path)

    for epoch in range(start_epoch, args.epochs):
        # Train step
        model.train()
        train_loss = []

        train_progress_bar = tqdm(train_dataloader, colour="green")
        for i, (images, labels) in enumerate(train_progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss_value = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss_value.backward()
            train_loss.append(loss_value.item())

            # Optimal
            optimizer.step()

            # Show information on train progress bar
            train_progress_bar.set_description(
                "Train epoch {}. Iteration {}/{} Loss {:.4f}: ".format(
                    epoch + 1, i + 1, len(train_dataloader), np.mean(train_loss)
                )
            )

            # Save training loss to tensorboard
            writer.add_scalar(
                "Train/Loss", np.mean(train_loss), i + epoch * len(train_dataloader)
            )

        # Validation step
        all_predictions = []
        all_labels = []
        test_loss = []
        model.eval()

        with torch.inference_mode():
            test_progress_bar = tqdm(test_dataloader, colour="cyan")
            for i, (images, labels) in enumerate(test_progress_bar):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss_value = criterion(outputs, labels)
                test_loss.append(loss_value.item())

                # Get the max predict value of the model
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())

                # Show information on test progress bar
                test_progress_bar.set_description(
                    "Valid epoch {}. Iteration {}/{} Loss {:.4f}: ".format(
                        epoch + 1, i + 1, len(test_dataloader), np.mean(test_loss)
                    )
                )

            # Save loss, confusion matrix to tensorboard
            conf_matrix = confusion_matrix(all_labels, all_predictions)
            acc_score = accuracy_score(all_labels, all_predictions)
            writer.add_scalar("Valid/Loss", np.mean(test_loss), epoch)
            writer.add_scalar("Valid/Acc", acc_score, epoch)
            plot_confusion_matrix(writer, conf_matrix, [i for i in classes_data], epoch)

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.save_path, "last.pt"))

        if np.mean(test_loss) < best_loss:
            best_loss = np.mean(test_loss)
            best_epoch = epoch
            torch.save(checkpoint, os.path.join(args.save_path, "best.pt"))

        if epoch - best_epoch == 5:  # Early Stopping
            print("Activate early stopping at epoch: ", epoch + 1)
            break
