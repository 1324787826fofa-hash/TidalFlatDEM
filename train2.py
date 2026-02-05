import os
import logging
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from unet_parts import UNet
from dataset import MultiSourceGeoDataset   # 你之前我给你的 Dataset


def compute_metrics(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision.item(), recall.item(), f1.item()



def train_net(
    net,
    device,
    data_path,
    epochs=50,
    batch_size=2,
    lr=1e-4,
    early_stop_patience=20,
    val_ratio=0.2,
):

    logging.basicConfig(
        filename="training.log",
        level=logging.INFO,
        format="%(asctime)s %(message)s"
    )


    dataset = MultiSourceGeoDataset(
        data_path=data_path,
        aux_features=["mndwi", "ndvi"]
    )

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)


    optimizer = optim.RMSprop(
        net.parameters(),
        lr=lr,
        weight_decay=1e-8,
        momentum=0.9
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0


    for epoch in range(epochs):
        net.train()
        train_loss = 0.0

        for image, label, _ in train_loader:
            image = image.to(device, dtype=torch.float)

            if label.ndim == 3:
                label = label.unsqueeze(1)

            label = label.to(device, dtype=torch.float)

            optimizer.zero_grad()
            pred = net(image)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)


        net.eval()
        val_loss = 0.0
        precision_list, recall_list, f1_list = [], [], []

        with torch.no_grad():
            for image, label, _ in val_loader:
                image = image.to(device, dtype=torch.float)

                if label.ndim == 3:
                    label = label.unsqueeze(1)

                label = label.to(device, dtype=torch.float)

                pred = net(image)
                loss = criterion(pred, label)
                val_loss += loss.item()

                p, r, f1 = compute_metrics(pred, label)
                precision_list.append(p)
                recall_list.append(r)
                f1_list.append(f1)

        val_loss /= len(val_loader)
        precision = sum(precision_list) / len(precision_list)
        recall = sum(recall_list) / len(recall_list)
        f1 = sum(f1_list) / len(f1_list)

        log_msg = (
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"P: {precision:.4f} R: {recall:.4f} F1: {f1:.4f}"
        )

        print(log_msg)
        logging.info(log_msg)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(net.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            logging.info("Early stopping triggered.")
            break



