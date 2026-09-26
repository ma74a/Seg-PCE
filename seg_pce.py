

"""
in this assignment we use segmentation problem, 
which we don't apply it with full mask image,
but using point sampling points that we apply for it
we use Partial Cross Entropy Loss instead of CE
to help use handling this problem

"""
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import transforms as T
import segmentation_models_pytorch as smp

from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

NUM_CLASSES=7
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=8
EPOCHS=30
LR=0.001
PERCENTAGES=[0.05, 0.10, 0.15, 0.20]

image_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

mask_transform = T.Compose([
    T.Resize(
        (512, 512),
        interpolation=T.InterpolationMode.NEAREST
    ),
])

def sample_point_labels_for_class(
    mask,
    sample_percentage=0.10,
    ignore_index=255
):
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.array(mask)

    # mask shape: [H, W]
    H, W = mask_np.shape

    # create a full image with just ignore index
    pointed_mask = np.full(shape=(H, W), fill_value=ignore_index, dtype=np.int64)

    # get the unique classes
    unique_classes = np.unique(mask_np)
    unique_classes = unique_classes[unique_classes != ignore_index]

    # loop through every class
    # get the total number of pixels for each
    # calculate the percentage for point sampling
    # randomly add point sampling for each class
    # put the selected point into point mask
    for cls in unique_classes:
        """
        get the ys, xs for each pixel for this class
        rows = [0, 1, 1]
        cols = [2, 1, 2]
        (0,2)
        (1,1)
        (1,2)
        """
        rows, cols = np.where(mask_np==cls)
        # number of available pixels for each class
        num_available = len(rows)
        # calculate the percentage 
        num_sample = int(sample_percentage * num_available)
        # make sure we at least have one pixel to sample
        num_sample = max(1, num_sample)
        # Don't sample more pixels than available
        num_sample = min(num_sample, num_available)
        # then randomly sampling the points
        chosen = np.random.choice(
            a=num_available,
            size=num_sample,
            replace=False
        )

        # then apply point mask
        pointed_mask[
            rows[chosen],
            cols[chosen]
        ] = cls

    return pointed_mask


class LoveDADataset(Dataset):
    """
    First I create the Custom dataset class I'm using LoveDa dataset
    When I created the dataset class I also apply point sampling for each class
    I don't apply sampling randomly I and I don't apply sampling randomly for each class
    I used a percentage like 10% for each class
    so I get the total pixels of each class and create the point sampling
    """
    def __init__(
        self, 
        data_root: str, 
        split: str="Train", 
        image_transform: T=None,
        mask_transform: T=None,
        sample_percentage: float = 0.10,
        ignore_index: int = 255,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.sample_percentage = sample_percentage
        self.ignore_index = ignore_index

        self.split_dir = self.data_root / self.split / self.split
        self.data = []

        for domain in ["Rural", "Urban"]:
            image_dir = self.split_dir / domain / "images_png"
            masks_dir = self.split_dir / domain / "masks_png"

            for img_path in image_dir.glob("*.png"):
                mask_path = masks_dir / img_path.name

                if mask_path.exists():
                    self.data.append(
                        (img_path, mask_path)
                    )

        print(f"{split}: {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.data[index]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.image_transform:
            img = self.image_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = torch.tensor(np.array(mask), dtype=torch.long)
        # Convert LoveDA labels to 0-6 + 255
        mask = torch.where(
        mask == 0,
        torch.tensor(self.ignore_index, dtype=torch.long, 
        device=mask.device),mask - 1)

        # Training → partial point supervision
        if self.split.lower() == "train":

            target = sample_point_labels_for_class(
                mask,
                sample_percentage=self.sample_percentage,
                ignore_index=self.ignore_index,
            )

        # Validation → full ground-truth mask
        else:
            target = mask

        return img, target




class PartialCrossEntropyLoss(nn.Module):
    """
    We will build PCE not CE, which in our condition is better
    for this We will combine PCE with focal loss which helps us with unbalanced dataset

    """
    def __init__(
            self,
            ignore_index=255,
            gamma=2.0
    ):
        super(PartialCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma

    def forward(self, predictions, targets):
        """
        predictions(logits): [B, C, H, W]
        targets: [B, H, W]
        
        convert logits into probs using log_softmax applying it to C 
        -> [B, C, H, W] -> [B, C, H, W]
        then we will get the pixels which have label
        valid will be [B, H, W]
        we will use valid to get the target labels
        we move C to the last in log_probs
        So every pixel contains its vector of class log-probabilities:
        pixel → [class0, class1, class2, ..., class6]
        we also apply valid into log_probs
        point_log_probs -> model's probabilities for each sampled pixel
        labels -> correct class for each sampled pixel
        We will Select probability of the correct class
        """

        # convert predictions logit into probs across C dim
        log_probs = F.log_softmax(input=predictions, dim=1)
        # get the pixels that have labels
        # creates a Boolean mask that tells us which pixels should be used in Partial CE.
        valid = targets != self.ignore_index

        # check if targets contain only 255
        # No labeled pixels
        if not valid.any():
            return predictions.sum() * 0.0
        
        # get the labels using valid applied it with target
        # Ground-truth labels of sampled pixels
        labels = targets[valid] # [N]
        # move C to the last
        log_probs = log_probs.permute(0, 2, 3, 1) # -> [B, H, W, C]
        # applying valid into log_probs
        # Keep only sampled pixels
        point_log_probs = log_probs[valid] # [N, C] # n-> num of sample points

        # Select probability of the correct class
        point_log_probs = point_log_probs[
                torch.arange(
                    point_log_probs.size(0),
                    device=predictions.device
                ),
                labels
            ]
        # Focal loss
        # p -> probability of the correct class (exp of point_log_probs)
        # (1 - p)^gamma is responsible for down weight easy pixels
        p = torch.exp(point_log_probs)
        focal_weight = (1 - p) ** self.gamma

        # CE = -log(P(correct class)), then apply focal weight
        losses = - focal_weight * point_log_probs

        # then take the average
        return losses.mean() # mean over N pixels only


def get_model():
    """
    using pretrained deeplabv3plus model for this task
    """
    model = model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=7,
    )

    return model


def train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    ):
    train_loss = 0
    model.train()
    for images, pointed_masks in tqdm(train_loader):
        images, pointed_masks = images.to(device), pointed_masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, pointed_masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader) 
    
def val_one_epoch(
        model,
        val_loader,
        criterion,
        device,
        ignore_index=255
    ):
    val_loss = 0
    intersection = torch.zeros(
        NUM_CLASSES,
        device=device
    )
    union = torch.zeros(
        NUM_CLASSES,
        device=device
    )

    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)

            # [B, C, H, W] -> [B, H, W]
            predictions = logits.argmax(dim=1)
            loss = criterion(logits, masks)
            # Calculate IoU for every class
            for cls in range(NUM_CLASSES):

                pred_class = predictions == cls
                target_class = masks == cls

                # Ignore invalid pixels
                valid = masks != ignore_index

                pred_class = pred_class & valid
                target_class = target_class & valid

                intersection[cls] += (
                    pred_class & target_class
                ).sum()

                union[cls] += (
                    pred_class | target_class
                ).sum()

            val_loss += loss.item()
    # IoU for each class
    iou = intersection / (union + 1e-7)

    # Mean IoU
    miou = iou.mean()

    return val_loss / len(val_loader), miou

def main():
    data_root = ""
    train_dataset = LoveDADataset(
    data_root=data_root,
    split="Train",
    image_transform=image_transform,
    mask_transform=mask_transform,
    sample_percentage=PERCENTAGES[0],
    )
    val_dataset = LoveDADataset(
        data_root=data_root,
        split="Val",
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = get_model()
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = PartialCrossEntropyLoss(ignore_index=255)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    history      = {'train_loss': [], 'val_loss': [], 'miou': []}
    best_miou    = 0.0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )
        val_loss, miou = val_one_epoch(
            model,
            val_loader,
            criterion,
            DEVICE
        )
        # Step scheduler
        scheduler.step()
 
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history["miou"].append(miou)

        print(f"\nEpoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} "
              f"mIoU: {miou:.4f}")

        if miou > best_miou:
            best_miou  = miou
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✓ Saved best model (mIoU={best_miou:.4f})")

    return history


def plot_losses(train_losses: List, val_losses: List) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("loss_plot.png")

# def run_experiments():
#     data_root = "path/to/LoveDA"
#     percentages = [0.005, 0.02, 0.10, 1.0]
#     results = {}

#     for p in percentages:
#         print(f"\n{'='*20} RUNNING FOR DENSITY: {p*100:.1f}% {'='*20}")
        
#         train_dataset = LoveDADataset(
#             data_root=data_root,
#             split="Train",
#             image_transform=image_transform,
#             mask_transform=mask_transform,
#             sample_percentage=p,
#         )
#         val_dataset = LoveDADataset(
#             data_root=data_root,
#             split="Val",
#             image_transform=image_transform,
#             mask_transform=mask_transform,
#         )

#         train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
#         val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

#         model = get_model().to(DEVICE)
#         optimizer = optim.Adam(model.parameters(), lr=LR)
#         criterion = PartialCrossEntropyLoss(ignore_index=255)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

#         best_miou = 0.0
#         for epoch in range(EPOCHS):
#             train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
#             val_loss, miou = val_one_epoch(model, val_loader, criterion, DEVICE)
#             scheduler.step()

#             if miou > best_miou:
#                 best_miou = miou.item()
#                 torch.save(model.state_dict(), f"best_model_p_{int(p*1000)}.pth")

#         results[f"{p*100:.1f}%"] = best_miou
#         print(f"Finished density {p*100:.1f}% -> Best mIoU: {best_miou:.4f}")

#     print("\n--- FINAL EXPERIMENT SUMMARY ---")
#     for density, score in results.items():
#         print(f"Density: {density:>6} | Best mIoU: {score:.4f}")

if __name__ == "__main__":
    history = main()
    plot_losses(history["train_loss"], history["val_loss"])
    