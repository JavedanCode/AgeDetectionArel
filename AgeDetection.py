import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class AFADDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        label = torch.tensor(int(row["age_group"]), dtype=torch.long)

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

def main():

    # Map age to age group
    def age_to_group(age):
        if age < 20:
            return 0  # 15–19
        elif age < 30:
            return 1  # 20–29
        elif age < 40:
            return 2  # 30–39
        elif age < 50:
            return 3  # 40–49
        else:
            return 4  # 50+

    image_dir = r"C:\Users\Soren\PycharmProjects\AgeDetectionArel\AFAD-Full"

    data = []
    # Traverse directories and collect image paths and age groups
    for age_folder in os.listdir(image_dir):
        age_path = os.path.join(image_dir, age_folder)

        if not os.path.isdir(age_path):
            continue

        try:
            age = int(age_folder)
        except:
            continue

        group = age_to_group(age)

        for inner_folder in os.listdir(age_path):
            inner_path = os.path.join(age_path, inner_folder)

            if not os.path.isdir(inner_path):
                continue

            for f in os.listdir(inner_path):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    rel_path = os.path.join(age_folder, inner_folder, f)
                    data.append([rel_path, group])
    # Create DataFrame
    df = pd.DataFrame(data, columns=["filename", "age_group"])
    print("Total images:", len(df))

    print("Groups BEFORE remap:", sorted(df["age_group"].unique()))
    print(df["age_group"].value_counts())
    # Remap age groups to consecutive integers starting from 0
    unique_labels = sorted(df["age_group"].unique())
    label_map = {old: new for new, old in enumerate(unique_labels)}
    df["age_group"] = df["age_group"].map(label_map).astype(int)
    # Verify remapping
    print("Groups AFTER remap:", sorted(df["age_group"].unique()))
    print(df["age_group"].value_counts())

    counts = df["age_group"].value_counts()
    if (counts < 2).any():
        stratify_val = None
        print("Warning: some classes have < 2 samples — disabling stratify for train_test_split.")
    else:
        stratify_val = df["age_group"]

    # Split dataset
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=stratify_val
    )
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    # Create Datasets
    train_ds = AFADDataset(train_df, image_dir, train_transform)
    val_ds = AFADDataset(val_df, image_dir, val_transform)
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    num_classes = df["age_group"].nunique()
    print("Number of classes:", num_classes)
    # Load pretrained EfficientNet-B2
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1
    model = efficientnet_b2(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        # Dropout layer for regularization
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)  # num_classes = 6
    )

    if device.type == "cuda":
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    # Define loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )
    # Define GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()
    # Check GPU availability
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    # Training epoch function
    def train_epoch():
        # Set model to training mode
        model.train()
        correct, total, running_loss = 0, 0, 0
        loop = tqdm(train_loader, desc="Training", leave=True)

        for imgs, labels in loop:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed precision context for forward pass and loss computation
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update metrics
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=(correct / total))

        return correct / total, running_loss / total

    def validate():
        # Set model to evaluation mode
        model.eval()
        correct, total, running_loss = 0, 0, 0

        loop = tqdm(val_loader, desc="Validating", leave=True)

        with torch.no_grad():
            for imgs, labels in loop:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # Use autocast for validation too
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                # Update metrics
                running_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                loop.set_postfix(loss=loss.item(), acc=(correct / total))

        return correct / total, running_loss / total

    # Training loop with early stopping
    model_path = "age_model_pt.pth"

    if not os.path.exists(model_path):
        print("Training model...")

        best_val_acc = 0.0
        patience = 3
        no_improve = 0

        # Training for a maximum of 15 epochs
        for epoch in range(15):

            train_acc, train_loss = train_epoch()
            val_acc, val_loss = validate()

            print(f"Epoch {epoch + 1}/15 | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            # Check for improvement and save the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved with val_acc = {best_val_acc:.4f}")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("⏹ Early stopping (no improvement).")
                    break

        print(f"Best validation accuracy: {best_val_acc:.4f}")
    else:
        print("Loading saved model...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        val_acc, _ = validate()
        print(f"Loaded model validation accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()








