import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.optim import Adam
import torchaudio
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from torchvision.models import resnet18
from torchvision.transforms import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report
import os

# Load the CSV file
df = pd.read_csv("snore_audio.csv")

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.4)

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
print(train_df["labels"].value_counts())


# Function to convert audio to melspectrogram
def get_melspectrogram_db(
    file_path,
    sr=22050,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    fmin=20,
    fmax=8300,
    top_db=80,
):
    # Load the audio file
    wav, sr = librosa.load(file_path, sr=sr)

    # Ensure the audio is at least 5 seconds long
    if wav.shape[0] < 5 * sr:
        wav = np.pad(wav, int(np.ceil((5 * sr - wav.shape[0]) / 2)), mode="reflect")
    else:
        wav = wav[: 5 * sr]

    # Compute the mel spectrogram
    spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )

    # Convert the mel spectrogram to decibel units
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db


def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    return spec_norm


# Custom data generator
class SnoreDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df["file_name"][idx]
        label = self.df["labels"][idx]

        image = get_melspectrogram_db(audio_path)
        image = spec_to_image(image)
        x = np.expand_dims(image, axis=0)  # Add channel dimension
        image = torch.tensor(x, device=device).float()
        label = torch.tensor(label, device=device).long()

        if self.transforms is not None:
            image = self.transforms(image)
        return image, label


# Load model
model = resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)  # Two output classes: non-snoring and snoring
model.conv1 = nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)
model = model.to(device)

# Define data augmentations
aug_train = transforms.Compose([])
aug_test = transforms.Compose([])

# Create datasets and dataloaders
dataset_train = SnoreDataset(df=train_df, transforms=aug_train)
dataset_test = SnoreDataset(df=test_df, transforms=aug_test)

train_loader = DataLoader(dataset=dataset_train, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = Adam(params=model.parameters(), lr=3e-3, amsgrad=False)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    avg_loss = 0.0
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("Training...")
    for i, (x_batch, y_batch) in enumerate(train_loader):
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item() / len(train_loader)
        if i % 10 == 0:
            print(f"Batch {i}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "snore_model.pth")

    print("Evaluating...")
    model.eval()
    real = []
    pred = []
    for i, (x_batch, y_batch) in enumerate(test_loader):
        with torch.no_grad():
            preds = model(x_batch)
            _, predicted = torch.max(preds, 1)

        real.append(y_batch.cpu().numpy()[0])
        pred.append(predicted.cpu().numpy()[0])

    print(classification_report(real, pred))
    scheduler.step()

