import torch
import torchaudioA
import librosa
import numpy as np
from torchvision.models import resnet18
import torch.nn as nn
import pathlib

base_dir = pathlib.Path(__file__).parent.resolve()

# Define function to convert audio to mel spectrogram
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
    wav, sr = librosa.load(file_path, sr=sr)
    if wav.shape[0] < 5 * sr:
        wav = np.pad(wav, int(np.ceil((5 * sr - wav.shape[0]) / 2)), mode="reflect")
    else:
        wav = wav[: 5 * sr]
    spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db


def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    return spec_norm


# Load the model
def load_model(model_path, device):
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(512, 2)  # Adjust for binary classification
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, file_path, device):
    # Convert audio to mel spectrogram
    spec = get_melspectrogram_db(file_path)
    spec = spec_to_image(spec)
    spec = np.expand_dims(spec, axis=0)  # Add channel dimension
    spec = np.expand_dims(spec, axis=0)  # Add batch dimension

    # Convert to tensor
    spec_tensor = torch.tensor(spec, device=device).float()

    # Make prediction
    with torch.no_grad():
        output = model(spec_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.item()


if __name__ == "__main__":
    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    model_path = "snore_model.pth"
    model = load_model(model_path, device)

    # Test files
    test_files = [
        base_dir / "tests/snore_test.wav",
        base_dir / "tests/snore_test_false.wav",
    ]

    # Make predictions
    for file in test_files:
        prediction = predict(model, file, device)
        label = "Snoring" if prediction == 1 else "Non-Snoring"
        print(f"File: {file}, Prediction: {label}")

