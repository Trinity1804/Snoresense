import torch
import numpy as np
import sounddevice as sd
import librosa
from torchvision.models import resnet18
import torch.nn as nn


# Model loading function
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


# Convert audio to mel spectrogram
def get_melspectrogram_db(
    audio,
    sr=22050,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    fmin=20,
    fmax=8300,
    top_db=80,
):
    if audio.shape[0] < 5 * sr:
        audio = np.pad(
            audio, int(np.ceil((5 * sr - audio.shape[0]) / 2)), mode="reflect"
        )
    else:
        audio = audio[: 5 * sr]
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db


# Normalize the spectrogram
def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    return spec_norm


# Make prediction from audio
def predict(model, audio, device):
    spec = get_melspectrogram_db(audio)
    spec = spec_to_image(spec)
    spec = np.expand_dims(spec, axis=0)  # Add channel dimension
    spec = np.expand_dims(spec, axis=0)  # Add batch dimension
    spec_tensor = torch.tensor(spec, device=device).float()

    with torch.no_grad():
        output = model(spec_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.item()


# Parameters
sr = 22050
duration = 3  # 5 seconds
buffer = np.zeros(sr * duration)


# Callback function for real-time audio processing
def audio_callback(indata, frames, time, status):
    global buffer
    if status:
        print(status)

    # Accumulate audio data in the buffer
    buffer[:-frames] = buffer[frames:]
    buffer[-frames:] = indata[:, 0]

    # Check if buffer is full (5 seconds of audio)
    if np.count_nonzero(buffer) > 0:
        prediction = predict(model, buffer, device)
        label = "Snoring" if prediction == 1 else "Non-Snoring"
        print(f"Prediction: {label}")


if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    model_path = "snore_model.pth"
    model = load_model(model_path, device)

    # Start the audio stream
    with sd.InputStream(
        callback=audio_callback, channels=1, samplerate=sr, blocksize=int(sr * 0.5)
    ):
        print("Listening... Press Ctrl+C to stop.")
        sd.sleep(1000000)  # Keep the stream open for a long time

