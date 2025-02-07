import os
import pandas as pd
import pathlib

# Get the base directory
base_dir = pathlib.Path(__file__).parent.resolve()

# Define paths to your data directories
non_snoring_path = base_dir / "dataset/0/"
snoring_path = base_dir / "dataset/1/"

# List of paths to iterate over
paths = [(non_snoring_path, 0), (snoring_path, 1)]

# Initialize a DataFrame
df = pd.DataFrame()

# Collect file names and labels
file_names = []
labels = []

for path, label in paths:
    for file_name in os.listdir(path):
        if file_name.endswith(".wav"):
            file_names.append(os.path.join(path, file_name))
            labels.append(label)

df["file_name"] = file_names
df["labels"] = labels
print(len(file_names), len(labels))
print(df.head())

# Save to CSV
df.to_csv("snore_audio.csv", index=False)
