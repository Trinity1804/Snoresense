# Snore Detection Project

This project contains a collection of scripts and data for training, testing, and running real-time snore detection using deep learning.

## Contents

- **Data**: The [dataset](dataset/) folder contains audio files and a text file with information about the snoring dataset.
- **Scripts:**

  - **`generate_dataset.py`**: Generates and organizes the dataset for training.
  - **`snore_detection_train.py`**: Trains the snore detection model.
  - **`snore_detection_realtime.py`**: Runs the snore detection in real time.
  - **`snore_test.py`**: Contains tests for the snore detection functionality.
- **Model:**

  - **`snore_model.pth`**: The trained PyTorch model for snore detection.
- **Tests**: The [tests](tests/) folder contains unit tests for the project.

## Setup

1. **Install Dependencies**
   Make sure you have Python and PyTorch installed. Then, install any required packages listed in the project requirements, for example:

   ```sh
   pip install -r requirements.txt
   ```
2. **Prepare the Data**
   Ensure that the dataset is organized correctly under the dataset folder. You can run [generate_dataset.py](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) to automatically download and organize the data if needed.

## Usage

* **Training:**
  Run the training script to train the snore detection model:

  **python** **snore_detection_train.py**
* **Real-time Detection:**
  To start the real-time snore detection:

  **python** **snore_detection_realtime.py**
* **Testing:**
  To run the tests for the project:

  **python** **-m** **unittest** **discover** **tests**

## License

This project is licensed under the terms of the GNU General Public License version 3.

## Acknowledgements

Dataset is taken from:

[https://www.kaggle.com/datasets/tareqkhanemu/snoring](Acknowledgements)
