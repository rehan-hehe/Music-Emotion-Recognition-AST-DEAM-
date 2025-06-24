# Music Emotion Recognition using Audio Spectrogram Transformers

This project fine-tunes a pre-trained Audio Spectrogram Transformer (AST) model on the DEAM (Database for Emotion Analysis in Music) dataset. The goal is to perform regression, predicting the continuous emotional dimensions of **valence** (how positive/negative an emotion is) and **arousal** (how calm/exciting it is) directly from audio clips.


*The Valence-Arousal space used for emotion modeling.*

---

## Table of Contents
- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architecture](#model-architecture)
- [Results](#results)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)

---

## Project Goal
The objective is to build a robust deep learning model that can analyze a music track and predict its emotional content. By using the continuous valence-arousal space instead of discrete emotion labels (e.g., 'happy', 'sad'), the model provides a more nuanced and accurate representation of the emotions evoked by music.

---

## Dataset
This project utilizes the [DEAM dataset](https://cvml.unige.ch/databases/DEAM), which contains 1,802 full-length popular music tracks with moment-by-moment and static valence-arousal annotations. For this project, the static, song-level averaged annotations were used.

---

## Methodology

### Data Preprocessing
A multi-step pipeline was created to transform raw audio into a format suitable for the Audio Spectrogram Transformer:

1.  **Audio Loading & Resampling:** Each `.mp3` file was loaded at its native 44.1kHz sample rate and downmixed from stereo to mono by averaging the channels.
2.  **Feature Extraction:** The raw audio waveform was converted into a **Mel Spectrogram** to represent the audio in the frequency domain. The spectrogram's magnitude was then converted to a decibel (dB) scale.
3.  **Data Cleaning & Truncation:** A histogram analysis of spectrogram lengths revealed a high concentration of samples around **9830** time frames. To ensure a uniform input size, samples shorter than this were discarded, and longer samples were truncated.
4.  **Target Normalization:** The original valence and arousal labels, rated on a scale of [1, 9], were normalized to a [0, 1] range.
5.  **Input Normalization:** Spectrograms were normalized using **Z-score normalization** based on the global mean and standard deviation of the entire dataset. This is a crucial step for transformer model stability.
6.  **Input Reshaping for AST:** The time dimension of the spectrograms was further truncated from 9830 to **9216** to be divisible by the model's patch size.
7.  **Data Augmentation via Chunking:** Each full `(128, 9216)` spectrogram was segmented into 9 smaller chunks of `(128, 1024)`. This significantly increased the dataset size and created inputs perfectly sized for the pre-trained AST model.

### Model Architecture
The core of the solution is a fine-tuned Hugging Face Transformer model.

- **Base Model:** `MIT/ast-finetuned-audioset-10-10-0.4593`, an Audio Spectrogram Transformer pre-trained on the large-scale AudioSet dataset.
- **Custom Regression Head:** A custom head was added on top of the AST's final hidden state. The sequence output is averaged across the time dimension (mean pooling) and then passed through a small MLP to produce the final regression outputs.
    - `Linear(768, 128)`
    - `ReLU()`
    - `Dropout(0.2)`
    - `Linear(128, 2)` -> `[valence, arousal]`
- **Training:** The model was trained for 15 epochs using the **Mean Squared Error (MSE)** loss function and the **Adam** optimizer with a learning rate of `1e-4`. The data was split into training (80%), validation (10%), and testing (10%) sets.

---

## Results
The model was evaluated on a held-out test set. The performance is measured by Mean Squared Error (MSE) and Mean Absolute Error (MAE) on the normalized [0, 1] scale.

| Metric      | Valence | Arousal |
|-------------|---------|---------|
| **MSE**     | 0.0080  | 0.0080  |
| **MAE**     | 0.0682  | 0.0689  |

These results demonstrate the model's strong capability to accurately predict both emotional dimensions from raw audio spectrograms. The MAE indicates that, on average, the model's predictions are within ~7% of the true value on the normalized scale.

---

## Technologies Used
- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **Librosa & TorchAudio** for audio processing
- **Scikit-learn** for data splitting and metrics
- **Pandas & NumPy** for data manipulation
- **Git & Git LFS** for version control
