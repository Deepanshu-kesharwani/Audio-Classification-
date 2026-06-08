# Audio Classification EDA

A machine learning project for Exploratory Data Analysis (EDA) and classification of audio files using the UrbanSound8K dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project performs Exploratory Data Analysis (EDA) on urban sound recordings and builds classification models to automatically categorize different types of sounds. The primary goal is to understand the characteristics of urban sounds and develop machine learning models that can accurately classify them into predefined categories.

**Key Objectives:**
- Analyze audio file characteristics and features
- Visualize sound patterns and spectrograms
- Build and train classification models
- Evaluate model performance on urban sounds

## 📊 Dataset

### UrbanSound8K
- **Source**: UrbanSound8K dataset
- **Size**: 6 GB (tar.gz archive)
- **Total Samples**: 8,732 labeled sound excerpts
- **Duration**: ~10 seconds per clip
- **Audio Format**: WAV
- **Sampling Rate**: 48 kHz

### Sound Categories
The dataset includes the following urban sound categories:
1. Air Conditioner
2. Car Horn
3. Children Playing
4. Dog Barking
5. Drilling
6. Engine Idling
7. Gunshot
8. Jackhammer
9. Siren
10. Street Music

## ✨ Features

- **Audio Loading & Processing**: Load and preprocess WAV files using librosa
- **Visualization**: Generate spectrograms and time-domain plots
- **Feature Extraction**: Extract audio features (MFCCs, spectral features, etc.)
- **EDA**: Comprehensive exploratory data analysis
- **Model Training**: Train classification models
- **Model Evaluation**: Performance metrics and visualization

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Audio_Classification.git
cd Audio_Classification
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
Download the UrbanSound8K dataset from [https://urbansounddataset.weebly.com/urbansound8k.html](https://urbansounddataset.weebly.com/urbansound8k.html)

Extract the dataset:
```bash
tar -xzf UrbanSound8K.tar.gz
```

## 📁 Project Structure

```
Audio_Classification/
│
├── Audio_Classification.ipynb       # Main Jupyter notebook for EDA
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── UrbanSound8K/                   # Dataset directory
│   ├── audio/                      # Audio files
│   │   ├── fold1/
│   │   ├── fold2/
│   │   └── ...
│   └── metadata/
│       └── UrbanSound8K.csv        # Metadata file
│
└── saved_models/                    # Directory for saved trained models
    ├── model_v1.pkl
    └── ...
```

## 💻 Usage

### 1. Basic Audio Loading
```python
import librosa
import matplotlib.pyplot as plt

# Load an audio file
filename = 'UrbanSound8K/UrbanSound8K/Chirping.wav'
y, sr = librosa.load(filename)

# Plot waveform
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Audio Waveform')
plt.show()
```

### 2. Generate Spectrogram
```python
# Compute mel-spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
S_db = librosa.power_to_db(S, ref=np.max)

# Display spectrogram
plt.figure(figsize=(12, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-spectrogram')
plt.show()
```

### 3. Extract Features
```python
# Extract MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Extract Mel Spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

# Extract Zero Crossing Rate
zcr = librosa.feature.zero_crossing_rate(y)

# Extract Spectral Centroid
spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
```

### 4. Run Full Analysis
Open and run the `Audio_Classification.ipynb` notebook:
```bash
jupyter notebook Audio_Classification.ipynb
```

## 📦 Dependencies

The project requires the following Python packages:

```
librosa==0.11.0
numpy>=1.22.3
scipy>=1.6.0
scikit-learn>=1.1.0
matplotlib>=3.0.0
pandas>=1.0.0
jupyter>=1.0.0
ipython>=7.0.0
audioread>=2.1.9
soundfile>=0.12.1
```



## 📈 Results

### Key Findings from EDA:
- **Sample Distribution**: Understand class imbalance across sound categories
- **Feature Analysis**: Identify important features for classification
- **Spectral Characteristics**: Visualize frequency patterns for different sound types
- **Temporal Patterns**: Analyze how sounds evolve over time

### Model Performance:
*(Add your actual results here after training)*
- **Accuracy**:  77.39%
- **Precision**:  79.95%
- **Recall**: 77.39%
- **F1-Score**: 77.90%

## 🔄 Preprocessing Pipeline

1. **Load Audio**: Read WAV files using librosa
2. **Normalize**: Standardize audio amplitude
3. **Feature Extraction**: Extract MFCC, Mel-spectrogram, and other features
4. **Scaling**: Standardize features for model input
5. **Train-Test Split**: Divide data for training and evaluation


## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)
- [Librosa Documentation](https://librosa.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Audio Signal Processing](https://en.wikipedia.org/wiki/Digital_signal_processing)

## 👥 Authors

- **Deepanshu Kesharwani** - [GitHub Profile](https://github.com/Deepanshu-kesharwani)



---

**Note**: This project is for educational purposes. Make sure you have the necessary permissions to use the UrbanSound8K dataset.

