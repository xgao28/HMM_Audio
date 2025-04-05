# HMMAudio

## Project Overview
HMMAudio is a package for audio classification using Hidden Markov Models (HMMs). It extracts Mel Frequency Cepstral Coefficients (MFCC) features from audio signals and uses HMMs to model the sequential nature of these features, enabling accurate audio classification tasks like emotion detection in speech.

## Key Features
- Feature extraction with MFCC coefficients
- Custom implementation of both discrete and continuous HMMs
- Built-in evaluation functions for model comparison
- Easy-to-use API for emotion classification

## Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/xgao28/HMM_Audio.git
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn tqdm matplotlib scipy
   ```

3. Run the demo notebook or initialize your own:
   ```python
   from hmmaudio.application import EmotionIdentifier

   train_path = "data/train"
   test_path = "data/test"
   model = EmotionIdentifier()
   model.train(train_path, limit=10)
   model.evaluate(test_path, limit=10)
   ```



## Core Components
- `features.py`: MFCC feature extraction with delta and delta-delta coefficients
- `hmm.py`: Implementation of both discrete and continuous HMMs
- `application.py`: High-level API for emotion identification
- `eval.py`: Functions for model evaluation
- `utils.py`: Utility functions for data loading and processing

## Data Format
The package expects the data folder structure as follows:
```
data/
├── train/
│   ├── Anger/
│   │   ├── audio1.wav
│   │   └── ...
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Neutral/
│   └── Sad/
└── test/
    ├── Anger/
    ├── Disgust/
    └── ...
```

## Datasets
Data are retrieved from:
- Training data: [CREMA-D dataset](https://www.kaggle.com/datasets/ejlok1/cremad)
- Testing data: [RAVDESS dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

## Performance Analysis
Our model demonstrates a common pattern in machine learning: high performance on limited training data but difficulty in generalizing to new data. 

With small training sets (10 samples per class), we achieve approximately 75% accuracy on training data but only 17% on test data, indicating overfitting. 

As training data increases to the full dataset, training accuracy decreases to around 40%, while test accuracy shows slight improvement, suggesting a more realistic model. 

This pattern reveals the challenge of cross-dataset emotion recognition, where differences between training (CREMA-D) and testing (RAVDESS) datasets create domain shift issues. 

Our custom HMM implementation shows comparable performance to standard libraries like hmmlearn, validating our approach while highlighting the need for better generalization strategies.

## Requirements
- For running the hmmaudio package and demo.ipynb:
  ```
  pip install numpy scikit-learn tqdm matplotlib scipy
  ```

- For running the baseline comparison or cross validation:
  ```
  pip install seaborn hmmlearn librosa
  ```

## Documentation
For more detailed information, see the example notebooks:
- `demo.ipynb`: Basic usage demonstration
- `compare.ipynb`: Comparing with standard HMM libraries
- `baseline.ipynb`: Baseline models using other packages

## License
This project is licensed under the MIT License - see the LICENSE file for details.