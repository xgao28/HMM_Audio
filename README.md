# HMMAudio

## Project Overview
HMMAudio is a package for audio classification using Hidden Markov Models. 
MFCC (Mel Frequency Cepstral Coefficients) features are extracted from audio signals to represent speech characteristics.
HMMs are used to model sequential audio features for effective emotion detection in speech.

## Requirements
For running the hmmaudio package and demo.ipynb:
```
pip install numpy scikit-learn tqdm matplotlib scipy
```

For running the baseline comparison or cross validation (as in baseline.ipynb and compare.ipynb):
```
pip install seaborn hmmlearn librosa
```

## Data
Data are retrieved through train data: https://www.kaggle.com/datasets/ejlok1/cremad and test data: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio. 

The package assumes the folder is organized as follows:

```
data/{data, test_data}/{Anger, Disgust, Fear, Happy, Neutral, Sad}/{wav files}
```

The data folder should contain two subfolders: `data` and `test_data`. Each of these folders should contain subfolders for each label category (Anger, Disgust, Fear, Happy, Neutral, Sad), and each emotion folder should contain the corresponding WAV files.


