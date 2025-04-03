from tqdm import tqdm
import os
import numpy as np
from hmmaudio.features import extract_features
from hmmaudio.hmm import HiddenMarkovModel, ContinuousHMM


def load_data(data_path, label, limit=None,                 
                include_mfcc=True,
                include_delta=True,
                include_delta2=True,
                num_cepstral=13,
                target_frames = None):
    """Load audio files for a specific label and extract features."""
    label_path = os.path.join(data_path, label)
    audio_files = [f for f in os.listdir(label_path) if f.endswith('.wav')]
    
    if limit:
        audio_files = audio_files[:limit]
    
    features_list = []
    file_names = []
    
    for audio_file in tqdm(audio_files, desc=f"Processing {label}"):
        file_path = os.path.join(label_path, audio_file)
        try:
            # Extract MFCC features with delta and delta-delta
            features = extract_features(
                file_path,
                include_mfcc=include_mfcc,
                include_delta=include_delta,
                include_delta2=include_delta2,
                num_cepstral=num_cepstral,
                target_frames=target_frames
            )
            features_list.append(features)
            file_names.append(audio_file)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    return features_list, file_names

def load_all_data(data_path, limit=None,                 
                include_mfcc=True,
                include_delta=True,
                include_delta2=True,
                num_cepstral=13, target_frames=None):
    """Load all audio files from the data path and extract features."""
    label_features = {}
    label_files = {}
    labels = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

    for label in labels:
        features_list, file_names = load_data(data_path, label,                 
                include_mfcc=include_mfcc,
                include_delta=include_delta,
                include_delta2=include_delta2,
                num_cepstral=num_cepstral, target_frames=target_frames)
        label_features[label] = features_list
        label_files[label] = file_names
        print(f"{label}: Loaded {len(features_list)} files")

    return label_features, label_files

def initialize_hmm(n_states, n_symbols):
    """Initialize a Hidden Markov Model with random parameters."""
    # Initialize transition matrix
    A = np.random.rand(n_states, n_states)
    # Add a bias towards diagonal and next state
    for i in range(n_states):
        if i < n_states - 1:
            A[i, i] += 1.0  # More likely to stay in same state
            A[i, i+1] += 0.5  # More likely to transition to next state
        else:
            A[i, i] += 1.5  # Last state more likely to stay
    # Normalize
    A = A / A.sum(axis=1, keepdims=True)
    
    # Initialize emission matrix
    B = np.random.rand(n_states, n_symbols)
    B = B / B.sum(axis=1, keepdims=True)
    
    # Initialize initial state distribution
    pi = np.zeros(n_states)
    pi[0] = 0.6  # High probability to start in first state
    pi[1:3] = 0.4 / 2  # Some probability to start in states 1 or 2
    pi = pi / pi.sum()
    
    return HiddenMarkovModel(A, B, pi)

def train_hmm(label_features, n_states, n_symbols, max_iter=100, continuous=True, diagonal_covariance=True):
    """Train a Hidden Markov Model for each label."""
    hmm_models = {}
    
    for label, features in label_features.items():
        print(f"Training HMM for {label}")
        if continuous:
            # Use Continuous HMM for continuous features
            hmm = ContinuousHMM(n_states, n_symbols, diagonal_covariance=diagonal_covariance)
        else:
            hmm = initialize_hmm(n_states, n_symbols)
        hmm.fit(features, max_iter=max_iter)
        hmm_models[label] = hmm
    
    return hmm_models

def score_observation(hmm_models, observation):
    """
    Calculate the log-likelihood of an observation sequence under each HMM model.
    
    Args:
        hmm_models: Dictionary of HMM models keyed by label
        observation: A single observation sequence as np.ndarray of shape (T, D)
    
    Returns:
        dict: Dictionary of log-likelihood scores keyed by label
    """
    scores = {}
    for label, model in hmm_models.items():
        scores[label] = model.score(observation)
    return scores

def predict_label(hmm_models, observation):
    """
    Predict the most likely label for an observation sequence.
    
    Args:
        hmm_models: Dictionary of HMM models keyed by label
        observation: A single observation sequence as np.ndarray of shape (T, D)
    
    Returns:
        tuple: (most_likely_label, scores_dict)
    """
    scores = score_observation(hmm_models, observation)
    most_likely_label = max(scores, key=scores.get)
    return most_likely_label, scores

