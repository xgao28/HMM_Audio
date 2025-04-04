import os
import numpy as np
from typing import Dict, Tuple
from .features import extract_features
from .hmm import ContinuousHMM
from .utils import load_all_data, predict_label, train_hmm
from .eval import evaluate_models

class EmotionIdentifier:
    """
    Audio emotion identifier using MFCC and HMMs.
    
    This class trains one HMM per emotion and uses them to identify
    emotions from audio samples.
    
    Attributes:
        models (Dict[str, HiddenMarkovModel]): Trained HMM for each emotion
        n_states (int): Number of states in each HMM
        feature_params (Dict[str, Any]): Parameters for feature extraction
    """
    
    def __init__(self, n_states: int = 5, **kwargs):
        """
        Initialize the emotion identification system.
        
        Args:
            n_states: Number of states in each HMM
            feature_params: Parameters for feature extraction
        """
        self.models = {}
        self.n_states = n_states
        self.feature_params = kwargs or {
            'include_mfcc': True,
            'include_delta': True,
            'include_delta2': True,
            'include_energy': True,
            'num_cepstral': 13,
            'frame_size': 0.025,
            'frame_stride': 0.01
        }
    
    def train(self, data_path: str, n_iterations: int = 10, diagonal_covariance: bool = True, evaluate: bool = False, limit: int = None) -> Dict[str, ContinuousHMM]:
        """
        Train emotion models using audio files.
        
        Args:
            data_path: String of the directory containing audio files
            n_iterations: Number of iterations for HMM training
            diagonal_covariance: Whether or not to use diagonal covariance for faster training
            evaluate: Whether to evaluate the models after training
        """
        # Load all data
        print("Extracting audio features...")
        features_dict, _ = load_all_data(data_path, limit, **self.feature_params)
        print("Audio features Extracted.")

        # Set HMM parameters
        n_features = 1
        for features in features_dict.values():
            n_features = features[0].shape[1]  # Number of features
            break  # Only need to check the first one

        # Train HMMs with diagonal covariance (faster)
        print("Training HMMs...")
        hmm_models = train_hmm(
            features_dict, 
            n_states=self.n_states, 
            n_symbols=n_features,
            max_iter=n_iterations,
            continuous=True,  # Use continuous HMM
            diagonal_covariance=diagonal_covariance,
        )
        print("HMMs trained.")

        self.models = hmm_models

        if evaluate:
            # Evaluate models
            accuracy, cm, _, _, _ = evaluate_models(
                self.models, 
                features_dict,
                normalize_by_length=True
            )
            print(f"Training Accuracy: {accuracy:.2f}")
            print("Training Confusion Matrix:")
            print(cm)

        return hmm_models
    
    def identify(self, audio_file: str, verbose: bool=False) -> Tuple[str, Dict[str, float]]:
        """
        Identify the emotion in an audio file.
        
        Args:
            audio_file: Path to the audio file
            verbose: Whether to print the predicted emotion and scores
            
        Returns:
            emotion: Identified emotion
            scores: Log-likelihood scores for each emotion
        """
        # Extract features from the audio file
        features = extract_features(audio_file, **self.feature_params)
        
        # Score and identify the most likely emotion
        predicted_emotion, scores = predict_label(self.models, features)

        # Normalize scores by sequence length for fair comparison
        normalized_scores = {emotion: score/len(features) for emotion, score in scores.items()}

        # Sort scores from highest to lowest
        sorted_scores = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

        if verbose:
            # Print results
            print(f"Predicted emotion: {predicted_emotion}\n")
            print("Scores for each emotion (log-likelihood/frame, higher is better):")

            for emotion, score in sorted_scores:
                print(f"{emotion}: {score:8.2f}")
        
        return predicted_emotion, sorted_scores
    
    def evaluate(self, data_path: str, normalize_by_length: bool=True, verbose: bool=False, limit: int=None) -> Dict[str, float]:
        """
        Evaluate the trained models on a dataset.
        
        Args:
            data_path: Path to the directory containing audio files for evaluation
            normalize_by_length: Whether to normalize log probabilities by sequence length
            verbose: Whether to print evaluation results
            
        Returns:
            accuracy: Overall accuracy of the models
            cm: Confusion matrix
        """
        # Load data for evaluation
        print("Extracting audio features...")
        features_dict, _ = load_all_data(data_path, limit, **self.feature_params)
        print("Audio features Extracted.")

        # Evaluate models
        accuracy, cm, _, _, _ = evaluate_models(
            self.models, 
            features_dict,
            normalize_by_length=normalize_by_length  # Normalize by sequence length to handle variable-length audio
        )

        if verbose:
            # Print evaluation results
            print(f"Accuracy: {accuracy:.2f}")
            print("Confusion Matrix:")
            print(cm)

        return accuracy, cm
    
    def save_models(self, directory: str):
        """
        Save trained models to disk.
        
        Args:
            directory: Directory to save models to
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Using NumPy's save function to store model parameters
        for emotion, model in self.models.items():
            model_file = os.path.join(directory, f"{emotion}.npz")
            np.savez(model_file,
                     n_states=model.N,
                     n_features=model.D,
                     transition_matrix=model.A,
                     initial_distribution=model.pi,
                     means=model.means,
                     covariances=model.covariances)
            print(f"Saved model for emotion: {emotion} to {model_file}")
    
    def load_models(self, directory: str):
        """
        Load trained models from disk.
        
        Args:
            directory: Directory containing saved models
        """
        self.models = {}
        
        # Load each model file
        for file_name in os.listdir(directory):
            if file_name.endswith('.npz'):
                emotion = os.path.splitext(file_name)[0]
                model_file = os.path.join(directory, file_name)
                
                # Load model parameters
                data = np.load(model_file)
                
                # Create and configure the model
                model = ContinuousHMM(
                    transition_matrix=data['transition_matrix'],
                    means=data['means'],
                    covars=data['covariances'],
                    initial_distribution=data['initial_distribution'],
                    n_states=int(data['n_states']), 
                    n_features=int(data['n_features']), 
                    diagonal_covariance=True
                )
                
                # Store the model
                self.models[emotion] = model

                print(f"Loaded model for emotion: {emotion}")

        return self.models
