from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_models(models, features_dict, normalize_by_length=True):
    """
    Evaluate the HMM models (discrete or continuous) on the given dataset and compute accuracy.
    
    Args:
        models: Dictionary of trained HMM models for each emotion
        features_dict: Dictionary of feature sequences for each emotion
        normalize_by_length: Whether to normalize log probabilities by sequence length (default: True)
        
    Returns:
        accuracy: Overall accuracy
        cm: Confusion matrix
        predictions: Dictionary of predictions for each emotion
        true_labels: List of true emotion labels
        pred_labels: List of predicted emotion labels
    """
    print("Evaluating HMMs...")

    true_labels = []
    pred_labels = []
    predictions = {}
    all_scores = []
    
    for true_emotion in features_dict.keys():
        emotion_preds = []
        emotion_scores = []
        
        for feature_seq in tqdm(features_dict[true_emotion], desc=f"Evaluating {true_emotion}"):
            # Calculate log probability for each emotion model
            log_probs = {}
            for emotion, model in models.items():
                if hasattr(model, 'score'):  # Use model.score() if available
                    log_prob = model.score(feature_seq)
                else:  # Otherwise, use model.forward()
                    _, log_prob = model.forward(feature_seq)
                
                # Optionally normalize by sequence length to handle variable-length sequences
                if normalize_by_length:
                    log_probs[emotion] = log_prob / len(feature_seq)
                else:
                    log_probs[emotion] = log_prob
            
            # Predict the emotion with highest log probability
            pred_emotion = max(log_probs, key=log_probs.get)
            
            true_labels.append(true_emotion)
            pred_labels.append(pred_emotion)
            emotion_preds.append(pred_emotion)
            emotion_scores.append(log_probs)
        
        predictions[true_emotion] = emotion_preds
        all_scores.append(emotion_scores)
    
    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=list(models.keys()))

    print("Evaluation complete.")
    
    return accuracy, cm, predictions, true_labels, pred_labels
