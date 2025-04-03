import os
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hmmaudio.hmm import HiddenMarkovModel
from hmmlearn import hmm
import copy
np.random.seed(42)

def generate_mfcc_like_data(n_frames=100, n_features=13, n_states=3):
    """
    Generate synthetic MFCC-like data for testing HMM models.
    
    Args:
        n_frames (int): Number of frames (time steps)
        n_features (int): Number of MFCC features per frame
        n_states (int): Number of hidden states to simulate
        
    Returns:
        tuple: (observations, states, true_trans)
            observations: List of quantized observation indices
            states: List of true hidden states used to generate observations
            true_trans: True transition probability matrix used to generate states
    """
    # Define state-dependent distributions for MFCC features
    means = np.array([
        np.random.normal(loc=i * 5.0, scale=1.0, size=n_features) for i in range(n_states)
    ])
    
    # Define the true transition probabilities
    true_trans = np.full((n_states, n_states), 0.1)
    np.fill_diagonal(true_trans, 0.7)
    for i in range(n_states):
        true_trans[i] /= true_trans[i].sum()  # Normalize rows to sum to 1
    
    initial_probs = np.full(n_states, 1.0 / n_states)
    
    # Generate state sequence
    states = []
    current_state = np.random.choice(n_states, p=initial_probs)
    states.append(current_state)
    
    for t in range(1, n_frames):
        current_state = np.random.choice(n_states, p=true_trans[current_state])
        states.append(current_state)
    
    # Generate MFCC features based on states
    raw_features = np.array([means[state] + np.random.normal(0, 1, n_features) for state in states])
    
    # Quantize features into discrete symbols (e.g., VQ codebook)
    # For simplicity, we'll just quantize each feature into 5 possible values
    num_symbols = 5
    min_val, max_val = raw_features.min(), raw_features.max()
    bins = np.linspace(min_val, max_val, num_symbols + 1)
    
    # For simplicity, we'll use only the first MFCC coefficient for quantization
    quantized = np.digitize(raw_features[:, 0], bins[1:-1])
    
    return quantized.tolist(), states, true_trans

# Generate MFCC-like data
n_states = 8
n_symbols = 5
observations, true_states, true_trans = generate_mfcc_like_data(n_frames=1000, n_states=n_states)

# Initialize models with wrong parameters
init_trans = np.full((n_states, n_states), 1/n_states)
init_emission = np.full((n_states, n_symbols), 1/n_symbols)
init_start = np.full(n_states, 1/n_states)

# Our HMM
our_hmm = HiddenMarkovModel(copy.deepcopy(init_trans), 
                            copy.deepcopy(init_emission),
                            copy.deepcopy(init_start))
our_hmm.fit(observations, max_iter=100)





# hmmlearn model
hmm_model = hmm.CategoricalHMM(n_components=n_states, n_iter=100,
                                init_params='', params='ste')
hmm_model.startprob_ = copy.deepcopy(init_start)
hmm_model.transmat_ = copy.deepcopy(init_trans)
hmm_model.emissionprob_ = copy.deepcopy(init_emission)
hmm_model.fit(np.array(observations).reshape(-1, 1))

# Results comparison
print("True Transition:\n", true_trans)
print("\nOur Learned Transition:\n", np.round(our_hmm.A, 2))
print("hmmlearn Transition:\n", np.round(hmm_model.transmat_, 2))

print("\nOur Learned Emission:\n", np.round(our_hmm.B, 2))
print("hmmlearn Emission:\n", np.round(hmm_model.emissionprob_, 2))

# Verify similarity
assert np.allclose(our_hmm.A, hmm_model.transmat_, atol=0.15), \
    "Transition matrices diverged"
assert np.allclose(our_hmm.B, hmm_model.emissionprob_, atol=0.15), \
    "Emission matrices diverged"
print("\nMFCC validation passed! Both implementations show similar behavior.")