"""
Core implementation of Hidden Markov Models for the HMMAudio package.

This module provides a from-scratch implementation of HMM algorithms including:
- Forward algorithm for computing the probability of an observation sequence
- Backward algorithm for computing the probability of an observation sequence in reverse
- Viterbi algorithm for finding the most likely state sequence given observations
- Baum-Welch (EM) algorithm for parameter estimation from observations
"""

import numpy as np
from typing import Tuple, List, Optional, Union

class HiddenMarkovModel:
    """
    Hidden Markov Model implementation with algorithms for inference and parameter estimation.
    
    This class implements core HMM algorithms including forward, backward, Viterbi,
    and Baum-Welch algorithms for discrete observation sequences.
    
    Attributes:
        A (np.ndarray): Transition probability matrix of shape (N, N) where N is the number of states.
                       A[i,j] represents the probability of transitioning from state i to state j.
        B (np.ndarray): Emission probability matrix of shape (N, M) where M is the number of possible
                       observation symbols. B[i,j] represents the probability of observing symbol j
                       while in state i.
        pi (np.ndarray): Initial state distribution of shape (N,). pi[i] represents the probability
                        of starting in state i.
        N (int): Number of hidden states in the model.
        M (int): Number of distinct observation symbols.
    """
    def __init__(self, transition_matrix: np.ndarray, emission_matrix: np.ndarray, initial_distribution: np.ndarray):
        """
        Initialize the Hidden Markov Model with given parameters.
        
        Args:
            transition_matrix (np.ndarray): Matrix of shape (N, N) representing state transition probabilities.
            emission_matrix (np.ndarray): Matrix of shape (N, M) representing emission probabilities.
            initial_distribution (np.ndarray): Vector of shape (N,) representing initial state probabilities.
        """
        self.A = np.clip(transition_matrix, 1e-20, 1.0)  # Prevent zeros
        self.B = np.clip(emission_matrix, 1e-20, 1.0)
        self.pi = np.clip(initial_distribution, 1e-20, 1.0)
        self.N = transition_matrix.shape[0]
        self.M = emission_matrix.shape[1]

    def forward(self, observations: List[int]) -> Tuple[np.ndarray, float]:
        """
        Compute the forward probabilities for a sequence of observations.
        
        Implements the forward algorithm to calculate alpha values, which represent
        the probability of observing the sequence up to time t and being in state i.
        
        Args:
            observations (List[int]): Sequence of observation indices (0-indexed).
        
        Returns:
            Tuple[np.ndarray, float]: A tuple containing:
                - alpha: Matrix of shape (T, N) where T is the length of observations and
                  N is the number of states. alpha[t,i] is the probability of observing the
                  sequence up to time t and being in state i.
                - log_prob: Log probability of the observation sequence under the model.
        """
        T = len(observations)
        alpha = np.zeros((T, self.N))
        alpha[0] = self.pi * self.B[:, observations[0]]
        alpha[0] /= np.sum(alpha[0])  # Normalization
        
        for t in range(1, T):
            alpha[t] = np.dot(alpha[t-1], self.A) * self.B[:, observations[t]]
            alpha[t] /= np.sum(alpha[t])  # Scaling to prevent underflow
            
        log_prob = np.sum(np.log(np.sum(alpha, axis=1)))
        return alpha, log_prob

    def backward(self, observations: List[int]) -> Tuple[np.ndarray, float]:
        """
        Compute the backward probabilities for a sequence of observations.
        
        Implements the backward algorithm to calculate beta values, which represent
        the probability of observing the sequence from time t+1 to the end, given state i at time t.
        
        Args:
            observations (List[int]): Sequence of observation indices (0-indexed).
        
        Returns:
            Tuple[np.ndarray, float]: A tuple containing:
                - beta: Matrix of shape (T, N) where T is the length of observations and
                  N is the number of states. beta[t,i] is the probability of observing the
                  sequence from time t+1 to the end, given state i at time t.
                - log_prob: Log probability of the observation sequence under the model.
        """
        T = len(observations)
        beta = np.ones((T, self.N))
        
        for t in range(T-2, -1, -1):
            beta[t] = np.dot(self.A, self.B[:, observations[t+1]] * beta[t+1])
            beta[t] /= np.sum(beta[t])  # Scaling
            
        log_prob = np.log(np.sum(self.pi * self.B[:, observations[0]] * beta[0]))
        return beta, log_prob

    def viterbi(self, observations: List[int]) -> Tuple[List[int], float]:
        """
        Find the most likely state sequence for a given observation sequence.
        
        Implements the Viterbi algorithm to find the sequence of states that maximizes
        the joint probability of the states and observations.
        
        Args:
            observations (List[int]): Sequence of observation indices (0-indexed).
        
        Returns:
            Tuple[List[int], float]: A tuple containing:
                - path: List of state indices (0-indexed) representing the most likely state sequence.
                - log_prob: Log probability of the most likely state sequence.
        """
        T = len(observations)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)
        delta[0] = self.pi * self.B[:, observations[0]]
        for t in range(1, T):
            trans_probs = delta[t-1].reshape(-1, 1) * self.A
            max_values = np.max(trans_probs, axis=0)
            delta[t] = max_values * self.B[:, observations[t]]
            psi[t] = np.argmax(trans_probs, axis=0)
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        log_prob = np.log(np.max(delta[-1]))
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        return path.tolist(), log_prob

    def baum_welch(self, observations: List[int], max_iter: int = 100) -> None:
        """
        Estimate model parameters using the Baum-Welch (EM) algorithm.
        
        Updates the model parameters (A, B, pi) to maximize the likelihood of the
        observed sequence. This is an implementation of the Expectation-Maximization
        algorithm for HMMs.
        
        Args:
            observations (List[int]): Sequence of observation indices (0-indexed).
            max_iter (int, optional): Maximum number of iterations. Defaults to 10.
            
        Note:
            This method modifies the model parameters in-place.
        """
        T = len(observations)
        eps = 1e-20  # Minimum probability threshold
        
        for _ in range(max_iter):
            # Forward-Backward with scaling
            alpha, _ = self.forward(observations)
            beta, _ = self.backward(observations)
            
            # Gamma calculation with stability
            gamma = alpha * beta
            gamma += eps
            gamma /= gamma.sum(axis=1, keepdims=True)
            
            # Xi calculation
            xi = np.zeros((T-1, self.N, self.N))
            for t in range(T-1):
                trans = self.A * np.outer(alpha[t], self.B[:, observations[t+1]] * beta[t+1])
                xi[t] = trans / (np.sum(trans) + eps)
            
            # Update parameters with smoothing
            self.A = np.clip(np.sum(xi, axis=0) / (np.sum(gamma[:-1], axis=0) + eps).reshape(-1, 1), 1e-5, 1.0)
            self.A /= self.A.sum(axis=1, keepdims=True)
            
            # Update emission probabilities
            for k in range(self.M):
                mask = (np.array(observations) == k)
                self.B[:, k] = np.clip(np.sum(gamma[mask], axis=0) / (np.sum(gamma, axis=0) + eps), 1e-5, 1.0)
            self.B /= self.B.sum(axis=1, keepdims=True)
            
            # Update initial probabilities
            self.pi = np.clip(gamma[0], 1e-5, 1.0)
            self.pi /= self.pi.sum()

    def fit(self, observations: List[int], max_iter: int = 100) -> None:
        """
        Fit the model to the observed data using the Baum-Welch algorithm.
        
        This is a convenience wrapper around the baum_welch method.
        """
        self.baum_welch(observations, max_iter)


    def generate_sequence(self, length: int) -> Tuple[List[int], List[int]]:
        """
        Generate a random observation sequence and corresponding state sequence.
        
        Samples a sequence of states and observations according to the model parameters.
        
        Args:
            length (int): Length of the sequence to generate.
            
        Returns:
            Tuple[List[int], List[int]]: A tuple containing:
                - observations: List of observation indices (0-indexed).
                - states: List of state indices (0-indexed).
        """
        observations = []
        states = []
        current_state = np.random.choice(self.N, p=self.pi)
        for _ in range(length):
            states.append(current_state)
            obs = np.random.choice(self.M, p=self.B[current_state])
            observations.append(obs)
            current_state = np.random.choice(self.N, p=self.A[current_state])
        return observations, states

if __name__ == "__main__":
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