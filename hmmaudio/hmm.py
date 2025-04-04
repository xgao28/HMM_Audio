"""
Core implementation of Hidden Markov Models for the HMMAudio package.

This module provides a from-scratch implementation of HMM algorithms including:
- Forward algorithm for computing the probability of an observation sequence
- Backward algorithm for computing the probability of an observation sequence in reverse
- Viterbi algorithm for finding the most likely state sequence given observations
- Baum-Welch (EM) algorithm for parameter estimation from observations
"""

import numpy as np
from typing import Tuple, List, Optional, Union, Dict
from tqdm import tqdm

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



class ContinuousHMM:

    """
    Hidden Markov Model with Gaussian emissions for continuous observations.

    This class implements an HMM for continuous observations using multivariate
    Gaussian distributions for each state's emissions. Core algorithms include
    the forward-backward algorithm, Viterbi algorithm, and Baum-Welch (EM) algorithm
    for parameter estimation.

    Attributes:
        A (np.ndarray): Transition probability matrix of shape (N, N).
        pi (np.ndarray): Initial state distribution of shape (N,).
        means (np.ndarray): Mean vectors for each state, shape (N, D).
        covariances (np.ndarray): Covariance matrices for each state, shape (N, D, D).
        N (int): Number of hidden states.
        D (int): Dimensionality of the observation vectors.
    """

    def __init__(self, transition_matrix: Optional[np.ndarray] = None, 
                    means: Optional[np.ndarray] = None,
                    covars: Optional[np.ndarray] = None, 
                    initial_distribution: Optional[np.ndarray] = None,
                    n_states: int = 5,
                    n_features: int = 39,
                    diagonal_covariance: bool = False):
        """
        Initialize a continuous HMM with given parameters or random initialization.
        
        Args:
            transition_matrix: Matrix of shape (N, N) for state transitions. If None, initialized randomly.
            means: Matrix of shape (N, D) for Gaussian means. If None, initialized randomly.
            covariance: Array of shape (N, D, D) for Gaussian covariances. If None, initialized as identity matrices.
            initial_distribution: Vector of shape (N,) for initial state probabilities. If None, initialized with bias towards first state.
            n_states: Number of hidden states (only used if matrices are None).
            n_features: Dimensionality of observation vectors (only used if matrices are None).
        """
        # Store whether to use diagonal covariance matrix
        self.diagonal_covariance = diagonal_covariance

        # If parameters are provided, use them
        if transition_matrix is not None and means is not None and covars is not None and initial_distribution is not None:
            self.A = transition_matrix
            self.means = means
            self.covariances = covars 
            self.pi = initial_distribution
            self.N = transition_matrix.shape[0]
            self.D = means.shape[1]

        # Otherwise initialize randomly
        else:
            self.N = n_states
            self.D = n_features
            
            # Initialize transition matrix with left-to-right bias
            self.A = np.zeros((n_states, n_states))
            for i in range(n_states):
                if i < n_states - 1:
                    self.A[i, i] = 0.7  # Stay in same state
                    self.A[i, i+1] = 0.3  # Move to next state
                else:
                    self.A[i, i] = 1.0  # Last state loops to itself
            
            # Initialize means randomly but spaced out
            self.means = np.zeros((n_states, n_features))
            for i in range(n_states):
                self.means[i] = np.random.randn(n_features) + i * np.ones(n_features)/n_states
            
            # Initialize covariances based on the covariance type
            if self.diagonal_covariance:
                # For diagonal, we just need a vector of variances for each state
                self.covariances = np.array([np.ones(n_features) for _ in range(n_states)])
            else:
                # For full covariance, we need a full matrix per state
                self.covariances = np.array([np.eye(n_features) for _ in range(n_states)])
            
            # Initialize initial state distribution with bias towards first state
            self.pi = np.zeros(n_states)
            self.pi[0] = 0.9
            self.pi[1:] = 0.1 / (n_states - 1)

    def _emission_log_prob(self, obs: np.ndarray, state: int) -> float:
        """
        Compute the log probability of an observation under a state's Gaussian.

        Args:
            obs (np.ndarray): Observation vector of shape (D,).
            state (int): Index of the state.

        Returns:
            float: Log probability of the observation given the state.
        """
        diff = obs - self.means[state]
        
        if self.diagonal_covariance:
            # For diagonal covariance, we can use vectorized operations
            # covars[state] is a vector of variances
            inv_var = 1.0 / (self.covariances[state] + 1e-10)  # Add small constant to avoid division by zero
            log_det = np.sum(np.log(self.covariances[state] + 1e-10))
            exponent = -0.5 * np.sum(diff * diff * inv_var)
        else:
            # For full covariance matrix
            inv_cov = np.linalg.inv(self.covariances[state])
            log_det = np.log(np.linalg.det(self.covariances[state]))
            exponent = -0.5 * np.dot(diff.T, np.dot(inv_cov, diff))
            
        log_prob = exponent - 0.5 * (self.D * np.log(2 * np.pi) + log_det)
        return log_prob

    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the forward probabilities and log likelihood using scaling.

        Args:
            observations (np.ndarray): Observation sequence of shape (T, D).
                Should be a single 2D numpy array, not a list of arrays.

        Returns:
            Tuple[np.ndarray, float]: Forward probabilities and log likelihood.
        """
        if not isinstance(observations, np.ndarray) or len(observations.shape) != 2:
            raise TypeError("Expected observations to be a 2D numpy array with shape (time_steps, features)")
        T = observations.shape[0]
        alpha = np.zeros((T, self.N))
        scale_factors = np.zeros(T)

        # Initialization
        log_emission = np.array([self._emission_log_prob(observations[0], i) for i in range(self.N)])
        alpha[0] = self.pi * np.exp(log_emission)
        sum_alpha = np.sum(alpha[0])
        alpha[0] /= sum_alpha
        scale_factors[0] = sum_alpha

        # Induction
        for t in range(1, T):
            for i in range(self.N):
                alpha[t, i] = np.sum(alpha[t-1] * self.A[:, i]) * np.exp(
                    self._emission_log_prob(observations[t], i))
            sum_alpha = np.sum(alpha[t])
            alpha[t] /= sum_alpha
            scale_factors[t] = sum_alpha

        log_likelihood = np.sum(np.log(scale_factors))
        return alpha, log_likelihood

    def backward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the backward probabilities using scaling.

        Args:
            observations (np.ndarray): Observation sequence of shape (T, D).
                Should be a single 2D numpy array, not a list of arrays.

        Returns:
            Tuple[np.ndarray, float]: Backward probabilities and log likelihood.
        """
        if not isinstance(observations, np.ndarray) or len(observations.shape) != 2:
            raise TypeError("Expected observations to be a 2D numpy array with shape (time_steps, features)")
        T = observations.shape[0]
        beta = np.zeros((T, self.N))
        beta[-1] = 1.0
        scale_factors = np.zeros(T)
        scale_factors[-1] = 1.0

        for t in range(T-2, -1, -1):
            emission_log_probs = np.array([self._emission_log_prob(observations[t+1], i) for i in range(self.N)])
            emission_probs = np.exp(emission_log_probs)
            beta[t] = np.sum(self.A * emission_probs * beta[t+1], axis=1)
            sum_beta = np.sum(beta[t])
            beta[t] /= sum_beta
            scale_factors[t] = sum_beta

        # Compute log likelihood (same as forward)
        log_emission_0 = np.array([self._emission_log_prob(observations[0], i) for i in range(self.N)])
        initial_probs = self.pi * np.exp(log_emission_0) * beta[0]
        log_likelihood = np.log(np.sum(initial_probs)) + np.sum(np.log(scale_factors[1:]))
        return beta, log_likelihood

    def viterbi(self, observations: np.ndarray) -> Tuple[List[int], float]:
        """
        Find the most likely state sequence using the Viterbi algorithm in log space.

        Args:
            observations (np.ndarray): Observation sequence of shape (T, D).
                Should be a single 2D numpy array, not a list of arrays.

        Returns:
            Tuple[List[int], float]: Most likely state sequence and log probability.
        """
        if not isinstance(observations, np.ndarray) or len(observations.shape) != 2:
            raise TypeError("Expected observations to be a 2D numpy array with shape (time_steps, features)")
        T = observations.shape[0]
        log_delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)

        # Initialization
        log_pi = np.log(self.pi + 1e-12)
        log_emission_0 = np.array([self._emission_log_prob(observations[0], i) for i in range(self.N)])
        log_delta[0] = log_pi + log_emission_0

        # Recursion
        for t in range(1, T):
            for i in range(self.N):
                log_trans = np.log(self.A[:, i] + 1e-12)
                log_prob = log_delta[t-1] + log_trans
                psi[t, i] = np.argmax(log_prob)
                log_delta[t, i] = log_prob[psi[t, i]] + self._emission_log_prob(observations[t], i)

        # Backtracking
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(log_delta[-1])
        log_prob = log_delta[-1, path[-1]]

        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]

        return path.tolist(), log_prob

    def baum_welch(self, observations, max_iter: int = 10) -> None:
        """
        Estimate model parameters using the Baum-Welch algorithm.

        Args:
            observations: Either a single observation sequence as np.ndarray of shape (T, D)
                        or a list of observation sequences, each of shape (T_i, D).
            max_iter (int): Maximum number of iterations.
        """
        # Check if observations is a list of sequences or a single sequence
        is_list = isinstance(observations, list)
        eps = 1e-12
        
        # If it's a single numpy array, convert it to a list with one element
        if not is_list:
            if not isinstance(observations, np.ndarray):
                raise TypeError("Observations must be a numpy array or a list of numpy arrays")
            observations = [observations]
        
        # Verify that all elements are numpy arrays with the expected dimensions
        if not all(isinstance(obs, np.ndarray) and len(obs.shape) == 2 for obs in observations):
            raise ValueError("All observations must be 2D numpy arrays with shape (time_steps, features)")
            
        # Get the feature dimension from the first observation
        D = observations[0].shape[1]
            
        for i in range(max_iter):
            print(f"Iteration {i+1}/{max_iter}")
            
            # Accumulators for parameters
            A_num = np.zeros((self.N, self.N))
            A_denom = np.zeros(self.N)
            means_num = np.zeros((self.N, D))
            gamma_sum = np.zeros(self.N)
            pi_sum = np.zeros(self.N)
            # First, we need to initialize covars_num correctly depending on diagonal_covariance
            if self.diagonal_covariance:
                # For diagonal covariances, only accumulate diagonal elements
                covars_num = np.zeros((self.N, D))
            else:
                # For full covariances, accumulate full matrices
                covars_num = np.zeros((self.N, D, D))
            
            # Process each observation sequence
            for obs in tqdm(observations, desc="Baum-Welch Training Progress"):
                T = len(obs)
                if T <= 1:  # Skip sequences that are too short
                    continue
                    
                # E-step: Compute alpha, beta, gamma, xi for this sequence
                alpha, _ = self.forward(obs)
                beta, _ = self.backward(obs)

                gamma = alpha * beta
                gamma /= np.sum(gamma, axis=1, keepdims=True) + eps

                xi = np.zeros((T-1, self.N, self.N))
                for t in range(T-1):
                    emission_probs = np.array([self._emission_log_prob(obs[t+1], j) for j in range(self.N)])
                    emission_probs = np.exp(emission_probs)
                    temp = alpha[t].reshape(-1, 1) * self.A * emission_probs.reshape(1, -1) * beta[t+1].reshape(1, -1)
                    xi[t] = temp / (np.sum(temp) + eps)

                # Accumulate statistics
                A_num += np.sum(xi, axis=0)
                A_denom += np.sum(gamma[:-1], axis=0)
                pi_sum += gamma[0]
                
                # Accumulate for means and covariances
                for i in range(self.N):
                    gamma_i = gamma[:, i]
                    gamma_sum[i] += np.sum(gamma_i)
                    means_num[i] += np.sum(gamma_i[:, np.newaxis] * obs, axis=0)
                    
                    # Compute covariance contribution for this sequence
                    for t in range(T):
                        diff = obs[t] - self.means[i]
                        if self.diagonal_covariance:
                            # For diagonal covariance, just square each component
                            covars_num[i] += gamma[t, i] * (diff * diff)
                        else:
                            # For full covariance, compute outer product
                            covars_num[i] += gamma[t, i] * np.outer(diff, diff)
            
            # M-step: Update parameters
            # Update transition matrix
            new_A = A_num / (A_denom.reshape(-1, 1) + eps)
            new_A = np.clip(new_A, 1e-12, 1.0)
            new_A /= new_A.sum(axis=1, keepdims=True)

            # Update initial distribution
            new_pi = pi_sum + eps
            new_pi /= new_pi.sum()

            # Update means and covariances
            new_means = np.zeros_like(self.means)
            new_covs = np.zeros_like(self.covariances)
            for i in range(self.N):
                if gamma_sum[i] < eps:
                    new_means[i] = self.means[i]
                    new_covs[i] = self.covariances[i]
                    continue
                    
                new_means[i] = means_num[i] / gamma_sum[i]
                new_covs[i] = covars_num[i] / gamma_sum[i]
                
                # Apply regularization based on covariance type
                if self.diagonal_covariance:
                    # Add a small constant to diagonal variances
                    new_covs[i] += 1e-12
                else:
                    # Add a small constant to diagonal of covariance matrix
                    new_covs[i] += np.eye(D) * 1e-12

            # Update parameters
            self.A = new_A
            self.pi = new_pi
            self.means = new_means
            self.covariances = new_covs

    def fit(self, observations, max_iter: int = 10) -> None:
        self.baum_welch(observations, max_iter)
        
    def score(self, observations) -> float:
        """
        Calculate the log-likelihood of observation(s) under the model.
        
        Args:
            observations: Either a single observation sequence as np.ndarray of shape (T, D)
                        or a list of observation sequences, each of shape (T_i, D).
        
        Returns:
            float: Log-likelihood of the observation(s) under the model.
        """
        # Check if observations is a list or a single sequence
        is_list = isinstance(observations, list)
        if not is_list:
            if not isinstance(observations, np.ndarray):
                raise TypeError("Observations must be a numpy array or a list of numpy arrays")
            observations = [observations]
        
        # Verify that all elements are numpy arrays with the expected dimensions
        if not all(isinstance(obs, np.ndarray) and len(obs.shape) == 2 for obs in observations):
            raise ValueError("All observations must be 2D numpy arrays with shape (time_steps, features)")
        
        # Calculate the log-likelihood for each sequence and sum them
        total_log_likelihood = 0.0
        for obs in observations:
            _, log_likelihood = self.forward(obs)
            total_log_likelihood += log_likelihood
        
        return total_log_likelihood

    def generate_sequence(self, length: int) -> Tuple[np.ndarray, List[int]]:
        """
        Generate a state and observation sequence from the model.

        Args:
            length (int): Length of the sequence to generate.

        Returns:
            Tuple[np.ndarray, List[int]]: Observations (shape (length, D)) and state sequence.
        """
        observations = []
        states = []
        current_state = np.random.choice(self.N, p=self.pi)
        for _ in range(length):
            states.append(current_state)
            
            if self.diagonal_covariance:
                # For diagonal covariance, create a diagonal matrix from the variance vector
                diag_cov = np.diag(self.covariances[current_state])
                obs = np.random.multivariate_normal(
                    self.means[current_state],
                    diag_cov
                )
            else:
                # For full covariance
                obs = np.random.multivariate_normal(
                    self.means[current_state],
                    self.covariances[current_state]
                )
                
            observations.append(obs)
            current_state = np.random.choice(self.N, p=self.A[current_state])
        return np.array(observations), states