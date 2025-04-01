"""
Audio feature extraction module for the HMMAudio package.

This module provides functions for extracting features from audio data,
including MFCCs (Mel-frequency cepstral coefficients) and other common
audio features for audio analysis.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.io import wavfile


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return the normalized signal data and sample rate.
    
    Args:
        file_path: Path to the audio file (WAV format)
        
    Returns:
        audio: Numpy array containing the audio samples
        sample_rate: Sample rate of the audio in Hz
    """
    sample_rate, audio = wavfile.read(file_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Normalize audio data
    audio = audio / np.max(np.abs(audio))
        
    return audio, sample_rate


def preemphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Apply pre-emphasis filter on the input signal.
    
    Args:
        signal: Input audio signal
        coeff: Pre-emphasis coefficient
        
    Returns:
        Pre-emphasized signal
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def frame_signal(signal: np.ndarray, frame_size: int, frame_stride: int, 
                 pad_end: bool = True) -> np.ndarray:
    """
    Split the signal into overlapping frames.
    
    Args:
        signal: Input signal (audio samples)
        frame_size: Size of each frame (in samples)
        frame_stride: Step size between consecutive frames (in samples)
        pad_end: Whether to pad the last frame if needed
        
    Returns:
        Framed signal as a 2D array where each row is a frame
    """
    signal_length = len(signal)
    if pad_end:
        # Pad signal to ensure we have complete frames
        num_frames = np.ceil(float(np.abs(signal_length - frame_size)) / frame_stride).astype(np.int32)
        pad_signal_length = num_frames * frame_stride + frame_size
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(signal, z)
    else:
        pad_signal = signal
    
    # Create indices for frames
    indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_stride, frame_stride), 
                      (frame_size, 1)).T
    
    # Extract frames using indices
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames


def power_spectrum(frames: np.ndarray, nfft: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the power spectrum of each frame.
    
    Args:
        frames: Audio frames
        nfft: FFT size
        
    Returns:
        magnitude: Magnitude of the FFT
        power: Power spectrum
    """
    # Apply window function (Hamming)
    frames = frames * np.hamming(frames.shape[1])
    
    # Calculate FFT
    complex_spec = np.fft.rfft(frames, nfft)
    magnitude = np.absolute(complex_spec)
    
    # Power spectrum
    power = ((1.0 / nfft) * (magnitude ** 2))
    
    return magnitude, power


def freq_to_mel(freq):
    """Convert frequency to Mel scale."""
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def met_to_freq(mels):
    """Convert Mel scale to frequency."""
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, nfft: int, n_filters: int = 40, 
                  f_min: int = 0, f_max: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a Mel filterbank.
    
    Args:
        sample_rate: Sample rate of the audio
        nfft: FFT size
        n_filters: Number of Mel filters
        f_min: Minimum frequency
        f_max: Maximum frequency (default: sample_rate/2)
        
    Returns:
        filterbank: Mel filterbank matrix (n_filters × (nfft/2+1))
        mel_freqs: Mel frequencies
    """
    f_max = f_max or sample_rate // 2
    
    # Convert min and max frequencies to Mel scale
    mel_min = freq_to_mel(f_min)
    mel_max = freq_to_mel(f_max)
    
    # Generate equally spaced points on the Mel scale
    mels = np.linspace(mel_min, mel_max, n_filters + 2)
    
    # Convert Mel points back to frequency
    freqs = met_to_freq(mels)
    
    # Calculate bin indices
    bin_indices = np.floor((nfft + 1) * freqs / sample_rate).astype(int)
    
    # Create filterbank
    filterbank = np.zeros((n_filters, nfft // 2 + 1))
    
    for i in range(n_filters):
        # For each filter, create a triangular filter
        filterbank[i, bin_indices[i]:bin_indices[i+1]] = np.linspace(0, 1, bin_indices[i+1] - bin_indices[i])
        filterbank[i, bin_indices[i+1]:bin_indices[i+2]] = np.linspace(1, 0, bin_indices[i+2] - bin_indices[i+1])
    
    return filterbank, freqs[1:-1]  # Return center frequencies as well


def extract_mfcc(signal: np.ndarray, sample_rate: int, 
                num_cepstral: int = 13, 
                frame_size: float = 0.025, 
                frame_stride: float = 0.01,
                preemphasis_coeff: float = 0.97,
                n_filters: int = 40,
                nfft: int = 512,
                lifter: bool = True,
                normalize: bool = True) -> np.ndarray:
    """
    Extract MFCC features from an audio signal.
    
    Args:
        signal: Input audio signal
        sample_rate: Sample rate of the audio
        num_cepstral: Number of cepstral coefficients to return
        frame_size: Size of each frame in seconds
        frame_stride: Step size between consecutive frames in seconds
        preemphasis_coeff: Pre-emphasis coefficient
        n_filters: Number of Mel filters
        nfft: FFT size
        lifter: Whether to apply liftering to the cepstral coefficients
        normalize: Whether to mean and variance normalize the MFCC features
    
    Returns:
        MFCC coefficients for each frame
    """
    # Pre-emphasis
    emphasized_signal = preemphasis(signal, preemphasis_coeff)
    
    # Framing
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    frames = frame_signal(emphasized_signal, frame_length, frame_step)
    
    # Power spectrum
    _, power_spec = power_spectrum(frames, nfft)
    
    # Mel filterbank
    mel_filter, _ = mel_filterbank(sample_rate, nfft, n_filters)
    
    # Apply filterbank to power spectrum
    mel_energies = np.dot(power_spec, mel_filter.T)
    
    # Log of the mel energies
    log_mel_energies = np.log(mel_energies + 1e-10)
    
    # DCT to get the cepstral coefficients
    mfcc = np.dot(log_mel_energies, np.cos(np.pi * np.outer(
        np.arange(0, log_mel_energies.shape[1]), 
        np.arange(0.5, num_cepstral + 0.5)) / log_mel_energies.shape[1]))
    
    # Liftering
    if lifter:
        n = np.arange(num_cepstral)
        lift = 1 + (22/2) * np.sin(np.pi * n / 22)
        mfcc *= lift
    
    # Mean and variance normalization
    if normalize:
        mfcc -= np.mean(mfcc, axis=0)
        mfcc /= np.std(mfcc, axis=0) + 1e-10
    
    return mfcc


def compute_delta(features: np.ndarray, N: int = 2) -> np.ndarray:
    """
    Compute delta features (first order derivatives).
    
    Args:
        features: Input feature matrix
        N: Number of frames over which to compute the delta
    
    Returns:
        Delta features
    """
    padded = np.pad(features, ((N, N), (0, 0)), mode='edge')
    delta = np.zeros_like(features)
    
    denominator = 2 * sum(n**2 for n in range(1, N+1))
    
    for t in range(features.shape[0]):
        delta[t] = np.sum(
            np.array([n * (padded[t+N+n] - padded[t+N-n]) for n in range(1, N+1)]), 
            axis=0
        ) / denominator
    
    return delta


def combine_features(feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Combine multiple feature matrices into a single matrix.
    
    Args:
        feature_dict: Dictionary of feature matrices
        
    Returns:
        Combined feature matrix
    """
    # Check if all features have the same number of frames
    if not feature_dict:
        return np.array([])
    
    num_frames = next(iter(feature_dict.values())).shape[0]
    all_same_length = all(f.shape[0] == num_frames for f in feature_dict.values())
    
    if not all_same_length:
        raise ValueError("All feature matrices must have the same number of frames")
    
    # Horizontally stack all feature matrices
    feature_list = [feature_dict[name] for name in sorted(feature_dict.keys())]
    return np.hstack(feature_list)


def extract_features(file_path: str, 
                     include_mfcc: bool = True,
                     include_delta: bool = True,
                     include_delta2: bool = True,
                     include_energy: bool = False,
                     num_cepstral: int = 13,
                     **kwargs) -> Dict[str, np.ndarray]:
    """
    Extract a comprehensive set of features from an audio file.
    
    Args:
        file_path: Path to the audio file
        include_mfcc: Whether to include MFCC features
        include_delta: Whether to include delta (first derivative) features
        include_delta2: Whether to include delta-delta (second derivative) features
        include_energy: Whether to include energy as a feature
        num_cepstral: Number of cepstral coefficients for MFCC
        **kwargs: Additional arguments passed to extract_mfcc
    
    Returns:
        Dictionary containing the extracted features
    """
    signal, sample_rate = load_audio(file_path)
    features = {}
    
    if include_mfcc:
        mfcc = extract_mfcc(signal, sample_rate, num_cepstral=num_cepstral, **kwargs)
        features['mfcc'] = mfcc
        
        if include_delta:
            delta = compute_delta(mfcc)
            features['delta'] = delta
            
            if include_delta2:
                delta_delta = compute_delta(delta)
                features['delta_delta'] = delta_delta
    
    if include_energy:
        # Compute frame energy
        frame_length = int(round(kwargs.get('frame_size', 0.025) * sample_rate))
        frame_step = int(round(kwargs.get('frame_stride', 0.01) * sample_rate))
        frames = frame_signal(signal, frame_length, frame_step)
        energy = np.sum(frames**2, axis=1)
        log_energy = np.log(energy + 1e-10)
        features['energy'] = log_energy.reshape(-1, 1)
    
    return combine_features(features)
