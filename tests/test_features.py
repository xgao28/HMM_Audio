"""
Tests for the feature extraction module.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hmmaudio.features import (
    load_audio, preemphasis, frame_signal, power_spectrum,
    mel_filterbank, extract_mfcc, compute_delta, extract_features
)

class FeatureTest(unittest.TestCase):
    """Test suite for the HMMAudio feature extraction module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a synthetic audio signal for testing
        self.sample_rate = 16000
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        
        # Create a simple sine wave with two frequencies
        self.signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        
        # Create a temporary WAV file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.wav_path = os.path.join(self.temp_dir.name, "test_audio.wav")
        wavfile.write(self.wav_path, self.sample_rate, self.signal)
        
        # Common parameters
        self.frame_size = 0.025  # 25ms
        self.frame_stride = 0.01  # 10ms
        self.nfft = 512
        self.num_cepstral = 13
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_load_audio(self):
        """Test the load_audio function."""
        audio, sr = load_audio(self.wav_path)
        
        self.assertEqual(sr, self.sample_rate)
        self.assertEqual(len(audio), len(self.signal))
        # Check if the loaded audio has been normalized
        self.assertAlmostEqual(np.max(np.abs(audio)), 1.0)
        
    def test_preemphasis(self):
        """Test the preemphasis function."""
        coeff = 0.97
        emphasized = preemphasis(self.signal, coeff)
        
        # Check output length
        self.assertEqual(len(emphasized), len(self.signal))
        # Check first sample (should remain unchanged)
        self.assertEqual(emphasized[0], self.signal[0])
        # Check a random sample to ensure the filter is applied
        idx = 1000
        expected = self.signal[idx] - coeff * self.signal[idx-1]
        self.assertAlmostEqual(emphasized[idx], expected)
        
    def test_frame_signal(self):
        """Test the frame_signal function."""
        frame_length = int(self.frame_size * self.sample_rate)
        frame_step = int(self.frame_stride * self.sample_rate)
        
        frames = frame_signal(self.signal, frame_length, frame_step)
        
        # Main test - check frame dimensions
        self.assertEqual(frames.shape[1], frame_length)
        
        # Check if the first frame contains the first frame_length samples
        np.testing.assert_array_almost_equal(frames[0], self.signal[:frame_length])
        
    def test_power_spectrum(self):
        """Test the power spectrum computation."""
        frame_length = int(self.frame_size * self.sample_rate)
        frame_step = int(self.frame_stride * self.sample_rate)
        
        frames = frame_signal(self.signal, frame_length, frame_step)
        magnitude, power = power_spectrum(frames, self.nfft)
        
        # Check dimensions
        self.assertEqual(magnitude.shape[0], frames.shape[0])
        self.assertEqual(magnitude.shape[1], self.nfft // 2 + 1)
        self.assertEqual(power.shape, magnitude.shape)
        
        # Power should be non-negative
        self.assertTrue(np.all(power >= 0))
        
        # Check relationship between magnitude and power
        expected_power = (1.0 / self.nfft) * (magnitude ** 2)
        np.testing.assert_array_almost_equal(power, expected_power)
        
    def test_mel_filterbank(self):
        """Test the mel filterbank generation."""
        n_filters = 40
        filterbank, mel_freqs = mel_filterbank(self.sample_rate, self.nfft, n_filters)
        
        # Check dimensions
        self.assertEqual(filterbank.shape, (n_filters, self.nfft // 2 + 1))
        self.assertEqual(len(mel_freqs), n_filters)
        
        # Check if filterbank sums approximately to triangular filters
        filter_sums = np.sum(filterbank, axis=1)
        self.assertTrue(np.all(filter_sums > 0))
        
        # Check if the generated frequencies are in ascending order
        self.assertTrue(np.all(np.diff(mel_freqs) > 0))
        
    def test_extract_mfcc(self):
        """Test MFCC extraction."""
        mfcc = extract_mfcc(
            self.signal, 
            self.sample_rate, 
            num_cepstral=self.num_cepstral,
            frame_size=self.frame_size,
            frame_stride=self.frame_stride,
            nfft=self.nfft
        )
        
        # Check that the number of frames is close to what we expect
        frame_length = int(self.frame_size * self.sample_rate)
        frame_step = int(self.frame_stride * self.sample_rate)
        signal_length = len(self.signal)
        min_expected_frames = (signal_length - frame_length) // frame_step
        max_expected_frames = min_expected_frames + 2  # Allow some flexibility
        
        self.assertGreaterEqual(mfcc.shape[0], min_expected_frames)
        self.assertLessEqual(mfcc.shape[0], max_expected_frames)
        
        # Check MFCC dimensions
        self.assertEqual(mfcc.shape[1], self.num_cepstral)
        
        # Check if normalized properly
        self.assertAlmostEqual(np.mean(mfcc), 0, delta=0.1)
        self.assertAlmostEqual(np.std(mfcc), 1, delta=0.5)
        
    def test_compute_delta(self):
        """Test delta feature computation."""
        # Create a sample feature matrix
        features = np.random.randn(100, 13)
        
        delta = compute_delta(features, N=2)
        
        # Check dimensions
        self.assertEqual(delta.shape, features.shape)
        
        # Manually calculate delta for a single frame to verify
        t = 50  # random frame index
        N = 2
        denominator = 2 * sum(n**2 for n in range(1, N+1))
        expected_delta = np.sum(
            np.array([n * (features[t+n] - features[t-n]) for n in range(1, N+1)]), 
            axis=0
        ) / denominator
        
        np.testing.assert_array_almost_equal(delta[t], expected_delta)
        
    def test_extract_features(self):
        """Test the full feature extraction pipeline."""
        features = extract_features(
            self.wav_path,
            include_mfcc=True,
            include_delta=True,
            include_delta2=True,
            include_energy=True,
            num_cepstral=self.num_cepstral,
            frame_size=self.frame_size,
            frame_stride=self.frame_stride,
            nfft=self.nfft
        )
        
        # Calculate expected dimension: MFCC + Delta + Delta-Delta + Energy
        expected_dim = self.num_cepstral * 3 + 1
        
        # Check features have reasonable number of frames
        frame_length = int(self.frame_size * self.sample_rate)
        frame_step = int(self.frame_stride * self.sample_rate)
        signal_length = len(self.signal)
        min_expected_frames = (signal_length - frame_length) // frame_step
        max_expected_frames = min_expected_frames + 2
        
        self.assertGreaterEqual(features.shape[0], min_expected_frames)
        self.assertLessEqual(features.shape[0], max_expected_frames)
        
        # Check dimensions
        self.assertEqual(features.shape[1], expected_dim)


class FeaturesExceptionTest(unittest.TestCase):
    """Test exception handling in the HMMAudio features module."""
    
    def test_combine_features_exception(self):
        """Test that combine_features raises an exception when features have different lengths."""
        from hmmaudio.features import combine_features
        
        # Create feature matrices with different lengths
        feature_dict = {
            'mfcc': np.random.randn(100, 13),
            'delta': np.random.randn(95, 13)  # Different length
        }
        
        with self.assertRaises(ValueError):
            combine_features(feature_dict)
            

def run_all_tests():
    """Run all tests with detailed output."""
    unittest.main(verbosity=2)
    
def run_visual_test():
    """Run a visual test that plots example features."""
    test = FeatureTest()
    test.setUp()
    
    # Extract and visualize features
    mfcc = extract_mfcc(
        test.signal, 
        test.sample_rate, 
        num_cepstral=test.num_cepstral,
        frame_size=test.frame_size,
        frame_stride=test.frame_stride,
        nfft=test.nfft
    )
    
    plt.figure(figsize=(12, 6))
    
    # Plot the audio signal
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(test.signal)) / test.sample_rate, test.signal)
    plt.title('Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Plot the MFCC features
    plt.subplot(2, 1, 2)
    plt.imshow(mfcc.T, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('MFCC Features')
    plt.xlabel('Frame')
    plt.ylabel('Coefficient')
    
    plt.tight_layout()
    plt.savefig('audio_analysis.png')
    plt.show()
    
    test.tearDown()

if __name__ == "__main__":
    # Run all tests
    run_all_tests()
    
    # Comment above and uncomment below to run visual test 
    # run_visual_test()