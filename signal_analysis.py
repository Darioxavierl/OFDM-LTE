"""
Signal Analysis Module

Provides utilities for calculating signal metrics:
- BER (Bit Error Rate)
- PAPR (Peak-to-Average Power Ratio)
- EVM (Error Vector Magnitude)
- CCDF (Complementary Cumulative Distribution Function)
- Power spectral density
- And other signal characteristics
"""

import numpy as np
from scipy import stats


class BERAnalyzer:
    """Bit Error Rate analysis utilities"""
    
    @staticmethod
    def calculate_ber(bits_tx, bits_rx):
        """
        Calculate BER between transmitted and received bits
        
        Args:
            bits_tx (ndarray): Transmitted bits
            bits_rx (ndarray): Received bits
        
        Returns:
            float: Bit Error Rate (0.0 to 1.0)
        """
        if len(bits_tx) == 0:
            return 0.0
        
        # Ensure same length
        min_len = min(len(bits_tx), len(bits_rx))
        bits_tx = bits_tx[:min_len]
        bits_rx = bits_rx[:min_len]
        
        errors = np.sum(bits_tx != bits_rx)
        ber = errors / len(bits_tx)
        return ber
    
    @staticmethod
    def calculate_ber_with_confidence(bits_tx, bits_rx, confidence=0.95):
        """
        Calculate BER with confidence interval
        
        Args:
            bits_tx (ndarray): Transmitted bits
            bits_rx (ndarray): Received bits
            confidence (float): Confidence level (0-1)
        
        Returns:
            dict: BER statistics with confidence interval
        """
        ber = BERAnalyzer.calculate_ber(bits_tx, bits_rx)
        n = len(bits_tx)
        
        # Wilson score interval
        z = stats.norm.ppf((1 + confidence) / 2)
        denominator = 1 + z**2 / n
        centre_adjusted = (ber + z**2 / (2*n)) / denominator
        adjusted_std = np.sqrt((ber * (1 - ber) + z**2 / (4*n)) / n) / denominator
        
        lower = max(0, centre_adjusted - z * adjusted_std)
        upper = min(1, centre_adjusted + z * adjusted_std)
        
        return {
            'ber': ber,
            'errors': np.sum(bits_tx != bits_rx),
            'total_bits': len(bits_tx),
            'ci_lower': lower,
            'ci_upper': upper,
            'ci_confidence': confidence
        }


class PAPRAnalyzer:
    """Peak-to-Average Power Ratio analysis utilities"""
    
    @staticmethod
    def calculate_papr(signal):
        """
        Calculate PAPR of a time-domain signal
        
        Args:
            signal (ndarray): Complex time-domain signal
        
        Returns:
            dict: PAPR statistics
                - papr_ratio: PAPR as linear ratio
                - papr_db: PAPR in dB
                - peak_power: Peak instantaneous power
                - avg_power: Average power
        """
        power = np.abs(signal) ** 2
        peak_power = np.max(power)
        avg_power = np.mean(power)
        
        papr_ratio = peak_power / avg_power if avg_power > 0 else 1.0
        papr_db = 10 * np.log10(papr_ratio)
        
        return {
            'papr_ratio': papr_ratio,
            'papr_db': papr_db,
            'peak_power': peak_power,
            'avg_power': avg_power
        }
    
    @staticmethod
    def calculate_ccdf(papr_values_db):
        """
        Calculate CCDF (Complementary Cumulative Distribution Function) of PAPR
        
        Args:
            papr_values_db (ndarray): PAPR values in dB
        
        Returns:
            dict: CCDF data
                - papr_x: PAPR values (sorted)
                - ccdf_y: Probability P(PAPR > x)
        """
        if len(papr_values_db) == 0:
            return {'papr_x': np.array([]), 'ccdf_y': np.array([])}
        
        # Sort PAPR values descending
        sorted_papr = np.sort(papr_values_db)[::-1]
        
        # Calculate CDF
        cdf = np.arange(1, len(sorted_papr) + 1) / len(sorted_papr)
        
        # CCDF = 1 - CDF = P(PAPR > x)
        ccdf = 1 - cdf
        
        return {
            'papr_x': sorted_papr,
            'ccdf_y': ccdf
        }
    
    @staticmethod
    def get_papr_percentile(papr_values_db, percentile=99):
        """
        Get PAPR value at specified percentile
        
        Args:
            papr_values_db (ndarray): PAPR values in dB
            percentile (float): Percentile (0-100)
        
        Returns:
            float: PAPR value at percentile
        """
        return np.percentile(papr_values_db, percentile)


class EVMAnalyzer:
    """Error Vector Magnitude analysis utilities"""
    
    @staticmethod
    def calculate_evm(symbols_tx, symbols_rx):
        """
        Calculate EVM (Error Vector Magnitude) as percentage
        
        Args:
            symbols_tx (ndarray): Transmitted symbols
            symbols_rx (ndarray): Received symbols
        
        Returns:
            dict: EVM statistics
                - evm_percent: EVM as percentage
                - evm_db: EVM in dB
                - rms_error: RMS error
        """
        if len(symbols_tx) == 0:
            return {'evm_percent': 0, 'evm_db': -np.inf, 'rms_error': 0}
        
        # Ensure same length
        min_len = min(len(symbols_tx), len(symbols_rx))
        symbols_tx = symbols_tx[:min_len]
        symbols_rx = symbols_rx[:min_len]
        
        # Error vector
        error_vector = symbols_rx - symbols_tx
        
        # RMS error
        rms_error = np.sqrt(np.mean(np.abs(error_vector) ** 2))
        
        # Reference power
        ref_power = np.sqrt(np.mean(np.abs(symbols_tx) ** 2))
        
        # EVM as percentage
        evm_percent = (rms_error / ref_power * 100) if ref_power > 0 else 0
        evm_db = 10 * np.log10(evm_percent / 100) if evm_percent > 0 else -np.inf
        
        return {
            'evm_percent': evm_percent,
            'evm_db': evm_db,
            'rms_error': rms_error,
            'ref_power': ref_power
        }


class ChannelAnalyzer:
    """Channel characteristics analysis"""
    
    @staticmethod
    def calculate_channel_impulse_response_stats(h):
        """
        Calculate statistics of channel impulse response
        
        Args:
            h (ndarray): Channel impulse response
        
        Returns:
            dict: Channel statistics
        """
        power = np.abs(h) ** 2
        
        return {
            'rms_delay_spread': ChannelAnalyzer._rms_delay_spread(power),
            'mean_excess_delay': ChannelAnalyzer._mean_excess_delay(power),
            'number_of_paths': np.sum(power > np.max(power) * 0.01),  # -20dB threshold
            'max_path_power_db': 10 * np.log10(np.max(power)),
            'coherence_bandwidth_hz': ChannelAnalyzer._coherence_bandwidth(power)
        }
    
    @staticmethod
    def _rms_delay_spread(power):
        """Calculate RMS delay spread"""
        tau = np.arange(len(power))
        mean_delay = np.sum(tau * power) / np.sum(power)
        rms_delay = np.sqrt(np.sum(((tau - mean_delay) ** 2) * power) / np.sum(power))
        return rms_delay
    
    @staticmethod
    def _mean_excess_delay(power):
        """Calculate mean excess delay"""
        if len(power) > 0:
            tau = np.arange(len(power))
            return np.sum(tau * power) / np.sum(power)
        return 0
    
    @staticmethod
    def _coherence_bandwidth(power, correlation_threshold=0.9):
        """Calculate coherence bandwidth"""
        rms_delay = ChannelAnalyzer._rms_delay_spread(power)
        if rms_delay > 0:
            return 1 / (5 * rms_delay)
        return np.inf


class PowerAnalyzer:
    """Power and spectral analysis utilities"""
    
    @staticmethod
    def calculate_power(signal):
        """
        Calculate signal power
        
        Args:
            signal (ndarray): Time-domain or frequency-domain signal
        
        Returns:
            dict: Power statistics
        """
        power = np.abs(signal) ** 2
        
        return {
            'avg_power': np.mean(power),
            'peak_power': np.max(power),
            'rms_power': np.sqrt(np.mean(power)),
            'crest_factor': np.max(np.abs(signal)) / np.sqrt(np.mean(power))
        }
    
    @staticmethod
    def calculate_snr(signal, noise):
        """
        Calculate SNR from signal and noise
        
        Args:
            signal (ndarray): Signal
            noise (ndarray): Noise
        
        Returns:
            float: SNR in dB
        """
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = np.mean(np.abs(noise) ** 2)
        
        if noise_power == 0:
            return np.inf
        
        snr = signal_power / noise_power
        return 10 * np.log10(snr)


class MetricsCollector:
    """Convenience class to collect all metrics"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics = {}
    
    def add_transmission(self, bits_tx, bits_rx, symbols_tx, symbols_rx, signal_tx, snr_db):
        """
        Add transmission metrics
        
        Args:
            bits_tx: Transmitted bits
            bits_rx: Received bits
            symbols_tx: Transmitted symbols
            symbols_rx: Received symbols
            signal_tx: Transmitted signal
            snr_db: SNR in dB
        """
        ber_stats = BERAnalyzer.calculate_ber_with_confidence(bits_tx, bits_rx)
        evm_stats = EVMAnalyzer.calculate_evm(symbols_tx, symbols_rx)
        papr_stats = PAPRAnalyzer.calculate_papr(signal_tx)
        power_stats = PowerAnalyzer.calculate_power(signal_tx)
        
        self.metrics[snr_db] = {
            'ber': ber_stats,
            'evm': evm_stats,
            'papr': papr_stats,
            'power': power_stats,
            'snr_db': snr_db
        }
    
    def get_all_metrics(self, snr_db):
        """Get all metrics for specific SNR"""
        return self.metrics.get(snr_db, {})
    
    def get_summary(self):
        """Get summary of all collected metrics"""
        return self.metrics
