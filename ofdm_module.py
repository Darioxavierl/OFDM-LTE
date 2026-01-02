"""
OFDM-SC Module - Main Interface (Backward Compatibility Layer)
Provides high-level API for OFDM transmission, reception, and analysis

This module maintains backward compatibility with the original OFDMModule
while leveraging the new modular architecture (OFDMSimulator, OFDMTransmitter,
OFDMReceiver, OFDMChannel) for improved scalability and extensibility.

For new code, consider using OFDMSimulator directly.
For SIMO/MIMO support, use the modular classes in core.ofdm_core.

Migration Path:
    Old: module = OFDMModule(config)
    New: simulator = OFDMSimulator(config)
    
    Old: result = module.transmit(bits, snr_db=10)
    New: result = simulator.simulate_siso(bits, snr_db=10)
"""
import numpy as np
import sys
import os

# Add parent directory to path for imports
_module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _module_path not in sys.path:
    sys.path.insert(0, _module_path)

from config import LTEConfig
from core.ofdm_core import OFDMSimulator


class OFDMModule:
    """
    High-level OFDM/SC-FDM Simulation Module (Backward Compatibility)
    
    This is now a wrapper around OFDMSimulator for backward compatibility.
    All functionality is preserved - internally uses the modular architecture.
    
    Provides simplified interface for transmitting and receiving signals
    through AWGN and Rayleigh multipath channels.

    Example:
        >>> config = LTEConfig(bandwidth=5.0, modulation='QPSK')
        >>> module = OFDMModule(config=config)
        >>> bits = np.random.randint(0, 2, 1000)
        >>> result = module.transmit(bits, snr_db=10)
        >>> print(f"BER: {result['ber']:.2e}")
    
    Attributes:
        config (LTEConfig): LTE configuration
        simulator (OFDMSimulator): Internal modular simulator
    """
    
    def __init__(self, config=None, channel_type='awgn', mode='lte',
                 enable_sc_fdm=False, enable_equalization=True):
        """
        Initialize OFDM Module
        
        Parameters:
        -----------
        config : LTEConfig, optional
            LTE configuration. Defaults to 5MHz QPSK if None
        channel_type : str
            'awgn' (Gaussian) or 'rayleigh_mp' (multipath)
        mode : str
            'lte' for LTE resource mapping, 'simple' for basic
        enable_sc_fdm : bool
            Enable SC-FDM (DFT precoding + IDFT demodulation)
        enable_equalization : bool
            Enable receiver equalization
        """
        if config is None:
            config = LTEConfig()
        
        self.config = config
        self.channel_type = channel_type
        self.mode = mode
        self.enable_sc_fdm = enable_sc_fdm
        self.enable_equalization = enable_equalization
        
        # Create internal simulator (SISO configuration)
        self.simulator = OFDMSimulator(
            config=config,
            channel_type=channel_type,
            mode=mode,
            enable_sc_fdm=enable_sc_fdm,
            enable_equalization=enable_equalization,
            num_channels=1
        )
        
        self.last_results = None
    
    
    def transmit(self, bits, snr_db=10.0):
        """
        Transmit bits through the OFDM system (SISO)
        
        Parameters:
        -----------
        bits : array-like
            Input bit array (0s and 1s)
        snr_db : float
            Signal-to-Noise Ratio in dB
            
        Returns:
        --------
        dict : Results containing:
            - transmitted_bits: Number of transmitted bits
            - received_bits: Number of received bits
            - bit_errors: Number of bit errors
            - ber: Bit Error Rate
            - snr_db: SNR used
            - papr_db: Peak-to-Average Power Ratio in dB
            - papr_linear: PAPR as linear ratio
            - signal_tx: Transmitted time-domain signal
            - signal_rx: Received time-domain signal
            - symbols_tx: Transmitted symbols
            - symbols_rx: Received detected symbols
        """
        # Delegate to simulator
        results = self.simulator.simulate_siso(bits, snr_db=snr_db)
        self.last_results = results
        return results
    
    
    def _calculate_papr(self, signal):
        """
        Calculate Peak-to-Average Power Ratio (PAPR)
        
        Parameters:
        -----------
        signal : array-like
            Time-domain signal (complex)
            
        Returns:
        --------
        dict : PAPR statistics
        """
        # Delegate to transmitter
        return self.simulator.tx.calculate_papr(signal)
    
    # ========================================================================
    # Backward Compatibility Properties
    # ========================================================================
    # These properties allow code written for the old OFDMModule to work
    # with the new modular architecture without modification.
    
    @property
    def channel(self):
        """Get first channel (for backward compatibility)"""
        return self.simulator.channels[0]
    
    @property
    def modulator(self):
        """Get transmitter's modulator (for backward compatibility)"""
        return self.simulator.tx.modulator
    
    @property
    def demodulator(self):
        """Get receiver's demodulator (for backward compatibility)"""
        return self.simulator.rx.demodulator
    
    @property
    def tx(self):
        """Get transmitter instance"""
        return self.simulator.tx
    
    @property
    def rx(self):
        """Get receiver instance"""
        return self.simulator.rx
    
    def run_ber_sweep(self, num_bits, snr_range, num_trials=1, progress_callback=None):
        """
        Perform BER sweep across SNR range
        
        Parameters:
        -----------
        num_bits : int
            Number of bits per trial
        snr_range : array-like
            Array of SNR values in dB to test
        num_trials : int
            Number of trials per SNR value
        progress_callback : callable, optional
            Function(percent, message) called for progress updates
            
        Returns:
        --------
        dict : Results with 'snr_db', 'ber_mean', 'ber_values'
        """
        # Delegate to simulator
        return self.simulator.run_ber_sweep(
            num_bits, 
            snr_range, 
            num_trials=num_trials, 
            progress_callback=progress_callback
        )
    
    
    def get_config(self):
        """Get LTE configuration"""
        return self.config
    
    def __repr__(self):
        mode = "SC-FDM" if self.enable_sc_fdm else "OFDM"
        return f"OFDMModule({self.config.modulation}, {mode}, {self.channel_type})"