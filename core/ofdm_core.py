"""
OFDM Core Module - Modular Architecture for SISO/SIMO/MIMO
===========================================================

Provides modular components for flexible OFDM system design:
- OFDMTransmitter: Handles transmission and signal modulation
- OFDMReceiver: Handles reception and signal demodulation
- OFDMChannel: Wraps channel simulation (AWGN, Rayleigh)
- OFDMSimulator: Orchestrator for SISO/SIMO/MIMO simulations

Architecture:
    SISO (Single-Input Single-Output):
        bits -> [TX] -> signal_tx -> [Channel] -> signal_rx -> [RX] -> bits_rx
    
    SIMO (Single-Input Multiple-Output) - Prepared:
        bits -> [TX] -> signal_tx -> [CH0, CH1, ...] -> [signals_rx] -> [MRC] -> [RX] -> bits_rx
    
    MIMO (Multiple-Input Multiple-Output) - Future:
        [bits] -> [TX0, TX1, ...] -> [signal_tx] -> [CH_matrix] -> [signals_rx] -> [RX] -> [bits_rx]

Usage Example:
    # SISO (current)
    sim = OFDMSimulator(config)
    result = sim.simulate_siso(bits, snr_db=10.0)
    
    # SIMO (prepared for future)
    result = sim.simulate_simo(bits, snr_db=10.0, num_rx=2, combining='mrc')
    
    # MIMO (future implementation)
    result = sim.simulate_mimo(bits, snr_db=10.0, num_tx=2, num_rx=2)
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from config import LTEConfig
from core.modulator import OFDMModulator
from core.demodulator import OFDMDemodulator
from core.channel import ChannelSimulator


class OFDMTransmitter:
    """
    OFDM Transmitter - Handles modulation and transmission
    
    Converts input bits to OFDM signal with configurable modulation,
    SC-FDM support, and LTE resource mapping.
    
    Attributes:
        config (LTEConfig): LTE configuration
        modulator (OFDMModulator): OFDM modulator instance
        mode (str): 'lte' or 'simple' mode
        enable_sc_fdm (bool): SC-FDM enable flag
    """
    
    def __init__(self, config: LTEConfig, mode: str = 'lte', 
                 enable_sc_fdm: bool = False):
        """
        Initialize OFDM Transmitter
        
        Parameters:
        -----------
        config : LTEConfig
            LTE configuration object
        mode : str
            'lte' for LTE resource mapping, 'simple' for basic
        enable_sc_fdm : bool
            Enable SC-FDM (DFT precoding)
        """
        self.config = config
        self.mode = mode
        self.enable_sc_fdm = enable_sc_fdm
        
        self.modulator = OFDMModulator(
            config,
            mode=mode,
            enable_sc_fdm=enable_sc_fdm
        )
        
        self.last_signal_tx = None
        self.last_symbols_tx = None
        self.last_mapping_infos = None
    
    def modulate(self, bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Modulate input bits to OFDM signal
        
        Parameters:
        -----------
        bits : np.ndarray
            Input bit array (0s and 1s)
        
        Returns:
        --------
        tuple : (signal_tx, symbols_tx, mapping_infos)
            - signal_tx: Time-domain transmitted signal (complex)
            - symbols_tx: OFDM symbols in frequency domain (complex)
            - mapping_infos: Dictionary with mapping information
        """
        if not isinstance(bits, np.ndarray):
            bits = np.array(bits, dtype=int)
        
        if bits.size == 0:
            raise ValueError("Bits array cannot be empty")
        
        signal_tx, symbols_tx, mapping_infos = self.modulator.modulate_stream(bits)
        
        self.last_signal_tx = signal_tx
        self.last_symbols_tx = symbols_tx
        self.last_mapping_infos = mapping_infos
        
        return signal_tx, symbols_tx, mapping_infos
    
    def calculate_papr(self, signal: np.ndarray) -> Dict:
        """
        Calculate Peak-to-Average Power Ratio (PAPR)
        
        Parameters:
        -----------
        signal : np.ndarray
            Time-domain signal (complex)
        
        Returns:
        --------
        dict : PAPR statistics
            - papr_db: PAPR in dB
            - papr_linear: PAPR as linear ratio
            - peak_power: Peak power
            - avg_power: Average power
        """
        power_inst = np.abs(signal) ** 2
        power_peak = np.max(power_inst)
        power_avg = np.mean(power_inst)
        
        if power_avg > 0:
            papr_linear = power_peak / power_avg
            papr_db = 10 * np.log10(papr_linear)
        else:
            papr_linear = 1.0
            papr_db = 0.0
        
        return {
            'papr_db': papr_db,
            'papr_linear': papr_linear,
            'peak_power': power_peak,
            'avg_power': power_avg
        }
    
    def get_config(self) -> LTEConfig:
        """Get transmitter configuration"""
        return self.config
    
    def __repr__(self) -> str:
        mode = "SC-FDM" if self.enable_sc_fdm else "OFDM"
        return f"OFDMTransmitter({self.config.modulation}, {mode})"


class OFDMReceiver:
    """
    OFDM Receiver - Handles demodulation and reception
    
    Demodulates received OFDM signal, detects symbols, decodes bits,
    and estimates channel characteristics.
    
    Attributes:
        config (LTEConfig): LTE configuration
        demodulator (OFDMDemodulator): OFDM demodulator instance
        mode (str): 'lte' or 'simple' mode
        enable_equalization (bool): Equalization enable flag
    """
    
    def __init__(self, config: LTEConfig, mode: str = 'lte',
                 enable_equalization: bool = True, 
                 enable_sc_fdm: bool = False):
        """
        Initialize OFDM Receiver
        
        Parameters:
        -----------
        config : LTEConfig
            LTE configuration object
        mode : str
            'lte' for LTE resource mapping, 'simple' for basic
        enable_equalization : bool
            Enable receiver equalization
        enable_sc_fdm : bool
            Enable SC-FDM support
        """
        self.config = config
        self.mode = mode
        self.enable_equalization = enable_equalization
        self.enable_sc_fdm = enable_sc_fdm
        
        self.demodulator = OFDMDemodulator(
            config,
            mode=mode,
            enable_equalization=enable_equalization,
            enable_sc_fdm=enable_sc_fdm
        )
        
        self.last_symbols_rx = None
        self.last_bits_rx = None
        self.channel_estimate = None
    
    def demodulate(self, signal_rx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Demodulate received OFDM signal
        
        Parameters:
        -----------
        signal_rx : np.ndarray
            Received time-domain signal (complex)
        
        Returns:
        --------
        tuple : (symbols_rx, bits_rx)
            - symbols_rx: Detected OFDM symbols in frequency domain
            - bits_rx: Decoded bit array
        """
        if signal_rx.size == 0:
            raise ValueError("Received signal cannot be empty")
        
        symbols_detected, bits_rx = self.demodulator.demodulate_stream(signal_rx)
        
        self.last_symbols_rx = symbols_detected
        self.last_bits_rx = bits_rx
        
        return symbols_detected, bits_rx
    
    def estimate_channel(self) -> Dict:
        """
        Get channel estimation information
        
        Returns:
        --------
        dict : Channel estimation data (currently placeholder)
        """
        # Placeholder for channel estimation
        # Future: Implement actual channel estimation from pilots
        return {
            'estimated': False,
            'method': 'none'
        }
    
    def calculate_ber(self, bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
        """
        Calculate Bit Error Rate
        
        Parameters:
        -----------
        bits_tx : np.ndarray
            Transmitted bits
        bits_rx : np.ndarray
            Received bits
        
        Returns:
        --------
        float : BER value
        """
        # Ensure same length
        min_len = min(len(bits_tx), len(bits_rx))
        bits_tx_aligned = bits_tx[:min_len]
        bits_rx_aligned = bits_rx[:min_len]
        
        bit_errors = np.sum(bits_tx_aligned != bits_rx_aligned)
        ber = bit_errors / len(bits_tx_aligned) if len(bits_tx_aligned) > 0 else 0.0
        
        return float(ber)
    
    def get_config(self) -> LTEConfig:
        """Get receiver configuration"""
        return self.config
    
    def __repr__(self) -> str:
        mode = "SC-FDM" if self.enable_sc_fdm else "OFDM"
        return f"OFDMReceiver({self.config.modulation}, {mode})"


class OFDMChannel:
    """
    OFDM Channel - Wraps channel simulation
    
    Supports AWGN and Rayleigh multipath channels.
    Extensible for SIMO (multiple receive channels) and 
    MIMO (channel matrix) support in future versions.
    
    Attributes:
        channel_type (str): 'awgn' or 'rayleigh_mp'
        channel (ChannelSimulator): Underlying channel instance
        profile (str): ITU channel profile for Rayleigh
    """
    
    def __init__(self, channel_type: str = 'awgn', snr_db: float = 10.0,
                 fs: float = 15.36e6, itu_profile: str = 'Pedestrian_A',
                 frequency_ghz: float = 2.0, velocity_kmh: float = 0):
        """
        Initialize OFDM Channel
        
        Parameters:
        -----------
        channel_type : str
            'awgn' or 'rayleigh_mp'
        snr_db : float
            Signal-to-Noise Ratio in dB
        fs : float
            Sampling frequency in Hz
        itu_profile : str
            ITU-R M.1225 profile for Rayleigh channel
        frequency_ghz : float
            Carrier frequency in GHz
        velocity_kmh : float
            User velocity in km/h (for Doppler)
        """
        self.channel_type = channel_type
        self.snr_db = snr_db
        self.fs = fs
        self.profile = itu_profile
        self.frequency_ghz = frequency_ghz
        self.velocity_kmh = velocity_kmh
        
        # Initialize underlying channel
        if channel_type == 'rayleigh_mp':
            self.channel = ChannelSimulator(
                channel_type='rayleigh_mp',
                snr_db=snr_db,
                fs=fs,
                itu_profile=itu_profile,
                frequency_ghz=frequency_ghz,
                velocity_kmh=velocity_kmh
            )
        else:
            self.channel = ChannelSimulator('awgn', snr_db=snr_db)
    
    def set_snr(self, snr_db: float) -> None:
        """
        Set channel SNR
        
        Parameters:
        -----------
        snr_db : float
            Signal-to-Noise Ratio in dB
        """
        self.snr_db = snr_db
        self.channel.set_snr(snr_db)
    
    def transmit(self, signal_tx: np.ndarray) -> np.ndarray:
        """
        Transmit signal through single channel (SISO)
        
        Parameters:
        -----------
        signal_tx : np.ndarray
            Transmitted time-domain signal (complex)
        
        Returns:
        --------
        np.ndarray : Received signal after channel effects
        """
        return self.channel.transmit(signal_tx)
    
    def transmit_simo(self, signal_tx: np.ndarray, num_rx: int = 2) -> List[np.ndarray]:
        """
        Transmit signal through multiple channels (SIMO) - Prepared for future
        
        Parameters:
        -----------
        signal_tx : np.ndarray
            Transmitted time-domain signal (complex)
        num_rx : int
            Number of receive channels
        
        Returns:
        --------
        list : List of received signals (one per RX antenna)
        
        Note:
            Currently returns same signal for all RX paths.
            Future: Implement independent channel paths for true SIMO.
        """
        # Placeholder: Return same signal for each RX path
        # Future: Each path will have independent fading
        signals_rx = [self.transmit(signal_tx) for _ in range(num_rx)]
        return signals_rx
    
    def get_config(self) -> Dict:
        """Get channel configuration"""
        return {
            'type': self.channel_type,
            'snr_db': self.snr_db,
            'fs': self.fs,
            'profile': self.profile,
            'frequency_ghz': self.frequency_ghz,
            'velocity_kmh': self.velocity_kmh
        }
    
    def __repr__(self) -> str:
        return f"OFDMChannel({self.channel_type}, SNR={self.snr_db}dB, {self.profile})"


class OFDMSimulator:
    """
    OFDM Simulator - Orchestrator for SISO/SIMO/MIMO simulations
    
    Manages transmitter, receiver, and channel instances.
    Provides high-level simulation interfaces for different antenna configurations.
    
    Currently supports:
        - SISO (Single-Input Single-Output): 1 TX, 1 RX
    
    Prepared for:
        - SIMO (Single-Input Multiple-Output): 1 TX, N RX with combining
        - MIMO (Multiple-Input Multiple-Output): N TX, M RX (future)
    
    Attributes:
        config (LTEConfig): LTE configuration
        tx (OFDMTransmitter): Transmitter instance
        rx (OFDMReceiver): Receiver instance
        channels (list): List of channel instances
    """
    
    def __init__(self, config: Optional[LTEConfig] = None, 
                 channel_type: str = 'awgn', mode: str = 'lte',
                 enable_sc_fdm: bool = False, 
                 enable_equalization: bool = True,
                 num_channels: int = 1):
        """
        Initialize OFDM Simulator
        
        Parameters:
        -----------
        config : LTEConfig, optional
            LTE configuration. Defaults to 5MHz QPSK if None
        channel_type : str
            'awgn' or 'rayleigh_mp'
        mode : str
            'lte' for LTE resource mapping, 'simple' for basic
        enable_sc_fdm : bool
            Enable SC-FDM
        enable_equalization : bool
            Enable receiver equalization
        num_channels : int
            Number of channel instances (for SIMO/MIMO)
        """
        if config is None:
            config = LTEConfig()
        
        self.config = config
        self.channel_type = channel_type
        self.mode = mode
        self.enable_sc_fdm = enable_sc_fdm
        self.enable_equalization = enable_equalization
        
        # Initialize TX and RX
        self.tx = OFDMTransmitter(
            config,
            mode=mode,
            enable_sc_fdm=enable_sc_fdm
        )
        
        self.rx = OFDMReceiver(
            config,
            mode=mode,
            enable_equalization=enable_equalization,
            enable_sc_fdm=enable_sc_fdm
        )
        
        # Initialize channel(s)
        fs = getattr(config, 'fs', 15.36e6)
        self.channels = []
        
        for i in range(num_channels):
            if channel_type == 'rayleigh_mp':
                ch = OFDMChannel(
                    channel_type='rayleigh_mp',
                    snr_db=10.0,
                    fs=fs,
                    itu_profile='Pedestrian_A',
                    frequency_ghz=2.0,
                    velocity_kmh=0
                )
            else:
                ch = OFDMChannel('awgn', snr_db=10.0, fs=fs)
            
            self.channels.append(ch)
        
        self.last_results = None
    
    def simulate_siso(self, bits: np.ndarray, snr_db: float = 10.0) -> Dict:
        """
        Simulate SISO (Single-Input Single-Output) transmission
        
        Standard SISO configuration: 1 transmitter, 1 channel, 1 receiver.
        This is the current working configuration matching original OFDMModule.
        
        Parameters:
        -----------
        bits : np.ndarray
            Input bit array (0s and 1s)
        snr_db : float
            Signal-to-Noise Ratio in dB
        
        Returns:
        --------
        dict : Complete simulation results
            - transmitted_bits: Number of bits transmitted
            - received_bits: Number of bits received
            - bit_errors: Number of bit errors
            - bits_received_array: Array of received bits
            - ber: Bit Error Rate
            - snr_db: SNR used
            - papr_db: Peak-to-Average Power Ratio in dB
            - papr_linear: PAPR as linear ratio
            - signal_tx: Transmitted time-domain signal
            - signal_rx: Received time-domain signal
            - symbols_tx: Transmitted OFDM symbols
            - symbols_rx: Received OFDM symbols
        """
        if not isinstance(bits, np.ndarray):
            bits = np.array(bits, dtype=int)
        
        if bits.size == 0:
            raise ValueError("Bits array cannot be empty")
        
        original_num_bits = len(bits)
        
        # Step 1: Transmit (modulate)
        signal_tx, symbols_tx, _ = self.tx.modulate(bits)
        
        # Step 2: Calculate PAPR
        papr_info = self.tx.calculate_papr(signal_tx)
        
        # Step 3: Transmit through channel
        self.channels[0].set_snr(snr_db)
        signal_rx = self.channels[0].transmit(signal_tx)
        
        # Step 4: Receive (demodulate)
        symbols_rx, bits_rx = self.rx.demodulate(signal_rx)
        
        # Step 5: Calculate BER
        if len(bits_rx) < original_num_bits:
            bits_rx = np.pad(bits_rx, (0, original_num_bits - len(bits_rx)), 'constant')
        else:
            bits_rx = bits_rx[:original_num_bits]
        
        ber = self.rx.calculate_ber(bits, bits_rx)
        bit_errors = np.sum(bits != bits_rx)
        
        results = {
            'transmitted_bits': int(original_num_bits),
            'received_bits': int(original_num_bits),
            'bits_received_array': bits_rx,
            'bit_errors': int(bit_errors),
            'errors': int(bit_errors),
            'ber': float(ber),
            'snr_db': float(snr_db),
            'papr_db': float(papr_info['papr_db']),
            'papr_linear': float(papr_info['papr_linear']),
            'signal_tx': signal_tx,
            'signal_rx': signal_rx,
            'symbols_tx': symbols_tx,
            'symbols_rx': symbols_rx
        }
        
        self.last_results = results
        return results
    
    def simulate_simo(self, bits: np.ndarray, snr_db: float = 10.0, 
                      num_rx: int = 2, combining: str = 'mrc') -> Dict:
        """
        Simulate SIMO (Single-Input Multiple-Output) transmission - Prepared
        
        Configuration: 1 transmitter, 1 channel split to N receivers, 1 receiver.
        Supports multiple receive antennas with diversity combining.
        
        Parameters:
        -----------
        bits : np.ndarray
            Input bit array (0s and 1s)
        snr_db : float
            Signal-to-Noise Ratio in dB
        num_rx : int
            Number of receive antennas
        combining : str
            Combining method: 'mrc' (Maximum Ratio Combining - default)
        
        Returns:
        --------
        dict : SIMO simulation results (same structure as SISO)
        
        Note:
            Currently implemented as multiple independent SISO paths.
            Future: Implement true spatial diversity and advanced combining.
        
        Implementation Priority:
            1. MRC (Maximum Ratio Combining) ← Next
            2. EGC (Equal Gain Combining)
            3. Selection Combining
            4. Advanced techniques (Alamouti, etc.)
        """
        if not isinstance(bits, np.ndarray):
            bits = np.array(bits, dtype=int)
        
        if bits.size == 0:
            raise ValueError("Bits array cannot be empty")
        
        original_num_bits = len(bits)
        
        # Step 1: Transmit (modulate)
        signal_tx, symbols_tx, _ = self.tx.modulate(bits)
        
        # Step 2: Calculate PAPR
        papr_info = self.tx.calculate_papr(signal_tx)
        
        # Step 3: Transmit through multiple channels
        signals_rx = self.channels[0].transmit_simo(signal_tx, num_rx=num_rx)
        for i, sig in enumerate(signals_rx):
            self.channels[0].set_snr(snr_db)
        
        # Step 4: Combine signals (placeholder - currently just uses first path)
        if combining == 'mrc':
            signal_combined = self._combine_mrc(signals_rx)
        else:
            signal_combined = signals_rx[0]  # Default: first path
        
        # Step 5: Receive (demodulate)
        symbols_rx, bits_rx = self.rx.demodulate(signal_combined)
        
        # Step 6: Calculate BER
        if len(bits_rx) < original_num_bits:
            bits_rx = np.pad(bits_rx, (0, original_num_bits - len(bits_rx)), 'constant')
        else:
            bits_rx = bits_rx[:original_num_bits]
        
        ber = self.rx.calculate_ber(bits, bits_rx)
        bit_errors = np.sum(bits != bits_rx)
        
        results = {
            'transmitted_bits': int(original_num_bits),
            'received_bits': int(original_num_bits),
            'bits_received_array': bits_rx,
            'bit_errors': int(bit_errors),
            'errors': int(bit_errors),
            'ber': float(ber),
            'snr_db': float(snr_db),
            'papr_db': float(papr_info['papr_db']),
            'papr_linear': float(papr_info['papr_linear']),
            'signal_tx': signal_tx,
            'signal_rx': signal_combined,
            'symbols_tx': symbols_tx,
            'symbols_rx': symbols_rx,
            'num_rx': num_rx,
            'combining_method': combining,
            'diversity_level': num_rx
        }
        
        self.last_results = results
        return results
    
    def simulate_mimo(self, bits: np.ndarray, snr_db: float = 10.0,
                     num_tx: int = 2, num_rx: int = 2) -> Dict:
        """
        Simulate MIMO (Multiple-Input Multiple-Output) transmission - Future
        
        Configuration: N transmitters, channel matrix, M receivers.
        Not yet implemented. Placeholder for future development.
        
        Parameters:
        -----------
        bits : np.ndarray
            Input bit array (0s and 1s)
        snr_db : float
            Signal-to-Noise Ratio in dB
        num_tx : int
            Number of transmit antennas
        num_rx : int
            Number of receive antennas
        
        Returns:
        --------
        dict : MIMO simulation results
        
        Raises:
            NotImplementedError: MIMO simulation not yet implemented
        
        Implementation Notes:
            1. Extend TX to support multiple antennas
            2. Create channel matrix H (num_rx × num_tx)
            3. Implement space-time coding (Alamouti for 2x2)
            4. Advanced: SVD-based precoding
        """
        raise NotImplementedError(
            f"MIMO simulation (TX={num_tx}, RX={num_rx}) not yet implemented.\n"
            "Roadmap: SISO (✓) -> SIMO (next) -> MIMO (future)"
        )
    
    def _combine_mrc(self, signals_rx: List[np.ndarray]) -> np.ndarray:
        """
        Maximum Ratio Combining (MRC) of multiple received signals
        
        Placeholder implementation: Currently returns first signal.
        Future: Implement proper MRC with channel estimation.
        
        Parameters:
        -----------
        signals_rx : list of np.ndarray
            List of received signals from different antennas
        
        Returns:
        --------
        np.ndarray : Combined signal
        """
        # Placeholder: Return first signal
        # Future: Implement actual MRC
        #   1. Estimate channel coefficients
        #   2. Weight each signal by conjugate of channel coefficient
        #   3. Sum weighted signals
        if len(signals_rx) == 0:
            raise ValueError("No received signals provided")
        
        return signals_rx[0]
    
    def run_ber_sweep(self, num_bits: int, snr_range: np.ndarray, 
                     num_trials: int = 1, 
                     progress_callback: Optional[callable] = None) -> Dict:
        """
        Perform BER sweep across SNR range
        
        Parameters:
        -----------
        num_bits : int
            Number of bits per trial
        snr_range : np.ndarray
            Array of SNR values in dB to test
        num_trials : int
            Number of trials per SNR value
        progress_callback : callable, optional
            Function(percent, message) called for progress updates
        
        Returns:
        --------
        dict : Results with 'snr_db', 'ber_mean', 'papr_values'
        """
        bits = np.random.randint(0, 2, num_bits)
        snr_values = np.atleast_1d(snr_range)
        ber_values = []
        papr_values = []
        
        total_trials = len(snr_values) * num_trials
        current_trial = 0
        
        for snr in snr_values:
            ber_trial = []
            papr_trial = []
            
            for trial in range(num_trials):
                result = self.simulate_siso(bits, snr_db=snr)
                ber_trial.append(result['ber'])
                papr_trial.append(result['papr_db'])
                
                current_trial += 1
                if progress_callback:
                    progress_pct = int((current_trial / total_trials) * 100)
                    progress_callback(progress_pct, f"SNR: {snr:.1f} dB - Trial {trial+1}/{num_trials}")
            
            ber_values.append(np.mean(ber_trial))
            papr_values.append(np.mean(papr_trial))
        
        return {
            'snr_db': snr_values,
            'ber_mean': np.array(ber_values),
            'ber_values': np.array(ber_values),
            'papr_values': np.array(papr_values)
        }
    
    def get_config(self) -> LTEConfig:
        """Get simulator configuration"""
        return self.config
    
    def __repr__(self) -> str:
        mode = "SC-FDM" if self.enable_sc_fdm else "OFDM"
        num_ch = len(self.channels)
        return f"OFDMSimulator({self.config.modulation}, {mode}, {self.channel_type}, {num_ch}ch)"
