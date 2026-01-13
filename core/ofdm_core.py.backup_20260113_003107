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
    
    def transmit_simo(self, signal_tx: np.ndarray, num_rx: int = 2) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Transmit signal through SIMO (Single-Input Multiple-Output) configuration
        
        SIMO Reception: All RX antennas receive through same fading channel,
        but with INDEPENDENT NOISE realizations.
        
        This models physical SIMO where:
        - Single transmit antenna
        - Multiple RX antennas spatially separated (uncorrelated noise)
        - Same fading environment (reasonable for short distances)
        
        SNR scaling: With N antennas and independent AWGN,
        combining provides ~10*log10(N) dB SNR improvement.
        
        Parameters:
        -----------
        signal_tx : np.ndarray
            Transmitted OFDM signal
        num_rx : int
            Number of receive antennas
        
        Returns:
        --------
        tuple : (signals_rx, channel_coefficients)
            - signals_rx: List of received signals (one per RX antenna)
            - channel_coefficients: List of channel gains
        """
        if num_rx < 1:
            raise ValueError("num_rx must be >= 1")
        
        signals_rx = []
        channel_coeffs = []
        
        # Apply channel to signal (shared fading)
        base_signal = self.transmit(signal_tx)
        
        # Add independent noise to each antenna
        for rx_idx in range(num_rx):
            if self.channel_type == 'awgn':
                # AWGN: add independent noise to each antenna
                signal_rx = base_signal.copy()
                # Add noise (channel already has noise, so add more)
                # Actually, ChannelSimulator.transmit already adds noise based on SNR
                # For multiple antennas: create new noisy version
                if num_rx > 1 and rx_idx > 0:
                    # Add additional independent noise for 2nd, 3rd, ... antennas
                    #Actually this is wrong - the base_signal already has noise
                    # Better: pass through channel multiple times for independent noise
                    ch_copy = ChannelSimulator(
                        channel_type=self.channel_type,
                        snr_db=self.snr_db,
                        fs=self.fs,
                        verbose=False  # Avoid repeated prints
                    )
                    signal_rx = ch_copy.transmit(signal_tx)
                signals_rx.append(signal_rx)
            else:
                # Rayleigh: same fading, independent noise
                # All antennas see same fading, but different noise
                ch_copy = ChannelSimulator(
                    channel_type=self.channel_type,
                    snr_db=self.snr_db,
                    fs=self.fs,
                    itu_profile=self.profile,
                    frequency_ghz=self.frequency_ghz,
                    velocity_kmh=self.velocity_kmh,
                    verbose=False  # Avoid repeated prints
                )
                signal_rx = ch_copy.transmit(signal_tx)
                signals_rx.append(signal_rx)
            
            # Estimate channel (assume unity gain after processing)
            channel_coeffs.append(1.0 + 0j)
        
        return signals_rx, channel_coeffs
        """
        Channel coefficient estimation
        
        For SIMO/MIMO: Estimate amplitude response per path.
        Uses energy-based estimation: stronger signal = better channel.
        
        Parameters:
        -----------
        signal_tx : np.ndarray
            Transmitted signal
        signal_rx : np.ndarray
            Received signal
        
        Returns:
        --------
        np.ndarray : Estimated channel amplitude/power
        """
        # Approach: Estimate channel gain from received vs transmitted power
        # h_estimate = sqrt(E[|rx|^2] / E[|tx|^2])
        
        min_len = min(len(signal_tx), len(signal_rx))
        tx = signal_tx[:min_len]
        rx = signal_rx[:min_len]
        
        # Power estimation
        tx_power = np.mean(np.abs(tx) ** 2)
        rx_power = np.mean(np.abs(rx) ** 2)
        
        if tx_power < 1e-12:
            # Zero transmitted power: return unity gain
            return np.ones(min_len, dtype=complex)
        
        # Channel amplitude estimate from power ratio
        # |h| = sqrt(rx_power / tx_power)
        channel_amplitude = np.sqrt(np.maximum(rx_power / tx_power, 1e-6))
        
        # For MRC: we need phase + amplitude
        # Use correlation to get phase
        # h = E[rx * tx*] / E[|tx|^2]
        h_est_complex = np.mean(rx * np.conj(tx)) / (tx_power + 1e-12)
        
        # Normalize: keep amplitude from power ratio, phase from correlation
        h_phase = np.angle(h_est_complex)
        h_normalized = channel_amplitude * np.exp(1j * h_phase)
        
        # Create array of this coefficient for all samples
        h_array = np.ones(len(signal_rx), dtype=complex) * h_normalized
        
        return h_array

    def transmit_mimo(self, signals_tx: List[np.ndarray], num_rx: int = 1) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Transmit signals through MIMO channel (num_tx × num_rx)
        
        AGREGAR ESTE MÉTODO A LA CLASE OFDMChannel
        
        Parameters:
        -----------
        signals_tx : list of np.ndarray
            List of transmitted signals, one per TX antenna
        num_rx : int
            Number of receive antennas
        
        Returns:
        --------
        tuple : (signals_rx, channel_matrix)
            - signals_rx: List of received signals (one per RX antenna)
            - channel_matrix: Channel coefficients, shape (num_rx, num_tx)
        """
        num_tx = len(signals_tx)
        
        if num_tx == 0:
            raise ValueError("No transmitted signals provided")
        
        signal_length = len(signals_tx[0])
        
        # Verify all TX signals have same length
        for tx_idx, sig in enumerate(signals_tx):
            if len(sig) != signal_length:
                raise ValueError(f"TX signal {tx_idx} length mismatch")
        
        # Initialize
        signals_rx = []
        channel_matrix = np.zeros((num_rx, num_tx), dtype=complex)
        
        # For each RX antenna
        for rx_idx in range(num_rx):
            rx_signal = np.zeros(signal_length, dtype=complex)
            
            for tx_idx in range(num_tx):
                # Create independent channel for this TX-RX link
                if self.channel_type == 'awgn':
                    h = 1.0 + 0j
                    signal_through_channel = signals_tx[tx_idx] * h
                else:
                    # Rayleigh: independent fading per path
                    from core.channel import ChannelSimulator
                    
                    link_channel = ChannelSimulator(
                        channel_type=self.channel_type,
                        snr_db=self.snr_db,
                        fs=self.fs,
                        itu_profile=self.profile,
                        frequency_ghz=self.frequency_ghz,
                        velocity_kmh=self.velocity_kmh,
                        verbose=False  # Avoid repeated prints
                    )
                    
                    signal_through_channel = link_channel.transmit(signals_tx[tx_idx])
                    
                    # Estimate channel coefficient
                    tx_power = np.mean(np.abs(signals_tx[tx_idx])**2)
                    rx_power = np.mean(np.abs(signal_through_channel)**2)
                    
                    if tx_power > 1e-12:
                        h_magnitude = np.sqrt(rx_power / tx_power)
                        correlation = np.mean(signal_through_channel * np.conj(signals_tx[tx_idx]))
                        h_phase = np.angle(correlation)
                        h = h_magnitude * np.exp(1j * h_phase)
                    else:
                        h = 1.0 + 0j
                
                channel_matrix[rx_idx, tx_idx] = h
                rx_signal += signal_through_channel
            
            # Add noise
            signal_power = np.mean(np.abs(rx_signal)**2)
            snr_linear = 10 ** (self.snr_db / 10)  # Convert dB to linear scale
            noise_power = signal_power / snr_linear
            
            noise_real = np.random.normal(0, np.sqrt(noise_power/2), signal_length)
            noise_imag = np.random.normal(0, np.sqrt(noise_power/2), signal_length)
            noise = noise_real + 1j * noise_imag
            
            rx_signal_noisy = rx_signal + noise
            signals_rx.append(rx_signal_noisy)
        
        return signals_rx, channel_matrix
    
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
                 num_channels: int = 1,
                 itu_profile: str = 'Pedestrian_A',
                 frequency_ghz: float = 2.0,
                 velocity_kmh: float = 0.0):
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
        itu_profile : str
            ITU profile for Rayleigh channel (default 'Pedestrian_A')
        frequency_ghz : float
            Carrier frequency in GHz (default 2.0)
        velocity_kmh : float
            Velocity in km/h for Doppler (default 0.0)
        """
        if config is None:
            config = LTEConfig()
        
        self.config = config
        self.channel_type = channel_type
        self.mode = mode
        self.enable_sc_fdm = enable_sc_fdm
        self.enable_equalization = enable_equalization
        self.itu_profile = itu_profile
        self.frequency_ghz = frequency_ghz
        self.velocity_kmh = velocity_kmh
        
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
                    itu_profile=itu_profile,
                    frequency_ghz=frequency_ghz,
                    velocity_kmh=velocity_kmh
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
    
    def _demodulate_with_channel_est(self, signal_rx: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Demodulate a single RX antenna signal and estimate its channel via CRS per OFDM symbol
        
        This performs:
        1. FFT demodulation of all OFDM symbols
        2. Channel estimation using CRS (Cell Reference Signals) pilotos PER SYMBOL
        3. Returns symbols WITHOUT equalization + channel estimates for each OFDM symbol
        
        CRITICAL: Returns symbols_data_only and channel_estimates_per_symbol
        (not just first symbol estimate like receive_and_decode does)
        
        Parameters:
        -----------
        signal_rx : np.ndarray
            Received time-domain signal from one antenna
        
        Returns:
        --------
        tuple : (symbols_rx_data, channel_estimates_per_symbol)
            - symbols_rx_data: Frequency-domain data symbols (complex array, NOT equalized)
            - channel_estimates_per_symbol: List of channel estimates [H_sym0, H_sym1, ...]
        """
        # Import LTEReceiver if not already available
        from core.lte_receiver import LTEReceiver
        
        # Create a temporary LTE receiver WITHOUT equalization
        # This gives us raw symbols and channel estimates for MRC combining
        lte_receiver_no_eq = LTEReceiver(
            self.config,
            cell_id=0,
            enable_equalization=False,  # CRITICAL: No ZF here, MRC will combine first
            enable_sc_fdm=self.enable_sc_fdm
        )
        
        # Demodulate all OFDM symbols in the signal
        all_received_symbols = lte_receiver_no_eq._demodulate_ofdm_stream(signal_rx)
        
        # Concatenate all symbols
        if all_received_symbols:
            received_symbols = np.concatenate(all_received_symbols)
        else:
            return np.array([], dtype=complex), [np.array([], dtype=complex)]
        
        # Estimate channel periodically (per OFDM symbol like LTE does)
        channel_estimates_per_symbol, _ = lte_receiver_no_eq._estimate_channel_periodic(
            all_received_symbols
        )
        
        # Extract data indices (positions in ONE OFDM symbol)
        data_indices = lte_receiver_no_eq.resource_grid.get_data_indices()
        
        # Extract only data symbols from ALL symbols
        num_ofdm_symbols = len(all_received_symbols)
        all_data_indices = []
        for sym_idx in range(num_ofdm_symbols):
            offset = sym_idx * self.config.N
            all_data_indices.extend(data_indices + offset)
        
        all_data_indices = np.array(all_data_indices)
        valid_indices = all_data_indices[all_data_indices < len(received_symbols)]
        symbols_data = received_symbols[valid_indices]
        
        return symbols_data, channel_estimates_per_symbol
    
    def _combine_symbols_mrc(self, symbols_rx_list: List[np.ndarray], 
                            h_estimates_per_antenna: List[List[np.ndarray]], 
                            regularization: float = 1e-10) -> np.ndarray:
        """
        Maximum Ratio Combining (MRC) in frequency domain
        
        Combines symbols from multiple RX antennas using optimal weights derived 
        from channel estimates. Works correctly with multiple OFDM symbols.
        
        MRC Formula (per subcarrier k in each OFDM symbol):
            w_i[k] = conj(H_i[k]) / |H_i[k]|²
            Y_combined[k] = sum_i(w_i[k] * Y_i[k])
        
        The weights w_i contain both phase and amplitude compensation, making 
        additional ZF equalization unnecessary.
        
        Parameters:
        -----------
        symbols_rx_list : list of np.ndarray
            Data symbols from each RX antenna (concatenated from all OFDM symbols)
            symbols_rx_list[i] has shape (num_total_data_symbols,)
        h_estimates_per_antenna : list of list of np.ndarray
            Channel estimates from each RX antenna for each OFDM symbol
            h_estimates_per_antenna[antenna_idx][symbol_idx] has shape (N,)
        regularization : float
            Small value to avoid division by zero (default 1e-10)
        
        Returns:
        --------
        np.ndarray : MRC-combined symbols, shape (num_total_data_symbols,)
        """
        from core.resource_mapper import LTEResourceGrid
        
        num_rx = len(symbols_rx_list)
        num_total_data_symbols = len(symbols_rx_list[0])
        
        # Get data indices for ONE OFDM symbol
        data_indices = LTEResourceGrid(self.config.N, self.config.Nc).get_data_indices()
        num_data_per_symbol = len(data_indices)
        
        # Calculate number of OFDM symbols
        num_ofdm_symbols = num_total_data_symbols // num_data_per_symbol
        remainder = num_total_data_symbols % num_data_per_symbol
        if remainder > 0:
            num_ofdm_symbols += 1
        
        # Initialize combined symbol array
        symbols_combined = np.zeros(num_total_data_symbols, dtype=complex)
        power_total = np.zeros(num_total_data_symbols, dtype=float)
        
        # MRC combining: process each RX antenna
        for antenna_idx in range(num_rx):
            Y_antenna = symbols_rx_list[antenna_idx]  # All data symbols from this antenna
            H_list = h_estimates_per_antenna[antenna_idx]  # Channel estimates per OFDM symbol
            
            # Process each OFDM symbol and its data
            data_idx = 0  # Index into symbols_data
            for sym_idx in range(num_ofdm_symbols):
                # Get channel estimate for this OFDM symbol and antenna
                if sym_idx < len(H_list):
                    H_full = H_list[sym_idx]  # Channel for all N subcarriers
                else:
                    # Use last available estimate if we run out
                    H_full = H_list[-1] if len(H_list) > 0 else np.ones(self.config.N, dtype=complex)
                
                # Extract channel at data positions for this symbol
                H_data = H_full[data_indices]
                
                # Number of data symbols in this OFDM symbol (may be less for last symbol)
                num_data_this_symbol = min(num_data_per_symbol, num_total_data_symbols - data_idx)
                
                # Apply MRC weighting for this symbol's data
                for local_k in range(num_data_this_symbol):
                    global_k = data_idx + local_k
                    
                    if global_k < len(Y_antenna):
                        y_k = Y_antenna[global_k]
                        h_k = H_data[local_k] if local_k < len(H_data) else 1.0
                        
                        # MRC weight: w = conj(H) / |H|²
                        h_mag_sq = np.abs(h_k) ** 2 + regularization
                        w_k = np.conj(h_k) / h_mag_sq
                        
                        # Accumulate
                        symbols_combined[global_k] += w_k * y_k
                        power_total[global_k] += h_mag_sq
                
                data_idx += num_data_this_symbol
        
        # Normalize: MRC sums weighted symbols from all antennas
        # For proper amplitude scaling, divide by num_rx
        # This gives: E[Y_combined] = E[Y] when channels are unit gain
        symbols_combined = symbols_combined / num_rx
        
        return symbols_combined
    
    def simulate_simo(self, bits: np.ndarray, snr_db: float = 10.0, 
                      num_rx: int = 2, combining: str = 'mrc',
                      parallel: bool = True) -> Dict:
        """
        Simulate SIMO (Single-Input Multiple-Output) transmission - CORRECTED
        
        Configuration: 1 transmitter, 1 channel split to N receivers with MRC combining.
        
        Correct signal processing flow:
            1. TX: Modulate bits to OFDM signal
            2. Channel: Transmit through N independent fading channels
            3. RX (per antenna, optionally parallel):
               a. FFT demodulation to get frequency-domain symbols
               b. CRS-based channel estimation (per-antenna via LTEChannelEstimator)
            4. MRC Combining (frequency domain):
               a. Compute optimal weights: w_i[k] = conj(H_i[k]) / |H_i[k]|²
               b. Combine: Y_combined[k] = sum_i(w_i[k] * Y_i[k])
            5. Symbol detection: Demodulate combined symbols to bits
        
        Key differences from previous (incorrect) implementation:
            ✓ Per-antenna channel estimation using CRS (not dummy estimates)
            ✓ Frequency-domain MRC with optimal weights (not time-domain EGC)
            ✓ MRC weights include equalization (no ZF after combining)
            ✓ Proper diversity gain (+3-10 dB SNR with multiple antennas)
        
        Parameters:
        -----------
        bits : np.ndarray
            Input bit array (0s and 1s)
        snr_db : float
            Signal-to-Noise Ratio in dB
        num_rx : int
            Number of receive antennas (default 2)
        combining : str
            Combining method: 'mrc' (Maximum Ratio Combining)
        parallel : bool
            Enable parallel processing of antenna demodulation via threads
        
        Returns:
        --------
        dict : SIMO simulation results
            - transmitted_bits, received_bits, bit_errors, ber
            - signal_tx, signal_rx_list (signals from all antennas)
            - symbols_tx, symbols_rx_combined (after MRC)
            - num_rx, combining_method, diversity_level
            - channel_estimates (for analysis)
        
        Implementation Notes:
            - Uses ThreadPoolExecutor for parallel per-antenna processing (optional)
            - Each antenna runs LTEChannelEstimator for CRS-based estimation
            - MRC is optimal for AWGN channels and achieves diversity gain
            - No additional equalization needed after MRC
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if not isinstance(bits, np.ndarray):
            bits = np.array(bits, dtype=int)
        
        if bits.size == 0:
            raise ValueError("Bits array cannot be empty")
        
        original_num_bits = len(bits)
        
        # Step 1: Transmit (modulate)
        signal_tx, symbols_tx, _ = self.tx.modulate(bits)
        
        # Step 2: Calculate PAPR
        papr_info = self.tx.calculate_papr(signal_tx)
        
        # Step 3: Transmit through multiple independent channels
        # Set SNR before transmitting
        self.channels[0].set_snr(snr_db)
        signals_rx, _ = self.channels[0].transmit_simo(signal_tx, num_rx=num_rx)
        
        # Step 4: Per-antenna demodulation and channel estimation
        # This can be done in parallel for efficiency
        if parallel and num_rx > 1:
            # Parallel processing with ThreadPoolExecutor
            symbols_rx_list = [None] * num_rx
            h_estimates_per_antenna = [None] * num_rx
            
            with ThreadPoolExecutor(max_workers=num_rx) as executor:
                futures = {}
                for i, signal_rx in enumerate(signals_rx):
                    future = executor.submit(self._demodulate_with_channel_est, signal_rx)
                    futures[i] = future
                
                # Collect results in order
                for antenna_idx in sorted(futures.keys()):
                    symbols_rx_list[antenna_idx], h_estimates_per_antenna[antenna_idx] = \
                        futures[antenna_idx].result()
        else:
            # Sequential processing
            symbols_rx_list = []
            h_estimates_per_antenna = []
            for signal_rx in signals_rx:
                symbols, h_est_list = self._demodulate_with_channel_est(signal_rx)
                symbols_rx_list.append(symbols)
                h_estimates_per_antenna.append(h_est_list)
        
        # Step 5: MRC combining in frequency domain
        # This includes equalization implicitly in the weights
        symbols_combined = self._combine_symbols_mrc(symbols_rx_list, h_estimates_per_antenna)
        
        # Step 6: Symbol detection to bits
        # Use the QAM demodulator to convert combined symbols to bits
        bits_rx = self.rx.demodulator.qam_demodulator.symbols_to_bits(symbols_combined)
        
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
            'signal_rx_list': signals_rx,
            'symbols_tx': symbols_tx,
            'symbols_rx_combined': symbols_combined,
            'symbols_rx_list': symbols_rx_list,
            'channel_estimates_per_antenna': h_estimates_per_antenna,
            'num_rx': num_rx,
            'combining_method': combining,
            'diversity_level': num_rx,
            'parallel_processing': parallel
        }
        
        self.last_results = results
        return results
    
    
    def _combine_bits_majority(self, bits_rx_list: List[np.ndarray]) -> np.ndarray:
        """
        Combine bits from multiple RX paths using majority voting
        
        For each bit position, take the most common value across all paths.
        This is a simple but effective hard-decision combining scheme.
        
        Parameters:
        -----------
        bits_rx_list : list of np.ndarray
            List of bit arrays from different RX paths
        
        Returns:
        --------
        np.ndarray : Combined bit array
        """
        if len(bits_rx_list) == 0:
            raise ValueError("No bit arrays provided")
        
        if len(bits_rx_list) == 1:
            return bits_rx_list[0]
        
        # Ensure all have same length
        min_len = min(len(b) for b in bits_rx_list)
        bits_rx_list = [b[:min_len] for b in bits_rx_list]
        
        # Stack all bits and perform majority voting
        bits_stacked = np.stack(bits_rx_list, axis=0)  # Shape: (num_rx, num_bits)
        
        # Majority voting: sum across RX paths, then threshold at 0.5
        bits_sum = np.sum(bits_stacked, axis=0)  # Sum votes
        bits_combined = (bits_sum >= len(bits_rx_list) / 2).astype(int)
        
        return bits_combined
    
    def _combine_mrc(self, signals_rx: List[np.ndarray], 
                     channel_coeffs: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Maximum Ratio Combining (MRC) of multiple received signals
        
        Combines multiple received signals optimally using channel estimation.
        Weight for each signal: w_i = h_i* / sum(|h_i|^2)
        Combined signal: y = sum(w_i * y_i)
        
        This provides diversity gain proportional to number of antennas.
        For N identical channels: SNR gain ≈ 10*log10(N)
        
        Parameters:
        -----------
        signals_rx : list of np.ndarray
            List of received signals from different antennas (complex)
        channel_coeffs : list of np.ndarray, optional
            List of estimated channel coefficients for each path
            If None, use equal gain combining (EGC)
        
        Returns:
        --------
        np.ndarray : Combined signal with improved SNR
        
        Theory:
            MRC is optimal for AWGN channels and maximizes SNR.
            Weight proportional to: conjugate(channel_coeff) / ||channel_coeff||^2
            All available SNR is combined coherently.
        """
        if len(signals_rx) == 0:
            raise ValueError("No received signals provided")
        
        if len(signals_rx) == 1:
            # Single antenna: no combining needed
            return signals_rx[0]
        
        # Ensure all signals have same length
        min_len = min(len(sig) for sig in signals_rx)
        signals_rx = [sig[:min_len] for sig in signals_rx]
        
        if channel_coeffs is None or len(channel_coeffs) == 0:
            # Equal Gain Combining (EGC): all weights = 1/N
            signal_combined = np.mean(signals_rx, axis=0)
            return signal_combined
        
        # Maximum Ratio Combining (MRC): weight by channel coefficient
        signal_combined = np.zeros(min_len, dtype=complex)
        total_power = np.zeros(min_len, dtype=float)
        
        for i, (signal, h_est) in enumerate(zip(signals_rx, channel_coeffs)):
            # Ensure h_est has same length as signal
            h_est = h_est[:min_len]
            
            # Scalar channel coefficient (from estimation)
            if len(h_est.shape) == 0 or h_est.shape[0] == 1:
                # Single value
                h_coeff = h_est if len(h_est.shape) == 0 else h_est[0]
                h_power = np.abs(h_coeff) ** 2
                
                if h_power > 1e-12:
                    # Weight: w = h* / |h|^2
                    weight = np.conj(h_coeff) / (h_power + 1e-12)
                    signal_combined += weight * signal
                    total_power += h_power
            else:
                # Array of channel coefficients (per sample)
                h_power = np.abs(h_est) ** 2
                weight = np.conj(h_est) / (h_power + 1e-12)
                signal_combined += weight * signal
                total_power += h_power
        
        # Normalize to maintain signal power
        total_power_mean = np.mean(total_power)
        if total_power_mean > 1e-12:
            signal_combined = signal_combined / np.sqrt(total_power_mean)
        
        return signal_combined
    
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



    def simulate_miso(self, bits: np.ndarray, snr_db: float = 10.0) -> Dict:
        """
        Simulate MISO (2 TX, 1 RX) transmission with SFBC Alamouti
        
        Supports multi-symbol transmission for large bit streams.
        
        Parameters:
        -----------
        bits : np.ndarray
            Input bit array (0s and 1s)
        snr_db : float
            Signal-to-Noise Ratio in dB
        
        Returns:
        --------
        dict : Simulation results
        """
        from core.sfbc_alamouti import SFBCAlamouti, SFBCResourceMapper
        from core.modulator import QAMModulator
        from core.resource_mapper import ResourceMapper
        
        if not isinstance(bits, np.ndarray):
            bits = np.array(bits, dtype=int)
        
        if bits.size == 0:
            raise ValueError("Bits array cannot be empty")
        
        original_num_bits = len(bits)
        
        print("\n[MISO] Creating transmitter with 2 TX antennas (SFBC Alamouti)...")
        
        # Initialize components
        qam_modulator = QAMModulator(self.config.modulation)
        resource_mapper = ResourceMapper(self.config)
        sfbc_mapper = SFBCResourceMapper(resource_mapper)
        sfbc_encoder = SFBCAlamouti(num_tx=2, enabled=True)
        
        # Calculate capacity per OFDM symbol
        bits_per_symbol = int(np.log2(len(qam_modulator.constellation)))
        symbols_per_ofdm = sfbc_mapper.num_data  # Even number of data subcarriers
        bits_per_ofdm = symbols_per_ofdm * bits_per_symbol
        
        num_ofdm_symbols = int(np.ceil(original_num_bits / bits_per_ofdm))
        
        print(f"  Data symbols per OFDM: {symbols_per_ofdm}")
        print(f"  Bits per OFDM symbol: {bits_per_ofdm}")
        print(f"  Number of OFDM symbols: {num_ofdm_symbols}")
        
        # Pad bits to multiple of bits_per_ofdm
        bits_padded = bits.copy()
        if len(bits_padded) < num_ofdm_symbols * bits_per_ofdm:
            bits_padded = np.pad(bits_padded, 
                                (0, num_ofdm_symbols * bits_per_ofdm - len(bits_padded)), 
                                'constant')
        
        self.channels[0].set_snr(snr_db)
        
        # Storage
        all_grids_tx0 = []
        all_grids_tx1 = []
        all_signals_tx0 = []
        all_signals_tx1 = []
        all_bits_chunks = []
        papr_tx0_list = []
        papr_tx1_list = []
        
        # Step 1: Prepare all OFDM symbols (modulation + encoding)
        for ofdm_idx in range(num_ofdm_symbols):
            # Extract bits for this OFDM symbol
            start_bit = ofdm_idx * bits_per_ofdm
            end_bit = (ofdm_idx + 1) * bits_per_ofdm
            bits_chunk = bits_padded[start_bit:end_bit]
            all_bits_chunks.append(bits_chunk)
            
            # QAM modulation
            qam_symbols = qam_modulator.bits_to_symbols(bits_chunk)
            
            # SFBC encoding
            tx0_data, tx1_data = sfbc_encoder.encode(qam_symbols)
            
            # Map to grids
            grid_tx0, grid_tx1 = sfbc_mapper.map_sfbc_to_grid(tx0_data, tx1_data)
            all_grids_tx0.append(grid_tx0)
            all_grids_tx1.append(grid_tx1)
            
            # IFFT
            time_tx0 = np.fft.ifft(grid_tx0) * np.sqrt(self.config.N)
            time_tx1 = np.fft.ifft(grid_tx1) * np.sqrt(self.config.N)
            
            # Add CP
            signal_tx0 = np.concatenate([time_tx0[-self.config.cp_length:], time_tx0])
            signal_tx1 = np.concatenate([time_tx1[-self.config.cp_length:], time_tx1])
            
            all_signals_tx0.append(signal_tx0)
            all_signals_tx1.append(signal_tx1)
            
            # Calculate PAPR
            power_tx0 = np.abs(signal_tx0) ** 2
            papr_tx0_db = 10 * np.log10(np.max(power_tx0) / np.mean(power_tx0))
            papr_tx0_list.append(papr_tx0_db)
            
            power_tx1 = np.abs(signal_tx1) ** 2
            papr_tx1_db = 10 * np.log10(np.max(power_tx1) / np.mean(power_tx1))
            papr_tx1_list.append(papr_tx1_db)
        
        # Step 2: Concatenate all signals for transmission
        signal_tx0_full = np.concatenate(all_signals_tx0)
        signal_tx1_full = np.concatenate(all_signals_tx1)
        
        # Step 3: Transmit through MIMO channel (all symbols at once)
        signals_rx, channel_matrix = self.channels[0].transmit_mimo(
            [signal_tx0_full, signal_tx1_full], num_rx=1
        )
        signal_rx = signals_rx[0]
        
        # Step 4: Demodulate and estimate channel periodically (like SISO/SIMO)
        from core.mimo_channel_estimator_periodic import MIMOChannelEstimatorPeriodic
        
        mimo_estimator = MIMOChannelEstimatorPeriodic(self.config, slot_size=14)
        all_grids_rx, H0_per_symbol, H1_per_symbol = mimo_estimator.demodulate_and_estimate_mimo(
            signal_rx, self.config.cp_length
        )
        
        # Step 5: Decode each OFDM symbol with its periodic channel estimate
        all_bits_rx = []
        total_bit_errors = 0
        
        for ofdm_idx in range(min(num_ofdm_symbols, len(all_grids_rx))):
            grid_rx = all_grids_rx[ofdm_idx]
            bits_chunk = all_bits_chunks[ofdm_idx]
            
            # Get channel estimates for this symbol
            H0_full = H0_per_symbol[ofdm_idx] if ofdm_idx < len(H0_per_symbol) else H0_per_symbol[-1]
            H1_full = H1_per_symbol[ofdm_idx] if ofdm_idx < len(H1_per_symbol) else H1_per_symbol[-1]
            
            # Extract channel at data positions
            H0_data = H0_full[sfbc_mapper.data_indices]
            H1_data = H1_full[sfbc_mapper.data_indices]
            
            # Extract data and decode
            data_rx = sfbc_mapper.extract_data_from_grid(grid_rx)
            decoded_symbols = sfbc_encoder.decode(data_rx, H0_data, H1_data)
            
            # Symbol detection
            constellation = qam_modulator.get_constellation()
            symbols_detected = np.zeros_like(decoded_symbols)
            
            for i, symbol in enumerate(decoded_symbols):
                distances = np.abs(constellation - symbol)
                nearest_idx = np.argmin(distances)
                symbols_detected[i] = constellation[nearest_idx]
            
            # Demodulate to bits
            bits_rx_chunk = qam_modulator.symbols_to_bits(symbols_detected)
            all_bits_rx.append(bits_rx_chunk)
            
            # Calculate errors for this chunk
            total_bit_errors += np.sum(bits_chunk != bits_rx_chunk)
        
        # Concatenate all received bits
        bits_rx = np.concatenate(all_bits_rx)[:original_num_bits]
        
        # Final BER calculation
        bit_errors = np.sum(bits[:original_num_bits] != bits_rx)
        ber = bit_errors / original_num_bits if original_num_bits > 0 else 0
        
        papr_tx0_avg = np.mean(papr_tx0_list)
        papr_tx1_avg = np.mean(papr_tx1_list)
        
        print(f"  PAPR TX0: {papr_tx0_avg:.2f} dB")
        print(f"  PAPR TX1: {papr_tx1_avg:.2f} dB")
        print(f"[MISO] Transmitting through MIMO channel (SNR={snr_db} dB)...")
        print(f"  Channel H[0,0]={channel_matrix[0,0]:.3f}, H[0,1]={channel_matrix[0,1]:.3f}")
        print("[MISO] Demodulating with SFBC decoding...")
        print(f"\n[MISO] BER: {ber:.4e}, Errors: {bit_errors}")
        
        results = {
            'transmitted_bits': int(original_num_bits),
            'received_bits': int(original_num_bits),
            'bits_received_array': bits_rx,
            'bit_errors': int(bit_errors),
            'errors': int(bit_errors),
            'ber': float(ber),
            'snr_db': float(snr_db),
            'num_tx': 2,
            'num_rx': 1,
            'mode': 'MISO-SFBC',
            'diversity_order': 2,
            'channel_matrix': channel_matrix,
            'papr_db_tx0': float(papr_tx0_avg),
            'papr_db_tx1': float(papr_tx1_avg),
            'papr_db': float(np.mean([papr_tx0_avg, papr_tx1_avg])),
            'papr_linear': 10 ** (np.mean([papr_tx0_avg, papr_tx1_avg]) / 10),
        }
        
        self.last_results = results
        return results


    def simulate_mimo(self, bits: np.ndarray, snr_db: float = 10.0, num_rx: int = 2) -> Dict:
        """
        Simulate MIMO (2 TX, N RX) transmission with SFBC + diversity
        
        Supports multi-symbol transmission for large bit streams.
        
        Parameters:
        -----------
        bits : np.ndarray
            Input bit array
        snr_db : float
            SNR in dB
        num_rx : int
            Number of RX antennas
        
        Returns:
        --------
        dict : Simulation results
        """
        from core.sfbc_alamouti import SFBCAlamouti, SFBCResourceMapper
        from core.modulator import QAMModulator
        from core.resource_mapper import ResourceMapper
        
        if not isinstance(bits, np.ndarray):
            bits = np.array(bits, dtype=int)
        
        original_num_bits = len(bits)
        
        print(f"\n[MIMO] Creating transmitter with 2 TX, {num_rx} RX (SFBC)...")
        
        # Initialize components
        qam_modulator = QAMModulator(self.config.modulation)
        resource_mapper = ResourceMapper(self.config)
        sfbc_mapper = SFBCResourceMapper(resource_mapper)
        sfbc_encoder = SFBCAlamouti(num_tx=2, enabled=True)
        
        # Calculate capacity per OFDM symbol
        bits_per_symbol = int(np.log2(len(qam_modulator.constellation)))
        symbols_per_ofdm = sfbc_mapper.num_data
        bits_per_ofdm = symbols_per_ofdm * bits_per_symbol
        
        num_ofdm_symbols = int(np.ceil(original_num_bits / bits_per_ofdm))
        
        # Pad bits
        bits_padded = bits.copy()
        if len(bits_padded) < num_ofdm_symbols * bits_per_ofdm:
            bits_padded = np.pad(bits_padded, 
                                (0, num_ofdm_symbols * bits_per_ofdm - len(bits_padded)), 
                                'constant')
        
        self.channels[0].set_snr(snr_db)
        
        # Storage
        all_grids_tx0 = []
        all_grids_tx1 = []
        all_signals_tx0 = []
        all_signals_tx1 = []
        all_bits_chunks = []
        papr_tx0_list = []
        papr_tx1_list = []
        
        # Step 1: Prepare all OFDM symbols (modulation + encoding)
        for ofdm_idx in range(num_ofdm_symbols):
            # Extract bits for this OFDM symbol
            start_bit = ofdm_idx * bits_per_ofdm
            end_bit = (ofdm_idx + 1) * bits_per_ofdm
            bits_chunk = bits_padded[start_bit:end_bit]
            all_bits_chunks.append(bits_chunk)
            
            # QAM modulation
            qam_symbols = qam_modulator.bits_to_symbols(bits_chunk)
            
            # SFBC encoding
            tx0_data, tx1_data = sfbc_encoder.encode(qam_symbols)
            
            # Map to grids
            grid_tx0, grid_tx1 = sfbc_mapper.map_sfbc_to_grid(tx0_data, tx1_data)
            all_grids_tx0.append(grid_tx0)
            all_grids_tx1.append(grid_tx1)
            
            # IFFT
            time_tx0 = np.fft.ifft(grid_tx0) * np.sqrt(self.config.N)
            time_tx1 = np.fft.ifft(grid_tx1) * np.sqrt(self.config.N)
            
            # Add CP
            signal_tx0 = np.concatenate([time_tx0[-self.config.cp_length:], time_tx0])
            signal_tx1 = np.concatenate([time_tx1[-self.config.cp_length:], time_tx1])
            
            all_signals_tx0.append(signal_tx0)
            all_signals_tx1.append(signal_tx1)
            
            # PAPR
            power_tx0 = np.abs(signal_tx0) ** 2
            papr_tx0_db = 10 * np.log10(np.max(power_tx0) / np.mean(power_tx0))
            papr_tx0_list.append(papr_tx0_db)
            
            power_tx1 = np.abs(signal_tx1) ** 2
            papr_tx1_db = 10 * np.log10(np.max(power_tx1) / np.mean(power_tx1))
            papr_tx1_list.append(papr_tx1_db)
        
        # Step 2: Concatenate all signals for transmission
        signal_tx0_full = np.concatenate(all_signals_tx0)
        signal_tx1_full = np.concatenate(all_signals_tx1)
        
        # Step 3: Transmit through MIMO channel (all symbols at once)
        signals_rx, channel_matrix = self.channels[0].transmit_mimo(
            [signal_tx0_full, signal_tx1_full], num_rx=num_rx
        )
        
        # Step 4: Demodulate and estimate channel periodically per RX antenna
        from core.mimo_channel_estimator_periodic import MIMOChannelEstimatorPeriodic
        
        mimo_estimator = MIMOChannelEstimatorPeriodic(self.config, slot_size=14)
        
        # Process each RX antenna separately
        all_grids_rx_per_antenna = []
        H0_per_symbol_per_antenna = []
        H1_per_symbol_per_antenna = []
        
        for rx_idx in range(num_rx):
            signal_rx = signals_rx[rx_idx]
            all_grids_rx, H0_per_symbol, H1_per_symbol = mimo_estimator.demodulate_and_estimate_mimo(
                signal_rx, self.config.cp_length
            )
            all_grids_rx_per_antenna.append(all_grids_rx)
            H0_per_symbol_per_antenna.append(H0_per_symbol)
            H1_per_symbol_per_antenna.append(H1_per_symbol)
        
        # Step 5: Decode each OFDM symbol with periodic estimates (combine across RX)
        all_bits_rx = []
        total_bit_errors = 0
        
        for ofdm_idx in range(min(num_ofdm_symbols, len(all_grids_rx_per_antenna[0]))):
            bits_chunk = all_bits_chunks[ofdm_idx]
            
            # Decode per RX antenna
            decoded_per_rx = []
            
            for rx_idx in range(num_rx):
                grid_rx = all_grids_rx_per_antenna[rx_idx][ofdm_idx]
                
                # Get channel estimates for this symbol and RX
                H0_full = H0_per_symbol_per_antenna[rx_idx][ofdm_idx] if ofdm_idx < len(H0_per_symbol_per_antenna[rx_idx]) else H0_per_symbol_per_antenna[rx_idx][-1]
                H1_full = H1_per_symbol_per_antenna[rx_idx][ofdm_idx] if ofdm_idx < len(H1_per_symbol_per_antenna[rx_idx]) else H1_per_symbol_per_antenna[rx_idx][-1]
                
                # Extract channel at data positions
                H0_data = H0_full[sfbc_mapper.data_indices]
                H1_data = H1_full[sfbc_mapper.data_indices]
                
                # Extract and decode
                data_rx = sfbc_mapper.extract_data_from_grid(grid_rx)
                decoded = sfbc_encoder.decode(data_rx, H0_data, H1_data)
                decoded_per_rx.append(decoded)
            
            # Combine (average across RX antennas)
            decoded_symbols = np.mean(decoded_per_rx, axis=0)
            
            # Detection
            constellation = qam_modulator.get_constellation()
            symbols_detected = np.zeros_like(decoded_symbols)
            
            for i, symbol in enumerate(decoded_symbols):
                distances = np.abs(constellation - symbol)
                nearest_idx = np.argmin(distances)
                symbols_detected[i] = constellation[nearest_idx]
            
            bits_rx_chunk = qam_modulator.symbols_to_bits(symbols_detected)
            all_bits_rx.append(bits_rx_chunk)
            
            # Calculate errors for this chunk
            total_bit_errors += np.sum(bits_chunk != bits_rx_chunk)
        
        # Concatenate all received bits
        bits_rx = np.concatenate(all_bits_rx)[:original_num_bits]
        
        # Final BER calculation
        bit_errors = np.sum(bits[:original_num_bits] != bits_rx)
        ber = bit_errors / original_num_bits if original_num_bits > 0 else 0
        
        diversity_order = 2 * num_rx
        
        papr_tx0_avg = np.mean(papr_tx0_list)
        papr_tx1_avg = np.mean(papr_tx1_list)
        
        print(f"[MIMO] Transmitting through 2×{num_rx} channel (SNR={snr_db} dB)...")
        print(f"  Channel matrix: {channel_matrix.shape}")
        print(f"[MIMO] Demodulating with {num_rx} RX antennas...")
        print(f"\n[MIMO] BER: {ber:.4e}, Diversity: {diversity_order}")
        
        results = {
            'transmitted_bits': int(original_num_bits),
            'received_bits': int(original_num_bits),
            'bits_received_array': bits_rx,
            'bit_errors': int(bit_errors),
            'errors': int(bit_errors),
            'ber': float(ber),
            'snr_db': float(snr_db),
            'num_tx': 2,
            'num_rx': num_rx,
            'mode': 'MIMO-SFBC',
            'diversity_order': diversity_order,
            'channel_matrix': channel_matrix,
            'papr_db_tx0': float(papr_tx0_avg),
            'papr_db_tx1': float(papr_tx1_avg),
            'papr_db': float(np.mean([papr_tx0_avg, papr_tx1_avg])),
            'papr_linear': 10 ** (np.mean([papr_tx0_avg, papr_tx1_avg]) / 10),
        }
        
        self.last_results = results
        return results
    
    def get_config(self) -> LTEConfig:
        """Get simulator configuration"""
        return self.config
    
    def __repr__(self) -> str:
        mode = "SC-FDM" if self.enable_sc_fdm else "OFDM"
        num_ch = len(self.channels)
        return f"OFDMSimulator({self.config.modulation}, {mode}, {self.channel_type}, {num_ch}ch)"