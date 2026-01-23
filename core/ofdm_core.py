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
from core.mimo_channel_estimator_periodic import MIMOChannelEstimatorPeriodic


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
        Transmit signal through SIMO (Single-Input Multiple-Output) configuration
        
        SIMO Reception: All RX antennas receive through INDEPENDENT channels
        (different fading, different noise).
        
        This models physical SIMO where:
        - Single transmit antenna
        - Multiple RX antennas spatially separated
        - Independent fading and noise per antenna (realistic for wide spacing)
        
        Channel estimation will be done using CRS (Cell-specific Reference Signals)
        in the receiver side.
        
        SNR scaling: With N antennas and MRC combining,
        provides ~10*log10(N) dB SNR improvement.
        
        Parameters:
        -----------
        signal_tx : np.ndarray
            Transmitted OFDM signal (with CRS pilots already inserted)
        num_rx : int
            Number of receive antennas
        
        Returns:
        --------
        List[np.ndarray] : List of received signals (one per RX antenna)
        """
        if num_rx < 1:
            raise ValueError("num_rx must be >= 1")
        
        signals_rx = []
        
        # For SIMO: each antenna sees independent channel realization
        for rx_idx in range(num_rx):
            # Create independent channel for each antenna
            ch_copy = ChannelSimulator(
                channel_type=self.channel_type,
                snr_db=self.snr_db,
                fs=self.fs,
                itu_profile=self.profile if self.channel_type == 'rayleigh_mp' else None,
                frequency_ghz=self.frequency_ghz if self.channel_type == 'rayleigh_mp' else None,
                velocity_kmh=self.velocity_kmh if self.channel_type == 'rayleigh_mp' else None,
                verbose=False  # Avoid repeated prints
            )
            
            # Transmit through independent channel
            signal_rx = ch_copy.transmit(signal_tx)
            signals_rx.append(signal_rx)
        
        return signals_rx
    
    def transmit(self, signal: np.ndarray) -> np.ndarray:
        """
        Transmit signal through the channel (SISO)
        
        Parameters:
        -----------
        signal : np.ndarray
            Transmitted signal
        
        Returns:
        --------
        np.ndarray : Received signal
        """
        return self.channel.transmit(signal)
    
    def set_snr(self, snr_db: float):
        """Set channel SNR"""
        self.snr_db = snr_db
        self.channel.set_snr(snr_db)

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
                    # LTE-like AWGN MIMO channel with spatial diversity
                    # Use different phase per TX antenna to enable diversity gain
                    # TX0: reference (0°), TX1: 90° phase, TX2: 180°, etc.
                    if tx_idx == 0:
                        h = 1.0 + 0j  # Reference antenna
                    else:
                        # Phase separation for spatial diversity
                        # For 2 TX: 0° and 90° (orthogonal)
                        # For 4 TX: 0°, 90°, 180°, 270° (orthogonal)
                        h_phase = (tx_idx * np.pi / 2)  # 90° per antenna
                        h = np.exp(1j * h_phase)
                    
                    signal_through_channel = signals_tx[tx_idx] * h
                else:
                    # Rayleigh: independent fading per path
                    # Use very high SNR (no noise) because noise will be added later
                    # This avoids double-noising and allows correct SNR control
                    from core.channel import ChannelSimulator
                    
                    link_channel = ChannelSimulator(
                        channel_type=self.channel_type,
                        snr_db=100.0,  # Very high SNR = fading only, no noise
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
            
            # Add noise based on channel type
            # For MIMO: normalize SNR by number of TX antennas to maintain fair comparison
            signal_power = np.mean(np.abs(rx_signal)**2)
            snr_linear = 10 ** (self.snr_db / 10)  # Convert dB to linear scale
            
            if self.channel_type == 'awgn':
                # For AWGN: Each TX transmits with power P/num_tx
                # Total received power is sum of all contributions
                noise_power = (signal_power / num_tx) / snr_linear
            else:
                # For Rayleigh: Fading channels created with SNR=100dB (no noise)
                # Add noise here with correct SNR (normalized by num_tx)
                noise_power = (signal_power / num_tx) / snr_linear
            
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
    
    def calculate_noise_var_zf(self, H_estimate: np.ndarray, snr_db: float) -> float:
        """
        Calcula la varianza de ruido efectiva después de ZF equalization
        
        Zero-Forcing amplifica el ruido inversamente proporcional a |H|².
        Para canales con selectividad de frecuencia (multipath), usamos un
        enfoque conservador basado en el promedio armónico de |H|² para
        capturar el peor caso (portadoras con canal débil).
        
        Formula:
            σ²_eff = σ²_n / |H|²_harmonic
        
        donde |H|²_harmonic = N / sum(1/|H|²) es más conservador que el promedio.
        
        Parameters:
        -----------
        H_estimate : np.ndarray
            Estimación del canal (complejos) para cada portadora
        snr_db : float
            SNR del canal en dB
        
        Returns:
        --------
        float : Varianza de ruido efectiva (σ²_eff)
        """
        if H_estimate.size == 0:
            # Sin estimación, usar solo el ruido base del canal
            return 1.0 / (10 ** (snr_db / 10))
        
        # Convertir a array si es escalar
        H_estimate = np.atleast_1d(H_estimate)
        
        # 1. Varianza de ruido base del canal
        noise_var_channel = 1.0 / (10 ** (snr_db / 10))
        
        # 2. Calcular potencia del canal |H|²
        H_power = np.abs(H_estimate) ** 2
        
        # Evitar divisiones por cero
        H_power = np.maximum(H_power, 1e-12)
        
        # 3. Si es un solo valor (flat fading), usar directamente
        if len(H_power) == 1:
            noise_var_eff = noise_var_channel / H_power[0]
        else:
            # Múltiples portadoras: usar promedio armónico (más conservador para multipath)
            # mean_harmonic = N / sum(1/x)
            H_power_harmonic = len(H_power) / np.sum(1.0 / H_power)
            noise_var_eff = noise_var_channel / H_power_harmonic
        
        return noise_var_eff
    
    def _calculate_llrs_qpsk(self, symbols: np.ndarray, noise_var: np.ndarray) -> np.ndarray:
        """
        Calcula LLRs para QPSK (2 bits por símbolo)
        
        QPSK Gray mapping:
        00 -> +1+1j  (I=+1, Q=+1)
        01 -> +1-1j  (I=+1, Q=-1)
        10 -> -1+1j  (I=-1, Q=+1)
        11 -> -1-1j  (I=-1, Q=-1)
        
        LLR(b) = log(P(b=0|y) / P(b=1|y))
        Para QPSK: LLR_I = 2*Re(y)*sqrt(2)/σ², LLR_Q = 2*Im(y)*sqrt(2)/σ²
        """
        scale = np.sqrt(2)
        llr_i = (2.0 / noise_var) * symbols.real * scale
        llr_q = (2.0 / noise_var) * symbols.imag * scale
        
        # Intercalar LLRs: [I0, Q0, I1, Q1, ...]
        llrs = np.zeros(2 * len(symbols), dtype=np.float64)
        llrs[0::2] = llr_i
        llrs[1::2] = llr_q
        
        return llrs
    
    def _calculate_llrs_16qam(self, symbols: np.ndarray, noise_var: np.ndarray) -> np.ndarray:
        """
        Calcula LLRs para 16-QAM usando método max-log-MAP
        
        16-QAM: mapeo binario directo (NO Gray) como en QAMModulator
        Índice binario [b3 b2 b1 b0] → símbolo constellation[índice]
        """
        # Normalización 16-QAM
        scale = np.sqrt(10)
        
        # Asegurar que noise_var es array
        if np.isscalar(noise_var):
            noise_var = np.full(len(symbols), noise_var)
        
        # Generar constelación exactamente como QAMModulator
        real_vals = np.array([-3, -1, 1, 3])
        imag_vals = np.array([-3, -1, 1, 3])
        constellation = []
        bit_map = []
        
        # Mapeo binario directo (mismo orden que QAMModulator)
        idx = 0
        for r in real_vals:
            for i in imag_vals:
                constellation.append((r + 1j * i) / scale)
                # Índice binario: bits = [b3, b2, b1, b0]
                bits = [(idx >> (3-b)) & 1 for b in range(4)]
                bit_map.append(bits)
                idx += 1
        
        constellation = np.array(constellation)
        bit_map = np.array(bit_map, dtype=int)
        
        # Calcular LLRs
        llrs = np.zeros(4 * len(symbols), dtype=np.float64)
        
        for sym_idx in range(len(symbols)):
            y = symbols[sym_idx]
            sigma2 = noise_var[sym_idx]
            
            for bit_pos in range(4):
                idx_bit0 = np.where(bit_map[:, bit_pos] == 0)[0]
                dist_bit0 = np.abs(y - constellation[idx_bit0])**2
                min_dist_bit0 = np.min(dist_bit0)
                
                idx_bit1 = np.where(bit_map[:, bit_pos] == 1)[0]
                dist_bit1 = np.abs(y - constellation[idx_bit1])**2
                min_dist_bit1 = np.min(dist_bit1)
                
                llr = (min_dist_bit1 - min_dist_bit0) / (2.0 * sigma2)
                llr = np.clip(llr, -10.0, 10.0)  # Clipping para evitar saturación del decoder
                llrs[sym_idx * 4 + bit_pos] = llr
        
        return llrs
    
    def _calculate_llrs_64qam(self, symbols: np.ndarray, noise_var: np.ndarray) -> np.ndarray:
        """
        Calcula LLRs para 64-QAM usando método max-log-MAP
        
        64-QAM: mapeo binario directo (NO Gray) como en QAMModulator
        Índice binario [b5 b4 b3 b2 b1 b0] → símbolo constellation[índice]
        """
        # Normalización 64-QAM
        scale = np.sqrt(42)
        
        # Asegurar que noise_var es array
        if np.isscalar(noise_var):
            noise_var = np.full(len(symbols), noise_var)
        
        # Generar constelación exactamente como QAMModulator
        real_vals = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        imag_vals = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        constellation = []
        bit_map = []
        
        # Mapeo binario directo (mismo orden que QAMModulator)
        idx = 0
        for r in real_vals:
            for i in imag_vals:
                constellation.append((r + 1j * i) / scale)
                # Índice binario: bits = [b5, b4, b3, b2, b1, b0]
                bits = [(idx >> (5-b)) & 1 for b in range(6)]
                bit_map.append(bits)
                idx += 1
        
        constellation = np.array(constellation)
        bit_map = np.array(bit_map, dtype=int)
        
        # Calcular LLRs
        llrs = np.zeros(6 * len(symbols), dtype=np.float64)
        
        for sym_idx in range(len(symbols)):
            y = symbols[sym_idx]
            sigma2 = noise_var[sym_idx]
            
            for bit_pos in range(6):
                idx_bit0 = np.where(bit_map[:, bit_pos] == 0)[0]
                dist_bit0 = np.abs(y - constellation[idx_bit0])**2
                min_dist_bit0 = np.min(dist_bit0)
                
                idx_bit1 = np.where(bit_map[:, bit_pos] == 1)[0]
                dist_bit1 = np.abs(y - constellation[idx_bit1])**2
                min_dist_bit1 = np.min(dist_bit1)
                
                llr = (min_dist_bit1 - min_dist_bit0) / (2.0 * sigma2)
                llr = np.clip(llr, -10.0, 10.0)  # Clipping para evitar saturación del decoder
                llrs[sym_idx * 6 + bit_pos] = llr
        
        return llrs
    
    def simulate_siso_coded(self, bits: np.ndarray, snr_db: float = 10.0) -> Dict:
        """
        Simulate SISO transmission WITH channel coding (CRC + Turbo)
        
        Este método implementa el MISMO flujo que simulate_siso() pero agregando:
        - TX: CRC-24A → Segmentation → Turbo Encoding → Rate Matching
        - RX: LLR generation → Rate Dematching → Turbo Decoding → CRC Check
        
        La infraestructura (TX, Channel, RX) es la MISMA que simulate_siso().
        Esto permite comparación directa entre coded vs uncoded.
        
        IMPORTANTE:
        - NO modifica simulate_siso() existente (backward compatible)
        - Usa misma modulación, resource mapping, channel estimation
        - Solo agrega capas de channel coding
        
        Parameters:
        -----------
        bits : np.ndarray
            Input bit array (0s and 1s) - Transport block
        snr_db : float
            Signal-to-Noise Ratio in dB
        
        Returns:
        --------
        dict : Complete simulation results
            - transmitted_bits: Number of transport block bits
            - received_bits: Number of decoded bits
            - bit_errors: Number of bit errors (before CRC check)
            - ber: Bit Error Rate
            - crc_pass: True if CRC check passed
            - snr_db: SNR used
            - papr_db: PAPR in dB
            - coded_bits_length: Length after channel coding
            - signal_tx: Transmitted signal
            - signal_rx: Received signal
            - symbols_tx: Transmitted symbols
            - symbols_rx: Received symbols (equalized)
            - H_estimate: Channel estimate
            - noise_var_eff: Effective noise variance for LLRs
        
        Example:
        --------
        >>> sim = OFDMSimulator(config)
        >>> bits = np.random.randint(0, 2, 1000)
        >>> result = sim.simulate_siso_coded(bits, snr_db=10.0)
        >>> result['crc_pass']
        True
        """
        from core.channel_coding.crc import attach_crc24a, check_crc24a
        from core.channel_coding.segmentation import segment_code_blocks, desegment_code_blocks
        from core.channel_coding.turbo_encoder import turbo_encode
        from core.channel_coding.turbo_decoder import turbo_decode
        from core.channel_coding.rate_matching import rate_match_turbo, rate_dematching_turbo
        from core.modulator import qpsk_to_llrs
        from core.lte_receiver import LTEReceiver
        
        if not isinstance(bits, np.ndarray):
            bits = np.array(bits, dtype=np.uint8)
        
        if bits.size == 0:
            raise ValueError("Bits array cannot be empty")
        
        # Soporta QPSK, 16-QAM y 64-QAM
        supported_modulations = ['QPSK', '16-QAM', '64-QAM']
        if self.config.modulation not in supported_modulations:
            raise ValueError(
                f"Modulation {self.config.modulation} not supported. "
                f"Supported: {supported_modulations}"
            )
        
        original_num_bits = len(bits)
        bits_per_symbol = self.config.bits_per_symbol  # 2, 4, o 6
        
        # =====================================================================
        # TRANSMITTER - Channel Coding Chain
        # =====================================================================
        
        # Step 1: Attach CRC-24A (Transport Block CRC)
        bits_with_crc = attach_crc24a(bits)
        
        # Step 2: Code Block Segmentation
        code_blocks, seg_metadata = segment_code_blocks(bits_with_crc)
        
        # Step 3: Turbo Encoding (rate 1/3)
        encoded_blocks = []
        for cb in code_blocks:
            encoded = turbo_encode(cb)
            encoded_blocks.append(encoded)
        
        # Step 4: Rate Matching (sub-block interleaving + bit collection)
        # Para este ejemplo, usamos E = len(encoded_block) (sin puncturing)
        rate_matched_blocks = []
        for cb, encoded in zip(code_blocks, encoded_blocks):
            K = len(cb)  # Longitud del code block original
            E = len(encoded)  # Sin acortamiento por ahora
            rate_matched = rate_match_turbo(encoded, E, K, rv_idx=0)
            rate_matched_blocks.append(rate_matched)
        
        # Concatenar todos los bloques codificados
        coded_bits = np.concatenate(rate_matched_blocks)
        coded_bits_length = len(coded_bits)
        
        # =====================================================================
        # TRANSMITTER - Modulation (igual que simulate_siso)
        # =====================================================================
        
        # Step 5: Modulate (QPSK/16-QAM/64-QAM)
        from .modulator import QAMModulator
        qam_modulator = QAMModulator(modulation_type=self.config.modulation)
        qam_symbols = qam_modulator.bits_to_symbols(coded_bits)
        
        # Step 5b: Time-Frequency Interleaving
        # Dispersar símbolos codificados para romper correlación de errores
        # Interleaver de bloque: escribir por filas, leer por columnas
        num_symbols = len(qam_symbols)
        num_data_per_ofdm = len(self.tx.modulator.resource_mapper.get_data_indices())  # Portadoras de datos por símbolo OFDM
        
        # Determinar tamaño de bloque: num_rows x num_cols
        # num_cols = portadoras de datos por símbolo OFDM
        # num_rows = ceil(num_symbols / num_cols)
        num_cols = num_data_per_ofdm
        num_rows = int(np.ceil(num_symbols / num_cols))
        
        # Padding si es necesario
        total_size = num_rows * num_cols
        if num_symbols < total_size:
            qam_symbols_padded = np.pad(qam_symbols, (0, total_size - num_symbols), 'constant', constant_values=0)
        else:
            qam_symbols_padded = qam_symbols[:total_size]
        
        # Reshape: escribir por filas (C order)
        interleaver_matrix = qam_symbols_padded.reshape(num_rows, num_cols)
        
        # Leer por columnas (Fortran order)
        interleaved_symbols = interleaver_matrix.T.flatten()
        
        # Step 5c: Map symbols to resource grid and transmit
        from .resource_mapper import ResourceMapper
        resource_mapper = ResourceMapper(self.config)
        
        # Crear señal OFDM de interleaved_symbols
        all_ofdm_signals = []
        num_data_subcarriers = len(resource_mapper.get_data_indices())
        
        # Dividir símbolos en bloques para cada símbolo OFDM
        num_ofdm_symbols_needed = int(np.ceil(len(interleaved_symbols) / num_data_subcarriers))
        
        for ofdm_idx in range(num_ofdm_symbols_needed):
            start_idx = ofdm_idx * num_data_subcarriers
            end_idx = min(start_idx + num_data_subcarriers, len(interleaved_symbols))
            
            # Símbolos para este símbolo OFDM
            data_symbols = interleaved_symbols[start_idx:end_idx]
            
            # Padding si es necesario
            if len(data_symbols) < num_data_subcarriers:
                data_symbols = np.pad(data_symbols, (0, num_data_subcarriers - len(data_symbols)), 'constant', constant_values=0)
            
            # Map symbols to resource grid
            grid_mapped, _ = resource_mapper.map_symbols(data_symbols)
            
            # IFFT
            time_domain = np.fft.ifft(grid_mapped) * np.sqrt(self.config.N)
            
            # Add cyclic prefix
            ofdm_signal = np.concatenate([
                time_domain[-self.config.cp_length:],
                time_domain
            ])
            
            all_ofdm_signals.append(ofdm_signal)
        
        signal_tx = np.concatenate(all_ofdm_signals)
        mapping_info = {'num_ofdm_symbols': num_ofdm_symbols_needed}
        
        # Step 6: Calculate PAPR
        papr_info = self.tx.calculate_papr(signal_tx)
        
        # =====================================================================
        # CHANNEL - Transmission (igual que simulate_siso)
        # =====================================================================
        
        # Step 7: Transmit through channel
        self.channels[0].set_snr(snr_db)
        signal_rx = self.channels[0].transmit(signal_tx)
        
        # =====================================================================
        # RECEIVER - Demodulation with Channel Estimation
        # =====================================================================
        
        # Step 8: Create LTE Receiver with channel estimation enabled
        lte_receiver = LTEReceiver(
            self.config,
            cell_id=0,
            enable_equalization=True,  # ZF equalization
            enable_sc_fdm=self.enable_sc_fdm
        )
        
        # Step 9: Demodulate OFDM stream (FFT + remove CP)
        all_ofdm_symbols = lte_receiver._demodulate_ofdm_stream(signal_rx)
        
        if not all_ofdm_symbols:
            raise ValueError("No OFDM symbols received")
        
        # Step 10: Estimate channel periodically (per OFDM symbol)
        channel_estimates_per_symbol, channel_snr_db = lte_receiver._estimate_channel_periodic(
            all_ofdm_symbols
        )
        
        # Step 11: Equalize symbols (ZF) and collect H estimates for data subcarriers
        data_indices = lte_receiver.resource_grid.get_data_indices()
        
        # IMPORTANTE: Calcular num_coded_symbols ANTES de de-interleaving
        # Usar bits_per_symbol del config (2 para QPSK, 4 para 16-QAM, 6 para 64-QAM)
        num_coded_symbols = coded_bits_length // bits_per_symbol
        
        equalized_symbols_per_symbol = []
        H_estimates_data = []  # H estimates para portadoras de datos
        
        for i, ofdm_sym in enumerate(all_ofdm_symbols):
            if i < len(channel_estimates_per_symbol):
                H_est = channel_estimates_per_symbol[i]
            else:
                # Fallback: usar última estimación disponible
                H_est = channel_estimates_per_symbol[-1]
            
            eq_sym = lte_receiver.equalizer.equalize(ofdm_sym, H_est)
            equalized_symbols_per_symbol.append(eq_sym)
            
            # Extraer H estimates solo para las portadoras de datos
            if isinstance(H_est, np.ndarray) and len(H_est) > 1:
                H_data = H_est[data_indices]
                H_estimates_data.append(H_data)
            else:
                # Si H_est es escalar o mono-elemento, replicar para todas las portadoras de datos
                H_scalar = H_est if np.isscalar(H_est) else (H_est[0] if len(H_est) == 1 else np.mean(H_est))
                H_data = np.full(len(data_indices), H_scalar)
                H_estimates_data.append(H_data)
        
        # Step 12: Extract data symbols (remove pilots)
        all_data_symbols = []
        for eq_sym in equalized_symbols_per_symbol:
            data_syms = eq_sym[data_indices]
            all_data_symbols.append(data_syms)
        
        symbols_rx_interleaved = np.concatenate(all_data_symbols)
        H_estimates_all_data_interleaved = np.concatenate(H_estimates_data)
        
        # Step 12b: Time-Frequency De-Interleaving
        # Revertir el interleaving: leer por columnas, escribir por filas
        num_data_per_ofdm = len(data_indices)
        
        # Calcular dimensiones del interleaver (mismas que TX)
        num_cols = num_data_per_ofdm
        num_rows = int(np.ceil(num_coded_symbols / num_cols))
        total_size = num_rows * num_cols
        
        # Truncar o pad símbolos recibidos si es necesario
        if len(symbols_rx_interleaved) < total_size:
            symbols_rx_padded = np.pad(symbols_rx_interleaved, (0, total_size - len(symbols_rx_interleaved)), 'constant', constant_values=0)
            H_padded = np.pad(H_estimates_all_data_interleaved, (0, total_size - len(H_estimates_all_data_interleaved)), 'edge')
        else:
            symbols_rx_padded = symbols_rx_interleaved[:total_size]
            H_padded = H_estimates_all_data_interleaved[:total_size]
        
        # Reshape según columnas (símbolos leídos por columnas en TX)
        deinterleaver_matrix = symbols_rx_padded.reshape(num_cols, num_rows)
        H_matrix = H_padded.reshape(num_cols, num_rows)
        
        # Leer por filas (revertir T.flatten())
        symbols_rx_data = deinterleaver_matrix.T.flatten()[:num_coded_symbols]
        H_estimates_all_data = H_matrix.T.flatten()[:num_coded_symbols]
        
        # Validar longitud final
        if len(symbols_rx_data) > num_coded_symbols:
            symbols_rx_data = symbols_rx_data[:num_coded_symbols]
            H_estimates_all_data = H_estimates_all_data[:num_coded_symbols]
        elif len(symbols_rx_data) < num_coded_symbols:
            pad_length = num_coded_symbols - len(symbols_rx_data)
            symbols_rx_data = np.pad(symbols_rx_data, (0, pad_length), 'constant', constant_values=0.0)
            if len(H_estimates_all_data) > 0:
                H_estimates_all_data = np.pad(H_estimates_all_data, (0, pad_length), 'edge')
        
        # =====================================================================
        # RECEIVER - LLR Generation (SOFT DEMODULATION)
        # =====================================================================
        
        # Step 13: Calculate effective noise variance PER SUBCARRIER for LLRs
        # Step 13: Calculate effective noise variance PER SUBCARRIER for LLRs
        # 
        # En canal multipath con ZF equalization:
        #   - Símbolos equalizados: y_eq[k] = s[k] + n[k]/H[k]
        #   - Varianza del ruido amplificado: σ²_eff[k] = σ²_n / |H[k]|²
        #   - LLR debe reflejar confiabilidad por portadora
        #
        # CRÍTICO: Cada portadora tiene SNR diferente debido a desvanecimiento selectivo
        # Los LLRs deben ser proporcionales al SNR efectivo de cada símbolo
        
        sigma2_n = 1.0 / (10 ** (snr_db / 10))  # Varianza de ruido base
        
        if hasattr(self.channels[0], 'channel_type') and self.channels[0].channel_type == 'awgn':
            # AWGN: noise_var constante para todos los símbolos
            noise_var_per_symbol = np.full(len(symbols_rx_data), sigma2_n)
        else:
            # Rayleigh/Multipath: calcular noise_var_eff por portadora
            # Después de ZF: σ²_eff[k] = σ²_n / |H[k]|²
            H_power = np.abs(H_estimates_all_data) ** 2
            # Evitar divisiones por cero y limitar valores extremos
            H_power = np.clip(H_power, 1e-6, 1e6)  # Limitar rango dinámico
            noise_var_per_symbol = sigma2_n / H_power
            
            # CRÍTICO: Limitar LLRs para evitar saturación del decoder
            # Estrategia: Limitar amplificación conservadoramente
            # En SNR bajos (≤6dB), el ruido domina y clipping muy agresivo descarta info útil
            # En SNR altos (≥9dB), el clipping protege contra deep fades
            # Compromiso: 6dB máximo (4x amplificación)
            noise_var_min = sigma2_n / 4.0  # Max 6dB de amplificación por ZF
            noise_var_per_symbol = np.maximum(noise_var_per_symbol, noise_var_min)
        
        # Step 14: Generate LLRs from symbols usando noise_var por símbolo
        # Soporta QPSK, 16-QAM y 64-QAM
        if self.config.modulation == 'QPSK':
            llrs = self._calculate_llrs_qpsk(symbols_rx_data, noise_var_per_symbol)
        elif self.config.modulation == '16-QAM':
            llrs = self._calculate_llrs_16qam(symbols_rx_data, noise_var_per_symbol)
        elif self.config.modulation == '64-QAM':
            llrs = self._calculate_llrs_64qam(symbols_rx_data, noise_var_per_symbol)
        else:
            raise ValueError(f"Unsupported modulation: {self.config.modulation}")
        
        # Truncar LLRs a la longitud de bits codificados
        if len(llrs) > coded_bits_length:
            llrs = llrs[:coded_bits_length]
        elif len(llrs) < coded_bits_length:
            # Padding con LLRs neutrales (0)
            llrs = np.pad(llrs, (0, coded_bits_length - len(llrs)), 'constant', constant_values=0.0)
        
        # =====================================================================
        # RECEIVER - Channel Decoding Chain
        # =====================================================================
        
        # Step 15: Rate Dematching
        # Dividir LLRs en bloques (asumiendo misma estructura que TX)
        llr_blocks = []
        offset = 0
        for i, (cb, rm_block) in enumerate(zip(code_blocks, rate_matched_blocks)):
            E = len(rm_block)
            llr_block = llrs[offset:offset + E]
            offset += E
            
            # Rate dematch
            K = len(cb)  # Longitud del code block original
            llr_dematched = rate_dematching_turbo(llr_block, K, rv_idx=0)
            llr_blocks.append(llr_dematched)
        
        # Step 16: Turbo Decoding
        decoded_blocks = []
        for i, llr_block in enumerate(llr_blocks):
            K = len(code_blocks[i])  # Longitud del code block original
            decoded = turbo_decode(llr_block, K=K, num_iterations=8)
            decoded_blocks.append(decoded)
        
        # Step 17: Desegmentation
        decoded_bits_with_crc = desegment_code_blocks(decoded_blocks, seg_metadata)
        
        # Step 18: CRC Check
        crc_pass = check_crc24a(decoded_bits_with_crc)
        
        # Step 19: Remove CRC
        if len(decoded_bits_with_crc) >= 24:
            decoded_bits = decoded_bits_with_crc[:-24]
        else:
            decoded_bits = decoded_bits_with_crc
        
        # =====================================================================
        # Calculate BER
        # =====================================================================
        
        # Ajustar longitud para comparación
        if len(decoded_bits) < original_num_bits:
            decoded_bits = np.pad(decoded_bits, (0, original_num_bits - len(decoded_bits)), 'constant')
        else:
            decoded_bits = decoded_bits[:original_num_bits]
        
        bit_errors = np.sum(bits != decoded_bits)
        ber = bit_errors / original_num_bits
        
        # =====================================================================
        # Return Results
        # =====================================================================
        
        results = {
            'transmitted_bits': int(original_num_bits),
            'received_bits': int(original_num_bits),
            'bits_received_array': decoded_bits,
            'bit_errors': int(bit_errors),
            'ber': float(ber),
            'crc_pass': bool(crc_pass),
            'snr_db': float(snr_db),
            'papr_db': float(papr_info['papr_db']),
            'papr_linear': float(papr_info['papr_linear']),
            'coded_bits_length': int(coded_bits_length),
            'signal_tx': signal_tx,
            'signal_rx': signal_rx,
            'symbols_tx': qam_symbols,  # Símbolos QAM originales (antes de interleaving)
            'symbols_rx': symbols_rx_data,
            'H_estimate': H_estimates_all_data,  # Array con H por portadora de datos
            'channel_snr_db': float(channel_snr_db),
            'noise_var_mean': float(np.mean(noise_var_per_symbol))
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
        Maximum Ratio Combining (MRC) in frequency domain - LTE Standard Implementation
        
        Combines symbols from multiple RX antennas using optimal weights derived 
        from channel estimates. Implements correct MRC formula per 3GPP TS 36.211.
        
        MRC Formula (per subcarrier k in each OFDM symbol):
            Y_combined[k] = Σ_i [conj(H_i[k]) * Y_i[k]] / Σ_i |H_i[k]|²
        
        This provides optimal SNR when:
        - Noise is AWGN with equal power per antenna
        - Channel estimates are accurate
        - Antennas with weak channels get less weight (automatic noise suppression)
        
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
        
        Notes:
        ------
        - Numerator: sum of conj(H_i) * Y_i (co-phased and weighted)
        - Denominator: sum of |H_i|² (total received power)
        - No division by num_rx needed - weights are optimal
        - CRITICAL FIX: Validate and pad all symbol arrays to same length
        """
        from core.resource_mapper import LTEResourceGrid
        
        num_rx = len(symbols_rx_list)
        
        # Validate input and get maximum length
        max_symbols = max(len(s) for s in symbols_rx_list) if symbols_rx_list else 0
        
        if max_symbols == 0:
            raise ValueError("No symbols received from any antenna")
        
        # Normalize all symbol arrays to same length (CRITICAL FIX)
        symbols_rx_list_normalized = []
        for antenna_idx, symbols in enumerate(symbols_rx_list):
            if len(symbols) < max_symbols:
                # Pad with zeros if shorter
                symbols_padded = np.pad(symbols, (0, max_symbols - len(symbols)), 'constant')
            else:
                # Truncate if longer
                symbols_padded = symbols[:max_symbols]
            symbols_rx_list_normalized.append(symbols_padded)
        
        num_total_data_symbols = max_symbols
        
        # Get data indices for ONE OFDM symbol
        data_indices = LTEResourceGrid(self.config.N, self.config.Nc).get_data_indices()
        num_data_per_symbol = len(data_indices)
        
        # Calculate number of OFDM symbols
        num_ofdm_symbols = num_total_data_symbols // num_data_per_symbol
        remainder = num_total_data_symbols % num_data_per_symbol
        if remainder > 0:
            num_ofdm_symbols += 1
        
        # Initialize MRC accumulators
        # Numerator: Σ conj(H_i) * Y_i
        mrc_numerator = np.zeros(num_total_data_symbols, dtype=complex)
        # Denominator: Σ |H_i|²
        mrc_denominator = np.zeros(num_total_data_symbols, dtype=float)
        
        # MRC combining: process each RX antenna
        for antenna_idx in range(num_rx):
            Y_antenna = symbols_rx_list_normalized[antenna_idx]  # Normalized symbols from this antenna
            H_list = h_estimates_per_antenna[antenna_idx]  # Channel estimates per OFDM symbol
            
            # Process each OFDM symbol and its data
            data_idx = 0  # Index into symbols_data (resets for each antenna, which is correct)
            for sym_idx in range(num_ofdm_symbols):
                # Get channel estimate for this OFDM symbol and antenna
                if sym_idx < len(H_list):
                    H_full = H_list[sym_idx]  # Channel for all N subcarriers
                else:
                    # Use last available estimate if we run out
                    H_full = H_list[-1] if len(H_list) > 0 else np.ones(self.config.N, dtype=complex)
                
                # Ensure H_full has correct length
                if len(H_full) < self.config.N:
                    H_full = np.pad(H_full, (0, self.config.N - len(H_full)), 'constant')
                
                # Extract channel at data positions for this symbol
                H_data = H_full[data_indices]
                
                # Ensure H_data matches expected length
                if len(H_data) < num_data_per_symbol:
                    H_data = np.pad(H_data, (0, num_data_per_symbol - len(H_data)), 'constant')
                
                # Number of data symbols in this OFDM symbol (may be less for last symbol)
                num_data_this_symbol = min(num_data_per_symbol, num_total_data_symbols - data_idx)
                
                # Accumulate MRC numerator and denominator for this symbol's data
                for local_k in range(num_data_this_symbol):
                    global_k = data_idx + local_k
                    
                    # Ensure indices are valid
                    if global_k < len(Y_antenna) and global_k < len(mrc_numerator):
                        y_k = Y_antenna[global_k]
                        h_k = H_data[local_k] if local_k < len(H_data) else 1.0 + 0j
                        
                        # Accumulate numerator: += conj(H) * Y
                        mrc_numerator[global_k] += np.conj(h_k) * y_k
                        
                        # Accumulate denominator: += |H|²
                        mrc_denominator[global_k] += np.abs(h_k) ** 2
                
                data_idx += num_data_this_symbol
        
        # Final MRC combination: Y_combined = numerator / denominator
        # Add regularization to prevent division by zero
        # This is the CORRECT MRC formula per LTE standard
        symbols_combined = mrc_numerator / (mrc_denominator + regularization)
        
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
        signals_rx = self.channels[0].transmit_simo(
            signal_tx, num_rx=num_rx
        )
        
        # Step 4: Per-antenna demodulation WITH CRS-based channel estimation
        # This uses pilot symbols for realistic channel estimation
        # Can be done in parallel for efficiency
        if parallel and num_rx > 1:
            # Parallel processing with ThreadPoolExecutor
            symbols_rx_list = [None] * num_rx
            h_estimates_per_antenna = [None] * num_rx
            
            with ThreadPoolExecutor(max_workers=num_rx) as executor:
                futures = {}
                for i, signal_rx in enumerate(signals_rx):
                    # Demodulate with CRS-based channel estimation
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
        
        print(f"[MIMO] Transmitting through 2x{num_rx} channel (SNR={snr_db} dB)...")
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
    
    def simulate_beamforming(self, 
                            bits: np.ndarray, 
                            snr_db: float = 10.0, 
                            num_tx: int = 2,
                            num_rx: int = 1,
                            codebook_type: str = 'TM6',
                            velocity_kmh: float = 3.0,
                            update_mode: str = 'adaptive') -> Dict:
        """
        Simulate downlink transmission with beamforming precoding.
        
        Uses LTE codebook-based precoding with CSI feedback simulation.
        
        Parameters:
        -----------
        bits : np.ndarray
            Data bits to transmit
        snr_db : float
            Signal-to-noise ratio in dB
        num_tx : int
            Number of TX antennas (2, 4, 8)
        num_rx : int
            Number of RX antennas (1, 2, 4, 8)
        codebook_type : str
            'TM6' (rank-1) or 'TM4' (rank-1/2)
        velocity_kmh : float
            UE velocity in km/h (for adaptive precoder update)
        update_mode : str
            'adaptive' (based on coherence time) or 'static' (update once)
        
        Returns:
        --------
        dict : Simulation results with BER, beamforming gain, etc.
        """
        from core.beamforming_precoder import BeamformingPrecoder, AdaptiveBeamforming
        from core.csi_feedback import CSIFeedback
        
        print(f"\n{'='*70}")
        print(f"[BEAMFORMING] Simulation Start")
        print(f"{'='*70}")
        print(f"  Configuration: {num_tx}x{num_rx} ({num_tx} TX, {num_rx} RX)")
        print(f"  Codebook: {codebook_type}")
        print(f"  Velocity: {velocity_kmh} km/h")
        print(f"  SNR: {snr_db} dB")
        
        # Store original bits count
        original_num_bits = len(bits)
        
        # Initialize components
        csi_feedback = CSIFeedback(num_tx, num_rx, codebook_type=codebook_type)
        
        if update_mode == 'adaptive':
            precoder = AdaptiveBeamforming(
                num_tx=num_tx,
                velocity_kmh=velocity_kmh,
                frequency_ghz=2.0,  # Asume 2 GHz
                num_layers=1
            )
        else:
            precoder = BeamformingPrecoder(num_tx=num_tx, num_layers=1, precoder_type='MRT')
        
        # Create resource mapper
        from core.resource_mapper import ResourceMapper
        from core.modulator import QAMModulator
        resource_mapper = ResourceMapper(self.config)
        qam_modulator = QAMModulator(self.config.modulation)
        
        # Modulate bits
        qam_symbols = qam_modulator.bits_to_symbols(bits)
        print(f"  Data bits: {len(bits):,}")
        print(f"  QAM symbols: {len(qam_symbols):,}")
        
        # Calculate capacity and padding
        bits_per_symbol = int(np.log2(len(qam_modulator.constellation)))
        num_data_subcarriers = len(resource_mapper.get_data_indices())
        bits_per_ofdm = num_data_subcarriers * bits_per_symbol
        num_ofdm_symbols = int(np.ceil(len(bits) / bits_per_ofdm))
        
        print(f"  Bits per OFDM symbol: {bits_per_ofdm}")
        print(f"  Number of OFDM symbols: {num_ofdm_symbols}")
        
        # Pad bits if needed
        total_bits_needed = num_ofdm_symbols * bits_per_ofdm
        if len(bits) < total_bits_needed:
            bits_padded = np.concatenate([bits, np.zeros(total_bits_needed - len(bits), dtype=int)])
            qam_symbols = qam_modulator.bits_to_symbols(bits_padded)
        
        # Generate channel matrix
        channel_matrix = (np.random.randn(num_rx, num_tx) + 
                         1j * np.random.randn(num_rx, num_tx)) / np.sqrt(2)
        
        print(f"  Channel matrix shape: {channel_matrix.shape}")
        
        # Initialize arrays
        all_tx_signals = [[] for _ in range(num_tx)]
        all_rx_signals = []
        beamforming_gains = []
        pmi_history = []
        
        # Process OFDM symbols
        symbol_start_idx = 0
        for ofdm_idx in range(num_ofdm_symbols):
            # Get data symbols for this OFDM symbol
            symbol_end_idx = symbol_start_idx + num_data_subcarriers
            data_symbols = qam_symbols[symbol_start_idx:symbol_end_idx]
            
            # CSI Feedback: RX calcula PMI
            feedback = csi_feedback.generate_feedback(channel_matrix, noise_variance=1.0)
            pmi = feedback['pmi']
            W_precoder = feedback['precoder']
            pmi_history.append(pmi)
            
            # Apply beamforming precoding
            if update_mode == 'adaptive':
                # Adaptive: actualiza según coherence time
                precoder.update_precoder(channel_matrix, method='MRT')
                W_precoder = precoder.get_current_precoder()
            
            # Precoding: tx_signals = W @ data_symbols
            tx_signals_precoded = precoder.apply_precoding(data_symbols, W_precoder)
            
            # Calculate beamforming gain
            bf_gain = precoder.calculate_beamforming_gain(channel_matrix)
            beamforming_gains.append(bf_gain)
            
            # Store TX signals
            for tx_idx in range(num_tx):
                all_tx_signals[tx_idx].append(tx_signals_precoded[tx_idx, :])
            
            # Transmit through channel: rx = H @ tx + noise
            # rx_signal shape: [num_rx, num_data_subcarriers]
            rx_signal = np.zeros((num_rx, num_data_subcarriers), dtype=complex)
            for rx_idx in range(num_rx):
                for tx_idx in range(num_tx):
                    rx_signal[rx_idx, :] += channel_matrix[rx_idx, tx_idx] * tx_signals_precoded[tx_idx, :]
            
            # Add noise per RX antenna
            noise_variance = 10 ** (-snr_db / 10)
            noise = (np.random.randn(num_rx, num_data_subcarriers) + 
                    1j * np.random.randn(num_rx, num_data_subcarriers)) * np.sqrt(noise_variance / 2)
            rx_signal_noisy = rx_signal + noise
            
            all_rx_signals.append(rx_signal_noisy)
            
            symbol_start_idx = symbol_end_idx
        
        # Concatenate all RX signals: [num_ofdm_symbols, num_rx, num_data_subcarriers]
        rx_symbols_all = np.array(all_rx_signals)  # Shape: [num_ofdm, num_rx, num_data]
        
        # Equalization con MRC (Maximum Ratio Combining) para múltiples RX
        # H_eff = H @ W: [num_rx, num_tx] @ [num_tx, 1] = [num_rx, 1]
        H_eff = channel_matrix @ W_precoder  # [num_rx, 1]
        
        # MRC: combinar señales RX ponderadas por canal conjugado
        # s_est = (H^H @ y) / (H^H @ H) para cada subcarrier
        rx_symbols_equalized = []
        for ofdm_idx in range(num_ofdm_symbols):
            rx_per_ofdm = rx_symbols_all[ofdm_idx]  # [num_rx, num_data]
            
            # MRC combining: suma ponderada por H_eff conjugado
            combined = np.zeros(num_data_subcarriers, dtype=complex)
            for rx_idx in range(num_rx):
                combined += np.conj(H_eff[rx_idx, 0]) * rx_per_ofdm[rx_idx, :]
            
            # Normalización por potencia del canal
            power_norm = np.sum(np.abs(H_eff)**2)
            combined = combined / power_norm
            
            rx_symbols_equalized.append(combined)
        
        rx_symbols_equalized = np.concatenate(rx_symbols_equalized)
        
        # Demodulate
        bits_rx = qam_modulator.symbols_to_bits(rx_symbols_equalized)
        
        # Trim to original length
        bits_rx = bits_rx[:original_num_bits]
        bits_tx = bits[:original_num_bits]
        
        # Calculate BER
        bit_errors = np.sum(bits_tx != bits_rx)
        ber = bit_errors / original_num_bits
        
        # Statistics
        avg_bf_gain = np.mean(beamforming_gains)
        unique_pmis = len(set(pmi_history))
        
        print(f"\n{'='*70}")
        print(f"[BEAMFORMING] Results")
        print(f"{'='*70}")
        print(f"  BER: {ber:.4e}")
        print(f"  Bit errors: {bit_errors:,} / {original_num_bits:,}")
        print(f"  Avg Beamforming Gain: {avg_bf_gain:.2f} dB")
        print(f"  PMIs used: {unique_pmis} / {csi_feedback.codebook.codebook_size}")
        print(f"{'='*70}\n")
        
        # Results
        results = {
            'transmitted_bits': int(original_num_bits),
            'received_bits': int(original_num_bits),
            'bits_received_array': bits_rx,
            'bit_errors': int(bit_errors),
            'errors': int(bit_errors),
            'ber': float(ber),
            'snr_db': float(snr_db),
            'num_tx': num_tx,
            'num_rx': num_rx,
            'mode': 'Beamforming',
            'codebook_type': codebook_type,
            'beamforming_gain_db': float(avg_bf_gain),
            'channel_matrix': channel_matrix,
            'pmi_history': pmi_history,
            'unique_pmis': unique_pmis,
            'velocity_kmh': velocity_kmh,
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


def simulate_spatial_multiplexing(
    bits,
    num_tx=4,
    num_rx=2,
    rank='adaptive',
    detector_type='MMSE',
    modulation='64-QAM',
    snr_db=15,
    config=None,
    channel_type='awgn',
    itu_profile='Pedestrian_A',
    velocity_kmh=3,
    frequency_ghz=2.0,
    enable_csi_feedback=True,
    coherence_time_symbols=None,
    enable_parallel=False,
    codebook_type='TM4'
):
    """
    Simula transmisiÃ³n con Spatial Multiplexing (TM4) - VERSIÃ“N CORREGIDA
    
    Sigue el mismo patrÃ³n que simulate_mimo() usando:
    - ResourceMapper para crear grids OFDM con CRS
    - transmit_spatial_multiplexing() para canal MIMO
    - Perfect CSI (matriz H real) para detecciÃ³n
    
    Args:
        bits: Array de bits a transmitir
        num_tx: NÃºmero de antenas TX (2, 4, 8)
        num_rx: NÃºmero de antenas RX (>= rank)
        rank: NÃºmero de capas espaciales o 'adaptive'
        detector_type: 'MMSE', 'ZF', 'SIC', 'MRC'
        modulation: '64-QAM', '16-QAM', 'QPSK'
        snr_db: SNR en dB
        config: LTEConfig object
        channel_type: 'awgn' o 'rayleigh_mp'
        enable_csi_feedback: Si True, calcula RI/PMI Ã³ptimos
    
    Returns:
        dict con BER, bits recibidos, matriz H, etc.
    """
    from core.layer_mapper import LayerMapper
    from core.rank_adaptation import RankAdaptation
    from core.mimo_detector import MIMODetector
    from core.codebook_lte import LTECodebook
    from core.modulator import QAMModulator
    from core.resource_mapper import ResourceMapper
    
    print(f"\n{'='*70}")
    print(f"[SPATIAL MULTIPLEXING] TM4 Simulation")
    print(f"{'='*70}")
    print(f"  Config: {num_tx}x{num_rx} MIMO, Rank: {rank}, Detector: {detector_type}")
    print(f"  Modulation: {modulation}, SNR: {snr_db} dB")
    print(f"{'='*70}\n")
    
    # ConfiguraciÃ³n
    if config is None:
        from config import LTEConfig
        config = LTEConfig(modulation=modulation)
    
    original_num_bits = len(bits)
    
    # Componentes
    qam_modulator = QAMModulator(modulation)
    resource_mapper = ResourceMapper(config)
    data_indices = resource_mapper.get_data_indices()
    
    # Calcular capacidad
    bits_per_symbol = int(np.log2(len(qam_modulator.constellation)))
    num_data_subcarriers = len(data_indices)
    bits_per_ofdm = num_data_subcarriers * bits_per_symbol
    num_ofdm_symbols = int(np.ceil(original_num_bits / bits_per_ofdm))
    
    print(f"[1/7] QAM Modulation:")
    print(f"  Bits per OFDM symbol: {bits_per_ofdm}")
    print(f"  Number of OFDM symbols: {num_ofdm_symbols}")
    
    # Pad bits
    bits_padded = bits.copy()
    if len(bits_padded) < num_ofdm_symbols * bits_per_ofdm:
        bits_padded = np.pad(bits_padded, 
                            (0, num_ofdm_symbols * bits_per_ofdm - len(bits_padded)), 
                            'constant')
    
    # Generar canal MIMO inicial para rank adaptation
    H_initial = (np.random.randn(num_rx, num_tx) + 1j * np.random.randn(num_rx, num_tx)) / np.sqrt(2 * num_tx)
    
    # Decidir rank y PMI
    if rank == 'adaptive' and enable_csi_feedback:
        rank_adapter = RankAdaptation(num_tx, num_rx, snr_db=snr_db)
        feedback = rank_adapter.get_feedback(H_initial)
        rank_used = feedback['ri']
        pmi_used = feedback['pmi']
        W_precoder = feedback['W']
        print(f"[2/7] Rank Adaptation: RI={rank_used}, PMI={pmi_used}")
    else:
        rank_used = int(rank) if rank != 'adaptive' else min(num_tx, num_rx)
        codebook = LTECodebook(num_tx, transmission_mode='TM4', rank=rank_used)
        pmi_used = 0
        W_precoder = codebook.get_precoder(pmi_used)
        print(f"[2/7] Fixed Rank: RI={rank_used}, PMI={pmi_used}")
    
    # Layer mapper
    layer_mapper = LayerMapper(num_layers=rank_used)
    
    print(f"[3/7] Layer Mapping: {rank_used} spatial layers")
    
    # Preparar grids por sÃ­mbolo OFDM
    all_grids_tx = [[] for _ in range(num_tx)]
    all_signals_tx = [[] for _ in range(num_tx)]
    all_bits_chunks = []
    
    for ofdm_idx in range(num_ofdm_symbols):
        # Extraer bits para este sÃ­mbolo OFDM
        start_bit = ofdm_idx * bits_per_ofdm
        end_bit = (ofdm_idx + 1) * bits_per_ofdm
        bits_chunk = bits_padded[start_bit:end_bit]
        all_bits_chunks.append(bits_chunk)
        
        # QAM modulation
        qam_symbols = qam_modulator.bits_to_symbols(bits_chunk)
        
        # Layer mapping
        padded_length = layer_mapper.get_padded_length(len(qam_symbols))
        if padded_length > len(qam_symbols):
            qam_symbols = np.concatenate([qam_symbols, np.zeros(padded_length - len(qam_symbols), dtype=complex)])
        
        layers = layer_mapper.map_to_layers(qam_symbols)  # [rank_used, symbols_per_layer]
        
        # ===== ARQUITECTURA CORRECTA PARA TM4 SPATIAL MULTIPLEXING =====
        # 1. Mapear cada layer a grid (obtener índices de datos)
        # 2. Aplicar precoding SOLO en posiciones de datos
        # 3. Insertar pilotos CRS ortogonales por TX (sin precoding)
        
        # Obtener índices de recursos
        data_indices = resource_mapper.get_data_indices()
        
        # Crear grids TX inicializados (uno por antena física)
        grids_tx_ofdm = [np.zeros(config.N, dtype=complex) for _ in range(num_tx)]
        
        # Aplicar precoding por subportadora solo en posiciones de datos
        for data_idx, sc_idx in enumerate(data_indices):
            if data_idx < layers.shape[1]:  # Verificar que tenemos datos
                # Extraer símbolos de todas las layers en esta subportadora
                layers_k = layers[:, data_idx]  # [rank_used]
                
                # Precoding: x[k] = W @ layers[k]
                x_k = W_precoder @ layers_k  # [num_tx]
                
                # Asignar a cada antena TX
                for tx_idx in range(num_tx):
                    grids_tx_ofdm[tx_idx][sc_idx] = x_k[tx_idx]
        
        # Insertar pilotos CRS ortogonales usando MIMOChannelEstimatorPeriodic
        # Crear estimador temporal para obtener índices de pilotos ortogonales
        temp_estimator = MIMOChannelEstimatorPeriodic(
            config=config,
            num_tx=num_tx,
            num_rx=1  # No importa para obtener índices
        )
        pilot_indices_per_tx = temp_estimator.get_orthogonal_pilot_indices()
        
        for tx_idx in range(num_tx):
            pilot_indices_tx = pilot_indices_per_tx[tx_idx]
            pilots_tx = temp_estimator.pilot_patterns[tx_idx].generate_pilots(len(pilot_indices_tx))
            grids_tx_ofdm[tx_idx][pilot_indices_tx] = pilots_tx
        
        # Guardar grids y generar señales temporales
        for tx_idx in range(num_tx):
            all_grids_tx[tx_idx].append(grids_tx_ofdm[tx_idx])
            
            # IFFT + CP
            time_tx = np.fft.ifft(grids_tx_ofdm[tx_idx]) * np.sqrt(config.N)
            signal_tx = np.concatenate([time_tx[-config.cp_length:], time_tx])
            all_signals_tx[tx_idx].append(signal_tx)
    
    print(f"[4/7] Precoding: W[{num_tx}x{rank_used}] applied to {num_ofdm_symbols} OFDM symbols")
    
    # Concatenar seÃ±ales TX de todos los sÃ­mbolos
    signals_tx_full = [np.concatenate(all_signals_tx[tx_idx]) for tx_idx in range(num_tx)]
    
    # Transmitir a travÃ©s de canal MIMO con Spatial Multiplexing
    print(f"[5/7] Transmitting through MIMO channel (SNR={snr_db} dB)...")
    
    # Configurar canal
    from core.channel import ChannelSimulator
    fs = config.fs if hasattr(config, 'fs') else 15.36e6
    channel_sim = ChannelSimulator(
        channel_type=channel_type,
        snr_db=snr_db,
        fs=fs,
        itu_profile=itu_profile,
        frequency_ghz=frequency_ghz,
        velocity_kmh=velocity_kmh,
        verbose=False
    )
    
    # Transmitir con spatial multiplexing
    signals_rx, H_channel = channel_sim.transmit_spatial_multiplexing(signals_tx_full, num_rx=num_rx)
    
    print(f"  Channel matrix H shape: {H_channel.shape}")
    print(f"  H matrix norm: {np.linalg.norm(H_channel):.3f}")
    
    # Demodular cada RX antenna
    print(f"[6/7] Demodulating {num_rx} RX antennas...")
    
    all_grids_rx_per_antenna = []
    
    for rx_idx in range(num_rx):
        signal_rx = signals_rx[rx_idx]
        
        # Demodular OFDM symbols
        grids_rx = []
        signal_ptr = 0
        symbol_length = config.N + config.cp_length
        
        for _ in range(num_ofdm_symbols):
            if signal_ptr + symbol_length <= len(signal_rx):
                ofdm_symbol = signal_rx[signal_ptr:signal_ptr + symbol_length]
                
                # Remove CP
                ofdm_no_cp = ofdm_symbol[config.cp_length:]
                
                # FFT
                freq_symbol = np.fft.fft(ofdm_no_cp) / np.sqrt(config.N)
                grids_rx.append(freq_symbol)
                
                signal_ptr += symbol_length
        
        all_grids_rx_per_antenna.append(grids_rx)
    
    # ===== FASE 3: MIMO Detection con CRS Estimation H[k] =====
    print(f"[7/7] MIMO Detection ({detector_type}) with CRS Estimation H[k]...")
    
    # Crear estimador CRS para estimar H[rx, tx, k] por subportadora
    crs_estimator = MIMOChannelEstimatorPeriodic(
        config=config,
        num_tx=num_tx,
        num_rx=num_rx
    )
    
    mimo_detector = MIMODetector(
        num_rx=num_rx,
        num_layers=rank_used,
        detector_type=detector_type,
        constellation=qam_modulator.get_constellation()
    )
    
    noise_variance = 10 ** (-snr_db / 10)
    
    # Decodificar cada símbolo OFDM
    all_bits_rx = []
    total_bit_errors = 0
    
    for ofdm_idx in range(min(num_ofdm_symbols, len(all_grids_rx_per_antenna[0]))):
        bits_chunk = all_bits_chunks[ofdm_idx]
        
        # Estimar H[rx, tx, k] de pilotos CRS
        # Concatenar grids RX de todas las antenas para este símbolo OFDM
        grids_rx_ofdm = np.array([all_grids_rx_per_antenna[rx_idx][ofdm_idx] 
                                   for rx_idx in range(num_rx)])  # [num_rx, N]
        
        # Estimación CRS: retorna H[num_rx, num_tx, N]
        H_est_freq, _ = crs_estimator.estimate_channel_from_grid(grids_rx_ofdm, return_full_freq=True)
        
        print(f"  OFDM symbol {ofdm_idx}: H_est shape = {H_est_freq.shape}")
        
        # Extraer señales recibidas en subportadoras de datos
        y_data = grids_rx_ofdm[:, data_indices[:num_data_subcarriers]]  # [num_rx, num_data]
        H_data = H_est_freq[:, :, data_indices[:num_data_subcarriers]]  # [num_rx, num_tx, num_data]
        
        # MIMO Detection usando el detector configurado (MMSE/IRC o SIC)
        layers_rx = mimo_detector.detect(
            y_received=y_data,
            H_channel=H_data,
            noise_variance=noise_variance,
            W_precoder=W_precoder
        )  # [rank_used, num_data]
        
        # Layer demapping
        symbols_rx = layer_mapper.demap_from_layers(layers_rx, original_length=num_data_subcarriers)
        
        # QAM demodulation
        bits_rx_chunk = qam_modulator.symbols_to_bits(symbols_rx[:num_data_subcarriers])
        all_bits_rx.append(bits_rx_chunk[:bits_per_ofdm])
        
        # Calculate errors
        total_bit_errors += np.sum(bits_chunk[:bits_per_ofdm] != bits_rx_chunk[:bits_per_ofdm])
    
    # Concatenar bits recibidos
    bits_rx = np.concatenate(all_bits_rx)[:original_num_bits]
    
    # BER final
    bit_errors = np.sum(bits[:original_num_bits] != bits_rx)
    ber = bit_errors / original_num_bits if original_num_bits > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"  RESULTS:")
    print(f"{'='*70}")
    print(f"  BER: {ber:.4e}")
    print(f"  Bit errors: {bit_errors}/{original_num_bits}")
    print(f"  Rank used: {rank_used}")
    print(f"  Detector: {detector_type}")
    print(f"{'='*70}\n")
    
    results = {
        'transmitted_bits': int(original_num_bits),
        'received_bits': int(original_num_bits),
        'bits_received_array': bits_rx,
        'bit_errors': int(bit_errors),
        'errors': int(bit_errors),
        'ber': float(ber),
        'snr_db': float(snr_db),
        'num_tx': num_tx,
        'num_rx': num_rx,
        'rank': rank_used,
        'detector_type': detector_type,
        'mode': 'Spatial Multiplexing TM4',
        'codebook_type': codebook_type,
        'channel_matrix': H_channel,
        'precoder_matrix': W_precoder,
        'pmi_used': pmi_used,
        'velocity_kmh': velocity_kmh,
        'modulation': modulation,
    }
    
    return results
