"""
SFBC Alamouti Coding for OFDM - Space-Frequency Block Coding
=============================================================

Save this as: core/sfbc_alamouti.py

Implements Alamouti's Space-Frequency Block Coding scheme for OFDM
with 2 transmit antennas. Similar to LTE Transmission Mode 2.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class SFBCAlamouti:
    """
    SFBC Alamouti Encoder/Decoder for 2 TX antennas
    
    Encoding:
    ---------
    Input: [s0, s1, s2, s3, ...] (data symbols)
    
    For each pair (s0, s1):
        TX0: [s0, -conj(s1)]  → subcarriers [k, k+1]
        TX1: [s1,  conj(s0)]  → subcarriers [k, k+1]
    """
    
    def __init__(self, num_tx: int = 2, enabled: bool = True):
        """
        Initialize SFBC Alamouti coder
        
        Parameters:
        -----------
        num_tx : int
            Number of TX antennas (must be 2 for Alamouti)
        enabled : bool
            Enable/disable SFBC coding
        """
        if num_tx != 2:
            raise ValueError("Alamouti SFBC requires exactly 2 TX antennas")
        
        self.num_tx = num_tx
        self.enabled = enabled
    
    def encode(self, symbols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode symbols using Alamouti SFBC
        
        Parameters:
        -----------
        symbols : np.ndarray
            Input data symbols, shape (N,) where N is even
        
        Returns:
        --------
        tuple : (tx0_symbols, tx1_symbols)
        """
        if not self.enabled:
            return symbols.copy(), symbols.copy()
        
        N = len(symbols)
        if N % 2 != 0:
            raise ValueError(f"Number of symbols must be even for Alamouti coding, got {N}")
        
        tx0_symbols = np.zeros(N, dtype=complex)
        tx1_symbols = np.zeros(N, dtype=complex)
        
        for i in range(0, N, 2):
            s0 = symbols[i]
            s1 = symbols[i + 1]
            
            tx0_symbols[i] = s0
            tx1_symbols[i] = s1
            
            tx0_symbols[i + 1] = -np.conj(s1)
            tx1_symbols[i + 1] = np.conj(s0)
        
        return tx0_symbols, tx1_symbols
    
    def decode(self, 
               rx_symbols: np.ndarray,
               H0: np.ndarray,
               H1: np.ndarray,
               regularization: float = 1e-10) -> np.ndarray:
        """
        Decode received symbols using Alamouti combining (3GPP TS 36.211)
        
        Space-Frequency Block Coding (SFBC) with Alamouti scheme for 2 TX antennas.
        Uses per-subcarrier channel estimates for optimal combining.
        
        Parameters:
        -----------
        rx_symbols : np.ndarray
            Received symbols, shape (N,)
        H0 : np.ndarray
            Channel estimate for TX antenna 0, shape (N,)
        H1 : np.ndarray
            Channel estimate for TX antenna 1, shape (N,)
        regularization : float
            Small value to avoid division by zero (default: 1e-10)
        
        Returns:
        --------
        np.ndarray : Decoded data symbols, shape (N,)
        
        Notes:
        ------
        Alamouti transmission (subcarrier k, k+1):
          TX0: [  s0,  -conj(s1) ]
          TX1: [  s1,   conj(s0) ]
        
        Reception:
          r_k   = h0_k * s0 + h1_k * s1 + n_k
          r_k+1 = h0_k+1 * (-conj(s1)) + h1_k+1 * conj(s0) + n_k+1
        
        Optimal combining (MRC-like):
          s0 = [conj(h0_k)*r_k + h1_k+1*conj(r_k+1)] / (|h0|² + |h1|²)
          s1 = [conj(h1_k)*r_k - h0_k+1*conj(r_k+1)] / (|h0|² + |h1|²)
        """
        if not self.enabled:
            return rx_symbols.copy()
        
        N = len(rx_symbols)
        if N % 2 != 0:
            raise ValueError(f"Number of RX symbols must be even, got {N}")
        
        if len(H0) != N or len(H1) != N:
            raise ValueError(f"Channel estimates must have length {N}")
        
        decoded_symbols = np.zeros(N, dtype=complex)
        
        for i in range(0, N, 2):
            r_k = rx_symbols[i]
            r_k1 = rx_symbols[i + 1]
            
            # Use per-subcarrier channel estimates (more accurate for frequency-selective channels)
            h0_k = H0[i]
            h1_k = H1[i]
            h0_k1 = H0[i + 1]
            h1_k1 = H1[i + 1]
            
            # Alamouti combining (optimal MRC combining)
            # s0 contribution from both received symbols:
            #   From r_k:   conj(h0_k) * r_k
            #   From r_k+1: h1_k+1 * conj(r_k+1)
            # s1 contribution from both received symbols:
            #   From r_k:   conj(h1_k) * r_k
            #   From r_k+1: -h0_k+1 * conj(r_k+1)
            
            s0_combined = np.conj(h0_k) * r_k + h1_k1 * np.conj(r_k1)
            s1_combined = np.conj(h1_k) * r_k - h0_k1 * np.conj(r_k1)
            
            # Normalization: divide by sum of squared channel magnitudes
            # For flat fading (h0_k ≈ h0_k+1, h1_k ≈ h1_k+1), this simplifies to |h0|² + |h1|²
            # For frequency-selective channels, use average for normalization
            h0_avg = (h0_k + h0_k1) / 2
            h1_avg = (h1_k + h1_k1) / 2
            norm = np.abs(h0_avg)**2 + np.abs(h1_avg)**2 + regularization
            
            decoded_symbols[i] = s0_combined / norm
            decoded_symbols[i + 1] = s1_combined / norm
        
        return decoded_symbols
    
    def get_statistics(self) -> Dict:
        """Return SFBC statistics"""
        return {
            'enabled': self.enabled,
            'num_tx': self.num_tx,
            'coding_scheme': 'Alamouti SFBC',
            'rate': 1.0,
            'diversity_order': 2
        }


class SFBCResourceMapper:
    """
    Resource mapper for SFBC-OFDM
    Integrates with existing ResourceMapper
    """
    
    def __init__(self, resource_mapper):
        """
        Initialize SFBC resource mapper
        
        Parameters:
        -----------
        resource_mapper : ResourceMapper
            Existing resource mapper instance
        """
        self.resource_mapper = resource_mapper
        self.data_indices = resource_mapper.get_data_indices()
        self.num_data = len(self.data_indices)
        
        # Ensure even number
        if self.num_data % 2 != 0:
            print(f"[WARNING] Odd number of data subcarriers ({self.num_data}). "
                  f"Last subcarrier will be nulled for SFBC.")
            self.num_data -= 1
            self.data_indices = self.data_indices[:self.num_data]
    
    def prepare_data_for_sfbc(self, qam_symbols: np.ndarray) -> np.ndarray:
        """Prepare QAM symbols for SFBC encoding"""
        if len(qam_symbols) < self.num_data:
            qam_symbols = np.pad(qam_symbols, (0, self.num_data - len(qam_symbols)), 
                                'constant', constant_values=0)
        elif len(qam_symbols) > self.num_data:
            qam_symbols = qam_symbols[:self.num_data]
        
        return qam_symbols
    
    def map_sfbc_to_grid(self, 
                         tx0_symbols: np.ndarray, 
                         tx1_symbols: np.ndarray,
                         pilot_symbols: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map SFBC-encoded symbols to full OFDM grid
        
        Parameters:
        -----------
        tx0_symbols : np.ndarray
            Encoded symbols for TX antenna 0
        tx1_symbols : np.ndarray
            Encoded symbols for TX antenna 1
        pilot_symbols : np.ndarray, optional
            Pilot symbols
        
        Returns:
        --------
        tuple : (grid_tx0, grid_tx1)
        """
        N = self.resource_mapper.config.N
        
        grid_tx0 = np.zeros(N, dtype=complex)
        grid_tx1 = np.zeros(N, dtype=complex)
        
        # Map data
        grid_tx0[self.data_indices] = tx0_symbols[:self.num_data]
        grid_tx1[self.data_indices] = tx1_symbols[:self.num_data]
        
        # Map pilots (ORTHOGONAL PILOTS for MIMO - LTE TM2 style)
        # TX0 uses even pilot positions with cell_id=0 pilots
        # TX1 uses odd pilot positions with cell_id=1 pilots
        pilot_indices = self.resource_mapper.grid.get_pilot_indices()
        
        # Split pilot positions: even for TX0, odd for TX1
        pilot_indices_tx0 = pilot_indices[::2]   # Even positions (0, 2, 4, ...)
        pilot_indices_tx1 = pilot_indices[1::2]  # Odd positions (1, 3, 5, ...)
        
        # Generate pilots with different cell_id for orthogonality
        from core.lte_receiver import PilotPattern
        pilot_pattern_tx0 = PilotPattern(cell_id=0)
        pilot_pattern_tx1 = PilotPattern(cell_id=1)
        
        pilot_symbols_tx0 = pilot_pattern_tx0.generate_pilots(len(pilot_indices_tx0))
        pilot_symbols_tx1 = pilot_pattern_tx1.generate_pilots(len(pilot_indices_tx1))
        
        # TX0: pilots only at even positions
        grid_tx0[pilot_indices_tx0] = pilot_symbols_tx0
        
        # TX1: pilots only at odd positions
        grid_tx1[pilot_indices_tx1] = pilot_symbols_tx1
        
        return grid_tx0, grid_tx1
    
    def extract_data_from_grid(self, rx_grid: np.ndarray) -> np.ndarray:
        """Extract data subcarriers from received OFDM grid"""
        return rx_grid[self.data_indices]
    
    def apply_generic_precoding(self, symbols: np.ndarray, W_matrix: np.ndarray) -> List[np.ndarray]:
        """
        Método genérico para aplicar cualquier precoder W.
        Usado para beamforming, spatial multiplexing, etc.
        
        Este método NO interfiere con el SFBC Alamouti original.
        Es un método adicional para esquemas de precoding genéricos.
        
        Parameters:
        -----------
        symbols : np.ndarray
            Símbolos QAM de entrada, shape (num_data,) o (num_layers, num_data)
        W_matrix : np.ndarray
            Matriz de precoding, shape (num_tx, num_layers)
            donde num_tx es el número de antenas TX
        
        Returns:
        --------
        list : Lista de arrays, uno por antena TX
            Cada elemento tiene shape (num_data,)
        
        Example:
        --------
        Para beamforming con 4 TX antennas:
            W = [[w0], [w1], [w2], [w3]]  # shape (4, 1)
            symbols = [s0, s1, s2, ...]    # shape (N,)
            tx_signals = apply_generic_precoding(symbols, W)
            # tx_signals[0] = w0 * symbols
            # tx_signals[1] = w1 * symbols
            # etc.
        """
        # Asegurar que symbols es 2D: (num_layers, num_data)
        if symbols.ndim == 1:
            symbols = symbols.reshape(1, -1)
        
        num_layers, num_data = symbols.shape
        num_tx = W_matrix.shape[0]
        
        # Verificar dimensiones
        if W_matrix.shape[1] != num_layers:
            raise ValueError(
                f"W_matrix shape {W_matrix.shape} no compatible con "
                f"{num_layers} layers"
            )
        
        # Aplicar precoding: tx_signals = W @ symbols
        # tx_signals[i] = sum_l(W[i, l] * symbols[l, :])
        tx_signals = []
        for tx_idx in range(num_tx):
            # Señal para antena tx_idx
            tx_signal = np.zeros(num_data, dtype=complex)
            for layer_idx in range(num_layers):
                tx_signal += W_matrix[tx_idx, layer_idx] * symbols[layer_idx, :]
            tx_signals.append(tx_signal)
        
        return tx_signals