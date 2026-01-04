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
               noise_var: float = 1e-6) -> np.ndarray:
        """
        Decode received symbols using Alamouti combining
        
        Parameters:
        -----------
        rx_symbols : np.ndarray
            Received symbols, shape (N,)
        H0 : np.ndarray
            Channel estimate for TX antenna 0, shape (N,)
        H1 : np.ndarray
            Channel estimate for TX antenna 1, shape (N,)
        noise_var : float
            Noise variance for normalization
        
        Returns:
        --------
        np.ndarray : Decoded data symbols, shape (N,)
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
            
            # For flat fading channel, h0_k ≈ h0_k1 and h1_k ≈ h1_k1
            # Use average of adjacent subcarriers
            h0 = (H0[i] + H0[i + 1]) / 2
            h1 = (H1[i] + H1[i + 1]) / 2
            
            # Alamouti combining (correct formula)
            # Received: r_k = h0*s0 + h1*s1
            #           r_k1 = -h0*conj(s1) + h1*conj(s0)
            # Decode:   s0 = conj(h0)*r_k + h1*conj(r_k1)
            #           s1 = conj(h1)*r_k - h0*conj(r_k1)
            s0_combined = np.conj(h0) * r_k + h1 * np.conj(r_k1)
            s1_combined = np.conj(h1) * r_k - h0 * np.conj(r_k1)
            
            # Normalization
            norm = np.abs(h0)**2 + np.abs(h1)**2 + noise_var
            
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