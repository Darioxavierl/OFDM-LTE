"""
MIMO Channel Estimator with Periodic Estimation

Extends LTE channel estimation to MIMO with orthogonal pilots.
Estimates H0 (TX0→RX) and H1 (TX1→RX) using periodic estimation like SISO/SIMO.

Author: AI Assistant
Date: 2026-01-04
"""

import numpy as np
from typing import List, Tuple, Dict
from core.lte_receiver import LTEChannelEstimator, PilotPattern, LTEResourceGrid


class MIMOChannelEstimatorPeriodic:
    """
    Estimador de canal periódico para MIMO 2x1 o 2xN (SFBC Alamouti)
    
    Estima H0 y H1 usando pilotos ortogonales:
    - TX0: pilotos en posiciones pares (cell_id=0)
    - TX1: pilotos en posiciones impares (cell_id=1)
    
    Implementa estimación periódica cada slot (como LTEReceiver) para:
    - Tracking temporal del canal
    - Reducción de varianza por promediado
    - Consistencia con SISO/SIMO
    """
    
    def __init__(self, config, slot_size: int = 14):
        """
        Inicializa estimador MIMO periódico
        
        Args:
            config: LTEConfig object
            slot_size: Número de símbolos OFDM por slot (default 14 como LTE)
        """
        self.config = config
        self.slot_size = slot_size
        
        # Estimadores para cada TX (con diferentes cell_id para pilotos ortogonales)
        self.estimator_tx0 = LTEChannelEstimator(config, cell_id=0)
        self.estimator_tx1 = LTEChannelEstimator(config, cell_id=1)
        
        # Patrones de pilotos ortogonales
        self.pilot_pattern_tx0 = PilotPattern(cell_id=0)
        self.pilot_pattern_tx1 = PilotPattern(cell_id=1)
        
        # Resource grid para obtener índices
        self.resource_grid = LTEResourceGrid(config.N, config.Nc)
    
    def get_orthogonal_pilot_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtiene índices de pilotos ortogonales para TX0 y TX1
        
        Returns:
            Tuple (pilot_indices_tx0, pilot_indices_tx1)
        """
        pilot_indices_all = self.resource_grid.get_pilot_indices()
        pilot_indices_tx0 = pilot_indices_all[::2]   # Pares: 0, 2, 4, ...
        pilot_indices_tx1 = pilot_indices_all[1::2]  # Impares: 1, 3, 5, ...
        
        return pilot_indices_tx0, pilot_indices_tx1
    
    def estimate_channel_from_grid(self, 
                                   grid_rx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Estima H0 y H1 de un grid recibido (frecuencia) usando pilotos ortogonales
        
        Args:
            grid_rx: Símbolos recibidos en dominio frecuencia (después de FFT)
                     Shape: (N,) donde N = número de subportadoras
        
        Returns:
            Tuple (H0_full, H1_full, info_dict):
                - H0_full: Estimación de canal TX0→RX para todas las subportadoras
                - H1_full: Estimación de canal TX1→RX para todas las subportadoras
                - info_dict: Información adicional (SNR, índices, etc.)
        """
        # Obtener índices de pilotos ortogonales
        pilot_indices_tx0, pilot_indices_tx1 = self.get_orthogonal_pilot_indices()
        
        # Generar pilotos conocidos
        pilots_tx0 = self.pilot_pattern_tx0.generate_pilots(len(pilot_indices_tx0))
        pilots_tx1 = self.pilot_pattern_tx1.generate_pilots(len(pilot_indices_tx1))
        
        # Extraer pilotos recibidos
        received_pilots_tx0 = grid_rx[pilot_indices_tx0]
        received_pilots_tx1 = grid_rx[pilot_indices_tx1]
        
        # Estimación LS en posiciones de pilotos
        H0_at_pilots = received_pilots_tx0 / pilots_tx0
        H1_at_pilots = received_pilots_tx1 / pilots_tx1
        
        # Interpolación a todas las subportadoras
        H0_full = self.estimator_tx0._interpolate_channel(
            pilot_indices_tx0, H0_at_pilots, self.config.N
        )
        H1_full = self.estimator_tx1._interpolate_channel(
            pilot_indices_tx1, H1_at_pilots, self.config.N
        )
        
        # Calcular SNR estimado de pilotos (opcional, para diagnóstico)
        noise_var_tx0 = np.var(H0_at_pilots - np.mean(H0_at_pilots))
        noise_var_tx1 = np.var(H1_at_pilots - np.mean(H1_at_pilots))
        signal_power = np.mean(np.abs(H0_at_pilots) ** 2 + np.abs(H1_at_pilots) ** 2) / 2
        noise_power = (noise_var_tx0 + noise_var_tx1) / 2
        pilot_snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        info = {
            'pilot_snr_db': pilot_snr_db,
            'pilot_indices_tx0': pilot_indices_tx0,
            'pilot_indices_tx1': pilot_indices_tx1,
            'num_pilots_tx0': len(pilot_indices_tx0),
            'num_pilots_tx1': len(pilot_indices_tx1)
        }
        
        return H0_full, H1_full, info
    
    def estimate_channel_periodic(self, 
                                  all_received_grids: List[np.ndarray]) -> Tuple[List[np.ndarray], 
                                                                                   List[np.ndarray], 
                                                                                   float]:
        """
        Estimación periódica de canal MIMO para múltiples símbolos OFDM
        
        Similar a LTEReceiver._estimate_channel_periodic pero para 2 TX.
        Estima canal cada slot (14 símbolos) y reutiliza para símbolos intermedios.
        
        Args:
            all_received_grids: Lista de grids recibidos (después de FFT)
                                Cada elemento es un array de shape (N,)
        
        Returns:
            Tuple (H0_estimates_per_symbol, H1_estimates_per_symbol, avg_snr_db):
                - H0_estimates_per_symbol: Lista con H0[k] para cada símbolo OFDM
                - H1_estimates_per_symbol: Lista con H1[k] para cada símbolo OFDM
                - avg_snr_db: SNR promedio estimado de los pilotos
        """
        num_symbols = len(all_received_grids)
        H0_estimates_per_symbol = []
        H1_estimates_per_symbol = []
        snr_list = []
        
        # Estimar canal cada slot (14 símbolos OFDM)
        for slot_start in range(0, num_symbols, self.slot_size):
            slot_end = min(slot_start + self.slot_size, num_symbols)
            slot_length = slot_end - slot_start
            
            # Estimar canal en el primer símbolo del slot
            if slot_start < num_symbols:
                H0_slot, H1_slot, info = self.estimate_channel_from_grid(
                    all_received_grids[slot_start]
                )
                snr_list.append(info['pilot_snr_db'])
                
                # Reutilizar esta estimación para todos los símbolos del slot
                for i in range(slot_length):
                    H0_estimates_per_symbol.append(H0_slot)
                    H1_estimates_per_symbol.append(H1_slot)
        
        # SNR promedio
        avg_snr_db = np.mean(snr_list) if snr_list else 0.0
        
        return H0_estimates_per_symbol, H1_estimates_per_symbol, avg_snr_db
    
    def demodulate_and_estimate_mimo(self, 
                                     signal_rx: np.ndarray,
                                     cp_length: int) -> Tuple[List[np.ndarray], 
                                                               List[np.ndarray], 
                                                               List[np.ndarray]]:
        """
        Demodula señal recibida y estima canales H0 y H1 periódicamente
        
        Procesa:
        1. Extrae símbolos OFDM (remueve CP + FFT)
        2. Estima H0 y H1 periódicamente cada slot
        3. Retorna grids recibidos y estimaciones de canal
        
        Args:
            signal_rx: Señal recibida en tiempo
            cp_length: Longitud del prefijo cíclico
        
        Returns:
            Tuple (all_grids_rx, H0_per_symbol, H1_per_symbol)
        """
        # Demodular todos los símbolos OFDM
        all_grids_rx = []
        
        symbol_length = self.config.N + cp_length
        num_symbols = len(signal_rx) // symbol_length
        
        for i in range(num_symbols):
            start_idx = i * symbol_length
            
            # Extraer símbolo OFDM (skip CP)
            ofdm_symbol = signal_rx[start_idx + cp_length:start_idx + symbol_length]
            
            # FFT para obtener dominio frecuencia
            grid_rx = np.fft.fft(ofdm_symbol) / np.sqrt(self.config.N)
            all_grids_rx.append(grid_rx)
        
        # Estimación periódica de canal
        H0_per_symbol, H1_per_symbol, _ = self.estimate_channel_periodic(all_grids_rx)
        
        return all_grids_rx, H0_per_symbol, H1_per_symbol
