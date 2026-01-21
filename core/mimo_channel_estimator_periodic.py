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
    Estimador de canal periódico para MIMO con múltiples TX (2, 4, 8 antenas)
    
    Usa pilotos CRS ortogonales en frecuencia mediante diferentes cell_id:
    - TX0: cell_id=0 (shift 0)
    - TX1: cell_id=1 (shift 3 subportadoras)
    - TX2: cell_id=2 (shift 6 subportadoras) 
    - TX3: cell_id=3 (shift 9 subportadoras)
    
    Para 8 TX, se reutilizan cell_id con diferentes símbolos OFDM.
    
    Implementa estimación periódica cada slot para:
    - Tracking temporal del canal
    - Reducción de varianza por promediado
    - Soporte Spatial Multiplexing y SFBC
    """
    
    def __init__(self, config, num_tx: int = 2, num_rx: int = 2, slot_size: int = 14):
        """
        Inicializa estimador MIMO periódico
        
        Args:
            config: LTEConfig object
            num_tx: Número de antenas TX (2, 4, o 8)
            num_rx: Número de antenas RX
            slot_size: Número de símbolos OFDM por slot (default 14 como LTE)
        """
        self.config = config
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.slot_size = slot_size
        
        if num_tx not in [2, 4, 8]:
            raise ValueError(f"num_tx debe ser 2, 4 o 8, recibido: {num_tx}")
        
        # Crear estimador y patrón de pilotos para cada TX
        self.estimators = []
        self.pilot_patterns = []
        
        for tx_idx in range(num_tx):
            # Usar cell_id diferente para ortogonalizar pilotos en frecuencia
            # LTE soporta cell_id 0-503, pero para ortogonalización usamos 0-3
            cell_id = tx_idx % 4
            
            estimator = LTEChannelEstimator(config, cell_id=cell_id)
            pilot_pattern = PilotPattern(cell_id=cell_id)
            
            self.estimators.append(estimator)
            self.pilot_patterns.append(pilot_pattern)
        
        # Resource grid para obtener índices
        self.resource_grid = LTEResourceGrid(config.N, config.Nc)
        
        print(f"[MIMOChannelEstimatorPeriodic] Inicializado:")
        print(f"  TX antennas: {num_tx}")
        print(f"  RX antennas: {num_rx}")
        print(f"  Cell IDs: {[tx_idx % 4 for tx_idx in range(num_tx)]}")
    
    def get_orthogonal_pilot_indices(self) -> List[np.ndarray]:
        """
        Obtiene índices de pilotos ortogonales para cada TX
        
        Usa shift en frecuencia basado en cell_id para separar pilotos:
        - TX0 (cell_id=0): posiciones base
        - TX1 (cell_id=1): posiciones base + shift
        - TX2 (cell_id=2): posiciones base + 2*shift
        - TX3 (cell_id=3): posiciones base + 3*shift
        
        Returns:
            List de arrays con índices de pilotos por TX
        """
        pilot_indices_per_tx = []
        
        for tx_idx in range(self.num_tx):
            # Obtener índices según cell_id
            cell_id = tx_idx % 4
            pilot_pattern = self.pilot_patterns[tx_idx]
            
            # Obtener todos los pilotos para este cell_id
            pilot_indices_all = self.resource_grid.get_pilot_indices()
            
            # Para ortogonalizar, tomar cada N-ésimo piloto con offset
            # Esto simula el pattern CRS de LTE con diferentes antenna ports
            step = self.num_tx if self.num_tx <= 4 else 4
            offset = tx_idx % step
            pilot_indices = pilot_indices_all[offset::step]
            
            pilot_indices_per_tx.append(pilot_indices)
        
        return pilot_indices_per_tx
    
    def estimate_channel_from_grid(self, 
                                   grid_rx: np.ndarray,
                                   return_full_freq: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Estima canal H de un grid recibido usando pilotos ortogonales CRS
        
        Args:
            grid_rx: Símbolos recibidos en dominio frecuencia
                     - Single RX: shape (N,) donde N = subportadoras
                     - Multi RX: shape (num_rx, N)
            return_full_freq: Si True, retorna H[k] por subportadora
                             Si False, retorna H escalar promedio
        
        Returns:
            Tuple (H_estimated, info_dict):
                - H_estimated: 
                  * return_full_freq=True: [num_rx, num_tx, N] por subportadora
                  * return_full_freq=False: [num_rx, num_tx] promedio escalar
                - info_dict: Información adicional (SNR, índices, etc.)
        """
        # Manejar single vs multi RX
        if grid_rx.ndim == 1:
            # Single RX: expandir a (1, N)
            grids_rx = grid_rx.reshape(1, -1)
            num_rx_actual = 1
        else:
            # Multi RX: usar como está
            grids_rx = grid_rx
            num_rx_actual = grids_rx.shape[0]
        
        N = grids_rx.shape[1]  # Número de subportadoras
        
        # Obtener índices de pilotos por TX
        pilot_indices_per_tx = self.get_orthogonal_pilot_indices()
        
        # Inicializar resultado
        if return_full_freq:
            H_estimated = np.zeros((num_rx_actual, self.num_tx, N), dtype=complex)
        else:
            H_estimated = np.zeros((num_rx_actual, self.num_tx), dtype=complex)
        
        # Estimar canal para cada combinación RX-TX
        for rx_idx in range(num_rx_actual):
            for tx_idx in range(self.num_tx):
                # Obtener pilotos para este TX
                pilot_indices = pilot_indices_per_tx[tx_idx]
                pilot_pattern = self.pilot_patterns[tx_idx]
                estimator = self.estimators[tx_idx]
                
                # Generar pilotos conocidos
                pilots_tx = pilot_pattern.generate_pilots(len(pilot_indices))
                
                # Extraer pilotos recibidos en este RX
                received_pilots = grids_rx[rx_idx, pilot_indices]
                
                # Estimación LS en posiciones de pilotos
                H_at_pilots = received_pilots / pilots_tx
                
                if return_full_freq:
                    # Interpolar a todas las subportadoras
                    H_full = estimator._interpolate_channel(
                        pilot_indices, H_at_pilots, N
                    )
                    H_estimated[rx_idx, tx_idx, :] = H_full
                else:
                    # Promedio escalar
                    H_estimated[rx_idx, tx_idx] = np.mean(H_at_pilots)
        
        # Calcular información adicional
        info = {
            'num_pilots_per_tx': [len(pilot_indices_per_tx[i]) for i in range(self.num_tx)],
            'pilot_indices': pilot_indices_per_tx,
            'num_rx': num_rx_actual,
            'num_tx': self.num_tx,
            'N': N
        }
        
        return H_estimated, info
    
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
