"""
Extensión de LTE Channel Estimator para MIMO 2x2
Usa los pilotos existentes para estimar canales individuales H0 y H1
"""
import numpy as np
from core.lte_receiver import LTEChannelEstimator

class MIMOChannelEstimator:
    """
    Estimador de canal MIMO que usa LTEChannelEstimator para cada enlace TX->RX
    
    Para SFBC Alamouti (2 TX):
    - Usa pilotos ortogonales para cada antena TX
    - Estima H[rx, tx] para cada combinación
    - Retorna canales por subportadora para decodificación correcta
    """
    
    def __init__(self, config, num_tx=2, num_rx=1):
        """
        Inicializa estimador MIMO
        
        Args:
            config: LTEConfig
            num_tx: Número de antenas TX (fijo en 2 para Alamouti)
            num_rx: Número de antenas RX
        """
        self.config = config
        self.num_tx = num_tx
        self.num_rx = num_rx
        
        # Crear un estimador por cada TX (usan cell_id diferente para pilotos ortogonales)
        self.estimators = []
        for tx_idx in range(num_tx):
            # cell_id diferente = pilotos ortogonales
            estimator = LTEChannelEstimator(config, cell_id=tx_idx)
            self.estimators.append(estimator)
    
    def estimate_mimo_channel(self, received_grids: list, 
                             transmitted_pilots: list = None) -> dict:
        """
        Estima matriz de canal MIMO H[num_rx, num_tx, num_subcarriers]
        
        En LTE TM2 (SFBC), los pilotos de cada TX son ortogonales en tiempo/frecuencia:
        - TX0 usa subportadoras pares de pilotos
        - TX1 usa subportadoras impares de pilotos
        
        Args:
            received_grids: Lista de grids recibidos, uno por RX antenna
                          Cada grid es np.ndarray de tamaño N (FFT bins)
            transmitted_pilots: Pilotos transmitidos (opcional)
            
        Returns:
            dict con:
                - 'channel_matrix': H[rx_idx, tx_idx, subcarrier]
                - 'channel_matrix_data': Solo subportadoras de datos
                - 'data_indices': Índices de subportadoras de datos
                - 'snr_db': SNR estimado
        """
        N = self.config.N
        
        # Matriz de canal: H[num_rx, num_tx, num_subcarriers]
        H_full = np.zeros((self.num_rx, self.num_tx, N), dtype=complex)
        snr_list = []
        
        for rx_idx in range(self.num_rx):
            grid_rx = received_grids[rx_idx]
            
            for tx_idx in range(self.num_tx):
                # Estimar canal TX[tx_idx] -> RX[rx_idx]
                # Usando el estimador con pilotos específicos de ese TX
                ch_info = self.estimators[tx_idx].estimate_channel(grid_rx)
                
                # Guardar estimación completa (todas las subportadoras)
                H_full[rx_idx, tx_idx, :] = ch_info['channel_estimate']
                snr_list.append(ch_info['pilot_snr_db'])
        
        # Extraer solo subportadoras de datos
        data_indices = self.estimators[0].resource_grid.get_data_indices()
        H_data = H_full[:, :, data_indices]  # [num_rx, num_tx, num_data_subcarriers]
        
        avg_snr = np.mean(snr_list) if snr_list else 0
        
        return {
            'channel_matrix': H_full,  # Completo
            'channel_matrix_data': H_data,  # Solo datos
            'data_indices': data_indices,
            'snr_db': avg_snr,
            'shape': H_full.shape
        }
    
    def extract_channel_for_alamouti(self, channel_matrix_data: np.ndarray, 
                                    rx_idx: int = 0) -> tuple:
        """
        Extrae canales H0 y H1 para decodificación Alamouti
        
        Args:
            channel_matrix_data: Matriz H[num_rx, num_tx, num_data_sc]
            rx_idx: Índice de antena RX
            
        Returns:
            tuple: (H0, H1) donde cada uno es array[num_data_subcarriers]
                  H0[k] = canal de TX0 a RX en subportadora k
                  H1[k] = canal de TX1 a RX en subportadora k
        """
        H0 = channel_matrix_data[rx_idx, 0, :]  # RX[rx_idx] <- TX0
        H1 = channel_matrix_data[rx_idx, 1, :]  # RX[rx_idx] <- TX1
        
        return H0, H1


def estimate_mimo_channel_simple(config, received_grids: list, num_tx: int = 2):
    """
    Función helper para estimar canal MIMO rápidamente
    
    Args:
        config: LTEConfig
        received_grids: Lista de grids recibidos (uno por RX)
        num_tx: Número de TX antennas
        
    Returns:
        dict con estimaciones de canal
    """
    num_rx = len(received_grids)
    estimator = MIMOChannelEstimator(config, num_tx=num_tx, num_rx=num_rx)
    return estimator.estimate_mimo_channel(received_grids)
