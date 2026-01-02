"""
DFT Precoding para SC-FDM (Single Carrier Frequency Division Multiplexing)

Implementa la precodificación con DFT que convierte OFDM en SC-FDM.
Esto reduce significativamente el PAPR (Peak-to-Average Power Ratio).

En SC-FDM:
1. Los símbolos de datos se precodifican con DFT de tamaño M
2. M es igual al número de subportadoras de datos útiles
3. La salida DFT se mapea a las subportadoras de datos
4. Pilotos y DC mantienen su mapeo original

Referencia: 3GPP LTE Uplink (TS 36.211)
"""

import numpy as np
from typing import Tuple, Dict


class DFTPrecodifier:
    """
    Implementa la precodificación DFT para SC-FDM
    
    La DFT de tamaño M se aplica a los símbolos de datos antes de mapearlos
    a las subportadoras. Esto produce una señal similar a single-carrier,
    reduciendo así el PAPR.
    """
    
    def __init__(self, M: int = None, enable: bool = True):
        """
        Inicializa el precodificador DFT
        
        Args:
            M: Tamaño de la DFT (debe ser igual al número de subportadoras de datos)
            enable: Si False, el precodificador devuelve los símbolos sin cambios
        """
        self.M = M
        self.enable = enable
        self._dft_matrix = None
        
        if M is not None and enable:
            self._compute_dft_matrix()
    
    def _compute_dft_matrix(self):
        """Precomputa la matriz DFT para eficiencia"""
        if self.M is None or not self.enable:
            return
        
        # Matriz DFT: W[k,n] = exp(-j*2*pi*k*n/M) / sqrt(M)
        # El factor 1/sqrt(M) normaliza la energía
        k = np.arange(self.M).reshape(-1, 1)
        n = np.arange(self.M).reshape(1, -1)
        
        self._dft_matrix = np.exp(-1j * 2 * np.pi * k * n / self.M) / np.sqrt(self.M)
    
    def set_size(self, M: int):
        """
        Configura el tamaño M de la DFT
        
        Args:
            M: Tamaño de la DFT (número de subportadoras de datos)
        """
        self.M = M
        if self.enable:
            self._compute_dft_matrix()
    
    def precoding(self, symbols: np.ndarray) -> np.ndarray:
        """
        Aplica precodificación DFT a los símbolos
        
        Args:
            symbols: Array de símbolos complejos (debe tener longitud M)
            
        Returns:
            Array de símbolos precodificados
        """
        if not self.enable or self.M is None:
            return symbols
        
        # Validar tamaño
        if len(symbols) != self.M:
            raise ValueError(
                f"Tamaño de símbolos ({len(symbols)}) debe ser igual a M ({self.M})"
            )
        
        # Aplicar DFT usando matriz precomputada
        if self._dft_matrix is not None:
            precoded = self._dft_matrix @ symbols
        else:
            # Fallback a numpy.fft si no hay matriz
            precoded = np.fft.fft(symbols) / np.sqrt(self.M)
        
        return precoded
    
    def precoding_ifft(self, symbols: np.ndarray) -> np.ndarray:
        """
        Aplica precodificación DFT usando IFFT (más rápido para M grande)
        
        Args:
            symbols: Array de símbolos complejos (debe tener longitud M)
            
        Returns:
            Array de símbolos precodificados
        """
        if not self.enable or self.M is None:
            return symbols
        
        # Validar tamaño
        if len(symbols) != self.M:
            raise ValueError(
                f"Tamaño de símbolos ({len(symbols)}) debe ser igual a M ({self.M})"
            )
        
        # DFT = IFFT conjugada
        # FFT: sum_{n=0}^{M-1} x[n] * exp(-j*2*pi*k*n/M)
        # IFFT: (1/M) * sum_{n=0}^{M-1} x[n] * exp(j*2*pi*k*n/M)
        # DFT = conj(IFFT(conj(x))) * M, pero aquí usamos dirección tradicional
        
        # Usar FFT con normalización 1/sqrt(M)
        precoded = np.fft.fft(symbols) / np.sqrt(self.M)
        
        return precoded
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas del precodificador"""
        return {
            'enabled': self.enable,
            'dft_size': self.M,
            'matrix_computed': self._dft_matrix is not None
        }


class IDFTDecodifier:
    """
    Implementa la decodificación IDFT inversa para SC-FDM
    
    Invierte el proceso de precodificación DFT en el receptor.
    Recupera los símbolos originales después de la demodulación FFT.
    """
    
    def __init__(self, M: int = None, enable: bool = True):
        """
        Inicializa el decodificador IDFT
        
        Args:
            M: Tamaño de la IDFT (debe ser igual al número de subportadoras de datos)
            enable: Si False, devuelve los símbolos sin cambios
        """
        self.M = M
        self.enable = enable
        self._idft_matrix = None
        
        if M is not None and enable:
            self._compute_idft_matrix()
    
    def _compute_idft_matrix(self):
        """Precomputa la matriz IDFT para eficiencia"""
        if self.M is None or not self.enable:
            return
        
        # Matriz IDFT (inversa de la DFT)
        # DFT normaliza por 1/sqrt(M): X[k] = (1/sqrt(M)) * sum x[n] * exp(-j*2*pi*k*n/M)
        # Para invertir exactamente: x[n] = (1/sqrt(M)) * sum X[k] * exp(+j*2*pi*k*n/M)
        # 
        # Esto mantiene la normalización unitaria:
        # Composición: x = (1/sqrt(M)) * exp(+j*...) * (1/sqrt(M)) * exp(-j*...) * M = I ✓
        # Porque: (1/M) * sum_k sum_n exp(j*...) * exp(-j*...) = (1/M) * M * M = M^2/M = M (hay M deltas)
        # Realmente: (1/M) * M * delta = 1 ✓
        
        k = np.arange(self.M).reshape(-1, 1)
        n = np.arange(self.M).reshape(1, -1)
        
        # IDFT con la misma normalización que DFT: 1/sqrt(M)
        # La transpuesta conjugada de exp(-j*2*pi*k*n/M) es exp(+j*2*pi*k*n/M)
        self._idft_matrix = np.exp(1j * 2 * np.pi * k * n / self.M) / np.sqrt(self.M)
    
    def set_size(self, M: int):
        """
        Configura el tamaño M de la IDFT
        
        Args:
            M: Tamaño de la IDFT (número de subportadoras de datos)
        """
        self.M = M
        if self.enable:
            self._compute_idft_matrix()
    
    def decoding(self, precoded_symbols: np.ndarray) -> np.ndarray:
        """
        Aplica decodificación IDFT a los símbolos precodificados
        
        Invierte la transformación DFT realizada en el transmisor.
        
        Args:
            precoded_symbols: Array de símbolos DFT-precodificados (longitud M)
            
        Returns:
            Array de símbolos originales recuperados
        """
        if not self.enable or self.M is None:
            return precoded_symbols
        
        # Validar tamaño
        if len(precoded_symbols) != self.M:
            raise ValueError(
                f"Tamaño de símbolos ({len(precoded_symbols)}) debe ser igual a M ({self.M})"
            )
        
        # Aplicar IDFT usando matriz precomputada
        if self._idft_matrix is not None:
            decoded = self._idft_matrix @ precoded_symbols
        else:
            # Fallback a numpy.ifft si no hay matriz
            decoded = np.fft.ifft(precoded_symbols) * np.sqrt(self.M)
        
        return decoded
    
    def decoding_fft(self, precoded_symbols: np.ndarray) -> np.ndarray:
        """
        Aplica decodificación IDFT usando IFFT (más rápido para M grande)
        
        Args:
            precoded_symbols: Array de símbolos DFT-precodificados (longitud M)
            
        Returns:
            Array de símbolos originales recuperados
        """
        if not self.enable or self.M is None:
            return precoded_symbols
        
        # Validar tamaño
        if len(precoded_symbols) != self.M:
            raise ValueError(
                f"Tamaño de símbolos ({len(precoded_symbols)}) debe ser igual a M ({self.M})"
            )
        
        # IDFT = IFFT escalada
        # IFFT: (1/M) * sum_{k=0}^{M-1} X[k] * exp(j*2*pi*k*n/M)
        # IDFT con normalización: (1/sqrt(M)) * sum_{k=0}^{M-1} X[k] * exp(j*2*pi*k*n/M)
        
        decoded = np.fft.ifft(precoded_symbols) * np.sqrt(self.M)
        
        return decoded
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas del decodificador"""
        return {
            'enabled': self.enable,
            'idft_size': self.M,
            'matrix_computed': self._idft_matrix is not None
        }


class SC_FDMPrecodifier:
    """
    Precodificador SC-FDM que encapsula la lógica completa de precodificación
    
    Combina:
    - Precodificación DFT
    - Mapeo a subportadoras de datos
    - Manejo de pilotos
    """
    
    def __init__(self, num_data_subcarriers: int, enable: bool = True):
        """
        Inicializa el precodificador SC-FDM
        
        Args:
            num_data_subcarriers: Número de subportadoras de datos (sin pilotos/DC/guardias)
            enable: Si True, aplica precodificación DFT
        """
        self.num_data_subcarriers = num_data_subcarriers
        self.enable = enable
        self.dft_precoder = DFTPrecodifier(M=num_data_subcarriers, enable=enable)
    
    def precoding(self, data_symbols: np.ndarray) -> np.ndarray:
        """
        Aplica precodificación SC-FDM a símbolos de datos
        
        Args:
            data_symbols: Array de símbolos para las subportadoras de datos
                         (longitud = número de subportadoras de datos)
            
        Returns:
            Array de símbolos precodificados (misma longitud)
        """
        return self.dft_precoder.precoding(data_symbols)
    
    def set_enable(self, enable: bool):
        """Activa/desactiva la precodificación"""
        self.enable = enable
        self.dft_precoder.enable = enable
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas"""
        return {
            'enabled': self.enable,
            'num_data_subcarriers': self.num_data_subcarriers,
            'dft_info': self.dft_precoder.get_statistics()
        }


class SC_FDMDecodifier:
    """
    Decodificador SC-FDM que encapsula la lógica completa de decodificación
    
    Combina:
    - Decodificación IDFT
    - Extracción de símbolos de datos
    - Manejo de pilotos
    """
    
    def __init__(self, num_data_subcarriers: int, enable: bool = True):
        """
        Inicializa el decodificador SC-FDM
        
        Args:
            num_data_subcarriers: Número de subportadoras de datos
            enable: Si True, aplica decodificación IDFT
        """
        self.num_data_subcarriers = num_data_subcarriers
        self.enable = enable
        self.idft_decoder = IDFTDecodifier(M=num_data_subcarriers, enable=enable)
    
    def decoding(self, precoded_symbols: np.ndarray) -> np.ndarray:
        """
        Aplica decodificación SC-FDM a símbolos precodificados
        
        Args:
            precoded_symbols: Array de símbolos precodificados (longitud M)
            
        Returns:
            Array de símbolos originales recuperados
        """
        return self.idft_decoder.decoding(precoded_symbols)
    
    def set_enable(self, enable: bool):
        """Activa/desactiva la decodificación"""
        self.enable = enable
        self.idft_decoder.enable = enable
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas"""
        return {
            'enabled': self.enable,
            'num_data_subcarriers': self.num_data_subcarriers,
            'idft_info': self.idft_decoder.get_statistics()
        }
