"""
Beamforming Precoder for LTE
Implementa precodificación espacial para transmisión con múltiples antenas.

Modos soportados:
- MRT (Maximum Ratio Transmission): Beamforming coherente óptimo
- Codebook-based: Precoding cuantizado (TM4/TM6)
- ZF (Zero-Forcing): Para multi-usuario (futuro)

Compatible con 2, 4, 8 antenas TX (genérico).
"""

import numpy as np


class BeamformingPrecoder:
    """
    Precoder genérico para beamforming LTE.
    Soporta múltiples antenas TX y diferentes estrategias de precoding.
    """
    
    def __init__(self, num_tx, num_layers=1, precoder_type='MRT'):
        """
        Args:
            num_tx (int): Número de antenas TX (2, 4, 8)
            num_layers (int): Número de capas espaciales (1 o 2)
            precoder_type (str): 'MRT', 'codebook', 'ZF'
        """
        self.num_tx = num_tx
        self.num_layers = num_layers
        self.precoder_type = precoder_type
        
        # Precoder actual (se actualiza periódicamente)
        self.W = None
        
        print(f"[BeamformingPrecoder] Inicializado:")
        print(f"  Antenas TX: {num_tx}")
        print(f"  Capas espaciales: {num_layers}")
        print(f"  Tipo: {precoder_type}")
    
    def calculate_mrt_weights(self, H_channel):
        """
        Calcula pesos óptimos usando Maximum Ratio Transmission (MRT).
        W = H* / ||H||  (conjugado normalizado)
        
        Args:
            H_channel: Canal estimado [num_rx, num_tx] para un subcarrier
        
        Returns:
            W: Vector de pesos [num_tx, 1]
        """
        # Para MRT con single layer, usamos el conjugado del canal
        # Si hay múltiples RX, promediamos los canales
        if H_channel.ndim == 2:
            # Promediar sobre receptores
            h_avg = np.mean(H_channel, axis=0)  # [num_tx]
        else:
            h_avg = H_channel
        
        # Conjugado del canal
        h_conj = np.conj(h_avg)
        
        # Normalizar por potencia total
        W = h_conj / np.sqrt(np.sum(np.abs(h_conj)**2))
        
        return W.reshape(-1, 1)  # [num_tx, 1]
    
    def calculate_eigenbeamforming(self, H_channel):
        """
        Calcula eigenvector dominante (máxima capacidad).
        W = eigenvector de (H^H * H) con mayor eigenvalue
        
        Args:
            H_channel: Canal [num_rx, num_tx]
        
        Returns:
            W: Vector de beamforming [num_tx, 1]
        """
        # H^H * H
        HH = H_channel.conj().T @ H_channel
        
        # Eigenvalues y eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(HH)
        
        # Eigenvector con mayor eigenvalue
        idx_max = np.argmax(np.abs(eigenvalues))
        W = eigenvectors[:, idx_max]
        
        # Normalizar
        W = W / np.sqrt(np.sum(np.abs(W)**2))
        
        return W.reshape(-1, 1)
    
    def apply_precoding(self, symbols, W_matrix=None):
        """
        Aplica precoding a los símbolos QAM.
        x = W @ s
        
        Args:
            symbols: Símbolos de entrada [num_symbols] o [num_layers, num_symbols]
            W_matrix: Matriz de precoding [num_tx, num_layers]
                      Si None, usa self.W
        
        Returns:
            tx_symbols: Símbolos precodificados [num_tx, num_symbols]
        """
        if W_matrix is None:
            if self.W is None:
                raise ValueError("Precoder W no ha sido calculado. Llamar a update_precoder() primero.")
            W_matrix = self.W
        
        # Asegurar que symbols es 2D
        if symbols.ndim == 1:
            symbols = symbols.reshape(1, -1)  # [1, num_symbols]
        
        # Aplicar precoding: x = W @ s
        tx_symbols = W_matrix @ symbols  # [num_tx, num_symbols]
        
        # Normalización de potencia: mantener potencia total = potencia entrada
        # En lugar de dividir por sqrt(num_tx), normalizamos por la norma de W
        # Esto asegura que ||x||² = ||s||² en promedio
        # power_scaling = np.sqrt(self.num_tx)
        # tx_symbols = tx_symbols / power_scaling
        
        return tx_symbols
    
    def update_precoder(self, H_channel, method='MRT'):
        """
        Actualiza el precoder W basado en el canal actual.
        
        Args:
            H_channel: Canal estimado [num_rx, num_tx, num_subcarriers]
                       o [num_rx, num_tx] para un subcarrier
            method: 'MRT' o 'eigen'
        """
        # Si H es 3D, promediar sobre subportadoras (simplificación)
        if H_channel.ndim == 3:
            H_avg = np.mean(H_channel, axis=2)  # [num_rx, num_tx]
        else:
            H_avg = H_channel
        
        # Calcular W según método
        if method == 'MRT':
            self.W = self.calculate_mrt_weights(H_avg)
        elif method == 'eigen':
            self.W = self.calculate_eigenbeamforming(H_avg)
        else:
            raise ValueError(f"Método '{method}' no soportado")
        
        return self.W
    
    def get_current_precoder(self):
        """Retorna el precoder actual W"""
        return self.W
    
    def get_effective_channel(self, H_channel):
        """
        Calcula el canal efectivo después del precoding.
        H_eff = H @ W
        
        Args:
            H_channel: Canal [num_rx, num_tx]
        
        Returns:
            H_eff: Canal efectivo [num_rx, num_layers]
        """
        if self.W is None:
            raise ValueError("Precoder W no disponible")
        
        H_eff = H_channel @ self.W
        return H_eff
    
    def calculate_beamforming_gain(self, H_channel):
        """
        Calcula la ganancia de beamforming en dB.
        Ganancia = ||H @ W||² / ||H||²_F
        
        Args:
            H_channel: Canal [num_rx, num_tx]
        
        Returns:
            gain_db: Ganancia en dB
        """
        if self.W is None:
            return 0.0
        
        # Canal efectivo
        H_eff = H_channel @ self.W
        
        # Potencia con beamforming
        power_bf = np.sum(np.abs(H_eff)**2)
        
        # Potencia sin beamforming (promedio sobre antenas)
        power_no_bf = np.sum(np.abs(H_channel)**2) / self.num_tx
        
        # Ganancia en dB
        gain_db = 10 * np.log10(power_bf / power_no_bf)
        
        return gain_db


class AdaptiveBeamforming(BeamformingPrecoder):
    """
    Beamforming adaptativo con actualización periódica basada en coherence time.
    """
    
    def __init__(self, num_tx, velocity_kmh, frequency_ghz, num_layers=1):
        """
        Args:
            num_tx (int): Antenas TX
            velocity_kmh (float): Velocidad del UE en km/h
            frequency_ghz (float): Frecuencia portadora en GHz
            num_layers (int): Capas espaciales
        """
        super().__init__(num_tx, num_layers, precoder_type='MRT')
        
        self.velocity_kmh = velocity_kmh
        self.frequency_ghz = frequency_ghz
        
        # Calcular período de actualización
        self.update_period = self._calculate_update_period()
        self.symbols_since_update = 0
        
        print(f"[AdaptiveBeamforming]")
        print(f"  Velocidad: {velocity_kmh} km/h")
        print(f"  Frecuencia: {frequency_ghz} GHz")
        print(f"  Actualizar W cada {self.update_period} símbolos OFDM")
    
    def _calculate_update_period(self):
        """
        Calcula período de actualización basado en coherence time.
        """
        # Convertir velocidad a m/s
        v_mps = self.velocity_kmh / 3.6
        
        # Frecuencia en Hz
        fc_hz = self.frequency_ghz * 1e9
        
        # Velocidad de la luz
        c = 3e8
        
        # Doppler máximo
        fd = v_mps * fc_hz / c
        
        if fd == 0:
            # Canal estático
            return 100  # Actualizar raramente
        
        # Coherence time (regla del 90%)
        Tc = 9 / (16 * np.pi * fd)
        
        # Actualizar cada 10% del coherence time (conservador)
        update_time = 0.1 * Tc
        
        # Duración de un símbolo OFDM (para 15 kHz spacing)
        symbol_duration = 1 / 15000  # 66.67 μs
        
        # Convertir a número de símbolos
        update_symbols = int(update_time / symbol_duration)
        
        # Limitar entre 1 y 140 (10 slots)
        return np.clip(update_symbols, 1, 140)
    
    def should_update(self):
        """
        Decide si es momento de actualizar el precoder.
        """
        return self.symbols_since_update >= self.update_period
    
    def process_symbol(self, symbols, H_channel):
        """
        Procesa un símbolo OFDM con actualización adaptativa.
        
        Args:
            symbols: Símbolos QAM [num_symbols]
            H_channel: Canal actual [num_rx, num_tx]
        
        Returns:
            tx_symbols: Símbolos precodificados [num_tx, num_symbols]
        """
        # Verificar si actualizar W
        if self.should_update() or self.W is None:
            self.update_precoder(H_channel, method='MRT')
            self.symbols_since_update = 0
        
        # Aplicar precoding
        tx_symbols = self.apply_precoding(symbols)
        
        # Incrementar contador
        self.symbols_since_update += 1
        
        return tx_symbols
