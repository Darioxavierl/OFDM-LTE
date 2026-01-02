"""
LTE Receiver - Receptor OFDM con soporte para mapeo LTE

Implementa:
- Extracción de pilotos
- Estimación de canal (Channel Estimation)
- Ecualización (Equalization)
- Extracción de datos
- Decodificación

Compatible con modo LTE del transmisor
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from core.resource_mapper import LTEResourceGrid, PilotPattern
from core.modulator import QAMModulator


class LTEChannelEstimator:
    """
    Estimador de canal basado en señales de referencia (pilotos) LTE
    
    Utiliza pilotos QPSK deterministas para estimar la respuesta del canal
    """
    
    def __init__(self, config, cell_id=0):
        """
        Inicializa el estimador de canal
        
        Args:
            config: Objeto LTEConfig
            cell_id: ID de celda para generar patrón de pilotos (0-167)
        """
        self.config = config
        self.cell_id = cell_id
        self.resource_grid = LTEResourceGrid(config.N, config.Nc)
        self.pilot_pattern = PilotPattern(cell_id)
    
    def estimate_channel(self, received_signal: np.ndarray, 
                        tx_signal: Optional[np.ndarray] = None) -> Dict:
        """
        Estima la respuesta del canal usando pilotos LTE
        
        Proceso:
        1. Demodular la señal recibida
        2. Extraer símbolos de piloto
        3. Comparar con pilotos conocidos (transmitidos)
        4. Estimar canal en posiciones de piloto (por interpolación)
        5. Interpolar canal a todas las subportadoras
        
        Args:
            received_signal: Señal demodulada recibida (dominio frecuencial)
            tx_signal: Señal transmitida (opcional, para LS estimation)
            
        Returns:
            dict con:
                - 'channel_estimate': Estimación de canal para cada SC
                - 'pilot_snr': SNR estimado de pilotos
                - 'interpolated': Si se usó interpolación
        """
        N = self.config.N
        
        # Generar patrón de pilotos conocidos
        pilot_indices = self.resource_grid.get_pilot_indices()
        num_pilots = len(pilot_indices)
        known_pilots = self.pilot_pattern.generate_pilots(num_pilots)
        
        # Extraer pilotos recibidos
        received_pilots = received_signal[pilot_indices]
        
        # LS (Least Squares) Channel Estimation en posiciones de piloto
        # H_est = Y / X (división elemento a elemento)
        # donde Y = símbolos piloto recibidos, X = símbolos piloto transmitidos
        pilot_channel_est = received_pilots / known_pilots
        
        # Estimar SNR de los pilotos
        pilot_power = np.mean(np.abs(received_pilots) ** 2)
        estimated_noise_power = np.mean(np.abs(received_pilots - known_pilots) ** 2)
        pilot_snr = pilot_power / (estimated_noise_power + 1e-10)
        
        # Interpolación lineal a todas las subportadoras
        channel_estimate = self._interpolate_channel(
            pilot_indices, 
            pilot_channel_est, 
            N
        )
        
        return {
            'channel_estimate': channel_estimate,
            'pilot_channel': pilot_channel_est,
            'pilot_indices': pilot_indices,
            'pilot_snr_linear': pilot_snr,
            'pilot_snr_db': 10 * np.log10(pilot_snr + 1e-10),
            'interpolated': True
        }
    
    def _interpolate_channel(self, pilot_indices: np.ndarray, 
                            pilot_values: np.ndarray, 
                            total_subcarriers: int) -> np.ndarray:
        """
        Interpola respuesta de canal a todas las subportadoras
        
        Usa interpolación lineal entre pilotos, extrapolación en los bordes
        
        Args:
            pilot_indices: Índices donde tenemos estimación de canal
            pilot_values: Valores de canal en esos índices
            total_subcarriers: Total de subportadoras
            
        Returns:
            Array con estimación de canal para todas las SC
        """
        # Crear array completo inicializado
        channel_full = np.zeros(total_subcarriers, dtype=complex)
        
        # Extrapolación: usar primer/último valor conocido en los bordes
        channel_full[:pilot_indices[0]] = pilot_values[0]
        channel_full[pilot_indices[-1]:] = pilot_values[-1]
        
        # Interpolación lineal entre pilotos
        for i in range(len(pilot_indices) - 1):
            idx1, idx2 = pilot_indices[i], pilot_indices[i + 1]
            val1, val2 = pilot_values[i], pilot_values[i + 1]
            
            # Número de puntos a interpolar
            num_points = idx2 - idx1 + 1
            
            # Interpolación lineal (también funciona para complejos)
            interpolated = np.linspace(val1, val2, num_points)
            channel_full[idx1:idx2 + 1] = interpolated
        
        return channel_full


class LTEEqualizerZF:
    """
    Ecualizador Zero-Forcing (ZF) para LTE
    
    Compensa la distorsión del canal estimado
    """
    
    def __init__(self, config, regularization=1e-6):
        """
        Inicializa el ecualizador ZF
        
        Args:
            config: Objeto LTEConfig
            regularization: Parámetro de regularización para evitar división por cero
        """
        self.config = config
        self.regularization = regularization
    
    def equalize(self, received_symbols: np.ndarray, 
                channel_estimate: np.ndarray) -> np.ndarray:
        """
        Aplica ecualización Zero-Forcing
        
        H_inv = 1 / H_est
        x_eq = Y * H_inv = Y / H_est
        
        Args:
            received_symbols: Símbolos recibidos (todo el espectro)
            channel_estimate: Estimación de canal
            
        Returns:
            Símbolos ecualizados
        """
        # Ecualización ZF simple: división por el canal estimado
        # Agregar regularización para evitar amplificación de ruido
        channel_mag = np.abs(channel_estimate)
        
        # Zero-Forcing directo
        equalized = received_symbols / (channel_estimate + self.regularization)
        
        # Opcional: normalizar potencia
        # (comentado: mantener potencia original)
        # equalized = equalized / np.mean(np.abs(equalized))
        
        return equalized


class LTEReceiver:
    """
    Receptor OFDM completo con soporte LTE
    
    Integra:
    - Demodulación FFT
    - Estimación de canal usando pilotos
    - Ecualización
    - Detección de símbolos
    - Descodificación QAM
    """
    
    def __init__(self, config, cell_id=0, enable_equalization=True, enable_sc_fdm=False):
        """
        Inicializa el receptor LTE
        
        Args:
            config: Objeto LTEConfig
            cell_id: ID de celda para generar pilotos (0-167)
            enable_equalization: Si True, usa ecualización; si False, asume canal AWGN
            enable_sc_fdm: Si True, aplica decodificación IDFT para SC-FDM
        """
        self.config = config
        self.cell_id = cell_id
        self.enable_equalization = enable_equalization
        self.enable_sc_fdm = enable_sc_fdm
        
        # Componentes
        self.resource_grid = LTEResourceGrid(config.N, config.Nc)
        self.pilot_pattern = PilotPattern(cell_id)
        self.channel_estimator = LTEChannelEstimator(config, cell_id)
        self.equalizer = LTEEqualizerZF(config)
        self.qam_demodulator = QAMModulator(config.modulation)
        
        # Inicializar SC-FDM decodificador si está habilitado
        if self.enable_sc_fdm:
            from core.dft_precoding import SC_FDMDecodifier
            num_data_subcarriers = len(self.resource_grid.get_data_indices())
            self.sc_fdm_decoder = SC_FDMDecodifier(
                num_data_subcarriers=num_data_subcarriers,
                enable=True
            )
        else:
            self.sc_fdm_decoder = None
        
        # Historial de estimaciones
        self.channel_estimates = []
        self.equalization_info = []
        
        # Parámetro para estimación periódica
        self.slot_size = 14  # Número de OFDM símbolos por slot en LTE
    
    def receive_and_decode(self, received_ofdm_signal: np.ndarray) -> Dict:
        """
        Recibe y decodifica una señal OFDM con mapeo LTE
        
        Proceso completo:
        1. FFT (demodulación OFDM de múltiples símbolos)
        2. Estimación de canal usando pilotos
        3. Ecualización (si está habilitada)
        4. Extracción de datos (ignorar DC, guardias, pilotos)
        5. Detección y decodificación
        
        Args:
            received_ofdm_signal: Señal OFDM recibida (dominio tiempo)
            
        Returns:
            dict con:
                - 'symbols_received': Símbolos recibidos (todos los SC)
                - 'symbols_equalized': Símbolos ecualizados (si aplica)
                - 'symbols_data_only': Solo símbolos de datos (sin pilotos/DC/guardias)
                - 'channel_estimate': Estimación de canal
                - 'channel_snr_db': SNR estimado del canal
                - 'bits': Bits decodificados
                - 'symbols_detected': Símbolos detectados
        """
        # Paso 1: Demodulación OFDM (FFT) para todos los símbolos
        all_received_symbols = self._demodulate_ofdm_stream(received_ofdm_signal)
        
        # Concatenar todos los símbolos en un array
        received_symbols = np.concatenate(all_received_symbols) if all_received_symbols else np.array([])
        
        if len(received_symbols) == 0:
            # Retornar arrays vacíos si no hay símbolos
            return {
                'symbols_received': received_symbols,
                'symbols_equalized': received_symbols,
                'symbols_data_only': received_symbols,
                'symbols_detected': received_symbols,
                'bits': np.array([]),
                'channel_estimate': received_symbols,
                'channel_snr_db': 0,
                'pilot_snr_db': 0,
                'num_data_symbols': 0,
                'num_pilot_symbols': 0,
                'equalization_enabled': self.enable_equalization
            }
        
        # Paso 2: Estimación periódica de canal (CADA SLOT como LTE)
        num_ofdm_symbols = len(all_received_symbols)
        
        # Realizar estimación periódica y obtener interpolación temporal
        channel_estimates_per_symbol, channel_snr_db = self._estimate_channel_periodic(
            all_received_symbols
        )
        
        # Para simplificar ecualizacion, usar el primer canal estimado
        # (se podría mejorar usando interpolación por símbolo)
        channel_estimate = channel_estimates_per_symbol[0]
        
        # Paso 3: Ecualización (opcional, usando estimaciones periódicas)
        if self.enable_equalization:
            symbols_equalized = self._equalize_with_periodic_estimates(
                all_received_symbols,
                channel_estimates_per_symbol
            )
        else:
            symbols_equalized = received_symbols
        
        # Paso 4: Extracción de datos (solo posiciones de datos)
        data_indices = self.resource_grid.get_data_indices()
        
        # Para múltiples símbolos OFDM, repetimos el patrón de índices
        num_ofdm_symbols = len(all_received_symbols)
        all_data_indices = []
        for sym_idx in range(num_ofdm_symbols):
            offset = sym_idx * self.config.N
            all_data_indices.extend(data_indices + offset)
        
        all_data_indices = np.array(all_data_indices)
        
        # Verificar que los índices están dentro del rango
        valid_indices = all_data_indices[all_data_indices < len(symbols_equalized)]
        symbols_data = symbols_equalized[valid_indices]
        
        # Aplicar IDFT si SC-FDM está habilitado (deprecodificación)
        if self.enable_sc_fdm and self.sc_fdm_decoder is not None:
            # Los símbolos de datos vienen en grupos de 249 (uno por símbolo OFDM)
            # Necesitamos aplicar IDFT a cada grupo
            symbols_data_list = []
            for sym_idx in range(num_ofdm_symbols):
                start_idx = sym_idx * len(data_indices)
                end_idx = start_idx + len(data_indices)
                if end_idx <= len(symbols_data):
                    data_chunk = symbols_data[start_idx:end_idx]
                    # Aplicar IDFT a este chunk
                    data_decoded = self.sc_fdm_decoder.decoding(data_chunk)
                    symbols_data_list.append(data_decoded)
            
            if symbols_data_list:
                symbols_data = np.concatenate(symbols_data_list)
        
        # Paso 5: Detección y decodificación
        symbols_detected = self._detect_symbols(symbols_data)
        bits = self.qam_demodulator.symbols_to_bits(symbols_detected)
        
        # Guardar en historial
        self.channel_estimates.append(channel_estimate)
        self.equalization_info.append({
            'channel_snr_db': channel_snr_db,
            'num_data_symbols': len(symbols_data)
        })
        
        return {
            'symbols_received': received_symbols,
            'symbols_equalized': symbols_equalized,
            'symbols_data_only': symbols_data,
            'symbols_detected': symbols_detected,
            'bits': bits,
            'channel_estimate': channel_estimate,
            'channel_snr_db': channel_snr_db,
            'pilot_snr_db': channel_snr_db,
            'num_data_symbols': len(symbols_data),
            'num_pilot_symbols': len(self.resource_grid.get_pilot_indices()) * num_ofdm_symbols,
            'equalization_enabled': self.enable_equalization
        }
    
    def _estimate_channel_periodic(self, all_received_symbols: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """
        Realiza estimación periódica de canal cada slot (14 símbolos OFDM) como LTE real.
        
        En LTE, el canal se estima en símbolos con pilotos (cada 2 símbolos) e interpola
        en tiempo para los símbolos sin pilotos. Aquí simplificamos a cada slot (14 símbolos).
        
        Args:
            all_received_symbols: Lista de símbolos OFDM recibidos en dominio frecuencia
            
        Returns:
            Tuple de:
            - channel_estimates_per_symbol: Lista con estimación de canal para cada símbolo
            - channel_snr_db: SNR estimado de los pilotos
        """
        num_symbols = len(all_received_symbols)
        channel_estimates_per_symbol = []
        pilot_snr_list = []
        
        # Estimar canal en cada slot (14 símbolos)
        for slot_start in range(0, num_symbols, self.slot_size):
            slot_end = min(slot_start + self.slot_size, num_symbols)
            slot_length = slot_end - slot_start
            
            # Estimar canal en el primer símbolo del slot que tenga pilotos
            # (Generalmente es el símbolo 0 del slot)
            symbol_idx = slot_start
            
            if symbol_idx < num_symbols:
                ch_info = self.channel_estimator.estimate_channel(
                    all_received_symbols[symbol_idx]
                )
                slot_channel_estimate = ch_info['channel_estimate']
                pilot_snr_list.append(ch_info['pilot_snr_db'])
                
                # Interpolar el canal para todos los símbolos del slot
                for i in range(slot_length):
                    sym_idx = slot_start + i
                    
                    if i == 0:
                        # Primer símbolo del slot: usar estimación directa
                        channel_estimates_per_symbol.append(slot_channel_estimate)
                    else:
                        # Símbolos posteriores en el slot: interpolar en tiempo
                        # Para simplificar, usar el mismo canal estimado
                        # (mejora: interpolar entre estimaciones de slots consecutivos)
                        channel_estimates_per_symbol.append(slot_channel_estimate)
        
        # SNR promedio de todos los slots
        channel_snr_db = np.mean(pilot_snr_list) if pilot_snr_list else 0.0
        
        return channel_estimates_per_symbol, channel_snr_db
    
    def _equalize_with_periodic_estimates(self, all_received_symbols: List[np.ndarray],
                                          channel_estimates_per_symbol: List[np.ndarray]) -> np.ndarray:
        """
        Aplica ecualización usando estimaciones periódicas de canal por símbolo.
        
        Cada símbolo OFDM se ecualiza con su estimación de canal correspondiente.
        
        Args:
            all_received_symbols: Lista de símbolos OFDM recibidos
            channel_estimates_per_symbol: Lista de estimaciones de canal por símbolo
            
        Returns:
            Array con símbolos ecualizados (concatenados)
        """
        symbols_equalized_list = []
        
        for sym_idx, received_sym in enumerate(all_received_symbols):
            # Obtener estimación de canal para este símbolo
            if sym_idx < len(channel_estimates_per_symbol):
                channel_est = channel_estimates_per_symbol[sym_idx]
            else:
                # Fallback: usar última estimación disponible
                channel_est = channel_estimates_per_symbol[-1]
            
            # Aplicar ecualización ZF a este símbolo
            sym_equalized = self.equalizer.equalize(received_sym, channel_est)
            symbols_equalized_list.append(sym_equalized)
        
        # Concatenar todos los símbolos ecualizados
        return np.concatenate(symbols_equalized_list) if symbols_equalized_list else np.array([])
    
    def _demodulate_ofdm_stream(self, received_signal: np.ndarray) -> List[np.ndarray]:
        """
        Demodula múltiples símbolos OFDM del flujo de recepción.
        
        Procesa la señal recibida en chunks de (N + CP) muestras, donde:
        - N = número de subportadoras (128)
        - CP = prefijo cíclico (32)
        
        Args:
            received_signal: Señal recibida completa (dominio tiempo)
            
        Returns:
            Lista de arrays con los símbolos OFDM en dominio frecuencia
            Cada elemento tiene shape (N,) = (128,)
        """
        symbol_length = self.config.N + self.config.cp_length  # 128 + 32 = 160
        num_symbols = len(received_signal) // symbol_length
        
        # Protección: asegurar que tenemos al menos 1 símbolo
        if num_symbols == 0:
            num_symbols = 1
        
        demodulated_symbols = []
        
        for sym_idx in range(num_symbols):
            # Extraer un símbolo OFDM (CP + datos)
            start_idx = sym_idx * symbol_length
            end_idx = start_idx + symbol_length
            
            # Protección: no pasar de los límites de la señal
            if end_idx > len(received_signal):
                end_idx = len(received_signal)
            
            received_ofdm = received_signal[start_idx:end_idx]
            
            # Si la longitud es insuficiente, rellenar con ceros
            if len(received_ofdm) < symbol_length:
                received_ofdm = np.pad(received_ofdm, (0, symbol_length - len(received_ofdm)), 'constant')
            
            # Remover prefijo cíclico
            signal_without_cp = received_ofdm[self.config.cp_length:]
            
            # FFT para obtener símbolo en dominio frecuencia
            frequency_domain = np.fft.fft(signal_without_cp) / np.sqrt(self.config.N)
            
            demodulated_symbols.append(frequency_domain)
        
        return demodulated_symbols
    
    def _demodulate_ofdm(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Demodulación OFDM: FFT en dominio frecuencial
        
        Asume sincronización perfecta (conocemos dónde está el símbolo OFDM)
        """
        symbols = self._demodulate_ofdm_stream(received_signal)
        # Concatenar todos los símbolos en un array
        if len(symbols) > 1:
            return np.concatenate(symbols)
        elif len(symbols) == 1:
            return symbols[0]
        else:
            return np.array([])
    
    def _detect_symbols(self, received_symbols: np.ndarray) -> np.ndarray:
        """
        Detección de símbolos con decisión dura (hard decision)
        Busca el símbolo de constelación más cercano
        """
        constellation = self.qam_demodulator.get_constellation()
        detected = np.zeros_like(received_symbols)
        
        for i, symbol in enumerate(received_symbols):
            distances = np.abs(constellation - symbol)
            nearest_idx = np.argmin(distances)
            detected[i] = constellation[nearest_idx]
        
        return detected
    
    def calculate_ber(self, transmitted_bits: np.ndarray, 
                      received_bits: np.ndarray) -> Dict:
        """
        Calcula tasa de error de bit (BER)
        
        Args:
            transmitted_bits: Bits transmitidos originales
            received_bits: Bits recibidos y decodificados
            
        Returns:
            dict con BER y errores
        """
        # Asegurar mismo tamaño
        min_len = min(len(transmitted_bits), len(received_bits))
        tx_bits = transmitted_bits[:min_len]
        rx_bits = received_bits[:min_len]
        
        # Contar errores
        errors = np.sum(tx_bits != rx_bits)
        ber = errors / min_len if min_len > 0 else 0
        
        return {
            'ber': ber,
            'errors': errors,
            'total_bits': min_len
        }
    
    def reset_history(self):
        """Limpia el historial de estimaciones"""
        self.channel_estimates = []
        self.equalization_info = []
    
    def get_channel_estimate_history(self) -> np.ndarray:
        """Retorna historial de estimaciones de canal"""
        return np.array(self.channel_estimates)
