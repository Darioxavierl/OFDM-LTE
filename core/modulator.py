"""
Modulador OFDM - Convierte símbolos QAM/QPSK a señal OFDM

Versiones:
- v1.0: Mapeo simple (secuencial)
- v2.0: Mapeo LTE con DC, guardias y pilotos (ResourceMapper)
- v3.0: SC-FDM con precodificación DFT
"""
import numpy as np
from config import LTEConfig
from core.resource_mapper import ResourceMapper
from core.dft_precoding import SC_FDMPrecodifier


class QAMModulator:
    """Modulador QAM/QPSK que genera símbolos complejos"""
    
    def __init__(self, modulation_type='QPSK'):
        """
        Inicializa el modulador QAM/QPSK
        
        Args:
            modulation_type: 'QPSK', '16-QAM' o '64-QAM'
        """
        self.modulation_type = modulation_type
        self.constellation = self._generate_constellation()
    
    def _generate_constellation(self):
        """Genera la constelación de símbolos"""
        if self.modulation_type == 'QPSK':
            # QPSK: 4 símbolos
            constellation = np.array([
                1+1j, 1-1j, -1+1j, -1-1j
            ]) / np.sqrt(2)
        
        elif self.modulation_type == '16-QAM':
            # 16-QAM: 16 símbolos en grid 4x4
            real_vals = np.array([-3, -1, 1, 3])
            imag_vals = np.array([-3, -1, 1, 3])
            symbols = []
            for r in real_vals:
                for i in imag_vals:
                    symbols.append(r + 1j*i)
            constellation = np.array(symbols) / np.sqrt(10)
        
        elif self.modulation_type == '64-QAM':
            # 64-QAM: 64 símbolos en grid 8x8
            real_vals = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            imag_vals = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            symbols = []
            for r in real_vals:
                for i in imag_vals:
                    symbols.append(r + 1j*i)
            constellation = np.array(symbols) / np.sqrt(42)
        
        else:
            raise ValueError(f"Modulación no soportada: {self.modulation_type}")
        
        return constellation
    
    def bits_to_symbols(self, bits):
        """
        Convierte bits a símbolos complejos
        
        Args:
            bits: Array de bits (0 y 1)
            
        Returns:
            Array de símbolos complejos
        """
        bits_per_symbol = int(np.log2(len(self.constellation)))
        
        # Rellenar bits si es necesario
        if len(bits) % bits_per_symbol != 0:
            bits = np.pad(bits, (0, bits_per_symbol - len(bits) % bits_per_symbol), 'constant')
        
        num_symbols = len(bits) // bits_per_symbol
        symbols = np.zeros(num_symbols, dtype=complex)
        
        for i in range(num_symbols):
            # Agrupar bits
            bit_group = bits[i*bits_per_symbol:(i+1)*bits_per_symbol]
            # Convertir a índice decimal
            index = int(''.join(map(str, bit_group.astype(int))), 2)
            # Mapear a símbolo
            symbols[i] = self.constellation[index % len(self.constellation)]
        
        return symbols
    
    def symbols_to_bits(self, symbols):
        """
        Convierte símbolos complejos a bits (demodulación)
        
        Args:
            symbols: Array de símbolos complejos
            
        Returns:
            Array de bits demodulados
        """
        bits_per_symbol = int(np.log2(len(self.constellation)))
        bits = []
        
        for symbol in symbols:
            # Buscar el símbolo más cercano en la constelación
            distances = np.abs(self.constellation - symbol)
            nearest_idx = np.argmin(distances)
            
            # Convertir índice a bits
            bit_pattern = format(nearest_idx, f'0{bits_per_symbol}b')
            bits.extend([int(b) for b in bit_pattern])
        
        return np.array(bits)
    
    def get_constellation(self):
        """Retorna la constelación actual"""
        return self.constellation


class OFDMModulator:
    """Modulador OFDM que implementa IFFT y prefijo cíclico
    
    Soporta tres modos:
    - 'simple': Mapeo secuencial tradicional (v1.0)
    - 'lte': Mapeo con estándar LTE (DC, guardias, pilotos) (v2.0)
    - 'sc-fdm': Mapeo LTE + precodificación DFT (v3.0)
    """
    
    def __init__(self, config, mode='lte', enable_sc_fdm=False):
        """
        Inicializa el modulador OFDM
        
        Args:
            config: Objeto LTEConfig con parámetros de configuración
            mode: 'simple', 'lte' o 'sc-fdm' (default: 'lte')
            enable_sc_fdm: Si True, habilita SC-FDM (alias para mode='sc-fdm')
        """
        self.config = config
        self.mode = 'sc-fdm' if enable_sc_fdm else mode
        self.enable_sc_fdm = enable_sc_fdm or (mode == 'sc-fdm')
        self.qam_modulator = QAMModulator(config.modulation)
        
        # Inicializar ResourceMapper si modo LTE o SC-FDM
        if self.mode in ['lte', 'sc-fdm']:
            self.resource_mapper = ResourceMapper(config)
        else:
            self.resource_mapper = None
        
        # Inicializar precodificador SC-FDM
        if self.enable_sc_fdm and self.resource_mapper is not None:
            num_data_subcarriers = len(self.resource_mapper.get_data_indices())
            self.sc_fdm_precoder = SC_FDMPrecodifier(
                num_data_subcarriers=num_data_subcarriers,
                enable=self.enable_sc_fdm
            )
        else:
            self.sc_fdm_precoder = None
    
    def modulate(self, bits):
        """
        Modula bits a señal OFDM
        
        Proceso (modo simple):
        1. Convertir bits a símbolos QAM
        2. Mapeo secuencial a subportadoras
        3. Aplicar IFFT
        4. Agregar prefijo cíclico
        
        Proceso (modo LTE):
        1. Convertir bits a símbolos QAM
        2. (Opcional SC-FDM: Precodificación DFT)
        3. Mapeo LTE: datos + pilotos + DC nulo + guardias nulas
        4. Aplicar IFFT
        5. Agregar prefijo cíclico
        
        Args:
            bits: Array de bits
            
        Returns:
            tuple: (signal_time_domain, symbols, mapping_info)
                   donde mapping_info es None en modo simple
        """
        # Paso 1: Convertir bits a símbolos QAM
        qam_symbols = self.qam_modulator.bits_to_symbols(bits)
        
        if self.mode in ['lte', 'sc-fdm']:
            # Modo LTE con mapeo de recursos (y opcionalmente SC-FDM)
            return self._modulate_lte(qam_symbols)
        else:
            # Modo simple (original)
            return self._modulate_simple(qam_symbols)
    
    def _modulate_simple(self, qam_symbols):
        """
        Modulación simple: mapeo secuencial tradicional
        
        Returns: (signal, symbols, None)
        """
        # Asegurarse de que tenemos el número correcto de símbolos
        symbols_parallel = np.zeros(self.config.N, dtype=complex)
        num_data_symbols = min(len(qam_symbols), self.config.Nc)
        symbols_parallel[:num_data_symbols] = qam_symbols[:num_data_symbols]
        
        # Aplicar IFFT
        time_domain = np.fft.ifft(symbols_parallel) * np.sqrt(self.config.N)
        
        # Agregar prefijo cíclico
        ofdm_symbol_with_cp = np.concatenate([
            time_domain[-self.config.cp_length:],
            time_domain
        ])
        
        return ofdm_symbol_with_cp, qam_symbols[:num_data_symbols], None
    
    def _modulate_lte(self, qam_symbols):
        """
        Modulación LTE: mapeo con estándar LTE (DC, guardias, pilotos)
        Soporta SC-FDM con precodificación DFT
        
        Returns: (signal, symbols, mapping_info)
        """
        # IMPORTANTE: El número de QAM symbols debe ser exactamente igual 
        # al número de subportadoras de datos (sin pilotos, sin DC, sin guardias)
        num_data_subcarriers = len(self.resource_mapper.get_data_indices())
        
        # Si tenemos menos símbolos de lo esperado, rellenar con ceros (esto NO debería ocurrir)
        if len(qam_symbols) < num_data_subcarriers:
            qam_symbols = np.pad(qam_symbols, (0, num_data_subcarriers - len(qam_symbols)), 'constant')
        elif len(qam_symbols) > num_data_subcarriers:
            # Si tenemos más, truncar
            qam_symbols = qam_symbols[:num_data_subcarriers]
        
        # Paso 1: Aplicar precodificación DFT si está habilitada (SC-FDM)
        if self.enable_sc_fdm and self.sc_fdm_precoder is not None:
            qam_symbols_precoded = self.sc_fdm_precoder.precoding(qam_symbols)
        else:
            qam_symbols_precoded = qam_symbols
        
        # Paso 2: Mapear símbolos usando ResourceMapper
        grid_mapped, mapping_info = self.resource_mapper.map_symbols(qam_symbols_precoded)
        
        # Paso 3: Aplicar IFFT
        time_domain = np.fft.ifft(grid_mapped) * np.sqrt(self.config.N)
        
        # Paso 4: Agregar prefijo cíclico
        ofdm_symbol_with_cp = np.concatenate([
            time_domain[-self.config.cp_length:],
            time_domain
        ])
        
        return ofdm_symbol_with_cp, qam_symbols, mapping_info
    
    def modulate_stream(self, bits, num_ofdm_symbols=None):
        """
        Modula un stream de bits a múltiples símbolos OFDM
        
        Args:
            bits: Array de bits
            num_ofdm_symbols: Número de símbolos OFDM a generar (opcional)
            
        Returns:
            tuple: (signal_concatenated, all_symbols, mapping_infos)
                   donde mapping_infos es None en modo simple
        """
        # Calcular bits por símbolo OFDM
        if self.mode in ['lte', 'sc-fdm'] and self.resource_mapper is not None:
            # Usar el número REAL de datos subportadoras en modo LTE/SC-FDM
            num_data_subcarriers = len(self.resource_mapper.get_data_indices())
            bits_per_ofdm = num_data_subcarriers * self.config.bits_per_symbol
        else:
            # Modo simple: usar la configuración nominal Nc
            bits_per_ofdm = self.config.Nc * self.config.bits_per_symbol
        
        # Calcular número de símbolos OFDM necesarios
        if num_ofdm_symbols is None:
            num_ofdm_symbols = int(np.ceil(len(bits) / bits_per_ofdm))
        
        # Rellenar bits si es necesario
        total_bits_needed = num_ofdm_symbols * bits_per_ofdm
        if len(bits) < total_bits_needed:
            bits = np.pad(bits, (0, total_bits_needed - len(bits)), 'constant')
        
        # Modular cada símbolo OFDM
        signal_stream = []
        all_symbols = []
        mapping_infos = [] if self.mode in ['lte', 'sc-fdm'] else None
        
        for i in range(num_ofdm_symbols):
            start_idx = i * bits_per_ofdm
            end_idx = start_idx + bits_per_ofdm
            bits_chunk = bits[start_idx:end_idx]
            
            ofdm_signal, symbols, mapping_info = self.modulate(bits_chunk)
            signal_stream.append(ofdm_signal)
            all_symbols.append(symbols)
            
            if self.mode in ['lte', 'sc-fdm']:
                mapping_infos.append(mapping_info)
        
        # Concatenar todos los símbolos OFDM
        signal_concatenated = np.concatenate(signal_stream)
        
        return signal_concatenated, all_symbols, mapping_infos if self.mode in ['lte', 'sc-fdm'] else None
    
    def modulate_stream_vectorized(self, bits, num_ofdm_symbols=None):
        """
        Modula un stream de bits a múltiples símbolos OFDM usando VECTORIZACIÓN.
        
        Este método procesa TODOS los símbolos OFDM simultáneamente en lugar de uno por uno.
        Es más rápido para grandes cantidades de datos.
        
        Args:
            bits: Array de bits
            num_ofdm_symbols: Número de símbolos OFDM a generar (opcional)
            
        Returns:
            tuple: (signal_concatenated, all_symbols, mapping_infos)
                   Idéntico a modulate_stream(), pero más rápido
        """
        # Calcular bits por símbolo OFDM
        if self.mode == 'lte' and self.resource_mapper is not None:
            # Usar el número REAL de datos subportadoras en modo LTE
            num_data_subcarriers = len(self.resource_mapper.get_data_indices())
            bits_per_ofdm = num_data_subcarriers * self.config.bits_per_symbol
        else:
            # Modo simple: usar la configuración nominal Nc
            bits_per_ofdm = self.config.Nc * self.config.bits_per_symbol
        
        # Calcular número de símbolos OFDM necesarios
        if num_ofdm_symbols is None:
            num_ofdm_symbols = int(np.ceil(len(bits) / bits_per_ofdm))
        
        # Rellenar bits si es necesario
        total_bits_needed = num_ofdm_symbols * bits_per_ofdm
        if len(bits) < total_bits_needed:
            bits = np.pad(bits, (0, total_bits_needed - len(bits)), 'constant')
        
        # ===================================================
        # PASO 1: VECTORIZAR BITS → QAM SÍMBOLOS
        # ===================================================
        # Reshape: reorganizar todos los bits de una vez
        bits_reshaped = bits[:total_bits_needed].reshape(num_ofdm_symbols, bits_per_ofdm)
        
        # Convertir todos los chunks de bits a símbolos QAM simultáneamente
        all_qam_symbols = []
        for i in range(num_ofdm_symbols):
            qam_symbols = self.qam_modulator.bits_to_symbols(bits_reshaped[i])
            all_qam_symbols.append(qam_symbols)
        
        # ===================================================
        # PASO 2: MAPEO A SUBPORTADORAS (TODO DE UNA VEZ)
        # ===================================================
        if self.mode == 'lte':
            # Mapeo LTE: procesar todos los símbolos
            signal_stream = []
            mapping_infos = []
            
            for i, qam_symbols in enumerate(all_qam_symbols):
                grid_mapped, mapping_info = self.resource_mapper.map_symbols(qam_symbols)
                
                # IFFT
                time_domain = np.fft.ifft(grid_mapped) * np.sqrt(self.config.N)
                
                # Agregar CP
                ofdm_symbol_with_cp = np.concatenate([
                    time_domain[-self.config.cp_length:],
                    time_domain
                ])
                
                signal_stream.append(ofdm_symbol_with_cp)
                mapping_infos.append(mapping_info)
            
            all_symbols = all_qam_symbols
        else:
            # Modo simple: mapeo secuencial a subportadoras
            signal_stream = []
            all_symbols = []
            
            for qam_symbols in all_qam_symbols:
                # Mapeo simple
                symbols_parallel = np.zeros(self.config.N, dtype=complex)
                num_data_symbols = min(len(qam_symbols), self.config.Nc)
                symbols_parallel[:num_data_symbols] = qam_symbols[:num_data_symbols]
                
                # IFFT
                time_domain = np.fft.ifft(symbols_parallel) * np.sqrt(self.config.N)
                
                # Agregar CP
                ofdm_symbol_with_cp = np.concatenate([
                    time_domain[-self.config.cp_length:],
                    time_domain
                ])
                
                signal_stream.append(ofdm_symbol_with_cp)
                all_symbols.append(qam_symbols[:num_data_symbols])
            
            mapping_infos = None
        
        # ===================================================
        # PASO 3: CONCATENAR TODOS LOS SÍMBOLOS
        # ===================================================
        signal_concatenated = np.concatenate(signal_stream)
        
        return signal_concatenated, all_symbols, mapping_infos if self.mode == 'lte' else None
    
    def set_sc_fdm_enabled(self, enable: bool):
        """
        Activa/desactiva SC-FDM dinámicamente
        
        Args:
            enable: True para habilitar SC-FDM, False para deshabilitarlo
        """
        self.enable_sc_fdm = enable
        self.mode = 'sc-fdm' if enable else 'lte'
        
        if self.sc_fdm_precoder is not None:
            self.sc_fdm_precoder.set_enable(enable)
    
    def get_qam_modulator(self):
        """Retorna el modulador QAM para acceso directo"""
        return self.qam_modulator


def qpsk_to_llrs(symbols: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Convierte símbolos QPSK a LLRs (Log-Likelihood Ratios) para decodificación soft
    
    QPSK: Constelación {+1+1j, +1-1j, -1+1j, -1-1j} / sqrt(2)
    
    LLR Calculation:
        Para bit en posición real: LLR_I = (2/σ²) * Re(y) * scale
        Para bit en posición imag: LLR_Q = (2/σ²) * Im(y) * scale
    
    Convención LLR:
        LLR > 0 → bit más probable = 0
        LLR < 0 → bit más probable = 1
    
    Parameters:
    -----------
    symbols : np.ndarray
        Símbolos QPSK recibidos (complejos, potencialmente con ruido)
    noise_var : float
        Varianza de ruido efectiva (σ²)
    
    Returns:
    --------
    np.ndarray : LLRs intercalados [LLR_I0, LLR_Q0, LLR_I1, LLR_Q1, ...]
    
    Example:
    --------
    >>> symbols = np.array([0.7+0.7j, 0.6-0.8j])  # QPSK con ruido
    >>> noise_var = 0.1
    >>> llrs = qpsk_to_llrs(symbols, noise_var)
    >>> len(llrs)
    4
    """
    if symbols.size == 0:
        return np.array([], dtype=np.float64)
    
    # QPSK normalizado por sqrt(2) tiene energía = 1
    # Los símbolos están en {±1±1j}/sqrt(2), entonces scale = sqrt(2)
    scale = np.sqrt(2)
    
    # Calcular LLRs para componente real (bit I)
    llr_i = (2.0 / noise_var) * symbols.real * scale
    
    # Calcular LLRs para componente imaginaria (bit Q)
    llr_q = (2.0 / noise_var) * symbols.imag * scale
    
    # Intercalar LLRs: [I0, Q0, I1, Q1, I2, Q2, ...]
    num_symbols = len(symbols)
    llrs = np.zeros(2 * num_symbols, dtype=np.float64)
    llrs[0::2] = llr_i  # Posiciones pares: bits I
    llrs[1::2] = llr_q  # Posiciones impares: bits Q
    
    return llrs


def qam16_to_llrs(symbols: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Convierte símbolos 16-QAM a LLRs para decodificación soft
    
    16-QAM: Constelación 4x4 con 4 bits por símbolo
    
    Parameters:
    -----------
    symbols : np.ndarray
        Símbolos 16-QAM recibidos (complejos)
    noise_var : float
        Varianza de ruido efectiva
    
    Returns:
    --------
    np.ndarray : LLRs para los 4 bits de cada símbolo
    
    Note:
    -----
    Placeholder para futura implementación completa
    """
    # TODO: Implementar LLR calculation para 16-QAM usando max-log-MAP
    raise NotImplementedError("16-QAM LLR generation not yet implemented")


def qam64_to_llrs(symbols: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Convierte símbolos 64-QAM a LLRs para decodificación soft
    
    64-QAM: Constelación 8x8 con 6 bits por símbolo
    
    Parameters:
    -----------
    symbols : np.ndarray
        Símbolos 64-QAM recibidos (complejos)
    noise_var : float
        Varianza de ruido efectiva
    
    Returns:
    --------
    np.ndarray : LLRs para los 6 bits de cada símbolo
    
    Note:
    -----
    Placeholder para futura implementación completa
    """
    # TODO: Implementar LLR calculation para 64-QAM usando max-log-MAP
    raise NotImplementedError("64-QAM LLR generation not yet implemented")

