"""
Sistema OFDM Completo - Integra modulador, canal y demodulador
Versión 2.0 con soporte LTE (mode='lte') e integración backward compatible
"""
import numpy as np
import time
import os
from core.modulator import OFDMModulator, QAMModulator
from core.demodulator import OFDMDemodulator, SymbolDetector
from core.channel import ChannelSimulator
from config import LTEConfig


class OFDMSystem:
    """Sistema OFDM completo que integra todos los componentes"""
    
    def __init__(self, config=None, channel_type='awgn', itu_profile=None, 
                 frequency_ghz=2.0, velocity_kmh=0, mode='lte', enable_equalization=True, enable_sc_fdm=False):
        """
        Inicializa el sistema OFDM
        
        Args:
            config: Objeto LTEConfig con parámetros de configuración
            channel_type: 'awgn' o 'rayleigh_mp'
            itu_profile: Perfil ITU para Rayleigh (si aplica)
            frequency_ghz: Frecuencia portadora en GHz (para Rayleigh)
            velocity_kmh: Velocidad en km/h (para Rayleigh)
            mode: 'lte' (nuevo mapeo LTE) o 'simple' (modo clásico)
            enable_equalization: Si True, habilita ecualización en modo LTE
            enable_sc_fdm: Si True, habilita SC-FDM (precodificación DFT)
        """
        if config is None:
            config = LTEConfig()
        
        self.config = config
        self.modulator = OFDMModulator(config, mode=mode, enable_sc_fdm=enable_sc_fdm)
        self.demodulator = OFDMDemodulator(config, mode=mode, enable_equalization=enable_equalization, enable_sc_fdm=enable_sc_fdm)
        self.channel_type = channel_type
        self.itu_profile = itu_profile
        self.frequency_ghz = frequency_ghz
        self.velocity_kmh = velocity_kmh
        self.mode = mode
        self.enable_equalization = enable_equalization
        self.enable_sc_fdm = enable_sc_fdm
        
        # Calcular Fs desde config
        fs = config.fs if hasattr(config, 'fs') else 15.36e6  # LTE default
        
        # Inicializar canal
        if channel_type == 'rayleigh_mp':
            self.channel = ChannelSimulator(
                channel_type='rayleigh_mp',
                snr_db=10.0,
                fs=fs,
                itu_profile=itu_profile,
                frequency_ghz=frequency_ghz,
                velocity_kmh=velocity_kmh
            )
        else:
            self.channel = ChannelSimulator('awgn', snr_db=10.0)
        
        self.symbol_detector = SymbolDetector(
            self.modulator.get_qam_modulator().get_constellation()
        )
        
        # Estadísticas
        self.statistics = {
            'transmitted_bits': 0,
            'received_bits': 0,
            'bit_errors': 0,
            'symbol_errors': 0,
            'total_symbols': 0,
            'transmission_time': 0
        }
        
        # Almacenar PAPR de transmisiones para CCDF
        self.papr_values_ofdm = []  # PAPR de OFDM estándar
        self.papr_values_sc_fdm = []  # PAPR de SC-FDM
    
    def calculate_papr(self, signal):
        """
        Calcula el PAPR (Peak-to-Average Power Ratio) de una señal OFDM
        
        PAPR = Potencia Pico / Potencia Promedio
        
        Args:
            signal: Señal transmitida en el dominio del tiempo (array complejo)
            
        Returns:
            dict: PAPR en dB, valores lineales, y potencias
        """
        # Calcular potencia instantánea |x[n]|²
        power_instantaneous = np.abs(signal) ** 2
        
        # Potencia pico: max(|x[n]|²)
        power_peak = np.max(power_instantaneous)
        
        # Potencia promedio: E[|x[n]|²]
        power_average = np.mean(power_instantaneous)
        
        # PAPR en forma lineal
        if power_average > 0:
            papr_linear = power_peak / power_average
            papr_db = 10 * np.log10(papr_linear)
        else:
            papr_linear = 0
            papr_db = 0
        
        return {
            'papr_db': papr_db,
            'papr_linear': papr_linear,
            'peak_power': power_peak,
            'avg_power': power_average
        }
    
    def calculate_papr_per_symbol(self, signal):
        """
        Calcula el PAPR por cada símbolo OFDM (más preciso)
        
        Args:
            signal: Señal completa transmitida
            
        Returns:
            dict: PAPR por símbolo y estadísticas
        """
        # Calcular número de símbolos OFDM
        samples_per_symbol = self.config.N + self.config.cp_length
        num_ofdm_symbols = len(signal) // samples_per_symbol
        
        papr_per_symbol = []
        power_avg_per_symbol = []
        power_peak_per_symbol = []
        
        # Calcular PAPR para cada símbolo OFDM
        for i in range(num_ofdm_symbols):
            start_idx = i * samples_per_symbol
            end_idx = start_idx + samples_per_symbol
            symbol_signal = signal[start_idx:end_idx]
            
            # Potencia del símbolo
            power_inst = np.abs(symbol_signal) ** 2
            power_peak = np.max(power_inst)
            power_avg = np.mean(power_inst)
            
            # PAPR de este símbolo
            if power_avg > 0:
                papr_linear = power_peak / power_avg
                papr_db = 10 * np.log10(papr_linear)
            else:
                papr_linear = 0
                papr_db = 0
            
            papr_per_symbol.append(papr_db)
            power_avg_per_symbol.append(power_avg)
            power_peak_per_symbol.append(power_peak)
        
        # Estadísticas generales
        papr_array = np.array(papr_per_symbol)
        power_peak_array = np.array(power_peak_per_symbol)
        power_avg_array = np.array(power_avg_per_symbol)
        
        return {
            'papr_per_symbol': papr_array,
            'power_peak_per_symbol': power_peak_array,
            'power_avg_per_symbol': power_avg_array,
            'papr_mean': np.mean(papr_array),
            'papr_max': np.max(papr_array),
            'papr_min': np.min(papr_array),
            'papr_std': np.std(papr_array),
            'num_symbols': num_ofdm_symbols
        }
    
    def calculate_papr_without_cp(self, signal):
        """
        Calcula el PAPR por cada símbolo OFDM SIN prefijo cíclico
        
        Este es el cálculo más preciso para OFDM, evaluando solo la parte útil.
        PAPR = Potencia Pico / Potencia Promedio (solo para la parte sin CP)
        
        Args:
            signal: Señal completa transmitida (con CP)
            
        Returns:
            dict: PAPR por símbolo (sin CP) y estadísticas
        """
        # Calcular número de símbolos OFDM
        samples_per_symbol = self.config.N + self.config.cp_length
        num_ofdm_symbols = len(signal) // samples_per_symbol
        
        papr_per_symbol_no_cp = []
        
        # Calcular PAPR para cada símbolo OFDM (sin CP)
        for i in range(num_ofdm_symbols):
            start_idx = i * samples_per_symbol
            
            # Saltar el CP: los primeros cp_length muestras son el CP
            # La parte útil comienza en start_idx + cp_length y tiene N muestras
            symbol_start = start_idx + self.config.cp_length
            symbol_end = symbol_start + self.config.N
            
            symbol_signal_no_cp = signal[symbol_start:symbol_end]
            
            # Potencia del símbolo SIN CP
            power_inst = np.abs(symbol_signal_no_cp) ** 2
            power_peak = np.max(power_inst)
            power_avg = np.mean(power_inst)
            
            # PAPR de este símbolo
            if power_avg > 0:
                papr_linear = power_peak / power_avg
                papr_db = 10 * np.log10(papr_linear)
            else:
                papr_linear = 0
                papr_db = 0
            
            papr_per_symbol_no_cp.append(papr_db)
        
        # Estadísticas generales
        papr_array = np.array(papr_per_symbol_no_cp)
        
        return {
            'papr_per_symbol': papr_array,  # PAPR de cada símbolo sin CP
            'papr_mean': np.mean(papr_array),
            'papr_max': np.max(papr_array),
            'papr_min': np.min(papr_array),
            'papr_std': np.std(papr_array),
            'num_symbols': num_ofdm_symbols,
            'papr_values': papr_array.tolist()  # Lista de PAPR para procesamiento posterior
        }
    
    def transmit(self, bits, snr_db=10.0, return_time=False, use_vectorized=False):
        """
        Transmite bits a través del sistema OFDM
        
        Proceso:
        1. Modular bits a señal OFDM (con soporte LTE si mode='lte')
        2. Calcular PAPR
        3. Transmitir a través del canal
        
        Args:
            bits: Array de bits a transmitir
            snr_db: SNR en dB
            return_time: Si True, retorna tiempo de transmisión en dict
            use_vectorized: Si True, usa método vectorizado (más rápido para datos grandes)
            
        Returns:
            dict: Resultados de transmisión con todas las métricas
        """
        start_time = time.time()
        
        # IMPORTANTE: Guardar número ORIGINAL de bits
        original_num_bits = len(bits)
        
        # Configurar SNR
        self.channel.set_snr(snr_db)
        
        # Modular bits (puede rellenar con ceros)
        # Nota: modulate_stream retorna (signal, symbols_list, mapping_infos)
        if use_vectorized and hasattr(self.modulator, 'modulate_stream_vectorized'):
            signal_transmitted, symbols_transmitted, mapping_infos = self.modulator.modulate_stream_vectorized(bits)
        else:
            signal_transmitted, symbols_transmitted, mapping_infos = self.modulator.modulate_stream(bits)
        
        # ✨ CALCULAR PAPR POR SÍMBOLO OFDM (más preciso) SIN CP
        papr_info_no_cp = self.calculate_papr_without_cp(signal_transmitted)
        
        # Almacenar PAPR para CCDF
        if self.enable_sc_fdm:
            self.papr_values_sc_fdm.extend(papr_info_no_cp['papr_values'])
        else:
            self.papr_values_ofdm.extend(papr_info_no_cp['papr_values'])
        
        # Transmitir a través del canal
        signal_received, _ = self.channel.channel.transmit(signal_transmitted)
        
        # Recibir
        bits_rx, symbols_detected, symbols_rx = self.receive(signal_received)
        
        transmission_time = time.time() - start_time
        self.statistics['transmission_time'] = transmission_time
        
        # IMPORTANTE: Retornar EXACTAMENTE el número original de bits
        # Truncar si recibimos más, rellenar con ceros si recibimos menos
        if len(bits_rx) < original_num_bits:
            bits_rx = np.pad(bits_rx, (0, original_num_bits - len(bits_rx)), 'constant')
        else:
            bits_rx = bits_rx[:original_num_bits]
        
        # Calcular errores solo con los bits originales
        bits_error = np.sum(bits != bits_rx)
        ber = bits_error / original_num_bits if original_num_bits > 0 else 0
        
        # Estadísticas de símbolos
        num_tx_symbols = sum(len(s) for s in symbols_transmitted)
        num_rx_symbols = len(symbols_detected)
        num_symbols_compare = min(num_tx_symbols, num_rx_symbols)
        
        symbols_tx_flat = np.concatenate(symbols_transmitted) if symbols_transmitted else np.array([])
        symbol_errors = 0
        if num_symbols_compare > 0:
            symbol_errors = np.sum(
                symbols_tx_flat[:num_symbols_compare] != symbols_detected[:num_symbols_compare]
            )
        ser = symbol_errors / num_symbols_compare if num_symbols_compare > 0 else 0
        
        results = {
            # Lowercase keys for GUI
            'snr_db': snr_db,
            'n_bits': original_num_bits,
            'transmitted_bits': original_num_bits,
            'received_bits': original_num_bits,
            'bit_errors': bits_error,
            'errors': bits_error,
            'ber': ber,
            'symbol_errors': symbol_errors,
            'ser': ser,
            'transmission_time': transmission_time,
            'evm': 0.0,
            
            # ✨ PAPR Metrics (SIN CP - más preciso para especificaciones)
            'papr_no_cp': papr_info_no_cp,
            
            # Uppercase keys for compatibility with simulator
            'SNR_dB': snr_db,
            'BER': ber,
            'SER': ser,
            
            # Common keys used by both
            'total_symbols': num_tx_symbols,
            'signal_tx': signal_transmitted,
            'tx_signal': signal_transmitted,  # Alias para GUI
            'signal_rx': signal_received,
            'symbols_tx': symbols_tx_flat[:num_symbols_compare] if num_symbols_compare > 0 else np.array([]),
            'symbols_rx': symbols_detected[:num_symbols_compare] if num_symbols_compare > 0 else np.array([]),
            'transmitted_symbols': symbols_tx_flat,
            'received_symbols': symbols_rx,  # ✨ Símbolos demodulados (con ruido) NO detectados
            'bits_tx': bits,  # Bits transmitidos originales
            'bits_rx': bits_rx,  # Bits recibidos (siempre mismo tamaño que bits_tx)
        }
        
        return results
    
    def receive(self, signal_received):
        """
        Recibe y demodula la señal OFDM
        
        Proceso:
        1. Demodular señal a símbolos
        2. Detectar símbolos más cercanos
        3. Convertir a bits
        
        Args:
            signal_received: Señal recibida del canal
            
        Returns:
            tuple: (bits_received, symbols_detected, symbols_demodulated)
        """
        # Calcular número de símbolos OFDM
        samples_per_symbol = self.config.N + self.config.cp_length
        num_ofdm_symbols = int(np.ceil(len(signal_received) / samples_per_symbol))
        
        # Demodular
        symbols_demodulated, bits_received = self.demodulator.demodulate_stream(
            signal_received, num_ofdm_symbols
        )
        
        # Detectar símbolos
        symbols_detected = self.symbol_detector.detect_batch(symbols_demodulated)
        
        return bits_received, symbols_detected, symbols_demodulated
    
    def simulate(self, bits, snr_db):
        """
        Alias para transmit() - mantiene compatibilidad con código anterior
        
        Args:
            bits: Array de bits a transmitir
            snr_db: SNR en dB
            
        Returns:
            dict: Resultados de la simulación
        """
        return self.transmit(bits, snr_db)
    
    # ============= MÉTODOS PARA CAMBIAR CANAL EN TIEMPO REAL =============
    
    def set_channel_type(self, channel_type, itu_profile=None):
        """
        Cambia el tipo de canal en tiempo real
        
        Args:
            channel_type: 'awgn' o 'rayleigh_mp'
            itu_profile: Perfil ITU (solo para rayleigh_mp)
        """
        self.channel_type = channel_type
        
        if channel_type == 'rayleigh_mp':
            if itu_profile is None:
                itu_profile = self.itu_profile
            else:
                self.itu_profile = itu_profile
            
            fs = self.config.fs if hasattr(self.config, 'fs') else 15.36e6
            self.channel = ChannelSimulator(
                channel_type='rayleigh_mp',
                snr_db=self.channel.channel.snr_db if hasattr(self.channel.channel, 'snr_db') else 10.0,
                fs=fs,
                itu_profile=itu_profile
            )
        else:
            # Cambiar a AWGN
            snr_db = self.channel.channel.snr_db if hasattr(self.channel.channel, 'snr_db') else 10.0
            self.channel = ChannelSimulator('awgn', snr_db=snr_db)
    
    def set_itu_profile(self, itu_profile):
        """
        Cambia el perfil ITU si el canal actual es Rayleigh
        
        Args:
            itu_profile: Nombre del perfil ITU-R M.1225
        """
        self.itu_profile = itu_profile
        if self.channel_type == 'rayleigh_mp':
            self.channel.set_itu_profile(itu_profile)
    
    def get_channel_info(self):
        """Retorna información del canal actual"""
        return self.channel.get_channel_info()
    
    # ====================================================================
    
    def run_ber_sweep(self, num_bits, snr_range, n_iterations, progress_callback=None, bits=None):
        """
        Realiza un barrido de SNR para calcular BER vs SNR
        
        Args:
            num_bits: Número de bits por simulación
            snr_range: Array de valores SNR en dB
            n_iterations: Número de iteraciones por SNR
            progress_callback: Función de callback para progreso
            bits: Array de bits específicos a transmitir (None = generar aleatorios)
            
        Returns:
            dict: Resultados del barrido (ber_mean, snr_range, etc.)
        """
        snr_range = np.atleast_1d(snr_range).astype(float)
        results = {
            'snr_db': snr_range,
            'n_bits': num_bits,
            'n_iterations': n_iterations,
            'ber_mean': [],
            'ber_std': [],
            'ber_ci_lower': [],
            'ber_ci_upper': [],
            'ser_mean': [],
            'ser_std': [],
            'ber_runs': [],
            'ser_runs': []
        }
        
        total_steps = len(snr_range)
        
        for idx, snr_db in enumerate(snr_range):
            ber_values = []
            ser_values = []
            
            # Ejecutar n_iterations corridas para cada SNR
            for run in range(n_iterations):
                # Usar bits proporcionados o generar aleatorios
                if bits is not None:
                    test_bits = bits.copy()
                else:
                    test_bits = np.random.randint(0, 2, num_bits)
                
                # Simular
                sim_results = self.transmit(test_bits, snr_db)
                
                ber_values.append(sim_results['ber'])
                ser_values.append(sim_results['ser'])
                
                # Reportar progreso
                progress = ((idx * n_iterations + run + 1) / (total_steps * n_iterations)) * 100
                if progress_callback:
                    progress_callback(progress, snr_db, run + 1)
            
            # Calcular estadísticas
            ber_array = np.array(ber_values)
            ser_array = np.array(ser_values)
            
            ber_mean = np.mean(ber_array)
            ber_std = np.std(ber_array)
            ser_mean = np.mean(ser_array)
            ser_std = np.std(ser_array)
            
            # Calcular intervalo de confianza (95%)
            from scipy import stats
            if len(ber_array) > 1:
                n = len(ber_array)
                t_value = stats.t.ppf(0.975, n - 1)
                margin = t_value * ber_std / np.sqrt(n)
                ber_ci_lower = max(0, ber_mean - margin)
                ber_ci_upper = ber_mean + margin
            else:
                ber_ci_lower = ber_ci_upper = ber_mean
            
            results['ber_mean'].append(ber_mean)
            results['ber_std'].append(ber_std)
            results['ber_ci_lower'].append(ber_ci_lower)
            results['ber_ci_upper'].append(ber_ci_upper)
            results['ser_mean'].append(ser_mean)
            results['ser_std'].append(ser_std)
            results['ber_runs'].append(ber_values)
            results['ser_runs'].append(ser_values)
        
        # Convertir a arrays numpy
        for key in results:
            if key not in ['ber_runs', 'ser_runs']:
                results[key] = np.array(results[key])
        
        return results
    
    def run_ber_sweep_all_modulations(self, num_bits, snr_range, n_iterations, 
                                      progress_callback=None, bits=None):
        """
        Realiza barrido de SNR para TODAS las modulaciones
        
        Args:
            num_bits: Número de bits por simulación
            snr_range: Array de valores SNR en dB
            n_iterations: Número de iteraciones por SNR
            progress_callback: Función de callback para progreso
            bits: Array de bits específicos a transmitir (None = generar aleatorios)
            
        Returns:
            dict: Resultados agrupados por modulación
        """
        from config import MODULATION_SCHEMES
        
        snr_range = np.atleast_1d(snr_range).astype(float)
        all_results = {}
        
        original_modulation = self.config.modulation
        original_bits_per_symbol = self.config.bits_per_symbol
        
        total_modulations = len(MODULATION_SCHEMES)
        total_snrs = len(snr_range)
        total_steps = total_modulations * total_snrs * n_iterations
        current_step = 0
        
        # Para cada modulación
        for mod_idx, modulation in enumerate(MODULATION_SCHEMES):
            # Cambiar modulación
            self.config.modulation = modulation
            self.config.bits_per_symbol = self._get_bits_per_symbol(modulation)
            
            # 🔧 IMPORTANTE: Reinicializar modulador y demodulador con la nueva modulación
            self.modulator = OFDMModulator(self.config, mode=self.mode)
            self.demodulator = OFDMDemodulator(self.config, mode=self.mode, enable_equalization=self.enable_equalization)
            self.symbol_detector = SymbolDetector(
                self.modulator.get_qam_modulator().get_constellation()
            )
            
            # Reportar inicio de modulación
            if progress_callback:
                progress = (current_step / total_steps) * 100
                progress_callback(int(progress), f"Modulación: {modulation}")
            
            # Crear callback interno para rastrear progreso
            def internal_progress_callback(inner_progress, snr, iteration):
                nonlocal current_step
                # Incrementar counter
                steps_in_this_snr = iteration
                modulation_progress = ((mod_idx * total_snrs * n_iterations + 
                                       (snr_range.tolist().index(snr) if snr in snr_range else 0) * n_iterations + 
                                       steps_in_this_snr) / total_steps) * 100
                if progress_callback:
                    msg = f"{modulation} - SNR: {snr:.1f} dB, Iter: {iteration}/{n_iterations}"
                    progress_callback(int(modulation_progress), msg)
            
            # Ejecutar barrido para esta modulación
            results = self.run_ber_sweep(
                num_bits=num_bits,
                snr_range=snr_range,
                n_iterations=n_iterations,
                progress_callback=internal_progress_callback,
                bits=bits
            )
            
            all_results[modulation] = results
            current_step += total_snrs * n_iterations
        
        # 🔧 Restaurar modulación original Y reinicializar modulador/demodulador
        self.config.modulation = original_modulation
        self.config.bits_per_symbol = original_bits_per_symbol
        self.modulator = OFDMModulator(self.config, mode=self.mode)
        self.demodulator = OFDMDemodulator(self.config, mode=self.mode, enable_equalization=self.enable_equalization)
        self.symbol_detector = SymbolDetector(
            self.modulator.get_qam_modulator().get_constellation()
        )
        
        return all_results
    
    def _get_bits_per_symbol(self, modulation):
        """Obtiene bits por símbolo para una modulación"""
        bits_map = {
            'QPSK': 2,
            '16-QAM': 4,
            '64-QAM': 6
        }
        return bits_map.get(modulation, 2)
    
    def calculate_transmission_metrics(self, num_bits):
        """
        Calcula métricas de transmisión para throughput
        
        Args:
            num_bits: Número de bits
            
        Returns:
            dict: Métricas de transmisión
        """
        bits_per_ofdm = self.config.Nc * self.config.bits_per_symbol
        num_ofdm_symbols = int(np.ceil(num_bits / bits_per_ofdm))
        duration = num_ofdm_symbols * (self.config.N + self.config.cp_length) * self.config.Ts
        throughput_mbps = (num_bits / duration) / 1e6 if duration > 0 else 0
        
        return {
            'n_ofdm_symbols': num_ofdm_symbols,
            'duration_seconds': duration,
            'throughput_mbps': throughput_mbps
        }
    
    def calculate_signal_power(self, signal):
        """
        Calcula la potencia de una señal
        
        Args:
            signal: Señal (real o compleja)
            
        Returns:
            float: Potencia media
        """
        if np.iscomplexobj(signal):
            return np.mean(np.abs(signal) ** 2)
        else:
            return np.mean(signal ** 2)
    
    def collect_papr_for_all_modulations(self, num_bits, n_simulations, snr_db=25.0, 
                                         progress_callback=None):
        """
        Colecta valores PAPR para todas las modulaciones (QPSK y 16-QAM) en OFDM y SC-FDM.
        
        PAPR es independiente de SNR, así que usa un SNR fijo (25 dB) para eficiencia.
        
        Args:
            num_bits: Número de bits por simulación
            n_simulations: Número de simulaciones para colectar PAPR
            snr_db: SNR fijo en dB (PAPR no depende de SNR)
            progress_callback: Función para reportar progreso
            
        Returns:
            dict: PAPR valores para cada {modulación}_{modo}:
                  Ejemplo: {'QPSK_OFDM': [...], 'QPSK_SC-FDM': [...], '16-QAM_OFDM': [...], ...}
        """
        from config import MODULATION_SCHEMES
        
        papr_results = {}
        
        original_modulation = self.config.modulation
        original_bits_per_symbol = self.config.bits_per_symbol
        original_sc_fdm = self.enable_sc_fdm
        
        total_configs = 4  # QPSK/16-QAM × OFDM/SC-FDM
        current_config = 0
        
        # Para cada modulación
        for modulation in ['QPSK', '16-QAM']:
            # Cambiar modulación
            self.config.modulation = modulation
            self.config.bits_per_symbol = self._get_bits_per_symbol(modulation)
            
            # Para cada modo (OFDM y SC-FDM)
            for sc_fdm_enabled in [False, True]:
                current_config += 1
                mode_label = "SC-FDM" if sc_fdm_enabled else "OFDM"
                config_label = f"{modulation}_{mode_label}"
                
                # Actualizar modo SC-FDM
                self.enable_sc_fdm = sc_fdm_enabled
                self.modulator = OFDMModulator(self.config, mode=self.mode, enable_sc_fdm=sc_fdm_enabled)
                self.demodulator = OFDMDemodulator(self.config, mode=self.mode, 
                                                  enable_equalization=self.enable_equalization,
                                                  enable_sc_fdm=sc_fdm_enabled)
                self.symbol_detector = SymbolDetector(
                    self.modulator.get_qam_modulator().get_constellation()
                )
                
                papr_values = []
                
                # Ejecutar n_simulations
                for sim in range(n_simulations):
                    # Generar bits aleatorios
                    bits = np.random.randint(0, 2, num_bits)
                    
                    # Transmitir
                    results = self.transmit(bits, snr_db=snr_db, return_time=False)
                    
                    # Extraer PAPR de la señal transmitida
                    if 'papr_no_cp' in results and 'papr_values' in results['papr_no_cp']:
                        papr_vals = results['papr_no_cp']['papr_values']
                        if papr_vals is not None:
                            papr_values.extend(papr_vals)
                    
                    # Reportar progreso
                    if progress_callback:
                        progress = ((current_config - 1) * n_simulations + sim + 1) / (total_configs * n_simulations) * 100
                        msg = f"PAPR: {config_label} - Sim {sim+1}/{n_simulations}"
                        progress_callback(int(progress), msg)
                
                # Almacenar valores PAPR para esta configuración
                papr_results[config_label] = np.array(papr_values) if papr_values else np.array([])
        
        # Restaurar configuración original
        self.config.modulation = original_modulation
        self.config.bits_per_symbol = original_bits_per_symbol
        self.enable_sc_fdm = original_sc_fdm
        self.modulator = OFDMModulator(self.config, mode=self.mode, enable_sc_fdm=original_sc_fdm)
        self.demodulator = OFDMDemodulator(self.config, mode=self.mode, 
                                         enable_equalization=self.enable_equalization,
                                         enable_sc_fdm=original_sc_fdm)
        self.symbol_detector = SymbolDetector(
            self.modulator.get_qam_modulator().get_constellation()
        )
        
        return papr_results
    
    def get_statistics(self):
        """Retorna las estadísticas del sistema"""
        return self.statistics.copy()
    
    def reset_statistics(self):
        """Reinicia las estadísticas"""
        for key in self.statistics:
            if isinstance(self.statistics[key], (int, float)):
                self.statistics[key] = 0
    
    def get_config_info(self):
        """Retorna información de la configuración actual"""
        return self.config.get_info()


class OFDMSystemManager:
    """Gestor del sistema OFDM que maneja múltiples configuraciones"""
    
    def __init__(self):
        """Inicializa el gestor del sistema"""
        self.current_system = None
        self.available_configs = {}
    
    def create_system(self, bandwidth, delta_f, modulation, cp_type):
        """
        Crea un nuevo sistema OFDM con la configuración especificada
        
        Args:
            bandwidth: Ancho de banda en MHz
            delta_f: Separación entre subportadoras en kHz
            modulation: Esquema de modulación
            cp_type: Tipo de prefijo cíclico
            
        Returns:
            OFDMSystem: Sistema OFDM creado
        """
        config = LTEConfig(bandwidth, delta_f, modulation, cp_type)
        system = OFDMSystem(config)
        self.current_system = system
        return system
    
    def get_current_system(self):
        """Retorna el sistema actual"""
        return self.current_system
    
    def update_system_snr(self, snr_db):
        """
        Actualiza el SNR del sistema actual
        
        Args:
            snr_db: SNR en dB
        """
        if self.current_system:
            self.current_system.channel.set_snr(snr_db)
    
    def get_available_presets(self):
        """Retorna los perfiles LTE disponibles"""
        from config import LTE_PROFILES
        return LTE_PROFILES

