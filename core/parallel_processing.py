"""
Parallel Processing Utilities para MIMO Multi-Antenna Systems

Paraleliza operaciones independientes por antena para mejorar performance
en sistemas con 4+ antenas TX/RX.

Operaciones paralelizables:
- OFDM modulation por antena TX
- OFDM demodulation por antena RX
- Canal por grupos de RX (opcional)

Operaciones NO paralelizables (secuenciales):
- Precoding (operación matricial pequeña, numpy ya optimizado)
- MIMO detection (requiere todas las antenas simultáneamente)
- Layer mapping/demapping (overhead > beneficio)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import time


class MIMOParallelProcessor:
    """
    Gestiona paralelización inteligente para procesamiento MIMO.
    
    Decide automáticamente si paralelizar según número de antenas
    y carga computacional. Threshold por defecto: 4 antenas.
    """
    
    def __init__(self, num_antennas, enable_parallel=True, threshold=4, max_workers=None):
        """
        Args:
            num_antennas (int): Número de antenas a procesar (TX o RX)
            enable_parallel (bool): Si False, siempre procesamiento secuencial
            threshold (int): Mínimo de antenas para activar paralelización
            max_workers (int): Máximo threads (None = min(num_antennas, 8))
        """
        self.num_antennas = num_antennas
        self.threshold = threshold
        
        # Decidir si paralelizar
        self.enable_parallel = enable_parallel and (num_antennas >= threshold)
        
        # Límite de workers (no crear más threads de los necesarios)
        if max_workers is None:
            self.max_workers = min(num_antennas, 8)  # Máximo razonable
        else:
            self.max_workers = min(max_workers, num_antennas)
        
        mode = "PARALELO" if self.enable_parallel else "SECUENCIAL"
        print(f"[ParallelProcessor] Modo: {mode} ({num_antennas} antenas, "
              f"threshold={threshold}, workers={self.max_workers if self.enable_parallel else 'N/A'})")
    
    def parallel_ofdm_modulate(self, modulator_func, freq_symbols_per_ant):
        """
        OFDM modulation en paralelo por antena TX.
        
        Cada antena TX hace su IFFT independientemente, ideal para paralelizar.
        
        Args:
            modulator_func: Función que toma freq_symbols y retorna time_signal
                           Ejemplo: lambda syms: modulator.modulate(syms)
            freq_symbols_per_ant: Lista de símbolos [ant][subcarriers]
        
        Returns:
            tx_signals: Lista de señales tiempo [ant][samples]
        
        Example:
            >>> modulator = OFDMModulator(config)
            >>> freq_syms = [syms_ant0, syms_ant1, syms_ant2, syms_ant3]
            >>> tx_sigs = processor.parallel_ofdm_modulate(
            ...     lambda s: modulator.modulate(s), freq_syms)
        """
        num_antennas = len(freq_symbols_per_ant)
        
        if not self.enable_parallel:
            # Modo secuencial
            return [modulator_func(syms) for syms in freq_symbols_per_ant]
        
        # Modo paralelo con ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit todas las tareas
            futures = {
                executor.submit(modulator_func, freq_symbols_per_ant[i]): i
                for i in range(num_antennas)
            }
            
            # Recolectar resultados en orden
            results = [None] * num_antennas
            for future in as_completed(futures):
                ant_idx = futures[future]
                try:
                    results[ant_idx] = future.result()
                except Exception as exc:
                    print(f"[ERROR] Antena TX {ant_idx} generó excepción: {exc}")
                    raise
        
        return results
    
    def parallel_ofdm_demodulate(self, demodulator_func, rx_signals_per_ant):
        """
        OFDM demodulation en paralelo por antena RX.
        
        Cada antena RX hace su FFT independientemente.
        
        Args:
            demodulator_func: Función que toma time_signal y retorna freq_symbols
                             Ejemplo: lambda sig: demodulator.demodulate(sig)
            rx_signals_per_ant: Lista de señales tiempo [ant][samples]
        
        Returns:
            freq_received: Lista de símbolos frecuencia [ant][subcarriers]
        
        Example:
            >>> demodulator = OFDMDemodulator(config)
            >>> rx_sigs = [sig_ant0, sig_ant1]
            >>> freq_rx = processor.parallel_ofdm_demodulate(
            ...     lambda s: demodulator.demodulate(s), rx_sigs)
        """
        num_antennas = len(rx_signals_per_ant)
        
        if not self.enable_parallel:
            # Modo secuencial
            return [demodulator_func(sig) for sig in rx_signals_per_ant]
        
        # Modo paralelo
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(demodulator_func, rx_signals_per_ant[i]): i
                for i in range(num_antennas)
            }
            
            results = [None] * num_antennas
            for future in as_completed(futures):
                ant_idx = futures[future]
                try:
                    results[ant_idx] = future.result()
                except Exception as exc:
                    print(f"[ERROR] Antena RX {ant_idx} generó excepción: {exc}")
                    raise
        
        return results
    
    def parallel_channel_estimation(self, estimator_func, received_per_ant, tx_pilots=None):
        """
        Estimación de canal en paralelo por antena RX.
        
        Args:
            estimator_func: Función que estima canal para una antena
            received_per_ant: Señales recibidas [ant][subcarriers]
            tx_pilots: Pilotos transmitidos (opcional)
        
        Returns:
            H_estimates: Lista de canales estimados [ant][subcarriers] o [ant][...]
        """
        num_antennas = len(received_per_ant)
        
        if not self.enable_parallel:
            return [estimator_func(rx, tx_pilots) for rx in received_per_ant]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            if tx_pilots is None:
                futures = {
                    executor.submit(estimator_func, received_per_ant[i]): i
                    for i in range(num_antennas)
                }
            else:
                futures = {
                    executor.submit(estimator_func, received_per_ant[i], tx_pilots): i
                    for i in range(num_antennas)
                }
            
            results = [None] * num_antennas
            for future in as_completed(futures):
                ant_idx = futures[future]
                results[ant_idx] = future.result()
        
        return results
    
    def benchmark_parallel_vs_sequential(self, func, data_list, iterations=5):
        """
        Compara performance entre ejecución paralela y secuencial.
        
        Args:
            func: Función a ejecutar por cada elemento
            data_list: Lista de datos de entrada
            iterations: Repeticiones para promediar
        
        Returns:
            dict: {'parallel_time': float, 'sequential_time': float, 'speedup': float}
        """
        print(f"\n[Benchmark] Comparando paralelo vs secuencial ({iterations} iteraciones)")
        
        # Benchmark secuencial
        t_seq = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = [func(data) for data in data_list]
            t_seq.append(time.perf_counter() - start)
        time_sequential = np.mean(t_seq)
        
        # Benchmark paralelo
        t_par = []
        for _ in range(iterations):
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(func, data) for data in data_list]
                _ = [f.result() for f in futures]
            t_par.append(time.perf_counter() - start)
        time_parallel = np.mean(t_par)
        
        speedup = time_sequential / time_parallel
        
        print(f"  Secuencial: {time_sequential*1000:.2f} ms")
        print(f"  Paralelo:   {time_parallel*1000:.2f} ms")
        print(f"  Speedup:    {speedup:.2f}x")
        
        return {
            'parallel_time': time_parallel,
            'sequential_time': time_sequential,
            'speedup': speedup
        }


def dummy_ofdm_process(symbols, delay_ms=10):
    """Función dummy para testing (simula IFFT/FFT)"""
    time.sleep(delay_ms / 1000)
    return np.fft.fft(symbols)


def test_parallel_processing():
    """
    Tests del MIMOParallelProcessor
    """
    print("\n=== Test MIMOParallelProcessor ===")
    
    # Test 1: Paralelización con 4 antenas (debe activarse)
    print("\n1. Test con 4 antenas (threshold=4):")
    processor4 = MIMOParallelProcessor(num_antennas=4, threshold=4)
    assert processor4.enable_parallel == True
    print("   ✓ Paralelización activada correctamente")
    
    # Test 2: Sin paralelización con 2 antenas
    print("\n2. Test con 2 antenas (threshold=4):")
    processor2 = MIMOParallelProcessor(num_antennas=2, threshold=4)
    assert processor2.enable_parallel == False
    print("   ✓ Modo secuencial activado correctamente")
    
    # Test 3: OFDM modulation paralelo
    print("\n3. Test OFDM modulation paralelo:")
    freq_symbols = [np.random.randn(64) + 1j*np.random.randn(64) for _ in range(4)]
    
    modulator_func = lambda syms: np.fft.ifft(syms)
    tx_signals = processor4.parallel_ofdm_modulate(modulator_func, freq_symbols)
    
    assert len(tx_signals) == 4
    assert all(len(sig) == 64 for sig in tx_signals)
    print("   ✓ OFDM modulation paralelo OK")
    
    # Test 4: OFDM demodulation paralelo
    print("\n4. Test OFDM demodulation paralelo:")
    demodulator_func = lambda sig: np.fft.fft(sig)
    freq_rx = processor4.parallel_ofdm_demodulate(demodulator_func, tx_signals)
    
    assert len(freq_rx) == 4
    # Verificar que FFT(IFFT(x)) ≈ x
    for i in range(4):
        assert np.allclose(freq_rx[i], freq_symbols[i], atol=1e-10)
    print("   ✓ OFDM demodulation paralelo OK")
    
    # Test 5: Modo secuencial forzado
    print("\n5. Test modo secuencial forzado:")
    processor_seq = MIMOParallelProcessor(num_antennas=8, enable_parallel=False)
    assert processor_seq.enable_parallel == False
    tx_signals_seq = processor_seq.parallel_ofdm_modulate(modulator_func, 
                                                           [np.random.randn(64) for _ in range(8)])
    assert len(tx_signals_seq) == 8
    print("   ✓ Modo secuencial forzado OK")
    
    print("\n=== Todos los tests pasaron ✓ ===\n")


if __name__ == '__main__':
    test_parallel_processing()
