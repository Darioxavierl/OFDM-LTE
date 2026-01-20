#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ventana principal del simulador OFDM con Spatial Multiplexing (LTE TM4)
Usa el core de spatial multiplexing de este proyecto (simulate_spatial_multiplexing)
"""
import sys
import os
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from pathlib import Path

# Agregar el directorio raíz al path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importar desde el proyecto actual
from config import LTEConfig
from core.ofdm_core import OFDMSimulator, simulate_spatial_multiplexing
from utils.image_processing import ImageProcessor
from Beamforming.gui.widgets import PlotWidget, MetricsPanel, ConfigInfoPanel, ImageComparisonWidget


class SimulationWorker(QThread):
    """Worker thread para ejecutar simulaciones de Spatial Multiplexing sin bloquear la GUI"""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    
    def __init__(self, ofdm_system, mode, params):
        super().__init__()
        self.ofdm_system = ofdm_system
        self.mode = mode
        self.params = params
    
    def run(self):
        """Ejecuta la simulación según el modo"""
        try:
            if self.mode == 'single':
                self._run_single_simulation()
            elif self.mode == 'sweep':
                self._run_sweep_simulation()
            elif self.mode == 'multiantenna':
                self._run_multiantenna_test()
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.finished.emit({'error': error_msg})
    
    def _run_single_simulation(self):
        """Ejecuta simulación única de Spatial Multiplexing usando método por bloques"""
        self.progress.emit(10, "Preparando datos...")
        
        if 'image_path' in self.params:
            # Transmisión de imagen
            bits, metadata = ImageProcessor.image_to_bits(self.params['image_path'])
            self.params['metadata'] = metadata
            
            print(f"\n[DEBUG Spatial Multiplexing Worker] Preparación de imagen:")
            print(f"  Imagen: {metadata['width']}x{metadata['height']}x{metadata['channels']}")
            print(f"  Bits TX: {len(bits):,}")
        else:
            # Bits aleatorios
            bits = np.random.randint(0, 2, self.params['n_bits'])
            print(f"[DEBUG Spatial Multiplexing] Bits aleatorios: {len(bits):,}")
        
        total_bits = len(bits)
        
        print(f"\n[DEBUG Spatial Multiplexing] Configuración:")
        print(f"  Tipo: {self.params['channel_type']}")
        print(f"  SNR: {self.params['snr_db']} dB")
        print(f"  Num TX: {self.params['num_tx']}")
        print(f"  Num RX: {self.params['num_rx']}")
        print(f"  Detector: {self.params.get('detector_type', 'MMSE')}")
        print(f"  Velocidad: {self.params.get('velocity_kmh', 3)} km/h")
        
        # Obtener config del sistema OFDM
        config = self.ofdm_system.config
        
        # Calcular capacidad por símbolo OFDM (249 subportadoras de datos)
        from core.modulator import QAMModulator
        qam_mod = QAMModulator(config.modulation)
        bits_per_symbol = int(np.log2(len(qam_mod.constellation)))
        data_subcarriers = 249  # Subportadoras de datos en LTE
        bits_per_ofdm = data_subcarriers * bits_per_symbol
        num_ofdm_symbols = int(np.ceil(total_bits / bits_per_ofdm))
        
        print(f"  Procesando en {num_ofdm_symbols} bloques OFDM ({bits_per_ofdm} bits/bloque)")
        print(f"  TIEMPO ESTIMADO: ~{(num_ofdm_symbols * 0.007):.1f} segundos")
        
        # Transmitir en bloques (método del test que funciona correctamente)
        all_bits_rx = []
        total_errors = 0
        all_symbols_tx = []
        all_symbols_rx = []
        
        for block_idx in range(num_ofdm_symbols):
            # Progreso cada 100 bloques para no ralentizar la GUI
            if block_idx % 100 == 0 or block_idx == num_ofdm_symbols - 1:
                progress_pct = 30 + int((block_idx / num_ofdm_symbols) * 40)
                self.progress.emit(progress_pct, f"Transmitiendo bloque {block_idx+1}/{num_ofdm_symbols}...")
            
            # Extraer bits para este bloque
            start_bit = block_idx * bits_per_ofdm
            end_bit = min(start_bit + bits_per_ofdm, total_bits)
            bits_block = bits[start_bit:end_bit]
            
            # Pad si es necesario
            if len(bits_block) < bits_per_ofdm:
                bits_block = np.concatenate([bits_block, np.zeros(bits_per_ofdm - len(bits_block), dtype=int)])
            
            # Simular transmisión de este bloque
            # Silenciar prints excepto cada 500 bloques para acelerar
            import sys
            import io
            should_print = (block_idx % 500 == 0 or block_idx == num_ofdm_symbols - 1)
            
            try:
                if not should_print:
                    # Redirigir stdout a /dev/null
                    old_stdout = sys.stdout
                    sys.stdout = io.StringIO()
                
                result_block = simulate_spatial_multiplexing(
                    bits=bits_block,
                    num_tx=self.params['num_tx'],
                    num_rx=self.params['num_rx'],
                    rank='adaptive',
                    detector_type=self.params.get('detector_type', 'MMSE'),
                    modulation=config.modulation,
                    snr_db=self.params['snr_db'],
                    config=config,
                    channel_type='rayleigh_mp',
                    itu_profile=self.params.get('itu_profile', 'Pedestrian_A'),
                    velocity_kmh=self.params.get('velocity_kmh', 3),
                    frequency_ghz=self.params.get('frequency_ghz', 2.0),
                    enable_csi_feedback=True
                )
                
                if not should_print:
                    # Restaurar stdout
                    sys.stdout = old_stdout
                
                # Acumular resultados
                all_bits_rx.append(result_block['bits_received_array'])
                total_errors += result_block['bit_errors']
                
                # Guardar algunos símbolos para constelación (primeros 1000)
                if len(all_symbols_tx) < 1000:
                    symbols_tx_block = qam_mod.bits_to_symbols(bits_block[:len(bits_block)])
                    symbols_rx_block = qam_mod.bits_to_symbols(result_block['bits_received_array'][:len(bits_block)])
                    all_symbols_tx.extend(symbols_tx_block[:1000 - len(all_symbols_tx)])
                    all_symbols_rx.extend(symbols_rx_block[:1000 - len(all_symbols_rx)])
                
                # Guardar rank del último bloque
                if block_idx == num_ofdm_symbols - 1:
                    rank_used = result_block.get('rank_used', result_block.get('rank', 1))
                    
            except Exception as e:
                if not should_print:
                    sys.stdout = old_stdout  # Restaurar en caso de error
                print(f"  [ERROR] Bloque {block_idx+1}: {e}")
                # Fallback: bits con errores
                all_bits_rx.append(np.random.randint(0, 2, len(bits_block)))
                total_errors += len(bits_block) // 2
        
        self.progress.emit(70, "Decodificando...")
        
        # Concatenar todos los bits recibidos
        bits_rx_full = np.concatenate(all_bits_rx)[:total_bits]
        ber = total_errors / total_bits if total_bits > 0 else 0.0
        
        # Crear resultados consolidados
        results = {
            'bits_received_array': bits_rx_full,
            'ber': ber,
            'bit_errors': total_errors,
            'rank_used': rank_used if 'rank_used' in locals() else 1,
            'symbols_tx': np.array(all_symbols_tx),
            'symbols_rx': np.array(all_symbols_rx)
        }
        
        print(f"\n[DEBUG Spatial Multiplexing] Resultados:")
        print(f"  BER: {results['ber']:.6f}")
        print(f"  Errores: {results['bit_errors']:,}")
        print(f"  Rank usado: {results.get('rank_used', 'N/A')}")
        print(f"  Símbolos para constelación: {len(results['symbols_tx'])}")
        
        # Reconstruir imagen si es necesario
        if 'metadata' in self.params:
            self.progress.emit(90, "Reconstruyendo imagen...")
            
            reconstructed_img = ImageProcessor.bits_to_image(
                results['bits_received_array'], 
                self.params['metadata']
            )
            results['reconstructed_image'] = reconstructed_img
            results['metadata'] = self.params['metadata']
            
            print(f"  Imagen reconstruida: {reconstructed_img.size}")
        
        self.progress.emit(100, "Completado!")
        self.finished.emit(results)
    
    def _run_sweep_simulation(self):
        """
        Ejecuta barrido completo de SNR para Spatial Multiplexing.
        
        Configuraciones:
        - Modulación: 64-QAM fija (más usada en LTE)
        - 4 configuraciones de antenas con 2 detectores cada una:
          * 2×2 (MMSE/SIC)
          * 4×2 (MMSE/SIC)
          * Mejora RX: 2×2 BF, 4×2 BF
          * Massive MIMO: 8×4 BF
        
        Genera 1 gráfica potente con 6-7 curvas diferenciadas por estilo.
        """
        self.progress.emit(5, "Preparando datos...")
        
        # Usar bits de imagen si existen, sino aleatorios
        if 'bits' in self.params and 'metadata' in self.params:
            bits_to_sweep = self.params['bits']
            metadata = self.params['metadata']
            print(f"\n{'='*70}")
            print(f"[DEBUG Spatial Multiplexing] Barrido SNR - IMAGEN CARGADA")
            print(f"{'='*70}")
            print(f"  Dimensiones imagen: {metadata['width']}×{metadata['height']}")
            print(f"  Total bits de imagen: {len(bits_to_sweep):,} bits")
            print(f"  Tamaño datos: {len(bits_to_sweep) / 8 / 1024:.2f} KB")
            print(f"  ✓ Se enviarán TODOS los bits de la imagen en cada SNR")
            print(f"{'='*70}")
        else:
            bits_to_sweep = np.random.randint(0, 2, 50000)
            print(f"\n{'='*70}")
            print(f"[DEBUG Spatial Multiplexing] Barrido SNR - SIN IMAGEN (usando bits aleatorios)")
            print(f"{'='*70}")
            print(f"  Total bits aleatorios: {len(bits_to_sweep):,} bits")
            print(f"  ⚠ Para usar imagen real, cargue una imagen primero")
            print(f"{'='*70}")
        
        # Configuraciones a comparar (TM4 Spatial Multiplexing)
        # Formato: (nombre, num_tx, num_rx, detector_type, estilo_línea)
        configurations = [
            ('2×2 IRC', 2, 2, 'MMSE', {'color': '#2196F3', 'linestyle': '-', 'linewidth': 2, 'marker': 'o'}),
            ('2×2 SIC', 2, 2, 'SIC', {'color': '#2196F3', 'linestyle': '--', 'linewidth': 2, 'marker': 's'}),
            ('4×2 IRC', 4, 2, 'MMSE', {'color': '#4CAF50', 'linestyle': '-', 'linewidth': 2.5, 'marker': 'o'}),
            ('4×2 SIC', 4, 2, 'SIC', {'color': '#4CAF50', 'linestyle': '--', 'linewidth': 2.5, 'marker': 's'}),
            ('4×4 IRC', 4, 4, 'MMSE', {'color': '#FF9800', 'linestyle': '-', 'linewidth': 2.5, 'marker': 'o'}),
            ('4×4 SIC', 4, 4, 'SIC', {'color': '#FF9800', 'linestyle': '--', 'linewidth': 2.5, 'marker': 's'}),
        ]
        
        snr_range = self.params['snr_range']
        n_iterations = self.params['n_iterations']
        
        print(f"  Modulación: 64-QAM (fija)")
        print(f"  Configuraciones: {len(configurations)} (4 antenas × 2 detectores)")
        print(f"  Rango SNR: {snr_range[0]:.1f} a {snr_range[-1]:.1f} dB")
        print(f"  Iteraciones: {n_iterations}")
        
        # Calcular pasos totales
        total_steps = len(configurations) * len(snr_range) * n_iterations
        current_step = 0
        
        # Resultados
        all_results = {}
        
        self.progress.emit(10, "Iniciando barrido de Spatial Multiplexing...")
        
        # Crear sistema con 64-QAM
        config = LTEConfig(
            bandwidth=float(self.params['bandwidth']),
            modulation='64-QAM',
            delta_f=float(self.params['delta_f']),
            cp_type=self.params['cp_type']
        )
        
        # Iterar sobre configuraciones
        for config_name, num_tx, num_rx, detector_type, style in configurations:
            print(f"\n=== {config_name} ===")
            
            # Crear sistema OFDM para esta configuración
            temp_system = OFDMSimulator(
                config=config,
                channel_type=self.params['channel_type'],
                mode='lte',
                enable_equalization=True,
                num_channels=1,
                itu_profile=self.params.get('itu_profile'),
                frequency_ghz=self.params.get('frequency_ghz', 2.0),
                velocity_kmh=self.params.get('velocity_kmh', 3.0)
            )
            
            ber_results = []
            snr_values_list = []
            
            # Iterar sobre SNR
            for snr_db in snr_range:
                ber_list = []
                
                # Múltiples iteraciones para promediar
                for iter_idx in range(n_iterations):
                    current_step += 1
                    progress_pct = int(10 + (current_step / total_steps) * 85)
                    self.progress.emit(progress_pct,
                                     f"{config_name} | SNR={snr_db:.1f}dB | {iter_idx+1}/{n_iterations}")
                    
                    # Spatial Multiplexing con rank adaptativo
                    result = simulate_spatial_multiplexing(
                        bits=bits_to_sweep,
                        num_tx=num_tx,
                        num_rx=num_rx,
                        rank='adaptive',
                        detector_type=detector_type,
                        modulation=config.modulation,
                        snr_db=snr_db,
                        config=config,
                        channel_type='rayleigh_mp',
                        itu_profile=self.params.get('itu_profile', 'Pedestrian_A'),
                        velocity_kmh=self.params.get('velocity_kmh', 3),
                        frequency_ghz=self.params.get('frequency_ghz', 2.0),
                        enable_csi_feedback=True
                    )
                    
                    ber_list.append(result['ber'])
                
                # Promedio de iteraciones
                avg_ber = np.mean(ber_list)
                ber_results.append(avg_ber)
                snr_values_list.append(snr_db)
            
            # Guardar resultados para esta configuración
            all_results[config_name] = {
                'snr_values': np.array(snr_values_list),
                'ber_values': np.array(ber_results),
                'style': style,
                'num_tx': num_tx,
                'num_rx': num_rx,
                'detector': detector_type
            }
            
            print(f"  BER: {min(ber_results):.2e} (mejor) a {max(ber_results):.2e} (peor)")
        
        self.progress.emit(95, "Finalizando simulación...")
        
        # Emitir resultados
        self.finished.emit({
            'type': 'sweep_spatial_multiplexing',
            'results': all_results,
            'configurations': [c[0] for c in configurations]
        })
    
    def _run_multiantenna_test(self):
        """
        Ejecuta prueba multiantena completa con Spatial Multiplexing TM4:
        
        3 modulaciones × 4 configuraciones × 2 detectores = 24 pruebas:
        - QPSK: 2×2, 4×2, 4×4, 8×4 (cada uno con MMSE y SIC)
        - 16-QAM: 2×2, 4×2, 4×4, 8×4 (cada uno con MMSE y SIC)
        - 64-QAM: 2×2, 4×2, 4×4, 8×4 (cada uno con MMSE y SIC)
        
        Genera 3 gráficos BER vs SNR (uno por modulación) con 8 líneas cada uno.
        """
        self.progress.emit(5, "Preparando prueba multiantena...")
        
        # Cargar imagen
        if 'image_path' not in self.params:
            self.finished.emit({'error': 'Se requiere una imagen para la prueba multiantena'})
            return
        
        bits, metadata = ImageProcessor.image_to_bits(self.params['image_path'])
        
        print(f"\n[DEBUG Spatial Multiplexing] Prueba Multiantena:")
        print(f"  Imagen: {metadata['width']}×{metadata['height']}×{metadata['channels']}")
        print(f"  Bits: {len(bits):,}")
        print(f"  === Parámetros de simulación ===")
        print(f"  Ancho de banda: {self.params['bandwidth']} MHz")
        print(f"  Delta-f: {self.params['delta_f']} kHz")
        print(f"  CP: {self.params['cp_type']}")
        print(f"  SNR: {self.params['snr_db']} dB")
        print(f"  Canal: {self.params['channel_type']}")
        print(f"  ITU Profile: {self.params.get('itu_profile', 'NO ESPECIFICADO')}")
        print(f"  Frequency: {self.params.get('frequency_ghz', 'NO ESPECIFICADO')} GHz")
        print(f"  Velocity: {self.params.get('velocity_kmh', 'NO ESPECIFICADO')} km/h")
        
        # Configuraciones a comparar (6 configuraciones totales: 3 antenas × 2 detectores)
        # Organizadas para mostrar: Fila 1 (IRC): 2×2, 4×2, 4×4 | Fila 2 (SIC): 2×2, 4×2, 4×4
        test_configs = [
            {'name': '2×2 IRC', 'num_tx': 2, 'num_rx': 2, 'detector': 'MMSE'},
            {'name': '4×2 IRC', 'num_tx': 4, 'num_rx': 2, 'detector': 'MMSE'},
            {'name': '4×4 IRC', 'num_tx': 4, 'num_rx': 4, 'detector': 'MMSE'},
            {'name': '2×2 SIC', 'num_tx': 2, 'num_rx': 2, 'detector': 'SIC'},
            {'name': '4×2 SIC', 'num_tx': 4, 'num_rx': 2, 'detector': 'SIC'},
            {'name': '4×4 SIC', 'num_tx': 4, 'num_rx': 4, 'detector': 'SIC'},
        ]
        
        results_list = []
        images_list = []
        
        total_steps = len(test_configs)
        
        # Crear configuración LTE
        from config import LTEConfig
        config = LTEConfig(
            bandwidth=float(self.params['bandwidth']),
            modulation=self.params['modulation'],
            delta_f=float(self.params['delta_f']),
            cp_type=self.params['cp_type']
        )
        
        # Calcular capacidad por símbolo OFDM
        from core.modulator import QAMModulator
        qam_mod = QAMModulator(config.modulation)
        bits_per_symbol = int(np.log2(len(qam_mod.constellation)))
        data_subcarriers = 249
        bits_per_ofdm = data_subcarriers * bits_per_symbol
        num_ofdm_symbols = int(np.ceil(len(bits) / bits_per_ofdm))
        total_bits = len(bits)
        
        print(f"  Bloques OFDM: {num_ofdm_symbols} ({bits_per_ofdm} bits/bloque)")
        print(f"  TOTAL DE SIMULACIONES: {len(test_configs)} configs × {num_ofdm_symbols} bloques = {len(test_configs) * num_ofdm_symbols:,} llamadas")
        print(f"  TIEMPO ESTIMADO: ~{(len(test_configs) * num_ofdm_symbols * 0.007):.0f} segundos (~{(len(test_configs) * num_ofdm_symbols * 0.007 / 60):.1f} minutos)")
        
        # Calcular pasos totales para progreso preciso
        total_blocks = len(test_configs) * num_ofdm_symbols
        blocks_processed = 0
        
        # Iterar sobre configuraciones
        for idx, test_config in enumerate(test_configs):
            print(f"\n  === {test_config['name']} ({test_config['num_tx']}×{test_config['num_rx']}) ===")
            
            num_tx = test_config['num_tx']
            num_rx = test_config['num_rx']
            detector_type = test_config['detector']
            
            # Transmitir en bloques (método del test)
            all_bits_rx = []
            total_errors = 0
            
            for block_idx in range(num_ofdm_symbols):
                # Actualizar progreso cada 100 bloques para no ralentizar
                if block_idx % 100 == 0 or block_idx == num_ofdm_symbols - 1:
                    progress_pct = int(10 + (blocks_processed / total_blocks) * 85)
                    self.progress.emit(progress_pct, 
                                     f"{test_config['name']}: {block_idx+1}/{num_ofdm_symbols} bloques")
                
                # Extraer bits del bloque
                start_bit = block_idx * bits_per_ofdm
                end_bit = min(start_bit + bits_per_ofdm, total_bits)
                bits_block = bits[start_bit:end_bit]
                
                # Pad si necesario
                if len(bits_block) < bits_per_ofdm:
                    bits_block = np.concatenate([bits_block, np.zeros(bits_per_ofdm - len(bits_block), dtype=int)])
                
                # Simular bloque (silenciar prints excepto cada 500 bloques)
                import sys
                import io
                should_print = (block_idx % 500 == 0 or block_idx == num_ofdm_symbols - 1)
                
                try:
                    if not should_print:
                        old_stdout = sys.stdout
                        sys.stdout = io.StringIO()
                    
                    result_block = simulate_spatial_multiplexing(
                        bits=bits_block,
                        num_tx=num_tx,
                        num_rx=num_rx,
                        rank='adaptive',
                        detector_type=detector_type,
                        modulation=config.modulation,
                        snr_db=self.params['snr_db'],
                        config=config,
                        channel_type='rayleigh_mp',
                        itu_profile=self.params.get('itu_profile', 'Pedestrian_A'),
                        velocity_kmh=self.params.get('velocity_kmh', 3),
                        frequency_ghz=self.params.get('frequency_ghz', 2.0),
                        enable_csi_feedback=True
                    )
                    
                    if not should_print:
                        sys.stdout = old_stdout
                    
                    all_bits_rx.append(result_block['bits_received_array'])
                    total_errors += result_block['bit_errors']
                    
                except Exception as e:
                    if not should_print:
                        sys.stdout = old_stdout
                    print(f"    [ERROR] Bloque {block_idx+1}: {e}")
                    all_bits_rx.append(np.random.randint(0, 2, len(bits_block)))
                    total_errors += len(bits_block) // 2
                
                blocks_processed += 1
            
            # Concatenar bits recibidos
            bits_rx_full = np.concatenate(all_bits_rx)[:total_bits]
            ber = total_errors / total_bits if total_bits > 0 else 0.0
            
            print(f"    BER: {ber:.6f} ({total_errors:,} errores)")
            
            # Crear resultado consolidado
            result = {
                'bits_received_array': bits_rx_full,
                'ber': ber,
                'bit_errors': total_errors
            }
            
            # Reconstruir imagen
            img_reconstructed = ImageProcessor.bits_to_image(
                result['bits_received_array'],
                metadata
            )
            
            # Calcular PSNR
            psnr = ImageProcessor.calculate_psnr_bits(bits, result['bits_received_array'])
            
            results_list.append({
                'name': test_config['name'],
                'num_tx': num_tx,
                'num_rx': num_rx,
                'detector': detector_type,
                'ber': result['ber'],
                'bit_errors': result['bit_errors'],
                'psnr': psnr,
                'rank_used': result.get('rank_used', 0)
            })
            
            images_list.append(img_reconstructed)
            
            print(f"    BER: {result['ber']:.6f}, Errores: {result['bit_errors']:,}, PSNR: {psnr:.1f} dB, Rank: {result.get('rank_used', 'N/A')}")
        
        self.progress.emit(95, "Generando imagen comparativa...")
        
        # Emitir resultados
        self.finished.emit({
            'type': 'multiantenna_test',
            'results': results_list,
            'images': images_list,
            'metadata': metadata,
            'snr_db': self.params['snr_db'],
            'channel_type': self.params['channel_type'],
            'velocity_kmh': self.params.get('velocity_kmh', 3)
        })


class SpatialMultiplexingGUI(QMainWindow):
    """Ventana principal del simulador OFDM con Spatial Multiplexing (LTE TM4)"""
    
    def __init__(self):
        super().__init__()
        self.ofdm_system = None
        self.current_image_path = None
        self.current_num_tx = 2
        self.current_num_rx = 2
        self.current_detector = 'MMSE'  # Internamente MMSE, se muestra como IRC
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        """Inicializa la interfaz de usuario"""
        self.setWindowTitle("Simulador OFDM-LTE (Spatial Multiplexing - TM4)")
        self.setGeometry(100, 100, 1600, 900)
        
        # Widget central con splitter horizontal de 3 paneles
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Crear splitter principal
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # PANEL 1: Control panel (izquierda)
        self.control_panel = self.create_control_panel()
        self.main_splitter.addWidget(self.control_panel)
        
        # PANEL 2: Info panel (centro)
        self.info_panel = ConfigInfoPanel()
        self.main_splitter.addWidget(self.info_panel)
        
        # PANEL 3: Results panel (derecha)
        self.results_panel = self.create_results_panel()
        self.main_splitter.addWidget(self.results_panel)
        
        # Establecer proporciones iniciales del splitter
        self.main_splitter.setSizes([400, 400, 800])
        
        # Barra de estado
        self.statusBar().showMessage("Listo - Spatial Multiplexing (TM4)")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    def create_control_panel(self):
        """Crea el panel de control (izquierda)"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Grupos de parámetros
        layout.addWidget(self.create_lte_parameters_group())
        layout.addWidget(self.create_simulation_parameters_group())
        layout.addWidget(self.create_image_group())
        layout.addWidget(self.create_simulation_buttons())
        
        layout.addStretch()
        
        # Establecer scroll area
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        return scroll
    
    def create_lte_parameters_group(self):
        """Crea grupo de parámetros LTE"""
        group = QGroupBox("Parámetros LTE")
        layout = QGridLayout()
        
        # Modulación
        layout.addWidget(QLabel("Modulación:"), 0, 0)
        self.modulation_combo = QComboBox()
        self.modulation_combo.addItems(['QPSK', '16-QAM', '64-QAM'])
        self.modulation_combo.setCurrentIndex(2)  # Default: 64-QAM (más desafiante)
        self.modulation_combo.currentTextChanged.connect(self.update_config)
        layout.addWidget(self.modulation_combo, 0, 1)
        
        # Ancho de banda
        layout.addWidget(QLabel("Ancho de banda (MHz):"), 1, 0)
        self.bandwidth_combo = QComboBox()
        self.bandwidth_combo.addItems(['1.25', '2.5', '5', '10', '15', '20'])
        self.bandwidth_combo.setCurrentText('10')  # Default 10 MHz para MIMO
        self.bandwidth_combo.currentTextChanged.connect(self.update_config)
        layout.addWidget(self.bandwidth_combo, 1, 1)
        
        # Separación subportadoras
        layout.addWidget(QLabel("Δf (kHz):"), 2, 0)
        self.delta_f_combo = QComboBox()
        self.delta_f_combo.addItems(['15.0', '7.5'])
        self.delta_f_combo.currentTextChanged.connect(self.update_config)
        layout.addWidget(self.delta_f_combo, 2, 1)
        
        # Tipo de CP
        layout.addWidget(QLabel("Prefijo Cíclico:"), 3, 0)
        self.cp_combo = QComboBox()
        self.cp_combo.addItems(['normal', 'extended'])
        self.cp_combo.currentTextChanged.connect(self.update_config)
        layout.addWidget(self.cp_combo, 3, 1)
        
        group.setLayout(layout)
        return group
    
    def create_simulation_parameters_group(self):
        """Crea grupo de parámetros de simulación"""
        group = QGroupBox("Parámetros de Simulación")
        layout = QGridLayout()
        
        # SNR para simulación única
        layout.addWidget(QLabel("SNR (dB):"), 0, 0)
        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(-10, 40)
        self.snr_spin.setValue(30.0)  # Default 30 dB para Spatial Multiplexing (mejor para configs complejas)
        self.snr_spin.setSingleStep(0.5)
        layout.addWidget(self.snr_spin, 0, 1)
        
        # Rango SNR para barrido
        layout.addWidget(QLabel("SNR Inicio (dB):"), 1, 0)
        self.snr_start_spin = QDoubleSpinBox()
        self.snr_start_spin.setRange(-10, 40)
        self.snr_start_spin.setValue(10.0)
        layout.addWidget(self.snr_start_spin, 1, 1)
        
        layout.addWidget(QLabel("SNR Fin (dB):"), 2, 0)
        self.snr_end_spin = QDoubleSpinBox()
        self.snr_end_spin.setRange(-10, 40)
        self.snr_end_spin.setValue(35.0)
        layout.addWidget(self.snr_end_spin, 2, 1)
        
        layout.addWidget(QLabel("SNR Paso (dB):"), 3, 0)
        self.snr_step_spin = QDoubleSpinBox()
        self.snr_step_spin.setRange(0.1, 10)
        self.snr_step_spin.setValue(2.0)
        layout.addWidget(self.snr_step_spin, 3, 1)
        
        # Iteraciones
        layout.addWidget(QLabel("Iteraciones:"), 4, 0)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 100)
        self.iterations_spin.setValue(3)  # 3 iteraciones por defecto para promediar resultados
        layout.addWidget(self.iterations_spin, 4, 1)
        
        # Número de transmisores
        layout.addWidget(QLabel("Num. Transmisores:"), 5, 0)
        self.num_tx_combo = QComboBox()
        self.num_tx_combo.addItems(['2', '4', '8'])
        self.num_tx_combo.setCurrentText('2')
        self.num_tx_combo.currentTextChanged.connect(self.update_config)
        layout.addWidget(self.num_tx_combo, 5, 1)
        
        # Número de receptores
        layout.addWidget(QLabel("Num. Receptores:"), 6, 0)
        self.num_rx_combo = QComboBox()
        self.num_rx_combo.addItems(['2', '4'])
        self.num_rx_combo.setCurrentText('2')
        self.num_rx_combo.currentTextChanged.connect(self.update_config)
        layout.addWidget(self.num_rx_combo, 6, 1)
        
        # Detector MIMO
        layout.addWidget(QLabel("Detector MIMO:"), 7, 0)
        self.detector_combo = QComboBox()
        self.detector_combo.addItems(['IRC', 'SIC'])
        self.detector_combo.setCurrentText('IRC')
        self.detector_combo.currentTextChanged.connect(self.update_config)
        layout.addWidget(self.detector_combo, 7, 1)
        
        # Label informativo
        info_label = QLabel("(Spatial Multiplexing TM4 con rank adaptativo)")
        info_label.setStyleSheet("color: #666; font-style: italic; font-size: 9pt;")
        layout.addWidget(info_label, 8, 0, 1, 2)
        
        # SNR recomendado por configuración
        snr_help = QLabel("💡 SNR recomendado: 2×2→20-25dB, 4×2→25-30dB, 4×4→30-35dB")
        snr_help.setStyleSheet("color: #0066cc; font-size: 9pt; padding: 5px;")
        snr_help.setWordWrap(True)
        layout.addWidget(snr_help, 9, 0, 1, 2)
        
        # Tipo de canal (SOLO MULTIPATH)
        layout.addWidget(QLabel("Tipo de Canal:"), 10, 0)
        self.channel_type_combo = QComboBox()
        self.channel_type_combo.addItems(['Rayleigh Multitrayecto'])
        self.channel_type_combo.currentTextChanged.connect(self.on_channel_type_changed)
        layout.addWidget(self.channel_type_combo, 10, 1)
        
        # Perfil ITU
        layout.addWidget(QLabel("Perfil ITU-R M.1225:"), 11, 0)
        self.itu_profile_combo = QComboBox()
        self.itu_profile_combo.addItems([
            'Pedestrian_A',
            'Pedestrian_B',
            'Vehicular_A',
            'Vehicular_B',
            'Typical_Urban',
            'Rural_Area'
        ])
        self.itu_profile_combo.setCurrentText('Pedestrian_A')  # Default para Spatial Multiplexing
        self.itu_profile_combo.currentTextChanged.connect(self.on_itu_profile_changed)
        self.itu_profile_combo.setEnabled(True)  # Siempre habilitado (solo multipath)
        layout.addWidget(self.itu_profile_combo, 11, 1)
        
        # Frecuencia portadora
        layout.addWidget(QLabel("Frecuencia (GHz):"), 12, 0)
        self.frequency_spin = QDoubleSpinBox()
        self.frequency_spin.setRange(0.5, 10.0)
        self.frequency_spin.setValue(2.0)
        self.frequency_spin.setSingleStep(0.1)
        self.frequency_spin.valueChanged.connect(self.on_frequency_changed)
        self.frequency_spin.setEnabled(True)  # Siempre habilitado
        layout.addWidget(self.frequency_spin, 12, 1)
        
        # Velocidad
        layout.addWidget(QLabel("Velocidad (km/h):"), 13, 0)
        self.velocity_spin = QSpinBox()
        self.velocity_spin.setRange(0, 500)
        self.velocity_spin.setValue(3)  # Default: 3 km/h (pedestrian)
        self.velocity_spin.valueChanged.connect(self.on_velocity_changed)
        self.velocity_spin.setEnabled(True)  # Siempre habilitado
        layout.addWidget(self.velocity_spin, 13, 1)
        
        group.setLayout(layout)
        return group
    
    def create_image_group(self):
        """Crea grupo para transmisión de imagen"""
        group = QGroupBox("Transmisión de Imagen")
        layout = QVBoxLayout()
        
        self.load_image_btn = QPushButton("Cargar Imagen")
        self.load_image_btn.clicked.connect(self.load_image)
        layout.addWidget(self.load_image_btn)
        
        self.image_label = QLabel("No hay imagen cargada")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(150)
        self.image_label.setStyleSheet("border: 1px solid gray")
        layout.addWidget(self.image_label)
        
        group.setLayout(layout)
        return group
    
    def create_simulation_buttons(self):
        """Crea botones de simulación"""
        group = QGroupBox("Simulación")
        layout = QVBoxLayout()
        
        self.single_sim_btn = QPushButton("Simulación Única")
        self.single_sim_btn.clicked.connect(self.run_single_simulation)
        layout.addWidget(self.single_sim_btn)
        
        self.sweep_sim_btn = QPushButton("Barrido en SNR")
        self.sweep_sim_btn.clicked.connect(self.run_sweep_simulation)
        layout.addWidget(self.sweep_sim_btn)
        
        self.multiantenna_btn = QPushButton("Prueba Multiantena")
        self.multiantenna_btn.clicked.connect(self.run_multiantenna_test)
        self.multiantenna_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        layout.addWidget(self.multiantenna_btn)
        
        group.setLayout(layout)
        return group
    
    def create_results_panel(self):
        '''Crea el panel de resultados (derecha)'''
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Tabs para diferentes resultados
        self.results_tabs = QTabWidget()
        
        # Tab 1: Transmisión/Recepción (Imágenes)
        self.image_comparison = ImageComparisonWidget()
        self.results_tabs.addTab(self.image_comparison, 'Transmisión/Recepción')
        
        # Tab 2: Constelación
        self.constellation_plot = PlotWidget(title='Constelación')
        self.results_tabs.addTab(self.constellation_plot, 'Constelación')
        
        # Tab 3: Curvas BER
        self.ber_plot = PlotWidget(title='BER vs SNR')
        self.results_tabs.addTab(self.ber_plot, 'Curvas BER')
        
        # Tab 4: Comparación MIMO
        self.mimo_comparison_plot = PlotWidget(title='Prueba Multiantena')
        self.results_tabs.addTab(self.mimo_comparison_plot, 'Prueba Multiantena')
        
        # Tab 5: Métricas
        self.metrics_panel = MetricsPanel()
        self.results_tabs.addTab(self.metrics_panel, 'Métricas')
        
        layout.addWidget(self.results_tabs)
        
        return panel
    
    def update_config(self):
        '''Actualiza la configuración del sistema Spatial Multiplexing'''
        try:
            bandwidth = float(self.bandwidth_combo.currentText())
            delta_f_khz = float(self.delta_f_combo.currentText())
            modulation = self.modulation_combo.currentText()
            cp_type = self.cp_combo.currentText()
            num_tx = int(self.num_tx_combo.currentText())
            num_rx = int(self.num_rx_combo.currentText())
            detector_type = self.detector_combo.currentText()
            
            # Mapear IRC a MMSE (IRC es solo el nombre visual)
            if detector_type == 'IRC':
                detector_type = 'MMSE'
            
            # Validar configuraciones válidas para TM4 Spatial Multiplexing
            valid_configs = [(2,2), (4,2), (4,4)]  # Sin 8×4
            if (num_tx, num_rx) not in valid_configs:
                QMessageBox.warning(
                    self, 
                    'Configuración Inválida',
                    f'La configuración {num_tx}×{num_rx} no es válida para TM4 Spatial Multiplexing.\n\n'
                    f'Configuraciones válidas: 2×2, 4×2, 4×4'
                )
                # Restaurar a configuración válida anterior o default
                self.num_tx_combo.setCurrentText('2')
                self.num_rx_combo.setCurrentText('2')
                return
            
            # Obtener parámetros de canal (siempre rayleigh_mp)
            itu_profile = self.itu_profile_combo.currentText()
            frequency_ghz = self.frequency_spin.value()
            velocity_kmh = self.velocity_spin.value()
            
            # Crear configuración LTE
            config = LTEConfig(
                bandwidth=bandwidth,
                delta_f=delta_f_khz,  # Ya está en kHz
                modulation=modulation,
                cp_type=cp_type
            )
            
            # Crear sistema OFDM
            self.ofdm_system = OFDMSimulator(
                config=config,
                channel_type='rayleigh_mp',
                mode='lte',
                enable_equalization=True,
                num_channels=1,
                itu_profile=itu_profile,
                frequency_ghz=frequency_ghz,
                velocity_kmh=velocity_kmh
            )
            
            # Guardar configuración actual
            self.current_num_tx = num_tx
            self.current_num_rx = num_rx
            self.current_detector = detector_type
            
            # Actualizar panel de información
            config_info = config.get_info()
            config_info[' Spatial Multiplexing (TM4) '] = ''
            config_info['Num. Transmisores'] = f'{num_tx} TX'
            config_info['Num. Receptores'] = f'{num_rx} RX'
            # Mostrar IRC en lugar de MMSE visualmente
            detector_display = 'IRC' if detector_type == 'MMSE' else detector_type
            config_info['Detector MIMO'] = detector_display
            config_info['Rank'] = 'Adaptativo (1-4 según canal)'
            config_info['Max Multiplexing Gain'] = f'{min(num_tx, num_rx)}x'
            config_info[' Canal Multipath '] = ''
            config_info['Perfil ITU'] = itu_profile
            config_info['Frecuencia'] = f'{frequency_ghz} GHz'
            config_info['Velocidad'] = f'{velocity_kmh} km/h'
            
            self.info_panel.update_config(config_info)
            
            self.statusBar().showMessage(
                f'Config: {num_tx}×{num_rx} Spatial Multiplexing, {detector_display}, {bandwidth} MHz, {modulation}, Rayleigh MP'
            )
            
        except Exception as e:
            import traceback
            error_msg = f'Error al actualizar configuración:\n{str(e)}\n\n{traceback.format_exc()}'
            QMessageBox.critical(self, 'Error', error_msg)
    
    def on_channel_type_changed(self):
        """Maneja el cambio de tipo de canal (siempre multipath en Spatial Multiplexing)"""
        # En spatial multiplexing siempre usamos multipath, todos los controles están habilitados
        self.update_config()
    
    def on_itu_profile_changed(self):
        """Maneja el cambio de perfil ITU"""
        self.update_config()
    
    def on_frequency_changed(self):
        """Maneja el cambio de frecuencia"""
        self.update_config()
    
    def on_velocity_changed(self):
        """Maneja el cambio de velocidad"""
        self.update_config()
    
    def load_image(self):
        '''Carga una imagen para transmitir'''
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Seleccionar Imagen', '',
            'Imágenes (*.png *.jpg *.jpeg *.bmp)'
        )
        
        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
            self.statusBar().showMessage(f'Imagen cargada: {os.path.basename(file_path)}')
    
    def run_single_simulation(self):
        """Ejecuta simulación única de Spatial Multiplexing"""
        if not self.ofdm_system:
            self.update_config()
        
        if not self.current_image_path:
            QMessageBox.warning(self, 'Advertencia', 
                              'Por favor carga una imagen para realizar la simulación')
            return
        
        # Preparar parámetros (solo multipath en spatial multiplexing)
        params = {
            'snr_db': self.snr_spin.value(),
            'image_path': self.current_image_path,
            'num_tx': self.current_num_tx,
            'num_rx': self.current_num_rx,
            'detector_type': self.current_detector,
            'channel_type': 'rayleigh_mp',
            'velocity_kmh': self.velocity_spin.value()
        }
        
        # Deshabilitar botones
        self.set_buttons_enabled(False)
        
        # Crear y ejecutar worker
        self.worker = SimulationWorker(self.ofdm_system, 'single', params)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_single_simulation_finished)
        self.worker.start()
    
    def run_sweep_simulation(self):
        """Ejecuta barrido de SNR para Spatial Multiplexing"""
        if not self.ofdm_system:
            self.update_config()
        
        print(f"\n[DEBUG] run_sweep_simulation() - Estado imagen:")
        print(f"  self.current_image_path = {self.current_image_path}")
        
        # Preparar parámetros
        snr_start = self.snr_start_spin.value()
        snr_end = self.snr_end_spin.value()
        snr_step = self.snr_step_spin.value()
        
        params = {
            'snr_range': np.arange(snr_start, snr_end + snr_step, snr_step),
            'n_iterations': self.iterations_spin.value(),
            'channel_type': 'rayleigh_mp',  # Solo multipath en spatial multiplexing
            'bandwidth': self.bandwidth_combo.currentText(),
            'delta_f': self.delta_f_combo.currentText(),
            'cp_type': self.cp_combo.currentText(),
            'itu_profile': self.itu_profile_combo.currentText(),
            'frequency_ghz': self.frequency_spin.value(),
            'velocity_kmh': self.velocity_spin.value()
        }
        
        # Cargar imagen para obtener bits (si existe)
        if self.current_image_path:
            try:
                bits, metadata = ImageProcessor.image_to_bits(self.current_image_path)
                params['bits'] = bits
                params['metadata'] = metadata
                print(f"[INFO] ✓ Imagen cargada exitosamente en params:")
                print(f"  - Ruta: {self.current_image_path}")
                print(f"  - Bits: {len(bits):,}")
                print(f"  - Dimensiones: {metadata['width']}×{metadata['height']}")
                print(f"  - params['bits'] agregado: {'bits' in params}")
                print(f"  - params['metadata'] agregado: {'metadata' in params}")
            except Exception as e:
                print(f"[WARNING] No se pudo cargar imagen, usando bits aleatorios: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[WARNING] No hay imagen cargada (self.current_image_path está vacío)")
            print(f"  → El barrido usará bits aleatorios")
        
        # Deshabilitar botones
        self.set_buttons_enabled(False)
        
        # Crear y ejecutar worker
        self.worker = SimulationWorker(self.ofdm_system, 'sweep', params)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_sweep_simulation_finished)
        self.worker.start()
    
    def run_multiantenna_test(self):
        '''Ejecuta prueba multiantena (8 configuraciones con SNR fijo)'''
        if not self.ofdm_system:
            self.update_config()
        
        if not self.current_image_path:
            QMessageBox.warning(self, 'Advertencia', 
                              'Por favor carga una imagen para realizar la prueba multiantena')
            return
        
        # Preparar parámetros (usa SNR único, no rango)
        params = {
            'image_path': self.current_image_path,
            'snr_db': self.snr_spin.value(),
            'channel_type': 'rayleigh_mp',
            'velocity_kmh': self.velocity_spin.value(),
            'itu_profile': self.itu_profile_combo.currentText(),
            'frequency_ghz': self.frequency_spin.value(),
            # Parámetros de configuración LTE
            'modulation': self.modulation_combo.currentText(),
            'bandwidth': self.bandwidth_combo.currentText(),
            'delta_f': self.delta_f_combo.currentText(),
            'cp_type': self.cp_combo.currentText()
        }
        
        # Deshabilitar botones
        self.set_buttons_enabled(False)
        
        # Crear y ejecutar worker
        self.worker = SimulationWorker(self.ofdm_system, 'multiantenna', params)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_multiantenna_finished)
        self.worker.start()
    
    def update_progress(self, value, message):
        '''Actualiza barra de progreso'''
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)
    
    def on_single_simulation_finished(self, results):
        '''Maneja finalización de simulación única'''
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)
        
        if 'error' in results:
            QMessageBox.critical(self, 'Error', f'Error en simulación:\n\n{results["error"]}')
            self.statusBar().showMessage('Simulación fallida')
            return
        
        # Actualizar métricas
        metrics = {
            'ber': results.get('ber', 0),
            'bit_errors': results.get('bit_errors', 0),
            'rank_used': results.get('rank_used', 0),
            'papr_db': results.get('papr_db', 0)
        }
        self.metrics_panel.update_metrics(metrics)
        
        # Mostrar imágenes si existen
        if 'reconstructed_image' in results and 'metadata' in results:
            self.show_image_comparison(results)
        
        # Graficar constelación si existe
        if 'symbols_rx' in results:
            # Para Spatial Multiplexing, usar símbolos recibidos
            self.plot_constellation(
                results.get('symbols_tx', results.get('qam_symbols')),
                results['symbols_rx']
            )
        
        self.statusBar().showMessage('Simulación completada exitosamente')
    
    def on_sweep_simulation_finished(self, results):
        '''Maneja finalización de barrido SNR'''
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)
        
        if 'error' in results:
            QMessageBox.critical(self, 'Error', f'Error en barrido:\n\n{results["error"]}')
            self.statusBar().showMessage('Barrido fallido')
            return
        
        # Graficar curvas BER (solo análisis, sin imagen)
        self.plot_sweep_ber_curves(results)
        
        self.statusBar().showMessage('Barrido completado exitosamente')
    
    def on_multiantenna_finished(self, results):
        '''Maneja finalización de prueba multiantena'''
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)
        
        if 'error' in results:
            QMessageBox.critical(self, 'Error', f'Error en prueba multiantena:\n\n{results["error"]}')
            self.statusBar().showMessage('Prueba multiantena fallida')
            return
        
        # Graficar prueba multiantena (12 configuraciones)
        self.plot_multiantenna_test(results)
        
        self.statusBar().showMessage('Prueba multiantena completada exitosamente')
    
    def plot_constellation(self, tx_symbols, rx_symbols):
        '''Grafica constelación'''
        fig = self.constellation_plot.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)
        
        # Convertir a numpy arrays
        tx_symbols = np.array(tx_symbols)
        rx_symbols = np.array(rx_symbols)
        
        # Muestrear si hay muchos símbolos
        max_symbols = 2000
        if len(tx_symbols) > max_symbols:
            indices = np.random.choice(len(tx_symbols), max_symbols, replace=False)
            tx_symbols = tx_symbols[indices]
            rx_symbols = rx_symbols[indices]
        
        ax.scatter(tx_symbols.real, tx_symbols.imag, alpha=0.5, s=20, label='TX', color='blue')
        ax.scatter(rx_symbols.real, rx_symbols.imag, alpha=0.3, s=15, label=f'RX ({self.current_detector})', color='red')
        ax.set_xlabel('I (In-Phase)')
        ax.set_ylabel('Q (Quadrature)')
        ax.set_title(f'Constelación - Spatial Multiplexing {self.current_num_tx}×{self.current_num_rx} (TM4, {self.current_detector})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        self.constellation_plot.get_canvas().draw()
        
        # Cambiar a tab de constelación
        self.results_tabs.setCurrentWidget(self.constellation_plot)
    
    def plot_sweep_ber_curves(self, results):
        '''Grafica curvas BER del barrido (8 configuraciones Spatial Multiplexing: 4 antenas × 2 detectores)'''
        fig = self.ber_plot.get_figure()
        fig.clear()
        
        # Configurar figura para un solo subplot
        fig.set_size_inches(12, 8)
        ax = fig.add_subplot(111)
        
        all_results = results['results']
        
        # Configuraciones en orden (deben coincidir con el worker)
        configs = [
            ('2×2 MMSE', 'blue', '-', 'o'),
            ('2×2 SIC', 'blue', '--', 's'),
            ('4×2 MMSE', 'green', '-', 'o'),
            ('4×2 SIC', 'green', '--', 's'),
            ('4×4 MMSE', 'orange', '-', 'o'),
            ('4×4 SIC', 'orange', '--', 's'),
            ('8×4 MMSE', 'red', '-', 'o'),
            ('8×4 SIC', 'red', '--', 's'),
        ]
        
        # Graficar cada configuración
        for config_key, color, linestyle, marker in configs:
            if config_key in all_results:
                config_data = all_results[config_key]
                ax.semilogy(
                    config_data['snr_values'],
                    config_data['ber_values'],
                    marker=marker,
                    label=config_key,
                    linewidth=2.5,
                    markersize=7,
                    color=color,
                    linestyle=linestyle,
                    alpha=0.8
                )
        
        ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('BER', fontsize=12, fontweight='bold')
        ax.set_title('Spatial Multiplexing Performance (TM4 - 64-QAM, Rank Adaptive)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9, ncol=2)
        ax.grid(True, which='both', alpha=0.3)
        
        # Añadir anotación
        ax.text(0.98, 0.97, 'Solid = MMSE, Dashed = SIC',
               transform=ax.transAxes, fontsize=10, style='italic',
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.tight_layout()
        self.ber_plot.get_canvas().draw()
        
        # Cambiar a tab de curvas BER
        self.results_tabs.setCurrentWidget(self.ber_plot)
    
    def plot_multiantenna_test(self, results):
        '''Grafica prueba multiantena (6 configuraciones: mosaico 2×3 - Fila 1: IRC, Fila 2: SIC)'''
        fig = self.mimo_comparison_plot.get_figure()
        fig.clear()
        
        # Cargar imagen original
        from PIL import Image as PILImage
        img_original = PILImage.open(self.current_image_path)
        if img_original.mode != 'RGB':
            img_original = img_original.convert('RGB')
        
        # Configurar figura para 2 filas × 3 columnas
        fig.set_size_inches(18, 12)
        
        # Obtener datos del worker
        results_list = results['results']  # Lista de diccionarios con resultados
        images_list = results['images']    # Lista de imágenes PIL
        
        # Crear grid 2×3 (IRC arriba, SIC abajo)
        for idx in range(min(6, len(results_list))):
            result_data = results_list[idx]
            img_recon = images_list[idx]
            
            ax = fig.add_subplot(2, 3, idx + 1)
            
            # Mostrar imagen reconstruida
            ax.imshow(img_recon)
            
            # Construir título con métricas
            name = result_data['name']
            num_tx = result_data['num_tx']
            num_rx = result_data['num_rx']
            ber = result_data['ber']
            
            title = f'{name}\nBER: {ber:.2e}'
            
            # Añadir rank si está disponible
            if result_data.get('rank_used', 0) > 0:
                title += f'\nRank: {result_data["rank_used"]}'
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Título general
        snr = results['snr_db']
        velocity = results.get('velocity_kmh', 'N/A')
        fig.suptitle(f'Prueba Multiantena Spatial Multiplexing (TM4) - SNR={snr}dB, Velocidad={velocity}km/h', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.mimo_comparison_plot.get_canvas().draw()
        
        # Cambiar a tab de comparación MIMO
        self.results_tabs.setCurrentWidget(self.mimo_comparison_plot)
    
    def show_image_comparison(self, results):
        '''Muestra comparación de imágenes original vs reconstruida'''
        try:
            original_pixmap = QPixmap(self.current_image_path)
            
            # Convertir PIL Image a QPixmap
            received_img = results['reconstructed_image']
            received_img_rgb = received_img.convert('RGB')
            data = received_img_rgb.tobytes('raw', 'RGB')
            qimage = QImage(data, received_img_rgb.width, received_img_rgb.height, 
                           received_img_rgb.width * 3, QImage.Format.Format_RGB888)
            received_pixmap = QPixmap.fromImage(qimage)
            
            # Calcular métricas
            from PIL import Image
            original_img = Image.open(self.current_image_path)
            psnr = ImageProcessor.calculate_psnr(original_img, received_img)
            ssim = ImageProcessor.calculate_ssim(original_img, received_img)
            
            self.image_comparison.set_images(original_pixmap, received_pixmap, psnr, ssim)
            
            # Cambiar a la pestaña de imágenes
            self.results_tabs.setCurrentWidget(self.image_comparison)
            
        except Exception as e:
            import traceback
            error_msg = f'Error al mostrar imágenes:\n{str(e)}\n\n{traceback.format_exc()}'
            QMessageBox.warning(self, 'Advertencia', error_msg)
    
    def set_buttons_enabled(self, enabled):
        '''Habilita/deshabilita botones'''
        self.single_sim_btn.setEnabled(enabled)
        self.sweep_sim_btn.setEnabled(enabled)
        self.multiantenna_btn.setEnabled(enabled)
        self.load_image_btn.setEnabled(enabled)
