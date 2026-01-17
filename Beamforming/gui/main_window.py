#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ventana principal del simulador OFDM con Beamforming (LTE TM6)
Usa el core de beamforming de este proyecto (simulate_beamforming)
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
from core.ofdm_core import OFDMSimulator
from utils.image_processing import ImageProcessor
from Beamforming.gui.widgets import PlotWidget, MetricsPanel, ConfigInfoPanel, ImageComparisonWidget


class SimulationWorker(QThread):
    """Worker thread para ejecutar simulaciones de Beamforming sin bloquear la GUI"""
    
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
        """Ejecuta simulación única de Beamforming"""
        self.progress.emit(10, "Preparando datos...")
        
        if 'image_path' in self.params:
            # Transmisión de imagen
            bits, metadata = ImageProcessor.image_to_bits(self.params['image_path'])
            self.params['metadata'] = metadata
            
            print(f"\n[DEBUG Beamforming Worker] Preparación de imagen:")
            print(f"  Imagen: {metadata['width']}x{metadata['height']}x{metadata['channels']}")
            print(f"  Bits TX: {len(bits):,}")
        else:
            # Bits aleatorios
            bits = np.random.randint(0, 2, self.params['n_bits'])
            print(f"[DEBUG Beamforming] Bits aleatorios: {len(bits):,}")
        
        self.progress.emit(30, f"Transmitiendo con Beamforming ({self.params['num_tx']}×{self.params['num_rx']})...")
        
        print(f"\n[DEBUG Beamforming] Configuración:")
        print(f"  Tipo: {self.params['channel_type']}")
        print(f"  SNR: {self.params['snr_db']} dB")
        print(f"  Num TX: {self.params['num_tx']}")
        print(f"  Num RX: {self.params['num_rx']}")
        print(f"  Velocidad: {self.params.get('velocity_kmh', 3)} km/h")
        
        # Beamforming: N TX × M RX
        results = self.ofdm_system.simulate_beamforming(
            bits=bits,
            snr_db=self.params['snr_db'],
            num_tx=self.params['num_tx'],
            num_rx=self.params['num_rx'],
            codebook_type='TM6',
            velocity_kmh=self.params.get('velocity_kmh', 3),
            update_mode='adaptive'
        )
        
        self.progress.emit(70, "Decodificando...")
        
        # Generar símbolos para constelación
        from core.modulator import QAMModulator
        qam_mod = QAMModulator(self.ofdm_system.config.modulation)
        
        # Tomar muestra de bits para constelación (máximo 1000 símbolos)
        max_symbols = 1000
        bits_per_symbol = int(np.log2(len(qam_mod.constellation)))
        max_bits = max_symbols * bits_per_symbol
        
        # Símbolos TX (desde bits originales)
        sample_bits_tx = bits[:min(max_bits, len(bits))]
        symbols_tx = qam_mod.bits_to_symbols(sample_bits_tx)
        
        # Símbolos RX (desde bits recibidos)
        bits_rx = results['bits_received_array']
        sample_bits_rx = bits_rx[:min(max_bits, len(bits_rx))]
        symbols_rx = qam_mod.bits_to_symbols(sample_bits_rx)
        
        # Agregar a resultados
        results['symbols_tx'] = symbols_tx
        results['symbols_rx'] = symbols_rx
        
        print(f"\n[DEBUG Beamforming] Resultados:")
        print(f"  BER: {results['ber']:.6f}")
        print(f"  Errores: {results['bit_errors']:,}")
        print(f"  Ganancia BF: {results.get('beamforming_gain_db', 0):.2f} dB")
        print(f"  Símbolos para constelación: {len(symbols_tx)}")
        
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
        Ejecuta barrido completo de SNR para Beamforming.
        
        OPCIÓN 3 - Enfoque Práctico Simplificado:
        - Modulación: 64-QAM fija (más usada en LTE)
        - Configuraciones escalonadas mostrando evolución tecnológica:
          * Baseline: 2×1 SFBC (TX Diversity)
          * Mejora TX: 4×1 BF, 8×1 BF
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
            print(f"[DEBUG Beamforming] Barrido SNR - IMAGEN CARGADA")
            print(f"{'='*70}")
            print(f"  Dimensiones imagen: {metadata['width']}×{metadata['height']}")
            print(f"  Total bits de imagen: {len(bits_to_sweep):,} bits")
            print(f"  Tamaño datos: {len(bits_to_sweep) / 8 / 1024:.2f} KB")
            print(f"  ✓ Se enviarán TODOS los bits de la imagen en cada SNR")
            print(f"{'='*70}")
        else:
            bits_to_sweep = np.random.randint(0, 2, 50000)
            print(f"\n{'='*70}")
            print(f"[DEBUG Beamforming] Barrido SNR - SIN IMAGEN (usando bits aleatorios)")
            print(f"{'='*70}")
            print(f"  Total bits aleatorios: {len(bits_to_sweep):,} bits")
            print(f"  ⚠ Para usar imagen real, cargue una imagen primero")
            print(f"{'='*70}")
        
        # Configuraciones a comparar (OPCIÓN 3)
        # Formato: (nombre, num_tx, num_rx, modo, estilo_línea)
        configurations = [
            ('2×1 SFBC (Baseline)', 2, 1, 'sfbc', {'color': '#999999', 'linestyle': '--', 'linewidth': 2}),
            ('2×1 Beamforming', 2, 1, 'bf', {'color': '#2196F3', 'linestyle': '-', 'linewidth': 2}),
            ('4×1 Beamforming', 4, 1, 'bf', {'color': '#4CAF50', 'linestyle': '-', 'linewidth': 2.5}),
            ('8×1 Beamforming', 8, 1, 'bf', {'color': '#FF9800', 'linestyle': '-', 'linewidth': 3}),
            ('2×2 Beamforming', 2, 2, 'bf', {'color': '#9C27B0', 'linestyle': '-.', 'linewidth': 2}),
            ('4×2 Beamforming', 4, 2, 'bf', {'color': '#00BCD4', 'linestyle': '-.', 'linewidth': 2.5}),
            ('8×4 Beamforming', 8, 4, 'bf', {'color': '#F44336', 'linestyle': ':', 'linewidth': 3}),
        ]
        
        snr_range = self.params['snr_range']
        n_iterations = self.params['n_iterations']
        
        print(f"  Modulación: 64-QAM (fija)")
        print(f"  Configuraciones: {len(configurations)}")
        print(f"  Rango SNR: {snr_range[0]:.1f} a {snr_range[-1]:.1f} dB")
        print(f"  Iteraciones: {n_iterations}")
        
        # Calcular pasos totales
        total_steps = len(configurations) * len(snr_range) * n_iterations
        current_step = 0
        
        # Resultados
        all_results = {}
        
        self.progress.emit(10, "Iniciando barrido de Beamforming...")
        
        # Crear sistema con 64-QAM
        config = LTEConfig(
            bandwidth=float(self.params['bandwidth']),
            modulation='64-QAM',
            delta_f=float(self.params['delta_f']),
            cp_type=self.params['cp_type']
        )
        
        # Iterar sobre configuraciones
        for config_name, num_tx, num_rx, mode, style in configurations:
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
                    
                    # Simular según modo
                    if mode == 'sfbc':
                        # TX Diversity (SFBC Alamouti)
                        result = temp_system.simulate_miso(bits_to_sweep, snr_db)
                    else:
                        # Beamforming
                        result = temp_system.simulate_beamforming(
                            bits=bits_to_sweep,
                            snr_db=snr_db,
                            num_tx=num_tx,
                            num_rx=num_rx,
                            codebook_type='TM6',
                            velocity_kmh=self.params.get('velocity_kmh', 3),
                            update_mode='adaptive'
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
                'num_rx': num_rx
            }
            
            print(f"  BER: {min(ber_results):.2e} (mejor) a {max(ber_results):.2e} (peor)")
        
        self.progress.emit(95, "Finalizando simulación...")
        
        # Emitir resultados
        self.finished.emit({
            'type': 'sweep_beamforming',
            'results': all_results,
            'configurations': [c[0] for c in configurations]
        })
    
    def _run_multiantenna_test(self):
        """
        Ejecuta prueba multiantena completa (similar a test_beamforming_image.py):
        
        3 filas × 4 configuraciones = 12 pruebas:
        - Fila 1 (1 RX): 2×1 SFBC, 2×1 BF, 4×1 BF, 8×1 BF
        - Fila 2 (2 RX): 2×2 SFBC, 2×2 BF, 4×2 BF, 8×2 BF
        - Fila 3 (4 RX): 2×4 SFBC, 2×4 BF, 4×4 BF, 8×4 BF
        
        Genera imagen comparativa en mosaico con métricas.
        """
        self.progress.emit(5, "Preparando prueba multiantena...")
        
        # Cargar imagen
        if 'image_path' not in self.params:
            self.finished.emit({'error': 'Se requiere una imagen para la prueba multiantena'})
            return
        
        bits, metadata = ImageProcessor.image_to_bits(self.params['image_path'])
        
        print(f"\n[DEBUG Beamforming] Prueba Multiantena:")
        print(f"  Imagen: {metadata['width']}×{metadata['height']}×{metadata['channels']}")
        print(f"  Bits: {len(bits):,}")
        print(f"  === Parámetros de simulación ===")
        print(f"  Modulación: {self.params['modulation']}")
        print(f"  Ancho de banda: {self.params['bandwidth']} MHz")
        print(f"  Delta-f: {self.params['delta_f']} kHz")
        print(f"  CP: {self.params['cp_type']}")
        print(f"  SNR: {self.params['snr_db']} dB")
        print(f"  Canal: {self.params['channel_type']}")
        print(f"  ITU Profile: {self.params.get('itu_profile', 'NO ESPECIFICADO')}")
        print(f"  Frequency: {self.params.get('frequency_ghz', 'NO ESPECIFICADO')} GHz")
        print(f"  Velocity: {self.params.get('velocity_kmh', 'NO ESPECIFICADO')} km/h")
        
        # Configuraciones a comparar (3 filas × 4 configs)
        test_configs = [
            # Fila 1: 1 RX
            {'name': '2×1 TX Diversity (SFBC)', 'num_tx': 2, 'num_rx': 1, 'mode': 'diversity'},
            {'name': '2×1 Beamforming', 'num_tx': 2, 'num_rx': 1, 'mode': 'beamforming'},
            {'name': '4×1 Beamforming', 'num_tx': 4, 'num_rx': 1, 'mode': 'beamforming'},
            {'name': '8×1 Beamforming', 'num_tx': 8, 'num_rx': 1, 'mode': 'beamforming'},
            
            # Fila 2: 2 RX
            {'name': '2×2 TX Diversity (SFBC)', 'num_tx': 2, 'num_rx': 2, 'mode': 'diversity'},
            {'name': '2×2 Beamforming', 'num_tx': 2, 'num_rx': 2, 'mode': 'beamforming'},
            {'name': '4×2 Beamforming', 'num_tx': 4, 'num_rx': 2, 'mode': 'beamforming'},
            {'name': '8×2 Beamforming', 'num_tx': 8, 'num_rx': 2, 'mode': 'beamforming'},
            
            # Fila 3: 4 RX
            {'name': '2×4 TX Diversity (SFBC)', 'num_tx': 2, 'num_rx': 4, 'mode': 'diversity'},
            {'name': '2×4 Beamforming', 'num_tx': 2, 'num_rx': 4, 'mode': 'beamforming'},
            {'name': '4×4 Beamforming', 'num_tx': 4, 'num_rx': 4, 'mode': 'beamforming'},
            {'name': '8×4 Beamforming', 'num_tx': 8, 'num_rx': 4, 'mode': 'beamforming'},
        ]
        
        results_list = []
        images_list = []
        
        total_configs = len(test_configs)
        
        for idx, test_config in enumerate(test_configs):
            progress_pct = int(10 + (idx / total_configs) * 80)
            self.progress.emit(progress_pct, f"Simulando {test_config['name']}...")
            
            print(f"\n  === {test_config['name']} ({test_config['num_tx']}×{test_config['num_rx']}) ===")
            
            num_tx = test_config['num_tx']
            num_rx = test_config['num_rx']
            mode = test_config['mode']
            
            # Crear sistema OFDM nuevo para cada configuración (para resetear estado del canal)
            from config import LTEConfig
            temp_config = LTEConfig(
                bandwidth=float(self.params['bandwidth']),
                modulation=self.params['modulation'],
                delta_f=float(self.params['delta_f']),
                cp_type=self.params['cp_type']
            )
            
            temp_system = OFDMSimulator(
                config=temp_config,
                channel_type='rayleigh_mp',
                mode='lte',
                enable_equalization=True,
                num_channels=1,
                itu_profile=self.params.get('itu_profile', 'Pedestrian_A'),
                frequency_ghz=self.params.get('frequency_ghz', 2.0),
                velocity_kmh=self.params.get('velocity_kmh', 3.0)
            )
            
            # Simular según modo
            if mode == 'diversity':
                # TX Diversity (SFBC Alamouti)
                result = temp_system.simulate_miso(bits, snr_db=self.params['snr_db'])
                gain_db = 0.0
            else:
                # Beamforming
                result = temp_system.simulate_beamforming(
                    bits=bits,
                    snr_db=self.params['snr_db'],
                    num_tx=num_tx,
                    num_rx=num_rx,
                    codebook_type='TM6',
                    velocity_kmh=self.params.get('velocity_kmh', 3),
                    update_mode='adaptive'
                )
                gain_db = result.get('beamforming_gain_db', 0.0)
            
            # DEBUG: Verificar resultado
            print(f"    [WORKER DEBUG] result['ber'] = {result.get('ber', 'NO EXISTE')}")
            print(f"    [WORKER DEBUG] result['bit_errors'] = {result.get('bit_errors', 'NO EXISTE')}")
            print(f"    [WORKER DEBUG] len(bits) = {len(bits):,}")
            print(f"    [WORKER DEBUG] len(result['bits_received_array']) = {len(result.get('bits_received_array', [])):,}")
            
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
                'mode': mode,
                'ber': result['ber'],
                'bit_errors': result['bit_errors'],
                'psnr': psnr,
                'gain_db': gain_db
            })
            
            images_list.append(img_reconstructed)
            
            print(f"    BER: {result['ber']:.6f}, Errores: {result['bit_errors']:,}, PSNR: {psnr:.1f} dB")
            if gain_db > 0:
                print(f"    Ganancia BF: {gain_db:.2f} dB")
        
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


class BeamformingGUI(QMainWindow):
    """Ventana principal del simulador OFDM con Beamforming (LTE TM6)"""
    
    def __init__(self):
        super().__init__()
        self.ofdm_system = None
        self.current_image_path = None
        self.current_num_tx = 2
        self.current_num_rx = 1
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        """Inicializa la interfaz de usuario"""
        self.setWindowTitle("Simulador OFDM-LTE (Beamforming - TM6)")
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
        self.statusBar().showMessage("Listo - Beamforming (TM6)")
        
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
        self.snr_spin.setValue(15.0)  # Default 15 dB para MIMO
        self.snr_spin.setSingleStep(0.5)
        layout.addWidget(self.snr_spin, 0, 1)
        
        # Rango SNR para barrido
        layout.addWidget(QLabel("SNR Inicio (dB):"), 1, 0)
        self.snr_start_spin = QDoubleSpinBox()
        self.snr_start_spin.setRange(-10, 40)
        self.snr_start_spin.setValue(0.0)
        layout.addWidget(self.snr_start_spin, 1, 1)
        
        layout.addWidget(QLabel("SNR Fin (dB):"), 2, 0)
        self.snr_end_spin = QDoubleSpinBox()
        self.snr_end_spin.setRange(-10, 40)
        self.snr_end_spin.setValue(20.0)
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
        self.iterations_spin.setValue(3)  # Menos iteraciones por defecto (Beamforming puede ser lento)
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
        self.num_rx_combo.addItems(['1', '2', '4'])
        self.num_rx_combo.setCurrentText('1')
        self.num_rx_combo.currentTextChanged.connect(self.update_config)
        layout.addWidget(self.num_rx_combo, 6, 1)
        
        # Label informativo
        info_label = QLabel("(Beamforming TM6 con precoding adaptativo)")
        info_label.setStyleSheet("color: #666; font-style: italic; font-size: 9pt;")
        layout.addWidget(info_label, 7, 0, 1, 2)
        
        # Tipo de canal (SOLO MULTIPATH)
        layout.addWidget(QLabel("Tipo de Canal:"), 8, 0)
        self.channel_type_combo = QComboBox()
        self.channel_type_combo.addItems(['Rayleigh Multitrayecto'])
        self.channel_type_combo.currentTextChanged.connect(self.on_channel_type_changed)
        layout.addWidget(self.channel_type_combo, 8, 1)
        
        # Perfil ITU
        layout.addWidget(QLabel("Perfil ITU-R M.1225:"), 9, 0)
        self.itu_profile_combo = QComboBox()
        self.itu_profile_combo.addItems([
            'Pedestrian_A',
            'Pedestrian_B',
            'Vehicular_A',
            'Vehicular_B',
            'Typical_Urban',
            'Rural_Area'
        ])
        self.itu_profile_combo.setCurrentText('Pedestrian_A')  # Default para Beamforming
        self.itu_profile_combo.currentTextChanged.connect(self.on_itu_profile_changed)
        self.itu_profile_combo.setEnabled(True)  # Siempre habilitado (solo multipath)
        layout.addWidget(self.itu_profile_combo, 9, 1)
        
        # Frecuencia portadora
        layout.addWidget(QLabel("Frecuencia (GHz):"), 10, 0)
        self.frequency_spin = QDoubleSpinBox()
        self.frequency_spin.setRange(0.5, 10.0)
        self.frequency_spin.setValue(2.0)
        self.frequency_spin.setSingleStep(0.1)
        self.frequency_spin.valueChanged.connect(self.on_frequency_changed)
        self.frequency_spin.setEnabled(True)  # Siempre habilitado
        layout.addWidget(self.frequency_spin, 10, 1)
        
        # Velocidad
        layout.addWidget(QLabel("Velocidad (km/h):"), 11, 0)
        self.velocity_spin = QSpinBox()
        self.velocity_spin.setRange(0, 500)
        self.velocity_spin.setValue(3)  # Default: 3 km/h (pedestrian)
        self.velocity_spin.valueChanged.connect(self.on_velocity_changed)
        self.velocity_spin.setEnabled(True)  # Siempre habilitado
        layout.addWidget(self.velocity_spin, 11, 1)
        
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
        '''Actualiza la configuración del sistema Beamforming'''
        try:
            bandwidth = float(self.bandwidth_combo.currentText())
            delta_f_khz = float(self.delta_f_combo.currentText())
            modulation = self.modulation_combo.currentText()
            cp_type = self.cp_combo.currentText()
            num_tx = int(self.num_tx_combo.currentText())
            num_rx = int(self.num_rx_combo.currentText())
            
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
            
            # Actualizar panel de información
            config_info = config.get_info()
            config_info[' Beamforming (TM6) '] = ''
            config_info['Num. Transmisores'] = f'{num_tx} TX'
            config_info['Num. Receptores'] = f'{num_rx} RX'
            config_info['Array Gain (teórico)'] = f'{10*np.log10(num_tx):.1f} dB (TX) + {10*np.log10(num_rx):.1f} dB (RX)'
            config_info[' Canal Multipath '] = ''
            config_info['Perfil ITU'] = itu_profile
            config_info['Frecuencia'] = f'{frequency_ghz} GHz'
            config_info['Velocidad'] = f'{velocity_kmh} km/h'
            
            self.info_panel.update_config(config_info)
            
            self.statusBar().showMessage(
                f'Config: {num_tx}×{num_rx} Beamforming, {bandwidth} MHz, {modulation}, Rayleigh MP'
            )
            
        except Exception as e:
            import traceback
            error_msg = f'Error al actualizar configuración:\n{str(e)}\n\n{traceback.format_exc()}'
            QMessageBox.critical(self, 'Error', error_msg)
    
    def on_channel_type_changed(self):
        """Maneja el cambio de tipo de canal (siempre multipath en Beamforming)"""
        # En beamforming siempre usamos multipath, todos los controles están habilitados
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
        """Ejecuta simulación única de Beamforming"""
        if not self.ofdm_system:
            self.update_config()
        
        if not self.current_image_path:
            QMessageBox.warning(self, 'Advertencia', 
                              'Por favor carga una imagen para realizar la simulación')
            return
        
        # Preparar parámetros (solo multipath en beamforming)
        params = {
            'snr_db': self.snr_spin.value(),
            'image_path': self.current_image_path,
            'num_tx': self.current_num_tx,
            'num_rx': self.current_num_rx,
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
        """Ejecuta barrido de SNR para Beamforming (OPCIÓN 3)"""
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
            'channel_type': 'rayleigh_mp',  # Solo multipath en beamforming
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
        '''Ejecuta prueba multiantena (12 configuraciones)'''
        if not self.ofdm_system:
            self.update_config()
        
        if not self.current_image_path:
            QMessageBox.warning(self, 'Advertencia', 
                              'Por favor carga una imagen para realizar la prueba multiantena')
            return
        
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
            'beamforming_gain_db': results.get('beamforming_gain_db', 0),
            'papr_db': results.get('papr_db', 0)
        }
        self.metrics_panel.update_metrics(metrics)
        
        # Mostrar imágenes si existen
        if 'reconstructed_image' in results and 'metadata' in results:
            self.show_image_comparison(results)
        
        # Graficar constelación si existe
        if 'symbols_rx' in results:
            # Para Beamforming, usar símbolos recibidos
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
        ax.scatter(rx_symbols.real, rx_symbols.imag, alpha=0.3, s=15, label='RX (Beamforming)', color='red')
        ax.set_xlabel('I (In-Phase)')
        ax.set_ylabel('Q (Quadrature)')
        ax.set_title(f'Constelación - Beamforming {self.current_num_tx}×{self.current_num_rx} (TM6)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        self.constellation_plot.get_canvas().draw()
        
        # Cambiar a tab de constelación
        self.results_tabs.setCurrentWidget(self.constellation_plot)
    
    def plot_sweep_ber_curves(self, results):
        '''Grafica curvas BER del barrido (OPCIÓN 3: single graph, 7 configs, 64-QAM)'''
        fig = self.ber_plot.get_figure()
        fig.clear()
        
        # Configurar figura para un solo subplot
        fig.set_size_inches(12, 8)
        ax = fig.add_subplot(111)
        
        all_results = results['results']
        
        # Configuraciones en orden de visualización (deben coincidir con el worker)
        configs = [
            ('2×1 SFBC (Baseline)', '2×1 SFBC (baseline)', 'gray', '-', 'o'),
            ('2×1 Beamforming', '2×1 Beamforming', 'blue', '-', 's'),
            ('4×1 Beamforming', '4×1 Beamforming', 'green', '-', '^'),
            ('8×1 Beamforming', '8×1 Beamforming', 'orange', '-', 'v'),
            ('2×2 Beamforming', '2×2 Beamforming', 'cyan', '-.', 's'),
            ('4×2 Beamforming', '4×2 Beamforming', 'magenta', '-.', '^'),
            ('8×4 Beamforming', '8×4 Beamforming (Massive MIMO)', 'red', ':', 'D')
        ]
        
        # Graficar cada configuración
        for config_key, label, color, linestyle, marker in configs:
            if config_key in all_results:
                config_data = all_results[config_key]
                ax.semilogy(
                    config_data['snr_values'],
                    config_data['ber_values'],
                    marker=marker,
                    label=label,
                    linewidth=2.5,
                    markersize=7,
                    color=color,
                    linestyle=linestyle,
                    alpha=0.8
                )
        
        ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('BER', fontsize=12, fontweight='bold')
        ax.set_title('Beamforming Performance: Technology Progression (64-QAM)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, which='both', alpha=0.3)
        
        # Añadir anotaciones de mejora tecnológica
        ax.text(0.98, 0.97, 'Baseline → Increased TX → Increased RX → Massive MIMO',
               transform=ax.transAxes, fontsize=9, style='italic',
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.tight_layout()
        self.ber_plot.get_canvas().draw()
        
        # Cambiar a tab de curvas BER
        self.results_tabs.setCurrentWidget(self.ber_plot)
    
    def plot_multiantenna_test(self, results):
        '''Grafica prueba multiantena (12 configuraciones: 3 filas × 4 columnas)'''
        fig = self.mimo_comparison_plot.get_figure()
        fig.clear()
        
        # Cargar imagen original
        from PIL import Image as PILImage
        img_original = PILImage.open(self.current_image_path)
        if img_original.mode != 'RGB':
            img_original = img_original.convert('RGB')
        
        # Configurar figura para 3 filas × 4 columnas
        fig.set_size_inches(18, 14)
        
        # Obtener datos del worker
        results_list = results['results']  # Lista de diccionarios con resultados
        images_list = results['images']    # Lista de imágenes PIL
        
        # Verificar que tenemos 12 configuraciones
        if len(results_list) != 12 or len(images_list) != 12:
            print(f"[WARNING] Se esperaban 12 configs, pero se recibieron {len(results_list)}")
        
        # Crear grid 3×4
        for idx in range(min(12, len(results_list))):
            result_data = results_list[idx]
            img_recon = images_list[idx]
            
            ax = fig.add_subplot(3, 4, idx + 1)
            
            # Mostrar imagen reconstruida
            ax.imshow(img_recon)
            
            # Construir título con métricas
            name = result_data['name']
            num_tx = result_data['num_tx']
            num_rx = result_data['num_rx']
            ber = result_data['ber']
            
            # Título corto para mejor legibilidad
            if 'Diversity' in name or 'SFBC' in name:
                short_name = f'{num_tx}×{num_rx} SFBC'
            else:
                short_name = f'{num_tx}×{num_rx} BF'
            
            title = f'{short_name}\nBER: {ber:.2e}'
            
            # Añadir gain si está disponible (solo para beamforming)
            if result_data['gain_db'] > 0:
                title += f'\nGain: {result_data["gain_db"]:.1f} dB'
            
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.axis('off')
        
        # Título general
        snr = results['snr_db']
        velocity = results.get('velocity_kmh', 'N/A')
        fig.suptitle(f'Prueba Multiantena Beamforming - SNR={snr}dB, Velocidad={velocity}km/h', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        self.mimo_comparison_plot.get_canvas().draw()
        
        # Cambiar a tab de comparación
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
