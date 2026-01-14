#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ventana principal del simulador OFDM con Transmit Diversity (SFBC Alamouti)
Usa el core MIMO de este proyecto (simulate_miso/simulate_mimo)
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
from Tx_div.gui.widgets import PlotWidget, MetricsPanel, ConfigInfoPanel, ImageComparisonWidget


class SimulationWorker(QThread):
    """Worker thread para ejecutar simulaciones MIMO sin bloquear la GUI"""
    
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
            elif self.mode == 'mimo_comparison':
                self._run_mimo_comparison()
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.finished.emit({'error': error_msg})
    
    def _run_single_simulation(self):
        """Ejecuta simulación única MIMO (MISO o MIMO según num_rx)"""
        self.progress.emit(10, "Preparando datos...")
        
        if 'image_path' in self.params:
            # Transmisión de imagen
            bits, metadata = ImageProcessor.image_to_bits(self.params['image_path'])
            self.params['metadata'] = metadata
            
            print(f"\n[DEBUG TxDiv Worker] Preparación de imagen:")
            print(f"  Imagen: {metadata['width']}x{metadata['height']}x{metadata['channels']}")
            print(f"  Bits TX: {len(bits):,}")
        else:
            # Bits aleatorios
            bits = np.random.randint(0, 2, self.params['n_bits'])
            print(f"[DEBUG TxDiv] Bits aleatorios: {len(bits):,}")
        
        self.progress.emit(30, "Transmitiendo con SFBC Alamouti...")
        
        print(f"\n[DEBUG TxDiv] Configuración de canal:")
        print(f"  Tipo: {self.params['channel_type']}")
        print(f"  SNR: {self.params['snr_db']} dB")
        
        # MIMO: 2 TX (SFBC) × N RX
        num_rx = self.params.get('num_rx', 1)
        
        print(f"\n[DEBUG TxDiv] Simulación MIMO:")
        print(f"  Num TX: 2 (SFBC Alamouti)")
        print(f"  Num RX: {num_rx}")
        print(f"  Modo: {'MISO' if num_rx == 1 else f'MIMO 2×{num_rx}'}")
        
        if num_rx == 1:
            # MISO (2 TX × 1 RX)
            results = self.ofdm_system.simulate_miso(bits, snr_db=self.params['snr_db'])
        else:
            # MIMO (2 TX × N RX)
            results = self.ofdm_system.simulate_mimo(
                bits,
                snr_db=self.params['snr_db'],
                num_rx=num_rx
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
        
        print(f"\n[DEBUG TxDiv] Resultados:")
        print(f"  BER: {results['ber']:.6f}")
        print(f"  Errores: {results['bit_errors']:,}")
        print(f"  Bits RX: {len(results['bits_received_array']):,}")
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
        Ejecuta barrido completo de SNR para MIMO:
        - 3 modulaciones (QPSK, 16-QAM, 64-QAM)
        - 4 configuraciones RX (1, 2, 4, 8 antenas) con 2 TX fijos
        - Rango de SNR configurable
        
        Genera 4 subplots apilados verticalmente, uno por num_rx,
        cada uno con 3 curvas (modulaciones)
        """
        self.progress.emit(5, "Preparando datos...")
        
        # Usar bits de imagen si existen, sino aleatorios
        if 'bits' in self.params and 'metadata' in self.params:
            bits_to_sweep = self.params['bits']
            metadata = self.params['metadata']
            print(f"\n{'='*70}")
            print(f"[DEBUG TxDiv] Barrido SNR - IMAGEN CARGADA")
            print(f"{'='*70}")
            print(f"  Dimensiones imagen: {metadata['width']}×{metadata['height']}")
            print(f"  Total bits de imagen: {len(bits_to_sweep):,} bits")
            print(f"  Tamaño datos: {len(bits_to_sweep) / 8 / 1024:.2f} KB")
            print(f"  ✓ Se enviarán TODOS los bits de la imagen en cada SNR")
            print(f"{'='*70}")
        else:
            bits_to_sweep = np.random.randint(0, 2, 50000)
            print(f"\n{'='*70}")
            print(f"[DEBUG TxDiv] Barrido SNR - SIN IMAGEN (usando bits aleatorios)")
            print(f"{'='*70}")
            print(f"  Total bits aleatorios: {len(bits_to_sweep):,} bits")
            print(f"  ⚠ Para usar imagen real, cargue una imagen primero")
            print(f"{'='*70}")
        
        # Parámetros del barrido
        modulations = ['QPSK', '16-QAM', '64-QAM']
        num_rx_values = [1, 2, 4, 8]  # 4 subplots
        snr_range = self.params['snr_range']
        n_iterations = self.params['n_iterations']
        
        print(f"  Modulaciones: {modulations}")
        print(f"  Num TX: 2 (SFBC fijo)")
        print(f"  Num RX: {num_rx_values}")
        print(f"  Rango SNR: {snr_range[0]:.1f} a {snr_range[-1]:.1f} dB")
        print(f"  Iteraciones: {n_iterations}")
        
        # Calcular pasos totales
        total_steps = len(num_rx_values) * len(modulations) * len(snr_range) * n_iterations
        current_step = 0
        
        # Resultados organizados por num_rx (para subplots verticales)
        all_results = {}
        
        self.progress.emit(10, "Iniciando barrido MIMO...")
        
        # Iterar sobre num_rx (cada uno será un subplot)
        for num_rx in num_rx_values:
            print(f"\n=== Configuración: 2×{num_rx} ===")
            
            rx_results = {}
            
            # Iterar sobre modulaciones (cada una será una curva en el subplot)
            for mod in modulations:
                print(f"  Modulación: {mod}")
                
                # Crear sistema con nueva modulación
                config = LTEConfig(
                    bandwidth=float(self.params['bandwidth']),
                    modulation=mod,
                    delta_f=float(self.params['delta_f']),  # Ya está en kHz
                    cp_type=self.params['cp_type']
                )
                
                # Crear nuevo sistema OFDM con esta configuración
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
                                         f"2×{num_rx} | {mod} | SNR={snr_db:.1f}dB | {iter_idx+1}/{n_iterations}")
                        
                        # Simular según num_rx
                        if num_rx == 1:
                            result = temp_system.simulate_miso(bits_to_sweep, snr_db)
                        else:
                            result = temp_system.simulate_mimo(
                                bits_to_sweep,
                                snr_db=snr_db,
                                num_rx=num_rx
                            )
                        
                        ber_list.append(result['ber'])
                    
                    # Promedio de iteraciones
                    avg_ber = np.mean(ber_list)
                    ber_results.append(avg_ber)
                    snr_values_list.append(snr_db)
                
                # Guardar resultados para esta modulación
                rx_results[mod] = {
                    'snr_values': np.array(snr_values_list),
                    'ber_values': np.array(ber_results)
                }
                
                print(f"    {mod} → BER: {min(ber_results):.2e} (mejor) a {max(ber_results):.2e} (peor)")
            
            # Guardar resultados de este num_rx
            all_results[f'2x{num_rx}'] = rx_results
        
        self.progress.emit(95, "Finalizando simulación...")
        
        # Emitir resultados (sin imagen, solo curvas BER)
        self.finished.emit({
            'type': 'sweep',
            'results': all_results,
            'num_rx_values': num_rx_values,
            'modulations': modulations
        })
    
    def _run_mimo_comparison(self):
        """
        Ejecuta comparación completa MIMO (similar a test_mimo_image.py):
        - SISO (1×1)
        - MISO (2×1)
        - MIMO-2×2
        - MIMO-2×4
        """
        self.progress.emit(5, "Preparando comparación MIMO...")
        
        # Cargar imagen
        if 'image_path' not in self.params:
            self.finished.emit({'error': 'Se requiere una imagen para la comparación MIMO'})
            return
        
        bits, metadata = ImageProcessor.image_to_bits(self.params['image_path'])
        
        print(f"\n[DEBUG TxDiv] Comparación MIMO:")
        print(f"  Imagen: {metadata['width']}×{metadata['height']}×{metadata['channels']}")
        print(f"  Bits: {len(bits):,}")
        print(f"  SNR: {self.params['snr_db']} dB")
        print(f"  Canal: {self.params['channel_type']}")
        
        # Configuraciones a comparar
        configs = [
            ("SISO", 1, 1),
            ("MISO", 2, 1),
            ("MIMO-2x2", 2, 2),
            ("MIMO-2x4", 2, 4)
        ]
        
        results = {}
        
        for idx, (config_name, num_tx, num_rx) in enumerate(configs):
            progress_pct = int(10 + (idx / len(configs)) * 80)
            self.progress.emit(progress_pct, f"Simulando {config_name}...")
            
            print(f"\n  === {config_name} ({num_tx}×{num_rx}) ===")
            
            if num_tx == 1:
                # SISO
                result = self.ofdm_system.simulate_siso(bits, snr_db=self.params['snr_db'])
            elif num_rx == 1:
                # MISO
                result = self.ofdm_system.simulate_miso(bits, snr_db=self.params['snr_db'])
            else:
                # MIMO
                result = self.ofdm_system.simulate_mimo(
                    bits,
                    snr_db=self.params['snr_db'],
                    num_rx=num_rx
                )
            
            # Reconstruir imagen
            img_reconstructed = ImageProcessor.bits_to_image(
                result['bits_received_array'],
                metadata
            )
            
            results[config_name] = {
                'ber': result['ber'],
                'bit_errors': result['bit_errors'],
                'image': img_reconstructed,
                'num_tx': num_tx,
                'num_rx': num_rx
            }
            
            print(f"    BER: {result['ber']:.6f}, Errores: {result['bit_errors']:,}")
        
        self.progress.emit(95, "Generando comparación...")
        
        # Emitir resultados
        self.finished.emit({
            'type': 'mimo_comparison',
            'results': results,
            'metadata': metadata,
            'snr_db': self.params['snr_db'],
            'channel_type': self.params['channel_type']
        })


class TxDiversityGUI(QMainWindow):
    """Ventana principal del simulador OFDM con Transmit Diversity (SFBC Alamouti)"""
    
    def __init__(self):
        super().__init__()
        self.ofdm_system = None
        self.current_image_path = None
        self.current_num_rx = 1
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        """Inicializa la interfaz de usuario"""
        self.setWindowTitle("Simulador OFDM-LTE (Transmit Diversity - SFBC Alamouti)")
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
        self.statusBar().showMessage("Listo - Transmit Diversity (2 TX SFBC)")
        
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
        self.iterations_spin.setValue(5)  # Menos iteraciones por defecto (MIMO es más lento)
        layout.addWidget(self.iterations_spin, 4, 1)
        
        # Número de receptores (con 2 TX fijos)
        layout.addWidget(QLabel("Num. Receptores:"), 5, 0)
        self.num_rx_combo = QComboBox()
        self.num_rx_combo.addItems(['1', '2', '4', '8'])
        self.num_rx_combo.setCurrentText('1')
        self.num_rx_combo.currentTextChanged.connect(self.update_config)
        layout.addWidget(self.num_rx_combo, 5, 1)
        
        # Label informativo de TX
        info_label = QLabel("(2 TX fijos con SFBC Alamouti)")
        info_label.setStyleSheet("color: #666; font-style: italic; font-size: 9pt;")
        layout.addWidget(info_label, 6, 0, 1, 2)
        
        # Tipo de canal
        layout.addWidget(QLabel("Tipo de Canal:"), 7, 0)
        self.channel_type_combo = QComboBox()
        self.channel_type_combo.addItems(['AWGN', 'Rayleigh Multitrayecto'])
        self.channel_type_combo.currentTextChanged.connect(self.on_channel_type_changed)
        layout.addWidget(self.channel_type_combo, 7, 1)
        
        # Perfil ITU
        layout.addWidget(QLabel("Perfil ITU-R M.1225:"), 8, 0)
        self.itu_profile_combo = QComboBox()
        self.itu_profile_combo.addItems([
            'Pedestrian_A',
            'Pedestrian_B',
            'Vehicular_A',
            'Vehicular_B',
            'Typical_Urban',
            'Rural_Area'
        ])
        self.itu_profile_combo.setCurrentText('Pedestrian_A')  # Default para MIMO tests
        self.itu_profile_combo.currentTextChanged.connect(self.on_itu_profile_changed)
        self.itu_profile_combo.setEnabled(False)
        layout.addWidget(self.itu_profile_combo, 8, 1)
        
        # Frecuencia portadora
        layout.addWidget(QLabel("Frecuencia (GHz):"), 9, 0)
        self.frequency_spin = QDoubleSpinBox()
        self.frequency_spin.setRange(0.5, 10.0)
        self.frequency_spin.setValue(2.0)
        self.frequency_spin.setSingleStep(0.1)
        self.frequency_spin.valueChanged.connect(self.on_frequency_changed)
        self.frequency_spin.setEnabled(False)
        layout.addWidget(self.frequency_spin, 9, 1)
        
        # Velocidad
        layout.addWidget(QLabel("Velocidad (km/h):"), 10, 0)
        self.velocity_spin = QSpinBox()
        self.velocity_spin.setRange(0, 500)
        self.velocity_spin.setValue(2)  # Default: 2 km/h (como en test MIMO)
        self.velocity_spin.valueChanged.connect(self.on_velocity_changed)
        self.velocity_spin.setEnabled(False)
        layout.addWidget(self.velocity_spin, 10, 1)
        
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
        
        self.multiantenna_btn = QPushButton("Comparación MIMO")
        self.multiantenna_btn.clicked.connect(self.run_mimo_comparison)
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
        self.mimo_comparison_plot = PlotWidget(title='Comparación MIMO')
        self.results_tabs.addTab(self.mimo_comparison_plot, 'Comparación MIMO')
        
        # Tab 5: Métricas
        self.metrics_panel = MetricsPanel()
        self.results_tabs.addTab(self.metrics_panel, 'Métricas')
        
        layout.addWidget(self.results_tabs)
        
        return panel
    
    def update_config(self):
        '''Actualiza la configuración del sistema MIMO'''
        try:
            bandwidth = float(self.bandwidth_combo.currentText())
            delta_f_khz = float(self.delta_f_combo.currentText())
            modulation = self.modulation_combo.currentText()
            cp_type = self.cp_combo.currentText()
            num_rx = int(self.num_rx_combo.currentText())
            
            # Obtener tipo de canal
            channel_type = 'awgn' if self.channel_type_combo.currentText() == 'AWGN' else 'rayleigh_mp'
            itu_profile = self.itu_profile_combo.currentText()
            frequency_ghz = self.frequency_spin.value() if channel_type == 'rayleigh_mp' else 2.0
            velocity_kmh = self.velocity_spin.value() if channel_type == 'rayleigh_mp' else 0
            
            # Crear configuración LTE
            config = LTEConfig(
                bandwidth=bandwidth,
                delta_f=delta_f_khz,  # Ya está en kHz
                modulation=modulation,
                cp_type=cp_type
            )
            
            # Crear sistema MIMO
            self.ofdm_system = OFDMSimulator(
                config=config,
                channel_type=channel_type,
                mode='lte',
                enable_equalization=True,
                num_channels=1,
                itu_profile=itu_profile,
                frequency_ghz=frequency_ghz,
                velocity_kmh=velocity_kmh
            )
            
            # Guardar num_rx para simulaciones
            self.current_num_rx = num_rx
            
            # Actualizar panel de información
            config_info = config.get_info()
            config_info[' MIMO (SFBC) '] = ''
            config_info['Num. Transmisores'] = '2 TX (SFBC Alamouti)'
            config_info['Num. Receptores'] = f'{num_rx} RX'
            config_info['Diversity Order'] = f'{2 * num_rx} (2 TX  {num_rx} RX)'
            config_info[' Canal '] = ''
            config_info['Tipo'] = channel_type.upper()
            
            if channel_type == 'rayleigh_mp':
                config_info['Perfil ITU'] = itu_profile
                config_info['Frecuencia'] = f'{frequency_ghz} GHz'
                config_info['Velocidad'] = f'{velocity_kmh} km/h'
            
            self.info_panel.update_config(config_info)
            
            self.statusBar().showMessage(
                f'Config: 2{num_rx} MIMO, {bandwidth} MHz, {modulation}, {channel_type.upper()}'
            )
            
        except Exception as e:
            import traceback
            error_msg = f'Error al actualizar configuración:\n{str(e)}\n\n{traceback.format_exc()}'
            QMessageBox.critical(self, 'Error', error_msg)
    
    def on_channel_type_changed(self):
        '''Maneja el cambio de tipo de canal'''
        is_rayleigh = self.channel_type_combo.currentText() == 'Rayleigh Multitrayecto'
        self.itu_profile_combo.setEnabled(is_rayleigh)
        self.frequency_spin.setEnabled(is_rayleigh)
        self.velocity_spin.setEnabled(is_rayleigh)
        self.update_config()
    
    def on_itu_profile_changed(self):
        '''Maneja el cambio de perfil ITU'''
        if self.channel_type_combo.currentText() == 'Rayleigh Multitrayecto':
            self.update_config()
    
    def on_frequency_changed(self):
        '''Maneja el cambio de frecuencia'''
        if self.channel_type_combo.currentText() == 'Rayleigh Multitrayecto':
            self.update_config()
    
    def on_velocity_changed(self):
        '''Maneja el cambio de velocidad'''
        if self.channel_type_combo.currentText() == 'Rayleigh Multitrayecto':
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
        '''Ejecuta simulación única MIMO'''
        if not self.ofdm_system:
            self.update_config()
        
        if not self.current_image_path:
            QMessageBox.warning(self, 'Advertencia', 
                              'Por favor carga una imagen para realizar la simulación')
            return
        
        # Preparar parámetros
        channel_type = 'awgn' if self.channel_type_combo.currentText() == 'AWGN' else 'rayleigh_mp'
        
        params = {
            'snr_db': self.snr_spin.value(),
            'image_path': self.current_image_path,
            'num_rx': self.current_num_rx,
            'channel_type': channel_type
        }
        
        # Deshabilitar botones
        self.set_buttons_enabled(False)
        
        # Crear y ejecutar worker
        self.worker = SimulationWorker(self.ofdm_system, 'single', params)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_single_simulation_finished)
        self.worker.start()
    
    def run_sweep_simulation(self):
        '''Ejecuta barrido de SNR para MIMO'''
        if not self.ofdm_system:
            self.update_config()
        
        print(f"\n[DEBUG] run_sweep_simulation() - Estado imagen:")
        print(f"  self.current_image_path = {self.current_image_path}")
        
        # Preparar parámetros
        snr_start = self.snr_start_spin.value()
        snr_end = self.snr_end_spin.value()
        snr_step = self.snr_step_spin.value()
        
        channel_type = 'awgn' if self.channel_type_combo.currentText() == 'AWGN' else 'rayleigh_mp'
        
        params = {
            'snr_range': np.arange(snr_start, snr_end + snr_step, snr_step),
            'n_iterations': self.iterations_spin.value(),
            'channel_type': channel_type,
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
    
    def run_mimo_comparison(self):
        '''Ejecuta comparación completa MIMO'''
        if not self.ofdm_system:
            self.update_config()
        
        if not self.current_image_path:
            QMessageBox.warning(self, 'Advertencia', 
                              'Por favor carga una imagen para realizar la comparación')
            return
        
        channel_type = 'awgn' if self.channel_type_combo.currentText() == 'AWGN' else 'rayleigh_mp'
        
        params = {
            'image_path': self.current_image_path,
            'snr_db': self.snr_spin.value(),
            'channel_type': channel_type
        }
        
        # Deshabilitar botones
        self.set_buttons_enabled(False)
        
        # Crear y ejecutar worker
        self.worker = SimulationWorker(self.ofdm_system, 'mimo_comparison', params)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_mimo_comparison_finished)
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
            'diversity_order': 2 * self.current_num_rx,  # 2 TX  N RX
            'papr_db': results.get('papr_db', 0)
        }
        self.metrics_panel.update_metrics(metrics)
        
        # Mostrar imágenes si existen
        if 'reconstructed_image' in results and 'metadata' in results:
            self.show_image_comparison(results)
        
        # Graficar constelación si existe
        if 'symbols_rx' in results:
            # Para MIMO, usar símbolos decodificados
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
    
    def on_mimo_comparison_finished(self, results):
        '''Maneja finalización de comparación MIMO'''
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)
        
        if 'error' in results:
            QMessageBox.critical(self, 'Error', f'Error en comparación:\n\n{results["error"]}')
            self.statusBar().showMessage('Comparación fallida')
            return
        
        # Graficar comparación MIMO
        self.plot_mimo_comparison(results)
        
        self.statusBar().showMessage('Comparación MIMO completada exitosamente')
    
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
        ax.scatter(rx_symbols.real, rx_symbols.imag, alpha=0.3, s=15, label='RX (SFBC decoded)', color='red')
        ax.set_xlabel('I (In-Phase)')
        ax.set_ylabel('Q (Quadrature)')
        ax.set_title(f'Constelación - MIMO 2{self.current_num_rx} (SFBC Alamouti)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        self.constellation_plot.get_canvas().draw()
        
        # Cambiar a tab de constelación
        self.results_tabs.setCurrentWidget(self.constellation_plot)
    
    def plot_sweep_ber_curves(self, results):
        '''Grafica curvas BER del barrido (4 subplots verticales)'''
        fig = self.ber_plot.get_figure()
        fig.clear()
        
        # Configurar figura para 4 subplots apilados verticalmente
        fig.set_size_inches(12, 16)  # Alto para 4 subplots
        
        num_rx_values = results['num_rx_values']
        modulations = results['modulations']
        all_results = results['results']
        
        # Colores y marcadores por modulación
        mod_colors = {'QPSK': 'blue', '16-QAM': 'green', '64-QAM': 'red'}
        mod_markers = {'QPSK': 'o', '16-QAM': 's', '64-QAM': '^'}
        
        # Crear un subplot por cada num_rx
        for idx, num_rx in enumerate(num_rx_values):
            ax = fig.add_subplot(4, 1, idx + 1)
            
            config_key = f'2x{num_rx}'
            if config_key in all_results:
                rx_data = all_results[config_key]
                
                # Graficar cada modulación
                for mod in modulations:
                    if mod in rx_data:
                        mod_data = rx_data[mod]
                        ax.semilogy(
                            mod_data['snr_values'],
                            mod_data['ber_values'],
                            marker=mod_markers[mod],
                            label=mod,
                            linewidth=2,
                            markersize=6,
                            color=mod_colors[mod]
                        )
                
                ax.set_xlabel('SNR (dB)', fontsize=11)
                ax.set_ylabel('BER', fontsize=11)
                ax.set_title(f'MIMO 2{num_rx} (Diversity Order {2*num_rx})', 
                           fontsize=12, fontweight='bold')
                ax.legend(fontsize=10, loc='best')
                ax.grid(True, which='both', alpha=0.3)
        
        fig.suptitle('BER vs SNR - MIMO Transmit Diversity (SFBC Alamouti)', 
                    fontsize=14, fontweight='bold', y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        
        self.ber_plot.get_canvas().draw()
        
        # Cambiar a tab de curvas BER
        self.results_tabs.setCurrentWidget(self.ber_plot)
    
    def plot_mimo_comparison(self, results):
        '''Grafica comparación MIMO (similar a test_mimo_image.py)'''
        fig = self.mimo_comparison_plot.get_figure()
        fig.clear()
        
        # Cargar imagen original
        from PIL import Image as PILImage
        img_original = PILImage.open(self.current_image_path)
        if img_original.mode != 'RGB':
            img_original = img_original.convert('RGB')
        
        # Configuraciones a mostrar
        configs = ['SISO', 'MISO', 'MIMO-2x2', 'MIMO-2x4']
        
        # Crear grid 2x4 (2 filas, 4 columnas)
        fig.set_size_inches(16, 8)
        
        for idx, config_name in enumerate(configs):
            # Fila 1: Original image (repetida)
            ax_orig = fig.add_subplot(2, 4, idx + 1)
            ax_orig.imshow(img_original)
            ax_orig.set_title('Original', fontsize=11)
            ax_orig.axis('off')
            
            # Fila 2: Reconstructed image
            ax_recon = fig.add_subplot(2, 4, idx + 5)
            
            if config_name in results['results']:
                config_data = results['results'][config_name]
                img_recon = config_data['image']
                ax_recon.imshow(img_recon)
                
                ber = config_data['ber']
                num_tx = config_data['num_tx']
                num_rx = config_data['num_rx']
                
                title = f'{config_name} ({num_tx}{num_rx})\nBER: {ber:.4e}'
            else:
                ax_recon.text(0.5, 0.5, 'Failed', ha='center', va='center',
                            transform=ax_recon.transAxes)
                title = f'{config_name} - Failed'
            
            ax_recon.set_title(title, fontsize=10)
            ax_recon.axis('off')
        
        snr = results['snr_db']
        channel = results['channel_type'].upper()
        fig.suptitle(f'Comparación MIMO - Canal {channel} @ SNR={snr}dB', 
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
