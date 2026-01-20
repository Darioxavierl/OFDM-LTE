#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ventana principal del simulador OFDM con configuraciones LTE
Replica la GUI del repositorio pero usa el core de este proyecto
"""
import sys
import os
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# Agregar el directorio raíz al path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importar desde el proyecto actual (NO del repo)
from config import LTEConfig  # Config de ESTE proyecto
from core.ofdm_core import OFDMSimulator, OFDMTransmitter, OFDMReceiver  # Core de ESTE proyecto
from utils.image_processing import ImageProcessor  # Utils de ESTE proyecto
from SIMO.gui.widgets import PlotWidget, MetricsPanel, ConfigInfoPanel, ImageComparisonWidget


class SimulationWorker(QThread):
    """Worker thread para ejecutar simulaciones sin bloquear la GUI"""
    
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
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.finished.emit({'error': error_msg})
    
    def _run_single_simulation(self):
        """Ejecuta simulación única SIMO"""
        self.progress.emit(10, "Preparando datos...")
        
        if 'image_path' in self.params:
            # Transmisión de imagen
            bits, metadata = ImageProcessor.image_to_bits(self.params['image_path'])
            self.params['metadata'] = metadata
            
            # DEBUG: Verificar longitud de bits
            expected_bits = metadata['height'] * metadata['width'] * metadata['channels'] * 8
            print(f"\n[DEBUG SimulationWorker] Preparación de imagen:")
            print(f"  Imagen: {metadata['width']}x{metadata['height']}x{metadata['channels']}")
            print(f"  Bits TX (originales): {len(bits):,}")
            print(f"  Bits esperados: {expected_bits:,}")
            print(f"  Match: {'✓' if len(bits) == expected_bits else '✗ MISMATCH!'}")
        else:
            # Bits aleatorios
            bits = np.random.randint(0, 2, self.params['n_bits'])
            print(f"[DEBUG] Bits aleatorios: {len(bits):,}")
        
        self.progress.emit(30, "Transmitiendo...")
        
        # DEBUG: Mostrar información del canal
        print(f"\n[DEBUG SimulationWorker] Configuración de canal:")
        print(f"  Tipo: {self.ofdm_system.channel_type}")
        print(f"  SNR: {self.params['snr_db']} dB")
        if hasattr(self.ofdm_system, 'itu_profile'):
            print(f"  Perfil ITU: {self.ofdm_system.itu_profile}")
            print(f"  Frecuencia: {self.ofdm_system.frequency_ghz} GHz")
            print(f"  Velocidad: {self.ofdm_system.velocity_kmh} km/h")
        
        # Usar simulate_simo o simulate_siso según num_rx
        num_rx = self.params.get('num_rx', 1)
        
        print(f"\n[DEBUG SimulationWorker] Simulación:")
        print(f"  Num RX: {num_rx}")
        print(f"  Modo: {'SISO' if num_rx == 1 else 'SIMO con MRC'}")
        
        if num_rx == 1:
            # SISO
            results = self.ofdm_system.simulate_siso(bits, snr_db=self.params['snr_db'])
        else:
            # SIMO con MRC y paralelismo
            results = self.ofdm_system.simulate_simo(
                bits, 
                snr_db=self.params['snr_db'],
                num_rx=num_rx,
                combining='mrc',
                parallel=True  # ✓ Habilitar paralelismo
            )
        
        self.progress.emit(70, "Procesando resultados...")
        
        # DEBUG: Verificar bits recibidos
        print(f"\n[DEBUG SimulationWorker] Después de simulación:")
        print(f"  Num RX usado: {results.get('num_rx', 1)}")
        print(f"  Bits RX (recibidos): {len(results['bits_received_array']):,}")
        print(f"  BER: {results['ber']:.2e}")
        print(f"  Errores: {results.get('bit_errors', 0):,}")
        print(f"  PAPR: {results.get('papr_db', 0):.2f} dB")
        
        if 'metadata' in self.params:
            # Reconstruir imagen
            print(f"\n[DEBUG SimulationWorker] Reconstrucción de imagen:")
            print(f"  Usando bits_received_array: {len(results['bits_received_array']):,}")
            
            reconstructed_img = ImageProcessor.bits_to_image(
                results['bits_received_array'], 
                self.params['metadata']
            )
            results['reconstructed_image'] = reconstructed_img
            results['metadata'] = self.params['metadata']
            print(f"  Imagen reconstruida: {reconstructed_img.size}")
        
        self.progress.emit(100, "Completado")
        self.finished.emit(results)
    
    def _run_sweep_simulation(self):
        """
        Ejecuta barrido completo de SNR:
        - 3 modulaciones (QPSK, 16-QAM, 64-QAM)
        - 4 configuraciones RX (1, 2, 4, 8 antenas)
        - Rango de SNR configurable
        
        Genera 3 subplots, uno por modulación, cada uno con 4 curvas (1/2/4/8 RX)
        """
        self.progress.emit(5, "Preparando datos...")
        
        # Extraer bits
        bits_to_sweep = None
        if 'image_path' in self.params:
            self.progress.emit(10, "Cargando imagen...")
            bits_to_sweep, metadata = ImageProcessor.image_to_bits(self.params['image_path'])
            self.params['metadata'] = metadata
            print(f"\n[DEBUG SimulationWorker] Barrido SNR Completo - Preparación:")
            print(f"  Imagen cargada: {len(bits_to_sweep):,} bits")
        else:
            # Bits aleatorios
            bits_to_sweep = np.random.randint(0, 2, 50000)
            print(f"\n[DEBUG SimulationWorker] Barrido SNR - Usando bits aleatorios: {len(bits_to_sweep):,}")
        
        # Parámetros del barrido
        modulations = ['QPSK', '16-QAM', '64-QAM']
        num_rx_values = [1, 2, 4, 8]
        snr_range = self.params['snr_range']
        n_iterations = self.params['n_iterations']
        
        print(f"  Modulaciones: {modulations}")
        print(f"  Num RX: {num_rx_values}")
        print(f"  Rango SNR: {snr_range[0]:.1f} a {snr_range[-1]:.1f} dB (paso {snr_range[1]-snr_range[0]:.1f})")
        print(f"  Iteraciones por punto: {n_iterations}")
        
        # Calcular pasos totales
        total_steps = len(modulations) * len(num_rx_values) * len(snr_range) * n_iterations
        current_step = 0
        
        # Guardar configuración original
        original_modulation = self.ofdm_system.config.modulation
        
        # Resultados organizados por modulación
        all_results = {}
        
        self.progress.emit(15, "Iniciando barrido completo...")
        
        # Iterar sobre modulaciones
        for mod in modulations:
            print(f"\n=== Modulación: {mod} ===")
            
            # Actualizar modulación en el sistema
            self.ofdm_system.config.modulation = mod
            self.ofdm_system.config._calculate_parameters()  # Recalcular parámetros derivados
            # Recrear TX y RX con nueva modulación
            self.ofdm_system.tx = OFDMTransmitter(
                self.ofdm_system.config,
                mode=self.ofdm_system.mode,
                enable_sc_fdm=self.ofdm_system.enable_sc_fdm
            )
            self.ofdm_system.rx = OFDMReceiver(
                self.ofdm_system.config,
                mode=self.ofdm_system.mode,
                enable_equalization=self.ofdm_system.enable_equalization,
                enable_sc_fdm=self.ofdm_system.enable_sc_fdm
            )
            
            mod_results = {}
            
            # Iterar sobre num_rx
            for num_rx in num_rx_values:
                print(f"  Testing {num_rx} RX...")
                
                ber_results = []
                snr_values_list = []
                
                # Iterar sobre SNR
                for snr_db in snr_range:
                    ber_list = []
                    
                    # Múltiples iteraciones para promediar
                    for iter_idx in range(n_iterations):
                        current_step += 1
                        progress_pct = int(15 + (current_step / total_steps) * 80)
                        self.progress.emit(progress_pct, 
                                          f"{mod} | {num_rx}RX | SNR={snr_db:.1f}dB | Iter {iter_idx+1}/{n_iterations}")
                        
                        # Simular según num_rx
                        if num_rx == 1:
                            result = self.ofdm_system.simulate_siso(bits_to_sweep, snr_db)
                        else:
                            result = self.ofdm_system.simulate_simo(
                                bits_to_sweep, 
                                snr_db,
                                num_rx=num_rx,
                                parallel=True
                            )
                        
                        ber_list.append(result['ber'])
                    
                    # Promedio de iteraciones
                    avg_ber = np.mean(ber_list)
                    ber_results.append(avg_ber)
                    snr_values_list.append(snr_db)
                
                # Guardar resultados para este num_rx
                mod_results[f'{num_rx}RX'] = {
                    'snr_values': np.array(snr_values_list),
                    'ber_values': np.array(ber_results),
                    'num_rx': num_rx
                }
                
                print(f"    {num_rx} RX → BER: {min(ber_results):.2e} (mejor) a {max(ber_results):.2e} (peor)")
            
            # Guardar resultados de esta modulación
            all_results[mod] = mod_results
        
        # Restaurar modulación original
        self.ofdm_system.config.modulation = original_modulation
        self.ofdm_system.config._calculate_parameters()  # Recalcular parámetros derivados
        # Recrear TX y RX con modulación original
        self.ofdm_system.tx = OFDMTransmitter(
            self.ofdm_system.config,
            mode=self.ofdm_system.mode,
            enable_sc_fdm=self.ofdm_system.enable_sc_fdm
        )
        self.ofdm_system.rx = OFDMReceiver(
            self.ofdm_system.config,
            mode=self.ofdm_system.mode,
            enable_equalization=self.ofdm_system.enable_equalization,
            enable_sc_fdm=self.ofdm_system.enable_sc_fdm
        )
        
        # Preparar resultados finales
        results = {
            'mode': 'sweep_full',
            'modulations': modulations,
            'num_rx_values': num_rx_values,
            'data': all_results
        }
        
        print(f"\n[DEBUG SimulationWorker] Barrido completo finalizado")
        print(f"  Total simulaciones: {current_step}")
        
        self.progress.emit(100, "Barrido completado")
        self.finished.emit(results)
            




class MultiantennaWorker(QThread):
    """Worker thread para prueba de diversidad multiantena"""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    
    def __init__(self, ofdm_system, image_path, snr_db):
        super().__init__()
        self.ofdm_system = ofdm_system
        self.image_path = image_path
        self.snr_db = snr_db
    
    def run(self):
        """Ejecuta prueba comparando 1, 2, 4, 8 RX"""
        try:
            self.progress.emit(5, "Cargando imagen...")
            bits, metadata = ImageProcessor.image_to_bits(self.image_path)
            
            # DEBUG: Verificar longitud de bits
            expected_bits = metadata['height'] * metadata['width'] * metadata['channels'] * 8
            print(f"\n[DEBUG MultiantennaWorker] Preparación de imagen:")
            print(f"  Imagen: {metadata['width']}x{metadata['height']}x{metadata['channels']}")
            print(f"  Bits TX (originales): {len(bits):,}")
            print(f"  Bits esperados: {expected_bits:,}")
            print(f"  Match: {'✓' if len(bits) == expected_bits else '✗ MISMATCH!'}")
            
            num_rx_values = [1, 2, 4, 8]
            results = {}
            
            for idx, num_rx in enumerate(num_rx_values):
                progress_pct = 10 + (idx * 20)
                self.progress.emit(progress_pct, f"Testeando {num_rx} RX...")
                
                print(f"\n[DEBUG MultiantennaWorker] Simulando {num_rx} RX...")
                print(f"  Paralelismo: {'✓ Habilitado' if num_rx > 1 else 'N/A (SISO)'}")
                
                if num_rx == 1:
                    result = self.ofdm_system.simulate_siso(bits, snr_db=self.snr_db)
                else:
                    # SIMO con paralelismo habilitado explícitamente
                    result = self.ofdm_system.simulate_simo(
                        bits, 
                        snr_db=self.snr_db,
                        num_rx=num_rx,
                        combining='mrc',
                        parallel=True  # ✓ Habilitar paralelismo
                    )
                
                print(f"  Bits RX: {len(result['bits_received_array']):,}")
                print(f"  BER: {result['ber']:.2e}")
                
                # Reconstruir imagen
                img_recon = ImageProcessor.bits_to_image(result['bits_received_array'], metadata)
                print(f"  Imagen reconstruida: {img_recon.size}")
                
                # Calcular métricas
                from PIL import Image as PILImage
                img_original = PILImage.open(self.image_path)
                psnr = ImageProcessor.calculate_psnr(img_original, img_recon)
                ssim = ImageProcessor.calculate_ssim(img_original, img_recon)
                
                results[f'{num_rx}RX'] = {
                    'num_rx': num_rx,
                    'ber': result['ber'],
                    'papr_db': result['papr_db'],
                    'image': img_recon,
                    'psnr': psnr,
                    'ssim': ssim
                }
            
            results['metadata'] = metadata
            results['test_type'] = 'multiantenna'
            
            self.progress.emit(100, "Prueba completada")
            self.finished.emit(results)
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.finished.emit({'error': error_msg})


class OFDMSimulatorGUI(QMainWindow):
    """Ventana principal del simulador OFDM"""
    
    def __init__(self):
        super().__init__()
        self.ofdm_system = None
        self.current_image_path = None
        self.current_num_rx = 1
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        """Inicializa la interfaz de usuario"""
        self.setWindowTitle("Simulador OFDM-LTE (SIMO)")
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
        self.statusBar().showMessage("Listo")
        
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
        """Crea grupo de parámetros LTE (IDÉNTICO AL REPO)"""
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
        self.bandwidth_combo.setCurrentText('5')
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
        """Crea grupo de parámetros de simulación (IDÉNTICO AL REPO)"""
        group = QGroupBox("Parámetros de Simulación")
        layout = QGridLayout()
        
        # SNR para simulación única
        layout.addWidget(QLabel("SNR (dB):"), 0, 0)
        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(-10, 40)
        self.snr_spin.setValue(10.0)
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
        self.iterations_spin.setValue(10)
        layout.addWidget(self.iterations_spin, 4, 1)
        
        # Número de receptores (SIMO)
        layout.addWidget(QLabel("Num. Receptores:"), 5, 0)
        self.num_rx_combo = QComboBox()
        self.num_rx_combo.addItems(['1', '2', '4', '8'])
        self.num_rx_combo.setCurrentText('1')
        self.num_rx_combo.currentTextChanged.connect(self.update_config)
        layout.addWidget(self.num_rx_combo, 5, 1)
        
        # Tipo de canal
        layout.addWidget(QLabel("Tipo de Canal:"), 6, 0)
        self.channel_type_combo = QComboBox()
        self.channel_type_combo.addItems(['AWGN', 'Rayleigh Multitrayecto'])
        self.channel_type_combo.currentTextChanged.connect(self.on_channel_type_changed)
        layout.addWidget(self.channel_type_combo, 6, 1)
        
        # Perfil ITU
        layout.addWidget(QLabel("Perfil ITU-R M.1225:"), 7, 0)
        self.itu_profile_combo = QComboBox()
        self.itu_profile_combo.addItems([
            'Pedestrian_A',
            'Pedestrian_B',
            'Vehicular_A',
            'Vehicular_B',
            'Typical_Urban',
            'Rural_Area'
        ])
        self.itu_profile_combo.setCurrentText('Vehicular_A')
        self.itu_profile_combo.currentTextChanged.connect(self.on_itu_profile_changed)
        self.itu_profile_combo.setEnabled(False)  # Deshabilitado por defecto
        layout.addWidget(self.itu_profile_combo, 7, 1)
        
        # Frecuencia portadora
        layout.addWidget(QLabel("Frecuencia (GHz):"), 8, 0)
        self.frequency_spin = QDoubleSpinBox()
        self.frequency_spin.setRange(0.5, 10.0)
        self.frequency_spin.setValue(2.0)
        self.frequency_spin.setSingleStep(0.1)
        self.frequency_spin.valueChanged.connect(self.on_frequency_changed)
        self.frequency_spin.setEnabled(False)
        layout.addWidget(self.frequency_spin, 8, 1)
        
        # Velocidad
        layout.addWidget(QLabel("Velocidad (km/h):"), 9, 0)
        self.velocity_spin = QSpinBox()
        self.velocity_spin.setRange(0, 500)
        self.velocity_spin.setValue(3)  # Valor por defecto: 3 km/h (peatón)
        self.velocity_spin.valueChanged.connect(self.on_velocity_changed)
        self.velocity_spin.setEnabled(False)
        layout.addWidget(self.velocity_spin, 9, 1)
        
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
        self.multiantenna_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        layout.addWidget(self.multiantenna_btn)
        
        group.setLayout(layout)
        return group
    
    def create_results_panel(self):
        """Crea el panel de resultados (derecha)"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Tabs para diferentes resultados
        self.results_tabs = QTabWidget()
        
        # Tab 1: Constelación
        self.constellation_plot = PlotWidget(title="Constelación")
        self.results_tabs.addTab(self.constellation_plot, "Constelación")
        
        # Tab 2: BER vs SNR
        self.ber_plot = PlotWidget(title="BER vs SNR")
        self.results_tabs.addTab(self.ber_plot, "BER vs SNR")
        
        # Tab 3: PAPR
        self.papr_plot = PlotWidget(title="PAPR")
        self.results_tabs.addTab(self.papr_plot, "PAPR")
        
        # Tab 4: Imágenes
        self.image_comparison = ImageComparisonWidget()
        self.results_tabs.addTab(self.image_comparison, "Imágenes")
        
        # Tab 5: Multiantena (SIMO)
        self.multiantenna_plot = PlotWidget(title="Prueba Multiantena")
        self.results_tabs.addTab(self.multiantenna_plot, "Multiantena")
        
        # Tab 6: Métricas
        self.metrics_panel = MetricsPanel()
        self.results_tabs.addTab(self.metrics_panel, "Métricas")
        
        layout.addWidget(self.results_tabs)
        
        return panel
    
    def update_config(self):
        """Actualiza la configuración del sistema (USANDO OFDMSimulator SIMO)"""
        try:
            bandwidth = float(self.bandwidth_combo.currentText())
            delta_f = float(self.delta_f_combo.currentText())
            modulation = self.modulation_combo.currentText()
            cp_type = self.cp_combo.currentText()
            num_rx = int(self.num_rx_combo.currentText())
            
            # Obtener tipo de canal
            channel_type = 'awgn' if self.channel_type_combo.currentText() == 'AWGN' else 'rayleigh_mp'
            itu_profile = self.itu_profile_combo.currentText()
            frequency_ghz = self.frequency_spin.value() if channel_type == 'rayleigh_mp' else 2.0
            velocity_kmh = self.velocity_spin.value() if channel_type == 'rayleigh_mp' else 0
            
            # Crear configuración LTE usando el CONFIG de ESTE proyecto
            config = LTEConfig(bandwidth, delta_f, modulation, cp_type)
            
            # Importar OFDMSimulator (core de ESTE proyecto)
            from core.ofdm_core import OFDMSimulator
            
            # Crear simulador SIMO usando el CORE de ESTE proyecto
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
            config_info['═══ SIMO ═══'] = ''
            config_info['Num. Receptores'] = f"{num_rx} RX"
            config_info['Combining'] = 'MRC (Maximum Ratio Combining)' if num_rx > 1 else 'N/A'
            config_info['═══ Canal ═══'] = ''
            config_info['Tipo'] = channel_type.upper()
            
            if channel_type == 'rayleigh_mp':
                config_info['Perfil ITU'] = itu_profile
                config_info['Frecuencia'] = f"{frequency_ghz} GHz"
                config_info['Velocidad'] = f"{velocity_kmh} km/h"
            
            self.info_panel.update_config(config_info)
            
            self.statusBar().showMessage(f"Config: {bandwidth} MHz, {modulation}, {num_rx} RX, {channel_type.upper()}")
            
        except Exception as e:
            import traceback
            error_msg = f"Error al actualizar configuración:\n{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
    
    def on_channel_type_changed(self):
        """Maneja el cambio de tipo de canal"""
        is_rayleigh = self.channel_type_combo.currentText() == 'Rayleigh Multitrayecto'
        self.itu_profile_combo.setEnabled(is_rayleigh)
        self.frequency_spin.setEnabled(is_rayleigh)
        self.velocity_spin.setEnabled(is_rayleigh)
        self.update_config()
    
    def on_itu_profile_changed(self):
        """Maneja el cambio de perfil ITU"""
        if self.channel_type_combo.currentText() == 'Rayleigh Multitrayecto':
            self.update_config()
    
    def on_frequency_changed(self):
        """Maneja el cambio de frecuencia"""
        if self.channel_type_combo.currentText() == 'Rayleigh Multitrayecto':
            self.update_config()
    
    def on_velocity_changed(self):
        """Maneja el cambio de velocidad"""
        if self.channel_type_combo.currentText() == 'Rayleigh Multitrayecto':
            self.update_config()
    
    def load_image(self):
        """Carga una imagen para transmitir"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
            self.statusBar().showMessage(f"Imagen cargada: {os.path.basename(file_path)}")
    
    def run_single_simulation(self):
        """Ejecuta simulación única"""
        if not self.ofdm_system:
            self.update_config()
        
        if not self.current_image_path:
            QMessageBox.warning(self, "Advertencia", 
                              "Por favor carga una imagen para realizar la simulación")
            return
        
        # Preparar parámetros
        params = {
            'snr_db': self.snr_spin.value(),
            'image_path': self.current_image_path,
            'num_rx': self.current_num_rx
        }
        
        # Deshabilitar botones
        self.set_buttons_enabled(False)
        
        # Crear y ejecutar worker
        self.worker = SimulationWorker(self.ofdm_system, 'single', params)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_single_simulation_finished)
        self.worker.start()
    
    def run_sweep_simulation(self):
        """Ejecuta barrido de SNR con num_rx fijo"""
        if not self.ofdm_system:
            self.update_config()
        
        # Preparar parámetros
        snr_start = self.snr_start_spin.value()
        snr_end = self.snr_end_spin.value()
        snr_step = self.snr_step_spin.value()
        
        params = {
            'snr_range': np.arange(snr_start, snr_end + snr_step, snr_step),
            'n_iterations': self.iterations_spin.value(),
            'num_rx': self.current_num_rx  # <-- AÑADIDO: usar num_rx seleccionado
        }
        
        # Si hay imagen, usarla
        if self.current_image_path:
            params['image_path'] = self.current_image_path
        else:
            QMessageBox.warning(self, "Advertencia", 
                              "Por favor carga una imagen para realizar el barrido")
            return
        
        # Deshabilitar botones
        self.set_buttons_enabled(False)
        
        # Crear y ejecutar worker
        self.worker = SimulationWorker(self.ofdm_system, 'sweep', params)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_sweep_simulation_finished)
        self.worker.start()
    
    def run_multiantenna_test(self):
        """Ejecuta prueba de diversidad multiantena (compara 1, 2, 4, 8 RX)"""
        if not self.ofdm_system:
            self.update_config()
        
        if not self.current_image_path:
            QMessageBox.warning(self, "Advertencia", 
                              "Por favor carga una imagen para realizar la prueba")
            return
        
        # Crear worker especial para multiantena
        self.worker = MultiantennaWorker(self.ofdm_system, self.current_image_path, 
                                        self.snr_spin.value())
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_multiantenna_test_finished)
        
        # Deshabilitar botones
        self.set_buttons_enabled(False)
        
        self.worker.start()
    
    def update_progress(self, value, message):
        """Actualiza barra de progreso"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)
    
    def on_single_simulation_finished(self, results):
        """Maneja finalización de simulación única"""
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)
        
        if 'error' in results:
            QMessageBox.critical(self, "Error", f"Error en simulación:\n\n{results['error']}")
            self.statusBar().showMessage("Simulación fallida")
            return
        
        # Actualizar métricas (solo BER, PAPR, Tiempo)
        metrics = {
            'ber': results.get('ber', 0),
            'papr_db': results.get('papr_db', 0),
            'transmission_time': results.get('transmission_time', 0)
        }
        self.metrics_panel.update_metrics(metrics)
        
        # Cambiar a tab de métricas
        self.results_tabs.setCurrentWidget(self.metrics_panel)
        
        # Graficar constelación
        # Para SISO: usa 'symbols_rx'
        # Para SIMO: usa 'symbols_rx_combined'
        symbols_rx = results.get('symbols_rx')
        if symbols_rx is None:
            symbols_rx = results.get('symbols_rx_combined')
        if 'symbols_tx' in results and symbols_rx is not None:
            self.plot_constellation(results['symbols_tx'], symbols_rx)
        
        # Graficar PAPR
        if 'papr_values' in results:
            self.plot_papr(results['papr_values'])
        
        # Mostrar imágenes
        if 'reconstructed_image' in results:
            self.show_image_comparison(results)
        
        self.statusBar().showMessage("Simulación completada exitosamente")
    
    def on_sweep_simulation_finished(self, results):
        """Maneja finalización de barrido SNR"""
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)
        
        if 'error' in results:
            QMessageBox.critical(self, "Error", f"Error en barrido:\n\n{results['error']}")
            self.statusBar().showMessage("Barrido fallido")
            return
        
        # Graficar curvas BER
        self.plot_ber_curves(results)
        
        self.statusBar().showMessage("Barrido completado exitosamente")
    
    def on_multiantenna_test_finished(self, results):
        """Maneja finalización de prueba multiantena"""
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)
        
        if 'error' in results:
            QMessageBox.critical(self, "Error", f"Error en prueba:\n\n{results['error']}")
            self.statusBar().showMessage("Prueba fallida")
            return
        
        # Crear visualización de comparación
        self.plot_multiantenna_comparison(results)
        
        # Actualizar métricas con baseline (1 RX)
        if '1RX' in results:
            metrics = {
                'ber': results['1RX']['ber'],
                'papr_db': results['1RX'].get('papr_db', 0),
                'transmission_time': results.get('transmission_time', 0)
            }
            self.metrics_panel.update_metrics(metrics)
        
        self.statusBar().showMessage("Prueba multiantena completada exitosamente")
    
    def plot_constellation(self, tx_symbols, rx_symbols):
        """Grafica constelación"""
        fig = self.constellation_plot.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)
        
        # Convertir a numpy arrays si no lo son
        tx_symbols = np.array(tx_symbols)
        rx_symbols = np.array(rx_symbols)
        
        # Muestrear símbolos si son muchos
        max_symbols = 1000
        if len(tx_symbols) > max_symbols:
            indices = np.random.choice(len(tx_symbols), max_symbols, replace=False)
            tx_symbols = tx_symbols[indices]
            rx_symbols = rx_symbols[indices]
        
        ax.scatter(tx_symbols.real, tx_symbols.imag, alpha=0.5, s=20, label='TX', color='blue')
        ax.scatter(rx_symbols.real, rx_symbols.imag, alpha=0.5, s=20, label='RX', color='red')
        ax.set_xlabel('I (In-Phase)')
        ax.set_ylabel('Q (Quadrature)')
        ax.set_title('Diagrama de Constelación')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        self.constellation_plot.get_canvas().draw()
    
    def plot_papr(self, papr_values):
        """Grafica PAPR"""
        fig = self.papr_plot.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)
        
        papr_db = 10 * np.log10(papr_values)
        
        ax.hist(papr_db, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('PAPR (dB)')
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'Distribución de PAPR (Media: {np.mean(papr_db):.2f} dB)')
        ax.grid(True, alpha=0.3)
        
        self.papr_plot.get_canvas().draw()
    
    def plot_ber_curves(self, results):
        """Grafica curvas BER"""
        fig = self.ber_plot.get_figure()
        fig.clear()
        
        # Caso 1: Barrido completo (3 modulaciones × 4 num_rx)
        if results.get('mode') == 'sweep_full':
            modulations = results['modulations']
            num_rx_values = results['num_rx_values']
            data = results['data']
            
            # Crear 3 subplots (uno por modulación)
            fig.set_size_inches(15, 5)
            
            colors = {1: 'blue', 2: 'green', 4: 'red', 8: 'purple'}
            markers = {1: 'o', 2: 's', 4: '^', 8: 'd'}
            
            for idx, mod in enumerate(modulations):
                ax = fig.add_subplot(1, 3, idx + 1)
                
                mod_data = data[mod]
                
                # Graficar cada num_rx
                for num_rx in num_rx_values:
                    key = f'{num_rx}RX'
                    if key in mod_data:
                        rx_data = mod_data[key]
                        ax.semilogy(rx_data['snr_values'], rx_data['ber_values'],
                                   marker=markers[num_rx], 
                                   label=f"{num_rx} RX",
                                   linewidth=2, 
                                   markersize=6, 
                                   color=colors[num_rx])
                
                ax.set_xlabel('SNR (dB)', fontsize=11)
                ax.set_ylabel('BER', fontsize=11)
                ax.set_title(f'{mod}', fontsize=12, fontweight='bold')
                ax.legend(fontsize=9, loc='best')
                ax.grid(True, which='both', alpha=0.3)
            
            fig.suptitle('BER vs SNR - Comparación de Diversidad SIMO por Modulación', 
                        fontsize=14, fontweight='bold', y=1.02)
            fig.tight_layout()
        
        # Caso 2: Barrido simple con num_rx fijo (formato antiguo, por si se necesita)
        elif 'mode' in results and results['mode'] == 'sweep':
            ax = fig.add_subplot(111)
            num_rx = results.get('num_rx', 1)
            snr_values = results['snr_values']
            ber_values = results['ber_values']
            
            # Determinar color según num_rx
            colors = {1: 'blue', 2: 'green', 4: 'red', 8: 'purple'}
            markers = {1: 'o', 2: 's', 4: '^', 8: 'd'}
            color = colors.get(num_rx, 'blue')
            marker = markers.get(num_rx, 'o')
            
            ax.semilogy(snr_values, ber_values, 
                       marker=marker, label=f"{num_rx} RX", 
                       linewidth=2, markersize=6, color=color)
            
            ax.set_xlabel('SNR (dB)', fontsize=12)
            ax.set_ylabel('BER', fontsize=12)
            ax.set_title(f'BER vs SNR ({num_rx} antena{"s" if num_rx > 1 else ""} receptora{"s" if num_rx > 1 else ""})', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, which='both', alpha=0.3)
        
        # Caso 3: Comparación de múltiples num_rx (formato muy antiguo)
        elif results.get('sweep_type') == 'num_rx_comparison':
            ax = fig.add_subplot(111)
            colors = ['blue', 'green', 'red', 'purple']
            markers = ['o', 's', '^', 'd']
            
            for idx, (key, color, marker) in enumerate(zip(['1RX', '2RX', '4RX', '8RX'], colors, markers)):
                if key in results:
                    rx_results = results[key]
                    ax.semilogy(rx_results['snr_values'], rx_results['ber_mean'], 
                               marker=marker, label=f"{rx_results['num_rx']} RX", 
                               linewidth=2, markersize=6, color=color)
            
            ax.set_xlabel('SNR (dB)', fontsize=12)
            ax.set_ylabel('BER', fontsize=12)
            ax.set_title('BER vs SNR - Comparación SIMO (Diversity Gain)', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, which='both', alpha=0.3)
        
        # Caso 4: Por modulación sin num_rx (formato obsoleto)
        else:
            ax = fig.add_subplot(111)
            modulations_found = []
            for mod in ['QPSK', '16-QAM', '64-QAM']:
                if mod in results:
                    mod_results = results[mod]
                    ax.semilogy(mod_results['snr_values'], mod_results['ber_mean'], 
                               marker='o', label=mod, linewidth=2, markersize=6)
                    modulations_found.append(mod)
            
            if modulations_found:
                ax.set_xlabel('SNR (dB)', fontsize=12)
                ax.set_ylabel('BER', fontsize=12)
                ax.set_title('BER vs SNR para diferentes modulaciones', 
                            fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, which='both', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No hay datos para graficar', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
        
        self.ber_plot.get_canvas().draw()
        
        # Cambiar a tab BER vs SNR
        self.results_tabs.setCurrentWidget(self.ber_plot)
    
    def plot_multiantenna_comparison(self, results):
        """
        Muestra comparación de diversidad multiantena EXACTAMENTE como test_simo_image.py
        Layout: 2 filas x 4 columnas
        - Fila 1: Original repetida 4 veces (una por cada configuración)
        - Fila 2: Imágenes reconstruidas con 1, 2, 4, 8 RX
        """
        fig = self.multiantenna_plot.get_figure()
        fig.clear()
        
        # Cargar imagen original
        from PIL import Image as PILImage
        img_original = PILImage.open(self.current_image_path)
        if img_original.mode != 'RGB':
            img_original = img_original.convert('RGB')
        
        # Configuraciones a mostrar
        num_rx_configs = [1, 2, 4, 8]
        
        # Crear grid 2x4 (2 filas, 4 columnas)
        for idx, num_rx in enumerate(num_rx_configs):
            # Fila 1: Original image (repetida en cada columna)
            ax_orig = fig.add_subplot(2, 4, idx + 1)
            ax_orig.imshow(img_original)
            ax_orig.set_title('Original Image', fontsize=11)
            ax_orig.axis('off')
            
            # Fila 2: Reconstructed image
            ax_recon = fig.add_subplot(2, 4, idx + 5)  # Segunda fila (4 + idx + 1)
            
            key = f'{num_rx}RX'
            if key in results:
                img_recon = results[key]['image']
                ax_recon.imshow(img_recon)
                ber = results[key]['ber']
                title = f"{num_rx} RX - BER: {ber:.4e}"
            else:
                ax_recon.text(0.5, 0.5, 'Failed', ha='center', va='center',
                            transform=ax_recon.transAxes)
                title = f"{num_rx} RX - Failed"
            
            ax_recon.set_title(title, fontsize=11)
            ax_recon.axis('off')
        
        fig.tight_layout()
        self.multiantenna_plot.get_canvas().draw()
        
        # Cambiar a tab de Multiantena
        self.results_tabs.setCurrentWidget(self.multiantenna_plot)
    
    def show_image_comparison(self, results):
        """Muestra comparación de imágenes"""
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
            error_msg = f"Error al mostrar imágenes:\n{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.warning(self, "Advertencia", error_msg)
    
    def set_buttons_enabled(self, enabled):
        """Habilita/deshabilita botones"""
        self.single_sim_btn.setEnabled(enabled)
        self.sweep_sim_btn.setEnabled(enabled)
        self.multiantenna_btn.setEnabled(enabled)
        self.load_image_btn.setEnabled(enabled)


def main():
    """Función principal"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = OFDMSimulatorGUI()
    window.show()
    window.update_config()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
