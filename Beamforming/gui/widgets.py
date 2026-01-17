"""
Widgets personalizados para la GUI MIMO - Transmit Diversity
"""
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QFrame, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class PlotWidget(QWidget):
    """Widget para gráficas con Matplotlib"""
    
    def __init__(self, parent=None, title=""):
        super().__init__(parent)
        self.title = title
        self.init_ui()
    
    def init_ui(self):
        """Inicializa el widget"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Crear figura de matplotlib
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(600, 400)
        
        # Barra de herramientas
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
    
    def get_figure(self):
        """Retorna la figura de matplotlib"""
        return self.figure
    
    def get_canvas(self):
        """Retorna el canvas"""
        return self.canvas
    
    def clear(self):
        """Limpia la gráfica"""
        self.figure.clear()
        self.canvas.draw()


class MetricsPanel(QWidget):
    """Panel para mostrar métricas de simulación"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Inicializa el widget"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Título
        title = QLabel("Métricas de Simulación")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Separador
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # Contenedor de métricas
        metrics_container = QWidget()
        metrics_layout = QVBoxLayout()
        metrics_container.setLayout(metrics_layout)
        
        # Labels para métricas (BER, PAPR, Beamforming Gain, Tiempo)
        self.ber_label = QLabel("BER: --")
        self.papr_label = QLabel("PAPR: -- dB")
        self.gain_label = QLabel("BF Gain: -- dB")
        self.time_label = QLabel("Tiempo: -- s")
        
        # Estilo para labels
        label_style = "font-size: 14pt; padding: 10px; font-weight: bold;"
        for label in [self.ber_label, self.papr_label, self.gain_label, self.time_label]:
            label.setStyleSheet(label_style)
            metrics_layout.addWidget(label)
        
        layout.addWidget(metrics_container)
        layout.addStretch()
    
    def update_metrics(self, metrics):
        """Actualiza las métricas mostradas (BER, PAPR, Beamforming Gain, Tiempo)"""
        if 'ber' in metrics:
            self.ber_label.setText(f"BER: {metrics['ber']:.2e}")
        if 'papr_db' in metrics:
            self.papr_label.setText(f"PAPR: {metrics['papr_db']:.2f} dB")
        if 'beamforming_gain_db' in metrics:
            self.gain_label.setText(f"BF Gain: {metrics['beamforming_gain_db']:.2f} dB")
        if 'transmission_time' in metrics:
            self.time_label.setText(f"Tiempo: {metrics['transmission_time']:.3f} s")
    
    def clear(self):
        """Limpia las métricas"""
        self.ber_label.setText("BER: --")
        self.papr_label.setText("PAPR: -- dB")
        self.gain_label.setText("BF Gain: -- dB")
        self.time_label.setText("Tiempo: -- s")


class ConfigInfoPanel(QWidget):
    """Panel para mostrar información de configuración"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Inicializa el widget"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Título
        title = QLabel("Información de Configuración")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Separador
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # Área de texto para configuración
        from PyQt6.QtWidgets import QTextEdit
        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #555;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
                background-color: #2b2b2b;
                color: #e0e0e0;
            }
        """)
        layout.addWidget(self.config_text)
    
    def update_config(self, config_dict):
        """Actualiza la información de configuración"""
        text = "═══ Configuración LTE-MIMO ═══\n\n"
        
        for key, value in config_dict.items():
            text += f"{key}: {value}\n"
        
        self.config_text.setPlainText(text)
    
    def clear(self):
        """Limpia la configuración"""
        self.config_text.clear()


class ImageComparisonWidget(QWidget):
    """Widget para comparar imágenes original y recibida"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Inicializa el widget"""
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # Panel izquierdo - Original
        left_panel = QVBoxLayout()
        left_title = QLabel("Imagen Original")
        left_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(300, 300)
        self.original_label.setStyleSheet("border: 1px solid gray;")
        
        left_panel.addWidget(left_title)
        left_panel.addWidget(self.original_label)
        
        # Panel derecho - Recibida
        right_panel = QVBoxLayout()
        right_title = QLabel("Imagen Recibida")
        right_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        
        self.received_label = QLabel()
        self.received_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.received_label.setMinimumSize(300, 300)
        self.received_label.setStyleSheet("border: 1px solid gray;")
        
        # Métricas de calidad
        self.psnr_label = QLabel("PSNR: --")
        
        metric_style = "font-size: 10pt; padding: 3px;"
        self.psnr_label.setStyleSheet(metric_style)
        
        right_panel.addWidget(right_title)
        right_panel.addWidget(self.received_label)
        right_panel.addWidget(self.psnr_label)
        
        # Agregar paneles al layout principal
        layout.addLayout(left_panel)
        layout.addLayout(right_panel)
    
    def set_images(self, original_pixmap, received_pixmap, psnr=None, ssim=None):
        """
        Establece las imágenes a comparar
        
        Args:
            original_pixmap: QPixmap de la imagen original
            received_pixmap: QPixmap de la imagen recibida
            psnr: Valor PSNR (opcional)
            ssim: Valor SSIM (opcional)
        """
        # Escalar imágenes manteniendo aspecto
        scaled_original = original_pixmap.scaled(
            300, 300, 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        scaled_received = received_pixmap.scaled(
            300, 300,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.original_label.setPixmap(scaled_original)
        self.received_label.setPixmap(scaled_received)
        
        # Actualizar métricas
        if psnr is not None:
            self.psnr_label.setText(f"PSNR: {psnr:.2f} dB")
    
    def clear(self):
        """Limpia las imágenes y widgets adicionales"""
        self.original_label.clear()
        self.received_label.clear()
        self.psnr_label.setText("PSNR: --")
        
        # Limpiar cualquier widget adicional (canvas de matplotlib)
        layout = self.layout()
        while layout.count() > 2:  # Mantener solo los 2 paneles originales
            item = layout.takeAt(2)
            if item.widget():
                item.widget().deleteLater()
