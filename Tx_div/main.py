#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Punto de entrada principal para la aplicación GUI Transmit Diversity (SFBC Alamouti)
"""
import sys
import os

# Agregar el directorio raíz al path para importar módulos del proyecto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from PyQt6.QtWidgets import QApplication
from Tx_div.gui.main_window import TxDiversityGUI


def main():
    """Función principal"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Estilo moderno
    
    window = TxDiversityGUI()
    window.show()
    window.update_config()  # Inicializar configuración
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
