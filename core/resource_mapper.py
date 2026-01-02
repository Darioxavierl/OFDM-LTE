"""
ResourceMapper - Mapeo de subportadoras siguiendo estándar LTE

Implementa:
- Subportadora DC (centro del espectro)
- Guard bands (bandas guardias)
- Reference Signals (pilotos/señales de referencia)
- Data subcarriers (datos)

Basado en 3GPP LTE Specification TS 36.211
"""

import numpy as np
from typing import Dict, Tuple, List


class LTEResourceGrid:
    """
    Grid de recursos LTE con estructura de subportadoras
    
    Estructura para un símbolo OFDM:
    ┌─────────────┬─────────────────┬──────────────┐
    │  Guard band │    Data + Pilot  │  Guard band  │
    │  (izquierdo)│                  │  (derecho)   │
    └─────────────┴─────────────────┴──────────────┘
    
    Dentro de Data + Pilot:
    - DC (subportadora de continua) en el centro
    - Pilotos cada 6 subportadoras (patrón LTE)
    - Datos en las posiciones restantes
    """
    
    def __init__(self, N: int, Nc: int):
        """
        Inicializa la grid de recursos LTE
        
        Args:
            N: Tamaño del FFT
            Nc: Número de subportadoras útiles
        """
        self.N = N  # Tamaño FFT
        self.Nc = Nc  # Subportadoras útiles
        
        # Cálculo de guardias (simétrico)
        self.num_guard_left = (N - Nc) // 2
        self.num_guard_right = N - Nc - self.num_guard_left
        
        # DC siempre en el centro del FFT (N/2)
        self.dc_index = N // 2
        
        # Patrón de pilotos LTE: cada 6 subportadoras
        self.pilot_spacing = 6
        
        # Inicializar tipos de subportadoras
        self._init_subcarrier_types()
    
    def _init_subcarrier_types(self):
        """Clasifica cada subportadora según su tipo"""
        self.subcarrier_types = {}  # k -> 'data', 'pilot', 'dc', 'guard'
        
        for k in range(self.N):
            if k < self.num_guard_left or k >= self.N - self.num_guard_right:
                # Guard bands
                self.subcarrier_types[k] = 'guard'
            elif k == self.dc_index:
                # Subportadora DC
                self.subcarrier_types[k] = 'dc'
            else:
                # Dentro del rango útil (sin DC)
                # Pilotos cada 6 subportadoras en patrón LTE
                if (k - self.num_guard_left) % self.pilot_spacing == self.pilot_spacing // 2:
                    self.subcarrier_types[k] = 'pilot'
                else:
                    self.subcarrier_types[k] = 'data'
    
    def get_subcarrier_type(self, k: int) -> str:
        """Retorna el tipo de subportadora en índice k"""
        return self.subcarrier_types.get(k, 'guard')
    
    def get_data_indices(self) -> np.ndarray:
        """Retorna índices de subportadoras de datos"""
        return np.array([k for k in range(self.N) 
                        if self.subcarrier_types[k] == 'data'])
    
    def get_pilot_indices(self) -> np.ndarray:
        """Retorna índices de subportadoras de pilotos"""
        return np.array([k for k in range(self.N) 
                        if self.subcarrier_types[k] == 'pilot'])
    
    def get_guard_indices(self) -> np.ndarray:
        """Retorna índices de subportadoras de guarda"""
        return np.array([k for k in range(self.N) 
                        if self.subcarrier_types[k] == 'guard'])
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas de la grid"""
        num_data = len(self.get_data_indices())
        num_pilot = len(self.get_pilot_indices())
        num_guard = len(self.get_guard_indices())
        
        return {
            'total_subcarriers': self.N,
            'useful_subcarriers': self.Nc,
            'data_subcarriers': num_data,
            'pilot_subcarriers': num_pilot,
            'guard_subcarriers': num_guard,
            'dc_subcarriers': 1,
            'guard_left': self.num_guard_left,
            'guard_right': self.num_guard_right,
            'pilot_spacing': self.pilot_spacing,
        }


class PilotPattern:
    """
    Patrón de pilotos de referencia LTE
    
    LTE usa Demodulation Reference Signals (DMRS) con patrón conocido
    Para simplificar, usamos pilotos determinísticos basados en PN sequence
    """
    
    def __init__(self, cell_id: int = 0, pilot_symbol_value: complex = None):
        """
        Inicializa el patrón de pilotos
        
        Args:
            cell_id: ID de celda (0-167 en LTE)
            pilot_symbol_value: Símbolo de piloto (si None, usa normalizado)
        """
        self.cell_id = cell_id
        # Símbolo de piloto normalizado QPSK
        if pilot_symbol_value is None:
            self.pilot_symbol_value = (1 + 1j) / np.sqrt(2)  # QPSK conocido
        else:
            self.pilot_symbol_value = pilot_symbol_value
    
    def generate_pilots(self, num_pilots: int) -> np.ndarray:
        """
        Genera secuencia de pilotos basada en PN (Pseudo-random Noise)
        
        Args:
            num_pilots: Número de pilotos a generar
            
        Returns:
            Array de símbolos piloto complejos
        """
        # PN sequence simple pero determinística
        np.random.seed(self.cell_id)  # Seed basado en cell_id
        phases = np.random.choice([1, -1], size=num_pilots)
        
        pilots = self.pilot_symbol_value * phases
        return pilots


class ResourceMapper:
    """
    Mapea datos y pilotos a la grid de recursos LTE
    
    Proceso:
    1. Obtener símbolos de datos (bits modulados)
    2. Obtener símbolos de piloto
    3. Mapear datos a posiciones de datos
    4. Mapear pilotos a posiciones de pilotos
    5. Dejar DC en cero (nulo)
    6. Dejar guardias en cero (nulo)
    """
    
    def __init__(self, config, cell_id: int = 0):
        """
        Inicializa el mapeador de recursos
        
        Args:
            config: Objeto LTEConfig
            cell_id: ID de celda para patrón de pilotos
        """
        self.config = config
        self.grid = LTEResourceGrid(config.N, config.Nc)
        self.pilot_pattern = PilotPattern(cell_id)
        self.stats = self.grid.get_statistics()
    
    def map_symbols(self, data_symbols: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Mapea símbolos de datos en la grid de recursos LTE
        
        Args:
            data_symbols: Array de símbolos QAM/QPSK para mapear
            
        Returns:
            tuple: (grid_mapped, mapping_info) donde:
                - grid_mapped: Vector de tamaño N con datos mapeados
                - mapping_info: Información sobre lo mapeado
        """
        # Inicializar grid
        grid_mapped = np.zeros(self.config.N, dtype=complex)
        
        # Índices disponibles
        data_indices = self.grid.get_data_indices()
        pilot_indices = self.grid.get_pilot_indices()
        
        # Mapear datos
        num_data_to_map = min(len(data_symbols), len(data_indices))
        grid_mapped[data_indices[:num_data_to_map]] = data_symbols[:num_data_to_map]
        
        # Generar y mapear pilotos
        pilots = self.pilot_pattern.generate_pilots(len(pilot_indices))
        grid_mapped[pilot_indices] = pilots
        
        # DC y guardias permanecen en cero (nulo)
        # (ya inicializados como cero)
        
        # Información de mapeo
        mapping_info = {
            'num_data_mapped': num_data_to_map,
            'num_pilots_mapped': len(pilot_indices),
            'num_nulls': len(self.grid.get_guard_indices()) + 1,  # +1 para DC
            'data_indices': data_indices[:num_data_to_map],
            'pilot_indices': pilot_indices,
            'guard_indices': self.grid.get_guard_indices(),
            'dc_index': self.grid.dc_index,
            'grid_statistics': self.stats,
        }
        
        return grid_mapped, mapping_info
    
    def extract_pilots(self, received_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae pilotos recibidos para estimación de canal
        
        Args:
            received_grid: Grid de símbolos recibidos
            
        Returns:
            tuple: (pilot_indices, pilot_symbols)
        """
        pilot_indices = self.grid.get_pilot_indices()
        pilot_symbols = received_grid[pilot_indices]
        
        return pilot_indices, pilot_symbols
    
    def get_data_indices(self) -> np.ndarray:
        """Retorna índices de datos para demodulación"""
        return self.grid.get_data_indices()
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas de la grid"""
        return self.stats
    
    def print_grid_structure(self):
        """Imprime visualización de la estructura de la grid"""
        print("\n" + "="*80)
        print("ESTRUCTURA DE GRID DE RECURSOS LTE")
        print("="*80)
        print(f"\nTamaño FFT (N): {self.config.N}")
        print(f"Subportadoras útiles (Nc): {self.config.Nc}")
        print(f"\nDistribución:")
        print(f"  Guard band izquierdo: {self.stats['guard_left']} subportadoras")
        print(f"  Datos: {self.stats['data_subcarriers']} subportadoras")
        print(f"  Pilotos: {self.stats['pilot_subcarriers']} subportadoras")
        print(f"  DC (nulo): 1 subportadora (índice {self.grid.dc_index})")
        print(f"  Guard band derecho: {self.stats['guard_right']} subportadoras")
        print(f"  Guardias totales (incluyendo DC): {self.stats['guard_subcarriers'] + 1}")
        print(f"\nEficiencia espectral:")
        print(f"  Datos útiles: {self.stats['data_subcarriers']} / {self.config.N} = "
              f"{self.stats['data_subcarriers']/self.config.N*100:.1f}%")
        print(f"  Overhead de pilotos: {self.stats['pilot_subcarriers']} / {self.config.N} = "
              f"{self.stats['pilot_subcarriers']/self.config.N*100:.1f}%")
        print("\n" + "="*80)


class EnhancedOFDMModulator:
    """
    Modulador OFDM mejorado con mapeo de recursos LTE
    
    Este es un wrapper que integra ResourceMapper en el proceso de modulación
    """
    
    def __init__(self, config, qam_modulator):
        """
        Inicializa el modulador mejorado
        
        Args:
            config: Objeto LTEConfig
            qam_modulator: Instancia de QAMModulator
        """
        self.config = config
        self.qam_modulator = qam_modulator
        self.resource_mapper = ResourceMapper(config)
    
    def modulate_with_mapping(self, bits: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Modula bits con mapeo de recursos LTE
        
        Proceso:
        1. Convertir bits a símbolos QAM
        2. Mapear a grid de recursos LTE (con pilotos)
        3. Aplicar IFFT
        4. Agregar prefijo cíclico
        
        Args:
            bits: Array de bits
            
        Returns:
            tuple: (ofdm_signal, mapping_info)
        """
        # Paso 1: Convertir bits a símbolos
        qam_symbols = self.qam_modulator.bits_to_symbols(bits)
        
        # Paso 2: Mapear a grid LTE
        grid_mapped, mapping_info = self.resource_mapper.map_symbols(qam_symbols)
        
        # Paso 3: IFFT
        time_domain = np.fft.ifft(grid_mapped) * np.sqrt(self.config.N)
        
        # Paso 4: Prefijo cíclico
        ofdm_signal = np.concatenate([
            time_domain[-self.config.cp_length:],
            time_domain
        ])
        
        return ofdm_signal, mapping_info
    
    def get_resource_mapper(self) -> ResourceMapper:
        """Retorna el mapeador de recursos para acceso directo"""
        return self.resource_mapper
