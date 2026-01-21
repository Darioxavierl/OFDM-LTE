"""
LTE Codebook Implementation (TS 36.211 Section 6.3.4.2.3)

Implementa codebooks para:
- TM6: Rank-1, closed-loop spatial multiplexing (single layer)
- TM4: Dual-layer precoding (preparado para futuro)

Soporta 2, 4, 8 antenas TX.
"""

import numpy as np


class LTECodebook:
    """
    Codebooks LTE para precoding cuantizado.
    Basado en TS 36.211 Table 6.3.4.2.3-1 y 6.3.4.2.3-2.
    """
    
    def __init__(self, num_tx, transmission_mode='TM6', rank=1):
        """
        Args:
            num_tx (int): Número de antenas TX (2, 4, 8)
            transmission_mode (str): 'TM6' (rank-1) o 'TM4' (rank-1/2/3/4)
            rank (int): Número de capas espaciales (1-4 para TM4, siempre 1 para TM6)
        """
        self.num_tx = num_tx
        self.transmission_mode = transmission_mode
        self.rank = rank
        
        # Validar rank según modo
        if transmission_mode == 'TM6' and rank != 1:
            raise ValueError(f"TM6 solo soporta rank=1, recibido rank={rank}")
        
        if transmission_mode == 'TM4' and (rank < 1 or rank > min(num_tx, 4)):
            raise ValueError(f"TM4 con {num_tx} antenas soporta rank 1-{min(num_tx, 4)}, recibido rank={rank}")
        
        # Generar codebook
        self.codebook = self._generate_codebook()
        self.codebook_size = len(self.codebook)
        
        print(f"[LTECodebook] Inicializado:")
        print(f"  Modo: {transmission_mode}, Rank: {rank}")
        print(f"  Antenas TX: {num_tx}")
        print(f"  Tamaño codebook: {self.codebook_size} matrices")
    
    def _generate_codebook(self):
        """
        Genera el codebook según el número de antenas.
        """
        if self.transmission_mode == 'TM6':
            return self._generate_tm6_codebook()
        elif self.transmission_mode == 'TM4':
            return self._generate_tm4_codebook()
        else:
            raise ValueError(f"Modo {self.transmission_mode} no soportado")
    
    def _generate_tm6_codebook(self):
        """
        TM6 Codebook (TS 36.211 Table 6.3.4.2.3-2)
        Rank-1 precoding para single layer.
        """
        if self.num_tx == 2:
            # 4 vectores para 2 antenas
            # Fase: 0°, 180°, 90°, -90°
            codebook = [
                np.array([[1], [1]]) / np.sqrt(2),      # W0: coherent sum
                np.array([[1], [-1]]) / np.sqrt(2),     # W1: coherent difference
                np.array([[1], [1j]]) / np.sqrt(2),     # W2: +90° phase
                np.array([[1], [-1j]]) / np.sqrt(2),    # W3: -90° phase
            ]
            
        elif self.num_tx == 4:
            # 16 vectores para 4 antenas (DFT-based)
            codebook = []
            
            # Vectores DFT base
            for i in range(16):
                # Fase lineal entre antenas
                phases = 2 * np.pi * i * np.arange(4) / 16
                W = np.exp(1j * phases).reshape(-1, 1) / 2
                codebook.append(W)
        
        elif self.num_tx == 8:
            # 8-16 vectores para 8 antenas (simplificado)
            codebook = []
            
            for i in range(16):
                phases = 2 * np.pi * i * np.arange(8) / 16
                W = np.exp(1j * phases).reshape(-1, 1) / np.sqrt(8)
                codebook.append(W)
        
        else:
            raise ValueError(f"num_tx={self.num_tx} no soportado en TM6")
        
        return codebook
    
    def _generate_tm4_codebook(self):
        """
        TM4 Codebook (TS 36.211 Table 6.3.4.2.3-1)
        Soporta Rank-1, Rank-2, Rank-3, Rank-4 según especificación LTE.
        """
        if self.rank == 1:
            return self._generate_tm4_rank1_codebook()
        elif self.rank == 2:
            return self._generate_tm4_rank2_codebook()
        elif self.rank == 3:
            return self._generate_tm4_rank3_codebook()
        elif self.rank == 4:
            return self._generate_tm4_rank4_codebook()
        else:
            raise ValueError(f"Rank {self.rank} no soportado en TM4")
    
    def _generate_tm4_rank1_codebook(self):
        """
        TM4 Rank-1: Similar a TM6 pero con índices diferentes.
        Usa mismos vectores DFT.
        """
        return self._generate_tm6_codebook()
    
    def _generate_tm4_rank2_codebook(self):
        """
        TM4 Rank-2 Codebook (TS 36.211 Table 6.3.4.2.3-1)
        Matrices [num_tx × 2] para transmisión de 2 capas paralelas.
        """
        if self.num_tx == 2:
            # 2×2: 3 matrices (TS 36.211 Table 6.3.4.2.3-1)
            # Normalización: ||W||²_F = 2 para rank=2
            codebook = [
                # W0: Identity (||W||²_F = 2)
                np.array([[1, 0],
                         [0, 1]]),
                
                # W1: Hadamard (||W||²_F = 4/2 = 2)
                np.array([[1, 1],
                         [1, -1]]) / np.sqrt(2),
                
                # W2: Phase rotation (||W||²_F = 4/2 = 2)
                np.array([[1, 1],
                         [1j, -1j]]) / np.sqrt(2),
            ]
        
        elif self.num_tx == 4:
            # 4×2: 16 matrices (TS 36.211 Table 6.3.4.2.3-1)
            codebook = []
            
            # Householder-based precoders para 4 antenas, 2 layers
            # Matrices ortogonales que maximizan diversidad
            
            # W0-W3: DFT-based con separación de capas
            for i in range(4):
                theta = 2 * np.pi * i / 4
                W = np.array([
                    [1, 0],
                    [np.exp(1j * theta), 0],
                    [0, 1],
                    [0, np.exp(1j * theta)]
                ]) / np.sqrt(2)
                codebook.append(W)
            
            # W4-W7: Rotaciones alternas
            for i in range(4):
                theta = 2 * np.pi * i / 4
                W = np.array([
                    [1, 1],
                    [np.exp(1j * theta), -np.exp(1j * theta)],
                    [1, -1],
                    [np.exp(1j * theta), np.exp(1j * theta)]
                ]) / 2
                codebook.append(W)
            
            # W8-W11: Configuraciones cruzadas
            for i in range(4):
                theta = 2 * np.pi * i / 4
                W = np.array([
                    [1, 0],
                    [0, 1],
                    [np.exp(1j * theta), 0],
                    [0, np.exp(1j * theta)]
                ]) / np.sqrt(2)
                codebook.append(W)
            
            # W12-W15: Hadamard variations
            for i in range(4):
                theta = 2 * np.pi * i / 4
                W = np.array([
                    [1, 1],
                    [1, -1],
                    [np.exp(1j * theta), np.exp(1j * theta)],
                    [np.exp(1j * theta), -np.exp(1j * theta)]
                ]) / 2
                codebook.append(W)
        
        elif self.num_tx == 8:
            # 8×2: 16 matrices (simplificado, extendiendo patrón de 4 antenas)
            codebook = []
            
            for i in range(16):
                theta = 2 * np.pi * i / 16
                # Patrón DFT extendido a 8 antenas
                W = np.zeros((8, 2), dtype=complex)
                W[0:4, 0] = np.exp(1j * theta * np.arange(4)) / np.sqrt(4)
                W[4:8, 1] = np.exp(1j * theta * np.arange(4)) / np.sqrt(4)
                codebook.append(W)
        
        else:
            raise ValueError(f"num_tx={self.num_tx} no soportado en TM4 Rank-2")
        
        return codebook
    
    def _generate_tm4_rank3_codebook(self):
        """
        TM4 Rank-3 Codebook
        Matrices [num_tx × 3] para 3 capas (requiere num_tx ≥ 4)
        """
        if self.num_tx < 4:
            raise ValueError(f"Rank-3 requiere al menos 4 antenas TX, disponibles: {self.num_tx}")
        
        codebook = []
        
        if self.num_tx == 4:
            # 4×3: 8 matrices
            for i in range(8):
                theta = 2 * np.pi * i / 8
                # ||W||²_F debe ser 3
                # Suma: 1 + 1 + 1 + 3 = 6, dividir por sqrt(2) da 3
                W = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [np.exp(1j * theta), np.exp(1j * theta), np.exp(1j * theta)]
                ]) / np.sqrt(2)
                codebook.append(W)
        
        elif self.num_tx == 8:
            # 8×3: 16 matrices
            # ||W||²_F debe ser 3: cada capa contribuye 1
            for i in range(16):
                theta = 2 * np.pi * i / 16
                W = np.zeros((8, 3), dtype=complex)
                # Distribuir 3 capas: cada una normalizada a potencia 1
                # Capa 0: antenas 0, 1, 2
                W[0:3, 0] = [1, np.exp(1j*theta), np.exp(1j*2*theta)] / np.sqrt(3)
                # Capa 1: antenas 3, 4, 5
                W[3:6, 1] = [1, np.exp(1j*theta), np.exp(1j*2*theta)] / np.sqrt(3)
                # Capa 2: antenas 5, 6, 7 (solapa con capa 1 en antena 5)
                W[5:8, 2] = [1, np.exp(1j*theta), np.exp(1j*2*theta)] / np.sqrt(3)
                codebook.append(W)
        
        else:
            raise ValueError(f"num_tx={self.num_tx} no soportado en TM4 Rank-3")
        
        return codebook
    
    def _generate_tm4_rank4_codebook(self):
        """
        TM4 Rank-4 Codebook
        Matrices [num_tx × 4] para 4 capas (requiere num_tx ≥ 4)
        """
        if self.num_tx < 4:
            raise ValueError(f"Rank-4 requiere al menos 4 antenas TX, disponibles: {self.num_tx}")
        
        codebook = []
        
        if self.num_tx == 4:
            # 4×4: 4 matrices (cuadradas)
            # Normalización: ||W||²_F = 4 para rank=4
            
            # W0: Identity (||W||²_F = 4 sin normalización)
            codebook.append(np.eye(4, dtype=complex))
            
            # W1: DFT matrix (||W||²_F = 16/4 = 4 con /2)
            W_dft = np.zeros((4, 4), dtype=complex)
            for i in range(4):
                for j in range(4):
                    W_dft[i, j] = np.exp(-2j * np.pi * i * j / 4)
            codebook.append(W_dft / 2)
            
            # W2: Hadamard-like
            codebook.append(np.array([
                [1, 1, 1, 1],
                [1, -1, 1, -1],
                [1, 1, -1, -1],
                [1, -1, -1, 1]
            ]) / 2)
            
            # W3: Phase rotations
            codebook.append(np.array([
                [1, 1, 1, 1],
                [1, 1j, -1, -1j],
                [1, -1, 1, -1],
                [1, -1j, -1, 1j]
            ]) / 2)
        
        elif self.num_tx == 8:
            # 8×4: 8 matrices
            for i in range(8):
                theta = 2 * np.pi * i / 8
                W = np.zeros((8, 4), dtype=complex)
                
                # Asignar 2 antenas por capa
                for layer in range(4):
                    ant_start = layer * 2
                    W[ant_start:ant_start+2, layer] = [1, np.exp(1j * theta * (layer+1))] / np.sqrt(2)
                
                codebook.append(W)
        
        else:
            raise ValueError(f"num_tx={self.num_tx} no soportado en TM4 Rank-4")
        
        return codebook
    
    def get_codebook(self):
        """Retorna el codebook completo"""
        return self.codebook
    
    def get_precoder(self, pmi):
        """
        Obtiene un precoder específico del codebook.
        
        Args:
            pmi (int): Precoding Matrix Indicator (0 a codebook_size-1)
        
        Returns:
            W: Matriz de precoding [num_tx, num_layers]
        """
        if pmi < 0 or pmi >= self.codebook_size:
            raise ValueError(f"PMI {pmi} fuera de rango [0, {self.codebook_size-1}]")
        
        return self.codebook[pmi]
    
    def select_best_pmi(self, H_channel, metric='capacity'):
        """
        Selecciona el mejor PMI del codebook para un canal dado.
        
        Args:
            H_channel: Canal [num_rx, num_tx]
            metric: 'capacity', 'sinr', o 'frobenius'
        
        Returns:
            best_pmi: Índice del mejor precoder
            best_metric: Valor de la métrica
        """
        best_pmi = 0
        best_metric = -np.inf
        
        for pmi, W in enumerate(self.codebook):
            # Canal efectivo con este precoder
            H_eff = H_channel @ W
            
            # Calcular métrica
            if metric == 'capacity':
                # Capacidad: log2(1 + SNR * ||H_eff||²)
                # Simplificado: solo maximizar ||H_eff||²
                current_metric = np.sum(np.abs(H_eff)**2)
            
            elif metric == 'sinr':
                # SINR efectivo
                current_metric = np.sum(np.abs(H_eff)**2)
            
            elif metric == 'frobenius':
                # Norma de Frobenius
                current_metric = np.linalg.norm(H_eff, 'fro')
            
            else:
                raise ValueError(f"Métrica '{metric}' no soportada")
            
            # Actualizar mejor
            if current_metric > best_metric:
                best_metric = current_metric
                best_pmi = pmi
        
        return best_pmi, best_metric
    
    def calculate_quantization_error(self, H_channel, pmi):
        """
        Calcula el error de cuantización al usar codebook.
        Error = ||H - H_quantized||² / ||H||²
        
        Args:
            H_channel: Canal real [num_rx, num_tx]
            pmi: PMI seleccionado
        
        Returns:
            error: Error de cuantización normalizado
        """
        # Precoder óptimo (sin cuantizar): MRT
        h_avg = np.mean(H_channel, axis=0)
        W_opt = np.conj(h_avg) / np.linalg.norm(h_avg)
        W_opt = W_opt.reshape(-1, 1)
        
        # Precoder cuantizado
        W_quant = self.get_precoder(pmi)
        
        # Error angular (product escalar)
        error = 1 - np.abs(np.vdot(W_opt.flatten(), W_quant.flatten()))**2
        
        return error
    
    def get_codebook_info(self):
        """
        Retorna información del codebook.
        """
        info = {
            'num_tx': self.num_tx,
            'transmission_mode': self.transmission_mode,
            'codebook_size': self.codebook_size,
            'num_layers': self.codebook[0].shape[1],
            'pmi_bits': int(np.ceil(np.log2(self.codebook_size)))
        }
        return info
    
    def print_codebook(self):
        """
        Imprime el codebook completo (para debugging).
        """
        print(f"\n{'='*60}")
        print(f"Codebook LTE - {self.transmission_mode} - {self.num_tx} TX")
        print(f"{'='*60}")
        
        for pmi, W in enumerate(self.codebook):
            print(f"\nPMI = {pmi}:")
            print(f"  W shape: {W.shape}")
            print(f"  W = ")
            for row in W:
                real = row.real if np.abs(row.real) > 1e-10 else 0
                imag = row.imag if np.abs(row.imag) > 1e-10 else 0
                if imag >= 0:
                    print(f"    {real:.3f} + {imag:.3f}j")
                else:
                    print(f"    {real:.3f} - {abs(imag):.3f}j")
        
        print(f"\n{'='*60}")
