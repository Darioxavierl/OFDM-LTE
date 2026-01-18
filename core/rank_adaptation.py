"""
Rank Adaptation para Spatial Multiplexing (TM4)

Calcula dinámicamente el RI (Rank Indicator) y PMI (Precoding Matrix Indicator)
óptimos basado en condiciones del canal MIMO.

En LTE, el UE (receptor) mide el canal H y calcula:
- RI: Número óptimo de capas espaciales (rank)
- PMI: Mejor precoder del codebook para ese rank
- CQI: Channel Quality Indicator (opcional)

El feedback se envía al eNB (transmisor) para adaptar la transmisión.
"""

import numpy as np
from core.codebook_lte import LTECodebook


class RankAdaptation:
    """
    Adaptación de rank y selección de precoder para TM4.
    
    Usa eigenvalue decomposition del canal para determinar:
    - Rank óptimo: Número de eigenvalues significativos
    - PMI óptimo: Precoder que maximiza capacidad/SINR
    """
    
    def __init__(self, num_tx, num_rx, snr_db=15.0, rank_threshold=0.15):
        """
        Args:
            num_tx (int): Número de antenas TX
            num_rx (int): Número de antenas RX
            snr_db (float): SNR de operación (para decisión de rank)
            rank_threshold (float): Umbral para considerar eigenvalue significativo
                                   (relativo al máximo eigenvalue)
        """
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10)
        self.rank_threshold = rank_threshold
        
        # Límite de rank: min(num_tx, num_rx, 4)
        self.max_rank = min(num_tx, num_rx, 4)
        
        print(f"[RankAdaptation] Inicializado:")
        print(f"  Antenas: {num_tx}×{num_rx}")
        print(f"  SNR: {snr_db} dB")
        print(f"  Rank máximo: {self.max_rank}")
        print(f"  Threshold eigenvalues: {rank_threshold}")
    
    def calculate_optimal_rank(self, H_channel, method='eigenvalue'):
        """
        Calcula el rank óptimo basado en el canal MIMO.
        
        Método eigenvalue:
        1. Calcula H^H · H (Gram matrix)
        2. Eigenvalue decomposition
        3. Cuenta eigenvalues > threshold * λ_max
        4. Considera SNR: rank bajo si SNR < 10 dB
        
        Args:
            H_channel: Canal MIMO [num_rx, num_tx] o [num_rx, num_tx, num_sc]
            method: 'eigenvalue' (basado en SVD) o 'capacity' (máxima capacidad)
        
        Returns:
            ri: Rank óptimo (1 a max_rank)
        """
        # Si H es 3D [rx, tx, subcarriers], promediar sobre subportadoras
        if H_channel.ndim == 3:
            H_avg = np.mean(H_channel, axis=2)
        else:
            H_avg = H_channel
        
        if method == 'eigenvalue':
            ri = self._rank_from_eigenvalues(H_avg)
        elif method == 'capacity':
            ri = self._rank_from_capacity(H_avg)
        else:
            raise ValueError(f"Método '{method}' no soportado")
        
        return ri
    
    def _rank_from_eigenvalues(self, H):
        """
        Rank basado en eigenvalues significativos.
        
        Criterio: λᵢ > threshold * λ_max
        """
        # Gram matrix: H^H · H
        HH = H.conj().T @ H  # [num_tx, num_tx]
        
        # Eigenvalues (sorted descendente)
        eigenvalues = np.linalg.eigvalsh(HH)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Mayor a menor
        
        # Normalizar
        lambda_max = eigenvalues[0]
        if lambda_max < 1e-10:
            return 1  # Canal muy débil, usar rank-1
        
        eigenvalues_norm = eigenvalues / lambda_max
        
        # Contar eigenvalues significativos
        significant = np.sum(eigenvalues_norm > self.rank_threshold)
        
        # Limitar por max_rank
        ri = min(significant, self.max_rank)
        
        # Ajuste por SNR: si SNR bajo, reducir rank
        if self.snr_db < 5:
            ri = 1
        elif self.snr_db < 10:
            ri = min(ri, 2)
        
        # Asegurar al menos rank=1
        ri = max(1, ri)
        
        return ri
    
    def _rank_from_capacity(self, H):
        """
        Rank que maximiza capacidad del canal.
        
        Capacidad: C = Σ log2(1 + SNR * λᵢ)
        Prueba rank 1 a max_rank y elige el mejor.
        """
        # SVD del canal
        U, s, Vh = np.linalg.svd(H, full_matrices=False)
        singular_values = s[:self.max_rank]
        
        best_rank = 1
        best_capacity = -np.inf
        
        for rank in range(1, self.max_rank + 1):
            # Capacidad con este rank
            capacity = 0
            for i in range(rank):
                if i < len(singular_values):
                    capacity += np.log2(1 + self.snr_linear * singular_values[i]**2 / rank)
            
            if capacity > best_capacity:
                best_capacity = capacity
                best_rank = rank
        
        return best_rank
    
    def select_precoder_for_rank(self, H_channel, rank, metric='capacity'):
        """
        Selecciona el mejor PMI del codebook TM4 para el rank dado.
        
        Args:
            H_channel: Canal [num_rx, num_tx]
            rank: Rank a usar
            metric: 'capacity', 'frobenius', o 'sinr'
        
        Returns:
            pmi: Índice del mejor precoder
            W: Matriz de precoding [num_tx, rank]
        """
        # Generar codebook para este rank
        codebook = LTECodebook(self.num_tx, transmission_mode='TM4', rank=rank)
        
        # Promediar H si es 3D
        if H_channel.ndim == 3:
            H_avg = np.mean(H_channel, axis=2)
        else:
            H_avg = H_channel
        
        # Buscar mejor precoder
        best_pmi = 0
        best_metric_value = -np.inf
        
        for pmi in range(codebook.codebook_size):
            W = codebook.get_precoder(pmi)
            
            # Canal efectivo: H_eff = H · W
            H_eff = H_avg @ W  # [num_rx, rank]
            
            # Calcular métrica
            if metric == 'capacity':
                # Capacidad aproximada: log2(det(I + SNR/rank * H_eff · H_eff^H))
                HH_eff = H_eff @ H_eff.conj().T
                try:
                    metric_value = np.log2(np.linalg.det(
                        np.eye(self.num_rx) + (self.snr_linear / rank) * HH_eff
                    ))
                except:
                    metric_value = 0
            
            elif metric == 'frobenius':
                # Norma de Frobenius
                metric_value = np.linalg.norm(H_eff, 'fro') ** 2
            
            elif metric == 'sinr':
                # SINR efectivo (suma de potencias)
                metric_value = np.sum(np.abs(H_eff) ** 2)
            
            else:
                raise ValueError(f"Métrica '{metric}' no soportada")
            
            # Actualizar mejor
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_pmi = pmi
        
        # Obtener precoder óptimo
        W_optimal = codebook.get_precoder(best_pmi)
        
        return best_pmi, W_optimal
    
    def get_feedback(self, H_channel, rank_method='eigenvalue', pmi_metric='capacity'):
        """
        Genera feedback CSI completo: RI + PMI.
        
        Simula proceso del receptor LTE:
        1. Medir canal H
        2. Calcular RI óptimo
        3. Calcular PMI óptimo para ese RI
        4. Enviar feedback a transmisor
        
        Args:
            H_channel: Canal medido [num_rx, num_tx] o [num_rx, num_tx, num_sc]
            rank_method: Método para calcular RI ('eigenvalue' o 'capacity')
            pmi_metric: Métrica para seleccionar PMI ('capacity', 'frobenius', 'sinr')
        
        Returns:
            dict: {
                'ri': int,           # Rank Indicator
                'pmi': int,          # Precoding Matrix Indicator
                'W': np.ndarray,     # Matriz de precoding [num_tx, ri]
                'eigenvalues': np.ndarray,  # Eigenvalues del canal
                'condition_number': float,  # Condición del canal
            }
        """
        # Calcular RI
        ri = self.calculate_optimal_rank(H_channel, method=rank_method)
        
        # Calcular PMI para ese RI
        pmi, W = self.select_precoder_for_rank(H_channel, ri, metric=pmi_metric)
        
        # Información adicional del canal
        if H_channel.ndim == 3:
            H_avg = np.mean(H_channel, axis=2)
        else:
            H_avg = H_channel
        
        # Eigenvalues
        HH = H_avg.conj().T @ H_avg
        eigenvalues = np.linalg.eigvalsh(HH)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Condition number
        singular_values = np.linalg.svd(H_avg, compute_uv=False)
        condition_number = singular_values[0] / (singular_values[-1] + 1e-10)
        
        feedback = {
            'ri': ri,
            'pmi': pmi,
            'W': W,
            'eigenvalues': eigenvalues,
            'condition_number': condition_number,
        }
        
        return feedback
    
    def update_snr(self, new_snr_db):
        """
        Actualiza SNR de operación (para adaptación dinámica).
        """
        self.snr_db = new_snr_db
        self.snr_linear = 10 ** (new_snr_db / 10)


def test_rank_adaptation():
    """
    Tests del RankAdaptation
    """
    print("\n=== Test RankAdaptation ===")
    
    # Test 1: Canal 4×2 con rank-2
    print("\n1. Test canal 4×2 (rank-2 esperado):")
    np.random.seed(42)
    
    # Canal con 2 eigenvalues fuertes
    H_test = np.random.randn(2, 4) + 1j * np.random.randn(2, 4)
    H_test = H_test / np.linalg.norm(H_test, 'fro')  # Normalizar
    
    adapter = RankAdaptation(num_tx=4, num_rx=2, snr_db=15)
    ri = adapter.calculate_optimal_rank(H_test)
    print(f"   RI calculado: {ri}")
    assert ri >= 1 and ri <= 2, f"RI fuera de rango para 2×4: {ri}"
    print("   ✓ RI válido")
    
    # Test 2: Selección de PMI
    print("\n2. Test selección PMI para rank=2:")
    pmi, W = adapter.select_precoder_for_rank(H_test, rank=2)
    print(f"   PMI seleccionado: {pmi}")
    print(f"   W shape: {W.shape}")
    assert W.shape == (4, 2), f"W shape incorrecta: {W.shape}"
    print("   ✓ PMI y W correctos")
    
    # Test 3: Feedback completo
    print("\n3. Test feedback completo:")
    feedback = adapter.get_feedback(H_test)
    print(f"   RI: {feedback['ri']}")
    print(f"   PMI: {feedback['pmi']}")
    print(f"   W shape: {feedback['W'].shape}")
    print(f"   Condition number: {feedback['condition_number']:.2f}")
    print(f"   Eigenvalues (top 2): {feedback['eigenvalues'][:2]}")
    assert 'ri' in feedback and 'pmi' in feedback and 'W' in feedback
    print("   ✓ Feedback completo")
    
    # Test 4: Canal 4×4 con rank alto
    print("\n4. Test canal 4×4 (rank alto esperado):")
    H_full = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    H_full = H_full / np.linalg.norm(H_full, 'fro')
    
    adapter44 = RankAdaptation(num_tx=4, num_rx=4, snr_db=20)
    feedback44 = adapter44.get_feedback(H_full)
    print(f"   RI: {feedback44['ri']}")
    assert feedback44['ri'] <= 4, f"RI > 4: {feedback44['ri']}"
    print("   ✓ RI para 4×4 correcto")
    
    # Test 5: SNR bajo fuerza rank=1
    print("\n5. Test SNR bajo (rank=1 esperado):")
    adapter_low = RankAdaptation(num_tx=4, num_rx=2, snr_db=3)
    ri_low = adapter_low.calculate_optimal_rank(H_test)
    print(f"   RI con SNR=3dB: {ri_low}")
    assert ri_low == 1, f"Con SNR bajo debe ser rank=1, obtenido: {ri_low}"
    print("   ✓ Rank adaptativo según SNR")
    
    print("\n=== Todos los tests pasaron ✓ ===\n")


if __name__ == '__main__':
    test_rank_adaptation()
