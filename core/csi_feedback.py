"""
CSI Feedback Simulator for LTE Beamforming

Simula el proceso de feedback de información de estado de canal (CSI):
- PMI (Precoding Matrix Indicator)
- CQI (Channel Quality Indicator)
- RI (Rank Indicator) - preparado para futuro

En LTE real:
1. RX estima canal H usando CRS
2. RX calcula PMI/CQI
3. RX envía PMI/CQI al TX por PUCCH/PUSCH
4. TX usa PMI para seleccionar precoder del codebook

En esta simulación (perfect CSI):
- Sin delay de feedback
- Sin errores en PMI/CQI
- TX conoce H perfectamente (solo para comparación)
"""

import numpy as np
from core.codebook_lte import LTECodebook


class CSIFeedback:
    """
    Simulador de feedback CSI (Channel State Information).
    """
    
    def __init__(self, num_tx, num_rx, codebook_type='TM6', feedback_mode='perfect'):
        """
        Args:
            num_tx (int): Antenas TX
            num_rx (int): Antenas RX
            codebook_type (str): 'TM6' o 'TM4'
            feedback_mode (str): 'perfect' (sin errores) o 'realistic' (con cuantización)
        """
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.codebook_type = codebook_type
        self.feedback_mode = feedback_mode
        
        # Codebook LTE
        self.codebook = LTECodebook(num_tx, transmission_mode=codebook_type)
        
        # Estadísticas
        self.total_feedbacks = 0
        self.pmi_history = []
        
        print(f"[CSIFeedback] Inicializado:")
        print(f"  Config: {num_tx}×{num_rx}")
        print(f"  Codebook: {codebook_type}")
        print(f"  Modo: {feedback_mode}")
    
    def calculate_pmi(self, H_channel):
        """
        Calcula el mejor PMI para un canal dado.
        
        Args:
            H_channel: Canal estimado [num_rx, num_tx]
        
        Returns:
            pmi: Índice del mejor precoder (0 a N-1)
        """
        # Seleccionar mejor PMI del codebook
        pmi, metric = self.codebook.select_best_pmi(H_channel, metric='capacity')
        
        # Guardar en historial
        self.pmi_history.append(pmi)
        self.total_feedbacks += 1
        
        return pmi
    
    def calculate_cqi(self, H_channel, pmi, noise_variance=1.0):
        """
        Calcula el CQI (Channel Quality Indicator).
        
        CQI cuantifica la calidad del canal efectivo después del precoding.
        Rango: 0-15 (16 niveles en LTE)
        
        Args:
            H_channel: Canal [num_rx, num_tx]
            pmi: PMI seleccionado
            noise_variance: Varianza del ruido
        
        Returns:
            cqi: Índice CQI (0-15)
            sinr_db: SINR efectivo en dB
        """
        # Obtener precoder
        W = self.codebook.get_precoder(pmi)
        
        # Canal efectivo
        H_eff = H_channel @ W
        
        # SINR post-precoding
        signal_power = np.sum(np.abs(H_eff)**2)
        sinr = signal_power / noise_variance
        sinr_db = 10 * np.log10(sinr)
        
        # Mapear SINR a CQI (tabla simplificada LTE)
        cqi = self._sinr_to_cqi(sinr_db)
        
        return cqi, sinr_db
    
    def _sinr_to_cqi(self, sinr_db):
        """
        Mapea SINR a CQI según tabla LTE (TS 36.213).
        
        CQI determina el MCS (Modulation and Coding Scheme) a usar.
        """
        # Tabla simplificada (valores aproximados)
        cqi_table = [
            (-np.inf, -6.0),   # CQI 0: out of range
            (-6.0, -4.0),      # CQI 1: QPSK 1/8
            (-4.0, -2.0),      # CQI 2: QPSK 1/5
            (-2.0, 0.0),       # CQI 3: QPSK 1/4
            (0.0, 2.0),        # CQI 4: QPSK 1/3
            (2.0, 4.0),        # CQI 5: QPSK 1/2
            (4.0, 6.0),        # CQI 6: QPSK 2/3
            (6.0, 8.0),        # CQI 7: 16QAM 1/2
            (8.0, 10.0),       # CQI 8: 16QAM 2/3
            (10.0, 12.0),      # CQI 9: 16QAM 3/4
            (12.0, 14.0),      # CQI 10: 64QAM 2/3
            (14.0, 16.0),      # CQI 11: 64QAM 3/4
            (16.0, 18.0),      # CQI 12: 64QAM 5/6
            (18.0, 20.0),      # CQI 13: 256QAM 3/4
            (20.0, 22.0),      # CQI 14: 256QAM 5/6
            (22.0, np.inf),    # CQI 15: 256QAM 9/10
        ]
        
        for cqi, (low, high) in enumerate(cqi_table):
            if low <= sinr_db < high:
                return cqi
        
        return 15  # Máximo CQI
    
    def calculate_rank_indicator(self, H_channel):
        """
        Calcula el RI (Rank Indicator) - número óptimo de capas espaciales.
        
        Args:
            H_channel: Canal [num_rx, num_tx]
        
        Returns:
            ri: Número óptimo de capas (1 o 2)
        """
        # Calcular eigenvalues del canal
        HH = H_channel.conj().T @ H_channel
        eigenvalues = np.linalg.eigvalsh(HH)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descendente
        
        # Criterio simple: si segundo eigenvalue > 20% del primero, usar rank 2
        if len(eigenvalues) >= 2:
            ratio = eigenvalues[1] / eigenvalues[0]
            ri = 2 if ratio > 0.2 else 1
        else:
            ri = 1
        
        return ri
    
    def generate_feedback(self, H_channel, noise_variance=1.0):
        """
        Genera feedback CSI completo (PMI + CQI + RI).
        
        Args:
            H_channel: Canal [num_rx, num_tx]
            noise_variance: Varianza del ruido
        
        Returns:
            feedback: Dict con PMI, CQI, RI, SINR
        """
        # PMI
        pmi = self.calculate_pmi(H_channel)
        
        # CQI
        cqi, sinr_db = self.calculate_cqi(H_channel, pmi, noise_variance)
        
        # RI
        ri = self.calculate_rank_indicator(H_channel)
        
        feedback = {
            'pmi': pmi,
            'cqi': cqi,
            'ri': ri,
            'sinr_db': sinr_db,
            'precoder': self.codebook.get_precoder(pmi)
        }
        
        return feedback
    
    def get_statistics(self):
        """
        Retorna estadísticas del feedback.
        """
        if len(self.pmi_history) == 0:
            return None
        
        stats = {
            'total_feedbacks': self.total_feedbacks,
            'unique_pmis': len(set(self.pmi_history)),
            'most_common_pmi': max(set(self.pmi_history), key=self.pmi_history.count),
            'pmi_distribution': np.bincount(self.pmi_history, minlength=self.codebook.codebook_size)
        }
        
        return stats
    
    def print_statistics(self):
        """
        Imprime estadísticas del feedback.
        """
        stats = self.get_statistics()
        if stats is None:
            print("[CSIFeedback] No hay estadísticas disponibles")
            return
        
        print(f"\n{'='*60}")
        print(f"Estadísticas CSI Feedback")
        print(f"{'='*60}")
        print(f"  Total feedbacks: {stats['total_feedbacks']}")
        print(f"  PMIs únicos usados: {stats['unique_pmis']} / {self.codebook.codebook_size}")
        print(f"  PMI más común: {stats['most_common_pmi']}")
        print(f"\n  Distribución PMI:")
        for pmi, count in enumerate(stats['pmi_distribution']):
            if count > 0:
                percentage = 100 * count / stats['total_feedbacks']
                print(f"    PMI {pmi}: {count} ({percentage:.1f}%)")
        print(f"{'='*60}\n")
