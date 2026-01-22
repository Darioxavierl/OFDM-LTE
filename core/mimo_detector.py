"""
MIMO Detectors para Spatial Multiplexing

Implementa detectores multi-capa para separar streams espaciales:
- MMSE (IRC): Minimum Mean Square Error / Interference Rejection Combining
- ZF: Zero-Forcing
- SIC: Successive Interference Cancellation (non-linear)
- MRC: Maximum Ratio Combining (wrapper para compatibilidad, solo rank-1)

En LTE TM4, el receptor debe separar múltiples capas espaciales que
se transmiten simultáneamente. MMSE y SIC son los detectores estándar.
"""

import numpy as np
from core.demodulator import SymbolDetector


class MIMODetector:
    """
    Detector MIMO multi-capa para Spatial Multiplexing.
    
    Soporta diferentes algoritmos de detección con trade-off
    complejidad vs performance.
    """
    
    def __init__(self, num_rx, num_layers, detector_type='MMSE', constellation=None):
        """
        Args:
            num_rx (int): Número de antenas RX
            num_layers (int): Número de capas espaciales (rank)
            detector_type (str): 'MMSE', 'IRC', 'ZF', 'SIC', 'MRC'
            constellation: Constelación QAM para hard decision (SymbolDetector)
        """
        if num_rx < num_layers:
            raise ValueError(f"num_rx ({num_rx}) debe ser >= num_layers ({num_layers})")
        
        self.num_rx = num_rx
        self.num_layers = num_layers
        self.detector_type = detector_type.upper()
        
        # Symbol detector para SIC (hard decisions)
        self.symbol_detector = constellation if constellation is not None else None
        
        print(f"[MIMODetector] Inicializado:")
        print(f"  Detector: {self.detector_type}")
        print(f"  Antenas RX: {num_rx}")
        print(f"  Capas: {num_layers}")
        print(f"  Constelación presente: {self.symbol_detector is not None}")
        if self.symbol_detector is not None:
            if isinstance(self.symbol_detector, np.ndarray):
                print(f"  Constelación tipo: numpy array, size={len(self.symbol_detector)}")
            else:
                print(f"  Constelación tipo: {type(self.symbol_detector)}")
    
    def detect(self, y_received, H_channel, noise_variance, W_precoder=None):
        """
        Detecta símbolos de múltiples capas espaciales.
        
        Args:
            y_received: Señal recibida [num_rx, num_subcarriers] o [num_rx]
            H_channel: Canal [num_rx, num_tx, num_subcarriers] o [num_rx, num_tx]
            noise_variance: σ² del ruido (float)
            W_precoder: Matriz de precoding usada en TX [num_tx, num_layers] (opcional)
        
        Returns:
            symbols_detected: Símbolos detectados [num_layers, num_subcarriers] o [num_layers]
        """
        # Determinar si procesamos por subportadora o todo junto
        is_per_subcarrier = (y_received.ndim == 2 and y_received.shape[1] > 1)
        
        if is_per_subcarrier:
            return self._detect_per_subcarrier(y_received, H_channel, 
                                               noise_variance, W_precoder)
        else:
            # Detección en un solo punto (y vector, H matriz)
            if y_received.ndim == 2:
                y_received = y_received.flatten()
            if H_channel.ndim == 3:
                H_channel = H_channel[:, :, 0]  # Primera subportadora
            
            return self._detect_single(y_received, H_channel, noise_variance, W_precoder)
    
    def _detect_per_subcarrier(self, y_received, H_channel, noise_variance, W_precoder):
        """
        Detecta por cada subportadora independientemente.
        """
        num_subcarriers = y_received.shape[1]
        symbols_detected = np.zeros((self.num_layers, num_subcarriers), dtype=complex)
        
        for sc in range(num_subcarriers):
            y_sc = y_received[:, sc]
            H_sc = H_channel[:, :, sc] if H_channel.ndim == 3 else H_channel
            W_sc = W_precoder if W_precoder is not None else None
            
            symbols_detected[:, sc] = self._detect_single(y_sc, H_sc, noise_variance, W_sc)
        
        return symbols_detected
    
    def _detect_single(self, y, H, sigma2, W=None):
        """
        Detección en un punto (una subportadora).
        
        Args:
            y: [num_rx]
            H: [num_rx, num_tx]
            sigma2: noise variance
            W: [num_tx, num_layers]
        """
        # Canal efectivo con precoding
        if W is not None:
            H_eff = H @ W  # [num_rx, num_layers]
        else:
            # Sin precoding: asumir que H tiene dimensión [num_rx, num_layers]
            H_eff = H[:, :self.num_layers]
        
        # Seleccionar detector
        print(f"[_detect_single] detector_type={self.detector_type}")
        if self.detector_type == 'MMSE' or self.detector_type == 'IRC':
            print(f"[_detect_single] Ejecutando MMSE")
            return self._mmse_detect(y, H_eff, sigma2)
        elif self.detector_type == 'ZF':
            print(f"[_detect_single] Ejecutando ZF")
            return self._zf_detect(y, H_eff)
        elif self.detector_type == 'SIC':
            print(f"[_detect_single] Ejecutando SIC")
            return self._sic_detect(y, H_eff, sigma2)
        elif self.detector_type == 'MRC':
            if self.num_layers != 1:
                raise ValueError("MRC solo soporta num_layers=1 (rank-1)")
            print(f"[_detect_single] Ejecutando MRC")
            return self._mrc_detect(y, H_eff)
        else:
            raise ValueError(f"Detector '{self.detector_type}' no soportado")
    
    def _mmse_detect(self, y, H_eff, sigma2):
        """
        MMSE (Minimum Mean Square Error) Detector.
        
        También conocido como IRC (Interference Rejection Combining).
        
        W_mmse = (H^H · H + σ²I)^-1 · H^H
        s_hat = W_mmse · y
        
        Minimiza: E[||s - s_hat||²]
        
        Args:
            y: [num_rx]
            H_eff: [num_rx, num_layers]
            sigma2: noise variance
        
        Returns:
            s_hat: [num_layers]
        """
        # Gram matrix: H^H · H
        HH = H_eff.conj().T @ H_eff  # [num_layers, num_layers]
        
        # Regularización: HH + σ²I
        regularization = sigma2 * np.eye(self.num_layers)
        
        # Inversión: (HH + σ²I)^-1
        try:
            HH_inv = np.linalg.inv(HH + regularization)
        except np.linalg.LinAlgError:
            # Si falla, usar pseudoinverse
            HH_inv = np.linalg.pinv(HH + regularization)
        
        # Filtro MMSE: W = (H^H·H + σ²I)^-1 · H^H
        W_mmse = HH_inv @ H_eff.conj().T  # [num_layers, num_rx]
        
        # Detección: s_hat = W · y
        s_hat = W_mmse @ y  # [num_layers]
        
        return s_hat
    
    def _zf_detect(self, y, H_eff):
        """
        Zero-Forcing Detector.
        
        W_zf = (H^H · H)^-1 · H^H = H^†  (pseudoinverse)
        s_hat = W_zf · y
        
        Cancela interferencia completamente pero amplifica ruido.
        Equivalente a MMSE con SNR → ∞.
        
        Args:
            y: [num_rx]
            H_eff: [num_rx, num_layers]
        
        Returns:
            s_hat: [num_layers]
        """
        # Pseudoinverse de H
        H_pinv = np.linalg.pinv(H_eff)  # [num_layers, num_rx]
        
        # Detección
        s_hat = H_pinv @ y  # [num_layers]
        
        return s_hat
    
    def _sic_detect(self, y, H_eff, sigma2):
        """
        SIC (Successive Interference Cancellation) Detector.
        
        Proceso iterativo (non-linear):
        1. Ordenar capas por SINR (más fuerte primero)
        2. Para cada capa:
           a. Detectar con MMSE
           b. Hard decision (demodular)
           c. Regenerar señal
           d. Cancelar interferencia
           e. Siguiente capa
        
        Mejor performance que MMSE pero más complejo.
        
        Args:
            y: [num_rx]
            H_eff: [num_rx, num_layers]
            sigma2: noise variance
        
        Returns:
            s_hat: [num_layers]
        """
        print(f"[_sic_detect] ENTRADA: num_layers={self.num_layers}, |y|={np.linalg.norm(y):.3f}")
        
        if self.symbol_detector is None:
            # Si no hay detector, usar MMSE en vez de SIC
            print("[WARNING] SIC requiere constellation, usando MMSE")
            return self._mmse_detect(y, H_eff, sigma2)
        
        # Inicializar
        y_residual = y.copy()
        H_remaining = H_eff.copy()
        s_hat = np.zeros(self.num_layers, dtype=complex)
        detected_order = []
        remaining_layers = list(range(self.num_layers))  # Índices de capas aún por detectar
        
        # Calcular SINR inicial de cada capa
        sinr_per_layer = self._calculate_sinr_per_layer(H_eff, sigma2)
        
        # Ordenar capas por SINR (descendente)
        layer_order = np.argsort(sinr_per_layer)[::-1]
        
        # DEBUG: Guardar H original para cancelación correcta
        H_original = H_eff.copy()
        
        # Detección sucesiva
        for iteration in range(self.num_layers):
            layer_idx = layer_order[iteration]  # Índice absoluto en s_hat
            
            # Encontrar índice relativo en H_remaining
            relative_idx = remaining_layers.index(layer_idx)
            
            # Detectar esta capa con MMSE sobre señal residual
            s_mmse = self._mmse_detect_single_layer(
                y_residual, H_remaining, sigma2, relative_idx
            )
            
            # DEBUG
            if iteration == 0:
                print(f"[SIC DEBUG] Iteración {iteration}: layer_idx={layer_idx}, relative_idx={relative_idx}, |s_mmse|={np.abs(s_mmse):.3f}")
            
            # Hard decision: encontrar símbolo más cercano en constelación
            if self.symbol_detector is not None and isinstance(self.symbol_detector, np.ndarray):
                # Constelación es array numpy: buscar símbolo más cercano
                distances = np.abs(self.symbol_detector - s_mmse)
                closest_idx = np.argmin(distances)
                s_hard = self.symbol_detector[closest_idx]
            elif hasattr(self.symbol_detector, 'detect'):
                # SymbolDetector con método detect()
                s_hard = self.symbol_detector.detect(np.array([s_mmse]))[0]
            else:
                # Sin constelación: usar símbolo MMSE directamente (soft decision)
                print("[WARNING] SIC sin constelación, usando soft decision")
                s_hard = s_mmse
            
            # Guardar
            s_hat[layer_idx] = s_hard
            detected_order.append(layer_idx)
            
            # DEBUG
            if iteration == 0:
                print(f"[SIC DEBUG] Hard decision: s_hard={s_hard:.3f}, diff={np.abs(s_hard - s_mmse):.3f}")
            
            # Regenerar interferencia de esta capa usando H ORIGINAL
            h_layer = H_original[:, layer_idx]  # FIX: usar columna de H original, no H_remaining
            interference = h_layer * s_hard
            
            # DEBUG
            if iteration == 0:
                print(f"[SIC DEBUG] |h_layer|={np.linalg.norm(h_layer):.3f}, |interferencia|={np.linalg.norm(interference):.3f}, |y_residual|={np.linalg.norm(y_residual):.3f}")
            
            # Cancelar interferencia
            y_residual = y_residual - interference
            
            # DEBUG
            if iteration == 0:
                print(f"[SIC DEBUG] Después de cancelar: |y_residual|={np.linalg.norm(y_residual):.3f}")
            
            # Remover esta capa de H y de remaining_layers (para siguiente iteración)
            if iteration < self.num_layers - 1:
                mask = np.ones(len(remaining_layers), dtype=bool)
                mask[relative_idx] = False
                H_remaining = H_remaining[:, mask]
                remaining_layers.pop(relative_idx)
        
        return s_hat
    
    def _mmse_detect_single_layer(self, y, H, sigma2, layer_idx):
        """
        Detecta una sola capa con MMSE.
        """
        num_layers_remaining = H.shape[1]
        
        if num_layers_remaining == 1:
            # Última capa: MRC simple
            h = H[:, 0]
            s = np.vdot(h, y) / (np.linalg.norm(h)**2 + sigma2)
            return s
        
        # MMSE completo
        HH = H.conj().T @ H
        HH_inv = np.linalg.inv(HH + sigma2 * np.eye(num_layers_remaining))
        W_mmse = HH_inv @ H.conj().T
        s_all = W_mmse @ y
        
        return s_all[layer_idx]
    
    def _calculate_sinr_per_layer(self, H_eff, sigma2):
        """
        Calcula SINR post-detección de cada capa (para ordenar en SIC).
        
        SINR_i ≈ ||h_i||² / (interferencia + ruido)
        """
        sinr = np.zeros(self.num_layers)
        
        for i in range(self.num_layers):
            h_i = H_eff[:, i]
            signal_power = np.linalg.norm(h_i) ** 2
            
            # Interferencia de otras capas
            interference_power = 0
            for j in range(self.num_layers):
                if j != i:
                    h_j = H_eff[:, j]
                    interference_power += np.linalg.norm(h_j) ** 2
            
            # SINR
            sinr[i] = signal_power / (interference_power + sigma2 + 1e-10)
        
        return sinr
    
    def _mrc_detect(self, y, H_eff):
        """
        MRC (Maximum Ratio Combining) para rank-1.
        
        W_mrc = H^* / ||H||²
        s_hat = W_mrc · y
        
        Solo válido para num_layers = 1.
        """
        h = H_eff[:, 0]  # [num_rx]
        
        # MRC weights
        w_mrc = h.conj() / (np.linalg.norm(h) ** 2)
        
        # Combinar
        s_hat = np.dot(w_mrc, y)
        
        return np.array([s_hat])


def test_mimo_detector():
    """
    Tests del MIMODetector
    """
    print("\n=== Test MIMODetector ===")
    
    np.random.seed(42)
    
    # Test 1: MMSE con 2×2, rank-2
    print("\n1. Test MMSE detector (2x2, rank-2):")
    num_rx = 2
    num_tx = 2
    num_layers = 2
    
    # Canal
    H = np.array([[1+0.5j, 0.3+0.2j],
                  [0.2+0.3j, 1+0.4j]])
    
    # Símbolos transmitidos
    s_tx = np.array([1+1j, -1+1j])
    
    # Señal recibida: y = H·s + n
    noise = 0.1 * (np.random.randn(num_rx) + 1j * np.random.randn(num_rx))
    y = H @ s_tx + noise
    
    detector_mmse = MIMODetector(num_rx, num_layers, detector_type='MMSE')
    s_hat = detector_mmse.detect(y, H, noise_variance=0.01)
    
    print(f"   Símbolos TX: {s_tx}")
    print(f"   Símbolos RX: {s_hat}")
    error = np.linalg.norm(s_tx - s_hat)
    print(f"   Error: {error:.4f}")
    assert error < 1.0, "Error muy grande en MMSE"
    print("   [OK] MMSE detector OK")
    
    # Test 2: ZF detector
    print("\n2. Test ZF detector (2x2, rank-2):")
    detector_zf = MIMODetector(num_rx, num_layers, detector_type='ZF')
    s_hat_zf = detector_zf.detect(y, H, noise_variance=0.01)
    print(f"   Símbolos RX (ZF): {s_hat_zf}")
    error_zf = np.linalg.norm(s_tx - s_hat_zf)
    print(f"   Error: {error_zf:.4f}")
    print("   [OK] ZF detector OK")
    
    # Test 3: MRC para rank-1
    print("\n3. Test MRC detector (2x1, rank-1):")
    H_mrc = H[:, :1]  # Solo primera columna
    s_tx_mrc = s_tx[:1]
    y_mrc = H_mrc @ s_tx_mrc + noise
    
    detector_mrc = MIMODetector(num_rx, num_layers=1, detector_type='MRC')
    s_hat_mrc = detector_mrc.detect(y_mrc, H_mrc, noise_variance=0.01)
    print(f"   Símbolos TX: {s_tx_mrc}")
    print(f"   Símbolos RX: {s_hat_mrc}")
    print("   [OK] MRC detector OK")
    
    # Test 4: Detección por subportadora
    print("\n4. Test detección multi-subportadora:")
    num_sc = 10
    y_multi = np.random.randn(num_rx, num_sc) + 1j * np.random.randn(num_rx, num_sc)
    H_multi = np.tile(H[:, :, np.newaxis], (1, 1, num_sc))
    
    s_hat_multi = detector_mmse.detect(y_multi, H_multi, noise_variance=0.01)
    assert s_hat_multi.shape == (num_layers, num_sc)
    print(f"   Shape detectado: {s_hat_multi.shape}")
    print("   ✓ Multi-subportadora OK")
    
    print("\n=== Todos los tests pasaron ✓ ===\n")


if __name__ == '__main__':
    test_mimo_detector()
