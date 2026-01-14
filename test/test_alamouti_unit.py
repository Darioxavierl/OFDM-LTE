"""
Test unitario simple para verificar Alamouti SFBC
"""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sfbc_alamouti import SFBCAlamouti

def test_alamouti_perfect_channel():
    """
    Test Alamouti con canal perfecto (sin ruido)
    Debería recuperar los símbolos exactamente
    """
    print("\n" + "="*70)
    print("TEST: Alamouti SFBC - Canal Perfecto")
    print("="*70)
    
    # Create encoder/decoder
    alamouti = SFBCAlamouti(num_tx=2, enabled=True)
    
    # Original symbols
    s0 = 1.0 + 1.0j  # Símbolo QAM
    s1 = -1.0 + 1.0j
    symbols_orig = np.array([s0, s1])
    
    print(f"\n[1] Símbolos originales:")
    print(f"    s0 = {s0}")
    print(f"    s1 = {s1}")
    
    # Encode
    tx0, tx1 = alamouti.encode(symbols_orig)
    
    print(f"\n[2] Después de encoding Alamouti:")
    print(f"    TX0[k]   = {tx0[0]}  (debería ser s0)")
    print(f"    TX0[k+1] = {tx0[1]}  (debería ser -conj(s1))")
    print(f"    TX1[k]   = {tx1[0]}  (debería ser s1)")
    print(f"    TX1[k+1] = {tx1[1]}  (debería ser conj(s0))")
    
    # Verify encoding
    expected_tx0_k = s0
    expected_tx0_k1 = -np.conj(s1)
    expected_tx1_k = s1
    expected_tx1_k1 = np.conj(s0)
    
    assert np.isclose(tx0[0], expected_tx0_k), f"TX0[k] incorrecto"
    assert np.isclose(tx0[1], expected_tx0_k1), f"TX0[k+1] incorrecto"
    assert np.isclose(tx1[0], expected_tx1_k), f"TX1[k] incorrecto"
    assert np.isclose(tx1[1], expected_tx1_k1), f"TX1[k+1] incorrecto"
    print(f"    ✅ Encoding correcto")
    
    # Channel (AWGN con fase diferente)
    h0 = 1.0 + 0j  # TX0 -> RX: canal 1 (0°)
    h1 = 0.0 + 1.0j  # TX1 -> RX: canal 2 (90°)
    
    print(f"\n[3] Canal MIMO:")
    print(f"    h0 = {h0} (magnitud = {abs(h0):.3f})")
    print(f"    h1 = {h1} (magnitud = {abs(h1):.3f})")
    
    # Received signal
    r_k = h0 * tx0[0] + h1 * tx1[0]
    r_k1 = h0 * tx0[1] + h1 * tx1[1]
    rx_symbols = np.array([r_k, r_k1])
    
    print(f"\n[4] Símbolos recibidos:")
    print(f"    r_k   = {r_k}")
    print(f"    r_k+1 = {r_k1}")
    
    # Manual calculation
    print(f"\n[5] Cálculo manual esperado:")
    print(f"    r_k   = h0*s0 + h1*s1")
    print(f"          = {h0}*{s0} + {h1}*{s1}")
    print(f"          = {h0*s0} + {h1*s1}")
    print(f"          = {h0*s0 + h1*s1}")
    print(f"    r_k+1 = h0*(-conj(s1)) + h1*conj(s0)")
    print(f"          = {h0}*{-np.conj(s1)} + {h1}*{np.conj(s0)}")
    print(f"          = {h0*(-np.conj(s1))} + {h1*np.conj(s0)}")
    print(f"          = {h0*(-np.conj(s1)) + h1*np.conj(s0)}")
    
    # Channel estimates (perfect)
    H0 = np.array([h0, h0])  # Mismo canal en k y k+1
    H1 = np.array([h1, h1])
    
    # Decode
    decoded = alamouti.decode(rx_symbols, H0, H1, regularization=1e-10)
    
    print(f"\n[6] Símbolos decodificados:")
    print(f"    s0_decoded = {decoded[0]}")
    print(f"    s1_decoded = {decoded[1]}")
    
    # Manual Alamouti combining
    print(f"\n[7] Cálculo manual Alamouti combining:")
    print(f"    s0_combined = conj(h0)*r_k + h1*conj(r_k+1)")
    print(f"                = {np.conj(h0)}*{r_k} + {h1}*{np.conj(r_k1)}")
    s0_manual = np.conj(h0)*r_k + h1*np.conj(r_k1)
    print(f"                = {s0_manual}")
    
    print(f"    s1_combined = conj(h1)*r_k - h0*conj(r_k+1)")
    print(f"                = {np.conj(h1)}*{r_k} - {h0}*{np.conj(r_k1)}")
    s1_manual = np.conj(h1)*r_k - h0*np.conj(r_k1)
    print(f"                = {s1_manual}")
    
    norm = abs(h0)**2 + abs(h1)**2
    print(f"\n    Normalización: norm = |h0|² + |h1|² = {abs(h0)**2} + {abs(h1)**2} = {norm}")
    print(f"    s0_final = s0_combined / norm = {s0_manual} / {norm} = {s0_manual/norm}")
    print(f"    s1_final = s1_combined / norm = {s1_manual} / {norm} = {s1_manual/norm}")
    
    # Check error
    error_s0 = abs(decoded[0] - s0)
    error_s1 = abs(decoded[1] - s1)
    
    print(f"\n[8] Error de decodificación:")
    print(f"    |s0_decoded - s0| = {error_s0:.6f}")
    print(f"    |s1_decoded - s1| = {error_s1:.6f}")
    
    if error_s0 < 1e-10 and error_s1 < 1e-10:
        print(f"\n✅ TEST PASSED: Alamouti decodifica correctamente (canal perfecto)")
        return True
    else:
        print(f"\n❌ TEST FAILED: Alamouti NO decodifica correctamente")
        print(f"    Esperado s0 = {s0}, obtenido = {decoded[0]}")
        print(f"    Esperado s1 = {s1}, obtenido = {decoded[1]}")
        return False


def test_alamouti_with_noise():
    """
    Test Alamouti con ruido
    """
    print("\n" + "="*70)
    print("TEST: Alamouti SFBC - Con Ruido (SNR=10dB)")
    print("="*70)
    
    np.random.seed(42)
    alamouti = SFBCAlamouti(num_tx=2, enabled=True)
    
    # Generate random QAM symbols
    N = 100
    symbols_orig = (np.random.choice([-1, 1], N) + 1j*np.random.choice([-1, 1], N)) / np.sqrt(2)
    
    # Encode
    tx0, tx1 = alamouti.encode(symbols_orig)
    
    # Channel
    h0 = 1.0 + 0j
    h1 = 0.0 + 1.0j
    
    # Received (with noise)
    rx_symbols = h0 * tx0 + h1 * tx1
    
    # Add AWGN noise
    signal_power = np.mean(np.abs(rx_symbols)**2)
    snr_db = 10.0
    snr_linear = 10**(snr_db/10)
    noise_power = signal_power / snr_linear
    noise = (np.random.randn(N) + 1j*np.random.randn(N)) * np.sqrt(noise_power/2)
    rx_symbols_noisy = rx_symbols + noise
    
    # Channel estimates (perfect knowledge)
    H0 = np.full(N, h0)
    H1 = np.full(N, h1)
    
    # Decode
    decoded = alamouti.decode(rx_symbols_noisy, H0, H1)
    
    # Calculate SER (Symbol Error Rate)
    errors = np.sum(~np.isclose(decoded.real, symbols_orig.real, atol=0.5) | 
                    ~np.isclose(decoded.imag, symbols_orig.imag, atol=0.5))
    ser = errors / N
    
    print(f"\n[Results]")
    print(f"  Símbolos: {N}")
    print(f"  SNR: {snr_db} dB")
    print(f"  SER: {ser:.4f} ({errors} errores)")
    print(f"  SER esperado @ 10dB BPSK: ~0.01")
    
    if ser < 0.10:
        print(f"\n✅ TEST PASSED: SER razonable para SNR=10dB")
        return True
    else:
        print(f"\n⚠️  WARNING: SER alto para SNR=10dB")
        return False


if __name__ == '__main__':
    print("\n")
    print("#"*70)
    print("# TESTS UNITARIOS: Alamouti SFBC")
    print("#"*70)
    
    test1_passed = test_alamouti_perfect_channel()
    test2_passed = test_alamouti_with_noise()
    
    print("\n" + "="*70)
    print("RESUMEN:")
    print(f"  Test 1 (Canal Perfecto): {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"  Test 2 (Con Ruido):      {'✅ PASS' if test2_passed else '⚠️  WARNING'}")
    print("="*70 + "\n")
    
    if test1_passed:
        print("✅ Decodificador Alamouti funciona correctamente")
        print("   El problema debe estar en otro lugar (canal, estimación, etc.)")
    else:
        print("❌ Decodificador Alamouti tiene errores")
        print("   Revisar fórmula de combining y normalización")
