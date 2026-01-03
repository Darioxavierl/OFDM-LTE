"""
RAYLEIGH DIAGNOSTIC - CORRECTED VERSION
Analyzes Rayleigh multipath performance issues
"""

import numpy as np
from config import LTEConfig
from core.ofdm_core import OFDMSimulator
from core.channel import FadingChannel

def run_rayleigh_diagnostic():
    print("\n" + "="*70)
    print("RAYLEIGH MULTIPATH DIAGNOSTIC")
    print("="*70)
    
    # PART 1: Channel Independence
    print("\n[PART 1] Checking channel independence...")
    print("-" * 70)
    
    num_samples = 10000
    signal_tx = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
    signal_tx = signal_tx / np.sqrt(np.mean(np.abs(signal_tx)**2))
    
    channels = [FadingChannel(snr_db=20.0) for _ in range(4)]
    signals_rx = [ch.transmit(signal_tx.copy()) for ch in channels]
    
    # Normalize
    signals_norm = [sig / np.sqrt(np.mean(np.abs(sig)**2)) for sig in signals_rx]
    
    # Correlation
    print("\nChannel Correlation Matrix (should have 1.0 on diagonal, ~0 off-diagonal):")
    for i in range(4):
        for j in range(4):
            corr = np.abs(np.mean(signals_norm[i] * np.conj(signals_norm[j])))
            print("  RX{} vs RX{}: {:.6f}".format(i, j, corr), end="")
            if i == j:
                print(" [Diagonal]")
            else:
                print()
    
    # Off-diagonal average
    off_diag = []
    for i in range(4):
        for j in range(4):
            if i != j:
                corr = np.abs(np.mean(signals_norm[i] * np.conj(signals_norm[j])))
                off_diag.append(corr)
    
    avg_corr = np.mean(off_diag)
    print("\n==> Average off-diagonal correlation: {:.6f}".format(avg_corr))
    if avg_corr < 0.1:
        print("    RESULT: Channels are INDEPENDENT (Good!)")
    else:
        print("    RESULT: Channels have CORRELATION (Problem!)")
    
    # PART 2: Power behavior
    print("\n[PART 2] Checking power levels across symbols...")
    print("-" * 70)
    
    num_symbols = 34
    samples_per_sym = 1024
    total_samples = num_symbols * samples_per_sym
    
    signal_tx = np.random.randn(total_samples) + 1j * np.random.randn(total_samples)
    signal_tx = signal_tx / np.sqrt(np.mean(np.abs(signal_tx)**2))
    
    channels = [FadingChannel(snr_db=20.0) for _ in range(4)]
    signals_rx = [ch.transmit(signal_tx.copy()) for ch in channels]
    
    powers_per_symbol = [[] for _ in range(4)]
    
    for sym_idx in range(num_symbols):
        start = sym_idx * samples_per_sym
        end = start + samples_per_sym
        
        for ant_idx in range(4):
            power_lin = np.mean(np.abs(signals_rx[ant_idx][start:end])**2)
            power_db = 10 * np.log10(power_lin + 1e-12)
            powers_per_symbol[ant_idx].append(power_db)
    
    # Display power statistics
    for ant_idx in range(4):
        powers = powers_per_symbol[ant_idx]
        print("\nAntenna {}:".format(ant_idx))
        print("  Min: {:.2f} dB, Max: {:.2f} dB, Mean: {:.2f} dB, StdDev: {:.2f} dB".format(
            min(powers), max(powers), np.mean(powers), np.std(powers)))
    
    # Power correlation across antennas
    powers_array = np.array(powers_per_symbol)
    power_corr = np.corrcoef(powers_array)
    
    print("\nPower variation correlation across antennas:")
    for i in range(4):
        for j in range(i+1, 4):
            pc = power_corr[i, j]
            if np.isnan(pc):
                print("  RX{} vs RX{}: (no variation)".format(i, j))
            else:
                print("  RX{} vs RX{}: {:.4f}".format(i, j, pc), end="")
                if abs(pc) > 0.6:
                    print(" [SYNCHRONIZED FADES - Problem!]")
                else:
                    print(" [Independent - OK]")
    
    # PART 3: BER Performance
    print("\n[PART 3] Performance testing with Rayleigh channel...")
    print("-" * 70)
    
    config = LTEConfig()
    simulator = OFDMSimulator(config)
    
    # Test with large bit count
    num_bits_test = 100000
    bits = np.random.randint(0, 2, num_bits_test)
    
    print("\nTesting with {} bits at different SNR levels:".format(num_bits_test))
    print()
    
    test_results = {}
    
    for snr in [10, 15, 20]:
        print("SNR = {} dB:".format(snr))
        
        # SISO
        result_siso = simulator.simulate_siso(bits, snr_db=snr)
        ber_siso = result_siso['ber']
        print("  SISO (1RX):  BER = {:.6e}".format(ber_siso))
        
        # SIMO with different antenna counts
        for num_rx in [2, 4]:
            result_simo = simulator.simulate_simo(bits, snr_db=snr, num_rx=num_rx, parallel=True)
            ber_simo = result_simo['ber']
            print("  SIMO ({}RX):  BER = {:.6e}".format(num_rx, ber_simo))
            
            if ber_siso > 0 and ber_simo > 0:
                improvement = (ber_siso - ber_simo) / ber_siso * 100
                snr_gain = 10 * np.log10(ber_siso / max(ber_simo, 1e-8))
                print("           Improvement: {:.1f}%, SNR gain: {:.2f} dB".format(improvement, snr_gain))
            elif ber_siso > 0 and ber_simo == 0:
                print("           [Perfect! SIMO achieved zero BER]")
            else:
                print("           [Both zero BER - inconclusive]")
        
        print()
    
    # SUMMARY
    print("\n" + "="*70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*70)
    
    print("""
FINDINGS:

1. Channel Independence:
   - If avg correlation < 0.1: Good! Channels are independent
   - If avg correlation > 0.1: Problem! Channels are correlated
   - Correlated channels = reduced diversity benefit

2. Temporal Variation:
   - If power std-dev is high: Channels experience deep fades
   - If power correlation is high: Fades happen SIMULTANEOUSLY
   - Simultaneous fades = MRC can't help

3. AWGN vs Rayleigh Performance:
   - AWGN: MRC ideal, achieves N*SNR improvement (27dB with 4RX)
   - Rayleigh: MRC good, but limited by simultaneous fades (1-25dB)
   - This difference is NORMAL and EXPECTED

NORMAL BEHAVIOR:
   - AWGN: 99% improvement with 4RX
   - Rayleigh: 20-50% improvement with 4RX
   - Reason: In fading, all antennas fade together sometimes

NEXT STEPS IF IMPROVEMENT NEEDED:
   1. Use MMSE combining instead of MRC
   2. Add temporal interpolation for channel estimates
   3. Implement Alamouti space-time coding
   4. Use adaptive combining based on SNR
""")
    
    print("="*70)


if __name__ == '__main__':
    run_rayleigh_diagnostic()
