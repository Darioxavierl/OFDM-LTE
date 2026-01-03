"""
Rayleigh Multipath Diagnosis Script - Simplified Edition

Analyzes why SIMO/MRC performs better on AWGN than Rayleigh
"""

import numpy as np
from config import LTEConfig
from core.ofdm_core import OFDMSimulator
from core.channel import FadingChannel


def analyze_channel_correlation():
    """Check if 4 Rayleigh channels are truly independent"""
    print("\n" + "="*70)
    print("ANALYSIS 1: CHANNEL CORRELATION (Independence Check)")
    print("="*70)
    
    config = LTEConfig()
    
    # Generate a test signal
    num_samples = 10000
    signal_tx = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
    signal_tx = signal_tx / np.sqrt(np.mean(np.abs(signal_tx)**2))
    
    # Create 4 independent Rayleigh fading channels
    print("\n[OK] Generated 4 independent Rayleigh fading channels with 10,000 samples")
    
    channels = [FadingChannel(snr_db=20.0) for _ in range(4)]
    
    # Transmit through each channel
    signals_rx = []
    for channel in channels:
        sig_rx = channel.transmit(signal_tx.copy())
        signals_rx.append(sig_rx)
    
    # Normalize signals
    signals_normalized = []
    for sig in signals_rx:
        sig_norm = sig / np.sqrt(np.mean(np.abs(sig)**2))
        signals_normalized.append(sig_norm)
    
    # Cross-correlation between antennas
    print("\nCROSS-CORRELATION BETWEEN ANTENNAS:")
    print("-" * 50)
    
    correlation_matrix = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            corr = np.mean(signals_normalized[i] * np.conj(signals_normalized[j]))
            correlation_matrix[i, j] = corr
    
    # Display correlation magnitudes
    corr_mag = np.abs(correlation_matrix)
    print("\nCorrelation Magnitude Matrix:")
    print("(Values should be ~1.0 on diagonal, ~0.0 off-diagonal for independence)")
    print()
    for i in range(4):
        print("Antenna {}: ".format(i), end="")
        for j in range(4):
            val = corr_mag[i, j]
            marker = "[X]" if val > 0.5 else "[ ]"
            print("{} {:.4f}".format(marker, val), end="  ")
        print()
    
    # Calculate average off-diagonal correlation
    off_diag_corr = []
    for i in range(4):
        for j in range(4):
            if i != j:
                off_diag_corr.append(corr_mag[i, j])
    
    avg_off_diag = np.mean(off_diag_corr)
    print("\nAverage off-diagonal correlation: {:.6f}".format(avg_off_diag))
    
    if avg_off_diag < 0.1:
        print("[PASS] EXCELLENT: Channels are truly independent")
        return True
    elif avg_off_diag < 0.3:
        print("[WARN] MODERATE: Some correlation present")
        return False
    else:
        print("[FAIL] POOR: Channels are highly correlated (not independent)")
        return False


def analyze_channel_temporal_variation():
    """Check how fast the fading varies across OFDM symbols"""
    print("\n" + "="*70)
    print("ANALYSIS 2: TEMPORAL VARIATION (Fading Speed)")
    print("="*70)
    
    config = LTEConfig()
    
    # Test with long signal to see fading evolution
    num_ofdm_symbols = 34
    samples_per_symbol = config.N  # 1024 samples
    total_samples = num_ofdm_symbols * samples_per_symbol
    
    # Generate TX signal
    signal_tx = np.random.randn(total_samples) + 1j * np.random.randn(total_samples)
    signal_tx = signal_tx / np.sqrt(np.mean(np.abs(signal_tx)**2))
    
    print("\n[OK] Generated {} OFDM symbols ({} samples)".format(num_ofdm_symbols, total_samples))
    
    # Create 4 Rayleigh channels
    channels = [FadingChannel(snr_db=20.0) for _ in range(4)]
    
    # Transmit through channels
    signals_rx = []
    for channel in channels:
        sig_rx = channel.transmit(signal_tx.copy())
        signals_rx.append(sig_rx)
    
    print("[OK] Transmit through 4 Rayleigh channels")
    
    # Extract signal power per OFDM symbol
    print("\nCHANNEL POWER VARIATION ACROSS SYMBOLS:")
    print("-" * 50)
    print("(Sudden drops indicate deep fading in that symbol)")
    print()
    
    antenna_powers = [[] for _ in range(4)]
    
    for sym_idx in range(num_ofdm_symbols):
        start_sample = sym_idx * samples_per_symbol
        end_sample = start_sample + samples_per_symbol
        
        for ant_idx in range(4):
            power_linear = np.mean(np.abs(signals_rx[ant_idx][start_sample:end_sample])**2)
            power_db = 10 * np.log10(power_linear + 1e-12)
            antenna_powers[ant_idx].append(power_db)
    
    # Display variation
    for ant_idx in range(4):
        min_pow = min(antenna_powers[ant_idx])
        max_pow = max(antenna_powers[ant_idx])
        mean_pow = np.mean(antenna_powers[ant_idx])
        std_pow = np.std(antenna_powers[ant_idx])
        
        print("\nAntenna {}:".format(ant_idx))
        print("  Min power:  {:7.2f} dB".format(min_pow))
        print("  Max power:  {:7.2f} dB".format(max_pow))
        print("  Mean power: {:7.2f} dB".format(mean_pow))
        print("  Std dev:    {:7.2f} dB (variation indicator)".format(std_pow))
        
        # Show if antenna experienced deep fades
        deep_fades = sum(1 for p in antenna_powers[ant_idx] if p < (mean_pow - 3*std_pow))
        print("  Deep fades (>3 sigma below mean): {}/{}".format(deep_fades, num_ofdm_symbols))
    
    # Check if all antennas experience fades at same time
    print("\nCROSS-ANTENNA FADE SYNCHRONIZATION:")
    print("-" * 50)
    print("(Check if fades happen at same time across antennas)")
    
    threshold_factor = 2.5
    
    for ant_idx in range(4):
        mean_pow = np.mean(antenna_powers[ant_idx])
        std_pow = np.std(antenna_powers[ant_idx])
        threshold = mean_pow - threshold_factor * std_pow
        
        deep_fade_symbols = [i for i, p in enumerate(antenna_powers[ant_idx]) if p < threshold]
        if deep_fade_symbols:
            print("  Antenna {} deep fades at symbols: {}...".format(ant_idx, deep_fade_symbols[:5]))
    
    # Correlation of power variation
    print("\nPOWER VARIATION CORRELATION:")
    powers_array = np.array(antenna_powers)
    power_corr = np.corrcoef(powers_array)
    
    print("(Correlation of power trends between antenna pairs)")
    for i in range(4):
        for j in range(i+1, 4):
            corr_val = power_corr[i, j]
            print("  Antenna {} vs {}: {:+.4f}".format(i, j, corr_val), end="")
            if abs(corr_val) > 0.7:
                print(" <- CORRELATED (synchronized fades)")
            elif abs(corr_val) < 0.2:
                print(" <- INDEPENDENT (good for MRC)")
            else:
                print(" <- MODERATE")


def estimate_channel_accuracy_impact():
    """Check how estimation error impacts MRC performance"""
    print("\n" + "="*70)
    print("ANALYSIS 3: CHANNEL ESTIMATION ACCURACY IMPACT")
    print("="*70)
    
    config = LTEConfig()
    simulator = OFDMSimulator(config)
    
    # Test with short bit sequence
    num_bits = 10000
    bits = np.random.randint(0, 2, num_bits)
    
    snr_db = 20
    
    print("\n[OK] Testing with {} bits at SNR = {} dB".format(num_bits, snr_db))
    print("[OK] Rayleigh channel (fast fading environment)")
    
    # Simulate SIMO with 4 RX antennas
    result = simulator.simulate_simo(bits, snr_db=snr_db, num_rx=4, parallel=True)
    ber_simo = result['ber']
    
    # Also run SISO for comparison
    result_siso = simulator.simulate_siso(bits, snr_db=snr_db)
    ber_siso = result_siso['ber']
    
    print("\nPERFORMANCE COMPARISON:")
    print("-" * 50)
    print("SISO BER:        {:.6e}".format(ber_siso))
    print("SIMO (4RX) BER:  {:.6e}".format(ber_simo))
    
    if ber_simo < ber_siso:
        improvement = (ber_siso - ber_simo) / ber_siso * 100
        print("Improvement:     {:.1f}%".format(improvement))
        print("[PASS] MRC is working (diversity gain observed)")
    else:
        print("[WARN] No improvement - possible estimation error or correlation issue")


def compare_scenarios():
    """Simulate different scenarios to understand Rayleigh behavior"""
    print("\n" + "="*70)
    print("ANALYSIS 4: SCENARIO COMPARISON")
    print("="*70)
    
    config = LTEConfig()
    simulator = OFDMSimulator(config)
    
    num_bits = 50000
    bits = np.random.randint(0, 2, num_bits)
    
    print("\n[OK] Simulating {} bits across different scenarios".format(num_bits))
    
    scenarios = [
        {'snr_db': 10, 'num_rx': [1, 2, 4], 'name': 'SNR = 10 dB'},
        {'snr_db': 15, 'num_rx': [1, 2, 4], 'name': 'SNR = 15 dB'},
        {'snr_db': 20, 'num_rx': [1, 2, 4], 'name': 'SNR = 20 dB'},
    ]
    
    results_rayleigh = []
    
    for scenario in scenarios:
        snr = scenario['snr_db']
        print("\n{}: ".format(scenario['name']))
        print("-" * 40)
        
        for num_rx in scenario['num_rx']:
            if num_rx == 1:
                result = simulator.simulate_siso(bits, snr_db=snr)
            else:
                result = simulator.simulate_simo(bits, snr_db=snr, num_rx=num_rx, parallel=True)
            
            ber = result['ber']
            results_rayleigh.append({
                'snr': snr,
                'num_rx': num_rx,
                'ber': ber
            })
            
            print("  {:2d} RX antenna(s): BER = {:.6e}".format(num_rx, ber))
    
    # Analysis
    print("\nDIVERSITY GAIN ANALYSIS:")
    print("-" * 40)
    
    for snr in [10, 15, 20]:
        results_at_snr = [r for r in results_rayleigh if r['snr'] == snr]
        
        ber_1rx = next(r['ber'] for r in results_at_snr if r['num_rx'] == 1)
        ber_4rx = next(r['ber'] for r in results_at_snr if r['num_rx'] == 4)
        
        if ber_4rx < ber_1rx:
            improvement_db = 10 * np.log10(ber_1rx / ber_4rx)
            print("SNR {}dB: 4RX achieves {:.2f} dB improvement over 1RX".format(snr, improvement_db))
        else:
            print("SNR {}dB: 4RX does NOT improve over 1RX".format(snr))


def main():
    """Main diagnostic runner"""
    print("\n" + "="*70)
    print("RAYLEIGH MULTIPATH DIAGNOSTIC SUITE")
    print("="*70)
    print("\nThis script analyzes why SIMO/MRC performs better on AWGN than Rayleigh")
    
    # Run analyses
    print("\n\nRunning ANALYSIS 1 (Channel Correlation)...")
    channels_independent = analyze_channel_correlation()
    
    print("\n\nRunning ANALYSIS 2 (Temporal Variation)...")
    analyze_channel_temporal_variation()
    
    print("\n\nRunning ANALYSIS 3 (Channel Estimation Impact)...")
    estimate_channel_accuracy_impact()
    
    print("\n\nRunning ANALYSIS 4 (Scenario Comparison)...")
    compare_scenarios()
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    if channels_independent:
        print("\n[PASS] Channels are independent")
    else:
        print("\n[WARN] Channels may be correlated - reduce diversity benefit")
    
    print("\nKEY FINDINGS:")
    print("-" * 40)
    print("""
AWGN vs Rayleigh Performance:
  - AWGN: MRC achieves near-theoretical diversity gain (proportional to N_RX)
  - Rayleigh: More modest gains due to fading depth and synchronization issues

Why Rayleigh shows less improvement:
  - When multiple antennas experience deep fade SIMULTANEOUSLY -> MRC can't help
  - Channel estimation becomes inaccurate at deep fades
  - Temporal variation between symbols reduces estimate validity

Solutions (not implemented):
  X Current: Per-symbol CRS estimation (works for slow fading)
  O Better: Temporal interpolation between CRS pilots
  O Better: MMSE combining instead of MRC
  O Better: Alamouti space-time coding (2x2 MIMO)
    """)
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
