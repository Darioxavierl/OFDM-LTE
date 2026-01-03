"""
Test MRC with identical signals (should give same BER as SISO)
"""
import numpy as np
from config import LTEConfig
from core.ofdm_core import OFDMSimulator

# Config
config = LTEConfig(bandwidth=10.0, delta_f=15.0, modulation='64-QAM')
sim = OFDMSimulator(config, channel_type='awgn', mode='lte')

# TX
bits = np.random.randint(0, 2, size=100000)
snr_db = 20.0

# SISO
print("="*80)
print("SISO")
print("="*80)
result_siso = sim.simulate_siso(bits, snr_db=snr_db)
ber_siso = result_siso['ber']
print(f"BER: {ber_siso:.6f}")

# SIMO with 2 RX (independent channels)
print("\n" + "="*80)
print("SIMO with 2 RX (independent channels)")
print("="*80)
result_simo_ind = sim.simulate_simo(bits, snr_db=snr_db, num_rx=2)
ber_simo_ind = result_simo_ind['ber']
print(f"BER: {ber_simo_ind:.6f}")

# Now test MRC logic: if we pass IDENTICAL signals, MRC should give SAME BER as SISO
# (no diversity gain, but also no degradation)
print("\n" + "="*80)
print("MRC TEST: Combining identical signals")
print("="*80)

# TX once
signal_tx, _, _ = sim.tx.modulate(bits)
sim.channels[0].set_snr(snr_db)

# Pass through ONE channel twice
signal_rx = sim.channels[0].transmit(signal_tx)
signals_rx = [signal_rx, signal_rx.copy()]  # Identical signals

# Demodulate each
symbols_rx_list = []
h_estimates_per_antenna = []
for i, sig in enumerate(signals_rx):
    symbols, h_est = sim._demodulate_with_channel_est(sig)
    symbols_rx_list.append(symbols)
    h_estimates_per_antenna.append(h_est)
    print(f"  Antenna {i}: {symbols.shape}")

# Combine
symbols_combined = sim._combine_symbols_mrc(symbols_rx_list, h_estimates_per_antenna)
print(f"  Combined: {symbols_combined.shape}")

# Demodulate
bits_rx = sim.rx.demodulator.qam_demodulator.symbols_to_bits(symbols_combined)
bits_test = bits[:min(len(bits), len(bits_rx))]
bits_rx_test = bits_rx[:len(bits_test)]
ber_mrc_identical = np.sum(bits_test != bits_rx_test) / len(bits_test)
print(f"BER with MRC (identical signals): {ber_mrc_identical:.6f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"SISO BER:              {ber_siso:.6f}")
print(f"SIMO (independent):    {ber_simo_ind:.6f}")
print(f"MRC (identical input): {ber_mrc_identical:.6f}")
print(f"\nExpected: MRC (identical) should ≈ SISO")
print(f"Actual:   MRC ({ber_mrc_identical:.4e}) vs SISO ({ber_siso:.4e})")
