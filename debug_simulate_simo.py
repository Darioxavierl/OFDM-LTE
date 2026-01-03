"""
Debug: Exact reproduction of simulate_simo logic
"""
import numpy as np
from config import LTEConfig
from core.ofdm_core import OFDMSimulator
from concurrent.futures import ThreadPoolExecutor, as_completed

# Create simulator with small test
config = LTEConfig(bandwidth=5.0, modulation='QPSK')
sim = OFDMSimulator(config, channel_type='awgn', mode='lte')

# Create test bits
bits = np.random.randint(0, 2, size=2000)
snr_db = 20.0

print("=" * 80)
print("SISO BASELINE")
print("=" * 80)

# SISO
result_siso = sim.simulate_siso(bits, snr_db=snr_db)
ber_siso = result_siso['ber']
print(f"SISO BER: {ber_siso:.6f}")

print("\n" + "=" * 80)
print("SIMO WITH 2 RX")
print("=" * 80)

# Now manually replicate simulate_simo logic
signal_tx, symbols_tx, _ = sim.tx.modulate(bits)
sim.channels[0].set_snr(snr_db)
signals_rx, _ = sim.channels[0].transmit_simo(signal_tx, num_rx=2)

print(f"\nSignals received:")
print(f"  signals_rx[0] shape: {signals_rx[0].shape}")
print(f"  signals_rx[1] shape: {signals_rx[1].shape}")

# Demodulate each antenna manually
symbols_rx_list = []
h_estimates = []
for i, signal_rx in enumerate(signals_rx):
    print(f"\n  Antenna {i}:")
    symbols, h_est = sim._demodulate_with_channel_est(signal_rx)
    print(f"    symbols shape: {symbols.shape}")
    print(f"    h_est shape: {h_est.shape}")
    symbols_rx_list.append(symbols)
    h_estimates.append(h_est)

# Combine with MRC
print(f"\n  MRC combining...")
symbols_combined = sim._combine_symbols_mrc(symbols_rx_list, h_estimates)
print(f"    symbols_combined shape: {symbols_combined.shape}")

# Demodulate combined symbols
print(f"\n  Demodulating combined symbols...")
bits_rx = sim.rx.demodulator.qam_demodulator.symbols_to_bits(symbols_combined)
print(f"    bits_rx shape: {bits_rx.shape}")

# Calculate BER
bits_test = bits[:min(len(bits), len(bits_rx))]
bits_rx_test = bits_rx[:len(bits_test)]
ber_manual = np.sum(bits_test != bits_rx_test) / len(bits_test)
print(f"\n  Manual SIMO BER: {ber_manual:.6f}")

# Compare with simulate_simo
print(f"\n" + "=" * 80)
print("USING simulate_simo() method")
print("=" * 80)
result_simo = sim.simulate_simo(bits, snr_db=snr_db, num_rx=2)
ber_simo = result_simo['ber']
print(f"SIMO BER: {ber_simo:.6f}")

print(f"\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"SISO BER:    {ber_siso:.6f}")
print(f"Manual SIMO: {ber_manual:.6f}")
print(f"Method SIMO: {ber_simo:.6f}")
