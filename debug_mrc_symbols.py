"""
Debug MRC: examine combined symbols
"""
import numpy as np
from config import LTEConfig
from core.ofdm_core import OFDMSimulator

# Tiny test for visibility
config = LTEConfig(bandwidth=5.0, modulation='QPSK')
sim = OFDMSimulator(config, channel_type='awgn', mode='lte')

# Small bits
bits = np.random.randint(0, 2, size=200)
snr_db = 20.0

# TX
signal_tx, _, _ = sim.tx.modulate(bits)
sim.channels[0].set_snr(snr_db)

# Get signal twice
signal_rx = sim.channels[0].transmit(signal_tx)
signals_rx = [signal_rx, signal_rx.copy()]

# Demodulate
symbols_rx_list = []
h_estimates_per_antenna = []
for i, sig in enumerate(signals_rx):
    symbols, h_est = sim._demodulate_with_channel_est(sig)
    symbols_rx_list.append(symbols)
    h_estimates_per_antenna.append(h_est)

# Check symbols before combining
print("="*80)
print("SYMBOLS BEFORE COMBINING")
print("="*80)
print(f"Antenna 0 first 5 symbols: {symbols_rx_list[0][:5]}")
print(f"Antenna 1 first 5 symbols: {symbols_rx_list[1][:5]}")
print(f"Are they identical? {np.allclose(symbols_rx_list[0], symbols_rx_list[1])}")

# Check channel estimates
print("\n" + "="*80)
print("CHANNEL ESTIMATES")
print("="*80)
print(f"Antenna 0, Symbol 0 first 5 H: {h_estimates_per_antenna[0][0][:5]}")
print(f"Antenna 1, Symbol 0 first 5 H: {h_estimates_per_antenna[1][0][:5]}")
print(f"Are they identical? {np.allclose(h_estimates_per_antenna[0][0], h_estimates_per_antenna[1][0])}")

# Combine
symbols_combined = sim._combine_symbols_mrc(symbols_rx_list, h_estimates_per_antenna)

print("\n" + "="*80)
print("SYMBOLS AFTER COMBINING")
print("="*80)
print(f"Combined first 5 symbols: {symbols_combined[:5]}")
print(f"Expected (if identical): {symbols_rx_list[0][:5] * 2}")  # Should be 2x for 2 identical antennas

# Try manual MRC for first symbol
print("\n" + "="*80)
print("MANUAL MRC FOR VERIFICATION")
print("="*80)
y1 = symbols_rx_list[0][0]
y2 = symbols_rx_list[1][0]
h1 = h_estimates_per_antenna[0][0][0]  # First subcarrier of first symbol
h2 = h_estimates_per_antenna[1][0][0]

w1 = np.conj(h1) / (np.abs(h1)**2 + 1e-10)
w2 = np.conj(h2) / (np.abs(h2)**2 + 1e-10)

y_mrc = w1 * y1 + w2 * y2
print(f"y1={y1:.6f}, y2={y2:.6f}")
print(f"h1={h1:.6f}, h2={h2:.6f}")
print(f"w1={w1:.6f}, w2={w2:.6f}")
print(f"y_mrc (manual) = {y_mrc:.6f}")
print(f"y_combined[0] = {symbols_combined[0]:.6f}")
