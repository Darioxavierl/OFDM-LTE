"""
Debug script to understand the SIMO shapes and data flow
"""
import numpy as np
from config import LTEConfig
from core.ofdm_core import OFDMSimulator

# Create minimal test
config = LTEConfig(bandwidth=5.0, modulation='QPSK')
sim = OFDMSimulator(config, channel_type='awgn', mode='lte')

# Create test bits
bits = np.random.randint(0, 2, size=100)

# TX
signal_tx, symbols_tx, _ = sim.tx.modulate(bits)
print(f"[TX] signal_tx shape: {signal_tx.shape}")
if isinstance(symbols_tx, list):
    print(f"[TX] symbols_tx is list of length: {len(symbols_tx)}")
    print(f"[TX] symbols_tx[0] shape: {symbols_tx[0].shape if hasattr(symbols_tx[0], 'shape') else len(symbols_tx[0])}")
else:
    print(f"[TX] symbols_tx shape: {symbols_tx.shape}")

# Channel
sim.channels[0].set_snr(20.0)
signals_rx, _ = sim.channels[0].transmit_simo(signal_tx, num_rx=2)
print(f"\n[Channel] signals_rx length: {len(signals_rx)}")
print(f"[Channel] signals_rx[0] shape: {signals_rx[0].shape}")
print(f"[Channel] signals_rx[1] shape: {signals_rx[1].shape}")

# Demodulate per antenna
print(f"\n[RX] Demodulating per antenna...")
for i, sig in enumerate(signals_rx):
    print(f"\n  Antenna {i}:")
    print(f"    Input signal shape: {sig.shape}")
    
    # Get LTE receiver
    lte_receiver = sim.rx.demodulator.lte_receiver
    result = lte_receiver.receive_and_decode(sig)
    
    print(f"    symbols_data_only shape: {result['symbols_data_only'].shape}")
    print(f"    channel_estimate shape: {result['channel_estimate'].shape}")
    print(f"    bits shape: {result.get('bits', np.array([])).shape}")

# Test MRC combining
print(f"\n[MRC] Testing combining...")
symbols_rx_list = []
h_estimates = []
for sig in signals_rx:
    lte_receiver = sim.rx.demodulator.lte_receiver
    result = lte_receiver.receive_and_decode(sig)
    symbols_rx_list.append(result['symbols_data_only'])
    h_estimates.append(result['channel_estimate'])

symbols_combined = sim._combine_symbols_mrc(symbols_rx_list, h_estimates)
print(f"symbols_combined shape: {symbols_combined.shape}")

# Demodulate combined
bits_rx = sim.rx.demodulator.qam_demodulator.symbols_to_bits(symbols_combined)
print(f"bits_rx shape: {bits_rx.shape}")
print(f"Expected bits: {len(bits)}")
