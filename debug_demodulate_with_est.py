"""
Debug: Check what _demodulate_with_channel_est returns
"""
import numpy as np
from config import LTEConfig
from core.ofdm_core import OFDMSimulator, OFDMTransmitter, OFDMChannel

# Config
config = LTEConfig(bandwidth=10.0, delta_f=15.0, modulation='64-QAM')

# TX
tx = OFDMTransmitter(config, mode='lte')
sim = OFDMSimulator(config, channel_type='awgn', mode='lte')

# Test with different bit counts
for num_bits in [10000, 100000]:
    bits = np.random.randint(0, 2, size=num_bits)
    signal_tx, _, _ = tx.modulate(bits)
    
    # Channel
    channel = OFDMChannel('awgn', snr_db=20.0)
    signal_rx = channel.transmit(signal_tx)
    
    # Use _demodulate_with_channel_est
    symbols_data, h_estimates_per_symbol = sim._demodulate_with_channel_est(signal_rx)
    
    print(f"\nBits: {num_bits}")
    print(f"  signal_rx shape: {signal_rx.shape}")
    print(f"  symbols_data shape: {symbols_data.shape}")
    print(f"  Number of OFDM symbols: {len(h_estimates_per_symbol)}")
    if len(h_estimates_per_symbol) > 0:
        print(f"  h_estimates_per_symbol[0] shape: {h_estimates_per_symbol[0].shape}")
    
    # Check if they're aligned properly
    from core.resource_mapper import LTEResourceGrid
    grid = LTEResourceGrid(config.N, config.Nc)
    data_indices_len = len(grid.get_data_indices())
    expected_symbols = len(h_estimates_per_symbol) * data_indices_len
    print(f"  Expected total data symbols: ~{expected_symbols}")
    print(f"  Actual data symbols: {len(symbols_data)}")
