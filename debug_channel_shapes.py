"""
Debug: Check channel estimate shapes with multiple OFDM symbols
"""
import numpy as np
from config import LTEConfig
from core.lte_receiver import LTEReceiver
from core.ofdm_core import OFDMTransmitter, OFDMChannel

# Config
config = LTEConfig(bandwidth=10.0, delta_f=15.0, modulation='64-QAM')

# TX
tx = OFDMTransmitter(config, mode='lte')

# Test with different bit counts
for num_bits in [1000, 10000, 100000]:
    bits = np.random.randint(0, 2, size=num_bits)
    signal_tx, _, _ = tx.modulate(bits)
    
    # Channel
    channel = OFDMChannel('awgn', snr_db=20.0)
    signal_rx = channel.transmit(signal_tx)
    
    # Receive without equalization
    lte_receiver = LTEReceiver(config, cell_id=0, enable_equalization=False)
    result = lte_receiver.receive_and_decode(signal_rx)
    
    print(f"\nBits: {num_bits}")
    print(f"  signal_rx shape: {signal_rx.shape}")
    print(f"  symbols_data_only shape: {result['symbols_data_only'].shape}")
    print(f"  channel_estimate shape: {result['channel_estimate'].shape}")
    
    # Check if channel estimate shape matches symbols shape
    if result['channel_estimate'].shape != result['symbols_data_only'].shape:
        print(f"  ⚠ SHAPE MISMATCH: channel_estimate has different shape than symbols_data_only")
