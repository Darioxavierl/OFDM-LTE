"""
Compare symbols from _demodulate_ofdm_stream vs receive_and_decode
"""
import numpy as np
from config import LTEConfig
from core.lte_receiver import LTEReceiver
from core.ofdm_core import OFDMTransmitter, OFDMChannel

# Config
config = LTEConfig(bandwidth=10.0, delta_f=15.0, modulation='64-QAM')
tx = OFDMTransmitter(config, mode='lte')

# TX signal
bits = np.random.randint(0, 2, size=10000)
signal_tx, _, _ = tx.modulate(bits)

# Channel
channel = OFDMChannel('awgn', snr_db=20.0)
signal_rx = channel.transmit(signal_tx)

# Method 1: receive_and_decode with enable_equalization=False
print("="*80)
print("Method 1: receive_and_decode(enable_equalization=False)")
print("="*80)
lte_no_eq = LTEReceiver(config, cell_id=0, enable_equalization=False)
result_no_eq = lte_no_eq.receive_and_decode(signal_rx)
symbols_data_no_eq = result_no_eq['symbols_data_only']
print(f"symbols_data_only shape: {symbols_data_no_eq.shape}")
print(f"Sample symbols: {symbols_data_no_eq[:5]}")

# Method 2: Manual _demodulate_ofdm_stream + data extraction
print("\n" + "="*80)
print("Method 2: _demodulate_ofdm_stream + manual data extraction")
print("="*80)
lte_manual = LTEReceiver(config, cell_id=0, enable_equalization=False)
all_symbols = lte_manual._demodulate_ofdm_stream(signal_rx)
print(f"Number of OFDM symbols: {len(all_symbols)}")
print(f"Shape of first OFDM symbol: {all_symbols[0].shape}")

# Concatenate all
received_symbols = np.concatenate(all_symbols)
print(f"Total symbols after concatenation: {len(received_symbols)}")

# Extract data
from core.resource_mapper import LTEResourceGrid
data_indices = LTEResourceGrid(config.N, config.Nc).get_data_indices()
print(f"Number of data indices per OFDM symbol: {len(data_indices)}")

# Build all data indices
num_ofdm_symbols = len(all_symbols)
all_data_indices = []
for sym_idx in range(num_ofdm_symbols):
    offset = sym_idx * config.N
    all_data_indices.extend(data_indices + offset)

all_data_indices = np.array(all_data_indices)
valid_indices = all_data_indices[all_data_indices < len(received_symbols)]
symbols_data_manual = received_symbols[valid_indices]
print(f"symbols_data_manual shape: {symbols_data_manual.shape}")
print(f"Sample symbols: {symbols_data_manual[:5]}")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"Shapes match? {symbols_data_no_eq.shape == symbols_data_manual.shape}")
print(f"Values match? {np.allclose(symbols_data_no_eq, symbols_data_manual)}")

if not np.allclose(symbols_data_no_eq, symbols_data_manual):
    diff = np.abs(symbols_data_no_eq - symbols_data_manual)
    print(f"Max absolute difference: {np.max(diff)}")
    print(f"Mean absolute difference: {np.mean(diff)}")
