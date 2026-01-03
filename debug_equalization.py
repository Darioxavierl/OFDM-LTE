"""
Debug: Check what receive_and_decode returns with enable_equalization=False
"""
import numpy as np
from config import LTEConfig
from core.lte_receiver import LTEReceiver
from core.ofdm_core import OFDMTransmitter, OFDMChannel

# Create config
config = LTEConfig(bandwidth=5.0, modulation='QPSK')

# Create TX
tx = OFDMTransmitter(config, mode='lte')
bits = np.random.randint(0, 2, size=1000)
signal_tx, _, _ = tx.modulate(bits)

# Send through channel
channel = OFDMChannel('awgn', snr_db=20.0)
signal_rx = channel.transmit(signal_tx)

# Receive WITH equalization
print("=" * 80)
print("WITH EQUALIZATION (enable_equalization=True)")
print("=" * 80)
lte_with_eq = LTEReceiver(config, cell_id=0, enable_equalization=True)
result_with_eq = lte_with_eq.receive_and_decode(signal_rx)

print(f"symbols_received shape: {result_with_eq['symbols_received'].shape}")
print(f"symbols_equalized shape: {result_with_eq['symbols_equalized'].shape}")
print(f"symbols_data_only shape: {result_with_eq['symbols_data_only'].shape}")
print(f"channel_estimate shape: {result_with_eq['channel_estimate'].shape}")
print(f"bits shape: {result_with_eq['bits'].shape}")

# Compute BER
# QAM demodulator generates more bits than symbols
# We need to use the number of output bits, not input bits
ber_with_eq = np.sum(bits[:len(result_with_eq['bits'])] != result_with_eq['bits'][:len(bits)]) / len(bits)
print(f"BER: {ber_with_eq:.6f}")

# Receive WITHOUT equalization
print("\n" + "=" * 80)
print("WITHOUT EQUALIZATION (enable_equalization=False)")
print("=" * 80)
lte_no_eq = LTEReceiver(config, cell_id=0, enable_equalization=False)
result_no_eq = lte_no_eq.receive_and_decode(signal_rx)

print(f"symbols_received shape: {result_no_eq['symbols_received'].shape}")
print(f"symbols_equalized shape: {result_no_eq['symbols_equalized'].shape}")
print(f"symbols_data_only shape: {result_no_eq['symbols_data_only'].shape}")
print(f"channel_estimate shape: {result_no_eq['channel_estimate'].shape}")
print(f"bits shape: {result_no_eq['bits'].shape}")

# Compute BER
ber_no_eq = np.sum(bits[:len(result_no_eq['bits'])] != result_no_eq['bits'][:len(bits)]) / len(bits)
print(f"BER: {ber_no_eq:.6f}")

# Compare symbols
print("\n" + "=" * 80)
print("SYMBOL COMPARISON")
print("=" * 80)
print(f"Same symbols_data_only? {np.allclose(result_with_eq['symbols_data_only'], result_no_eq['symbols_data_only'])}")
print(f"Same channel_estimate? {np.allclose(result_with_eq['channel_estimate'], result_no_eq['channel_estimate'])}")

# Manual MRC on no_eq symbols
print("\n" + "=" * 80)
print("MANUAL MRC TEST")
print("=" * 80)
from core.resource_mapper import LTEResourceGrid

# Extract data indices
data_indices = LTEResourceGrid(config.N, config.Nc).get_data_indices()
h_est_full = result_no_eq['channel_estimate']
h_est_data = h_est_full[data_indices]

# Pad if needed
if len(h_est_data) < len(result_no_eq['symbols_data_only']):
    h_est_data = np.pad(h_est_data, (0, len(result_no_eq['symbols_data_only']) - len(h_est_data)), 'constant', constant_values=1.0)

Y = result_no_eq['symbols_data_only']
H = h_est_data[:len(Y)]

# MRC weight
w = np.conj(H) / (np.abs(H)**2 + 1e-10)
Y_eq = w * Y

# Demodulate
from core.modulator import QAMModulator
qam = QAMModulator(config.modulation)
bits_mrc = qam.symbols_to_bits(Y_eq)

ber_mrc = np.sum(bits[:len(bits_mrc)] != bits_mrc[:len(bits)]) / len(bits)
print(f"BER with manual MRC: {ber_mrc:.6f}")
print(f"BER no_eq receiver: {ber_no_eq:.6f}")
print(f"BER with_eq receiver: {ber_with_eq:.6f}")
