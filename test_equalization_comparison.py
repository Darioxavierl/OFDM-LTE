"""
Compare: receive_and_decode WITH equalization vs _demodulate_with_channel_est
"""
import numpy as np
from config import LTEConfig
from core.lte_receiver import LTEReceiver
from core.ofdm_core import OFDMSimulator, OFDMTransmitter, OFDMChannel

# Config
config = LTEConfig(bandwidth=10.0, delta_f=15.0, modulation='64-QAM')
sim = OFDMSimulator(config, channel_type='awgn', mode='lte')
tx = OFDMTransmitter(config, mode='lte')

# TX
bits = np.random.randint(0, 2, size=100000)
signal_tx, _, _ = tx.modulate(bits)

# Channel
channel = OFDMChannel('awgn', snr_db=20.0)
signal_rx = channel.transmit(signal_tx)

# Method 1: Full SISO via sim (uses LTEReceiver with equalization)
print("="*80)
print("Method 1: SISO via sim.simulate_siso")
print("="*80)
result_siso = sim.simulate_siso(bits, snr_db=20.0)
ber_siso = result_siso['ber']
print(f"BER: {ber_siso:.6f}")

# Method 2: receive_and_decode with enable_equalization=True directly
print("\n" + "="*80)
print("Method 2: receive_and_decode(enable_equalization=True)")
print("="*80)
lte_with_eq = LTEReceiver(config, cell_id=0, enable_equalization=True)
result_with_eq = lte_with_eq.receive_and_decode(signal_rx)
bits_from_lte = result_with_eq['bits']
bits_test = bits[:min(len(bits), len(bits_from_lte))]
bits_lte_test = bits_from_lte[:len(bits_test)]
ber_lte = np.sum(bits_test != bits_lte_test) / len(bits_test)
print(f"BER: {ber_lte:.6f}")

# Method 3: _demodulate_with_channel_est (uses enable_equalization=False)
print("\n" + "="*80)
print("Method 3: _demodulate_with_channel_est(enable_equalization=False)")
print("="*80)
symbols_data, h_est_list = sim._demodulate_with_channel_est(signal_rx)
bits_from_raw = sim.rx.demodulator.qam_demodulator.symbols_to_bits(symbols_data)
bits_test = bits[:min(len(bits), len(bits_from_raw))]
bits_raw_test = bits_from_raw[:len(bits_test)]
ber_raw = np.sum(bits_test != bits_raw_test) / len(bits_test)
print(f"BER: {ber_raw:.6f}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"SISO (with eq):    {ber_siso:.6f}")
print(f"LTE (with eq):     {ber_lte:.6f}")
print(f"Raw (no eq):       {ber_raw:.6f}")
print(f"\nMethods 1 and 2 should be similar (both use equalization)")
print(f"Method 3 should be worse (no equalization)")
