"""
Debug: Test with larger bit count like the image test
"""
import numpy as np
from config import LTEConfig
from core.ofdm_core import OFDMSimulator

# Use similar config to image test
config = LTEConfig(
    bandwidth=10.0,
    delta_f=15.0,
    modulation='64-QAM',
    cp_type='normal'
)

sim = OFDMSimulator(config, channel_type='awgn', mode='lte', enable_equalization=True)

# Create many bits (like image test has ~4.86M)
bits = np.random.randint(0, 2, size=100000)
snr_db = 20.0

print("=" * 80)
print("SISO with large bitcount")
print("=" * 80)
result_siso = sim.simulate_siso(bits, snr_db=snr_db)
print(f"SISO BER: {result_siso['ber']:.6f}")
print(f"Bits: {len(bits)}")
print(f"Errors: {result_siso['bit_errors']}")

print("\n" + "=" * 80)
print("SIMO with 2 RX, large bitcount")
print("=" * 80)
result_simo = sim.simulate_simo(bits, snr_db=snr_db, num_rx=2)
print(f"SIMO (2RX) BER: {result_simo['ber']:.6f}")
print(f"Bits: {len(bits)}")
print(f"Errors: {result_simo['bit_errors']}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"SISO BER:    {result_siso['ber']:.6f}")
print(f"SIMO BER:    {result_simo['ber']:.6f}")
print(f"Expected:    SIMO BER should be <= SISO BER")
if result_simo['ber'] <= result_siso['ber']:
    print(f"✓ PASS: SIMO performs better or equal")
else:
    print(f"✗ FAIL: SIMO performs worse than SISO")
