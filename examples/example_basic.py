"""
Example 1: Basic OFDM Transmission

This example demonstrates:
- Creating an OFDM module
- Transmitting random bits
- Computing BER and PAPR
"""

import numpy as np
import sys
import os

# Add module path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from module import OFDMModule, LTEConfig


def main():
    print("\n" + "="*70)
    print("OFDM Module - Example 1: Basic Transmission")
    print("="*70)
    
    # Create OFDM module with default configuration (5MHz QPSK)
    print("\n[1] Creating OFDM module...")
    module = OFDMModule()
    print(f"    Config: {module}")
    
    # Create random bits
    print("\n[2] Creating test signal...")
    num_bits = 10000
    bits = np.random.randint(0, 2, num_bits)
    print(f"    Generated {num_bits} random bits")
    
    # Transmit at various SNR levels
    print("\n[3] Transmitting at different SNR levels...")
    snr_values = [5, 10, 15, 20]
    
    for snr in snr_values:
        results = module.transmit(bits.copy(), snr_db=snr)
        
        print(f"\n    SNR = {snr} dB:")
        print(f"      BER: {results['ber']:.6e}")
        print(f"      Errors: {results['errors']}")
        print(f"      PAPR: {results['papr_db']:.2f} dB")
    
    print("\n" + "="*70)
    print("✓ Example 1 Complete")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
