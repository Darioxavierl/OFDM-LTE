"""
Example 2: BER vs SNR Sweep

This example demonstrates:
- Running a BER sweep across SNR range
- Comparing OFDM vs SC-FDM
- Analyzing PAPR improvement
"""

import numpy as np
import sys
import os

# Add module path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from module import OFDMModule, LTEConfig


def run_sweep_comparison():
    print("\n" + "="*70)
    print("OFDM Module - Example 2: BER Sweep & OFDM vs SC-FDM")
    print("="*70)
    
    # Configuration
    snr_range = np.arange(0, 21, 5)  # 0 to 20 dB, step 5
    num_bits = 50000
    
    # Test configurations
    configs = [
        ('OFDM (QPSK)', {'enable_sc_fdm': False}),
        ('SC-FDM (QPSK)', {'enable_sc_fdm': True}),
    ]
    
    results_all = {}
    
    for config_name, kwargs in configs:
        print(f"\n[Testing] {config_name}...")
        
        # Create module
        module = OFDMModule(**kwargs)
        
        # Progress callback
        def progress(p, m):
            if p % 33 == 0:
                print(f"  [{p}%] {m}")
        
        # Run sweep
        results = module.run_ber_sweep(
            num_bits=num_bits,
            snr_range=snr_range,
            progress_callback=progress
        )
        
        results_all[config_name] = results
        
        # Display results
        print(f"\n  Results for {config_name}:")
        print(f"  {'SNR (dB)':<12} {'BER':<15}")
        print(f"  {'-'*27}")
        for snr, ber in zip(results['snr_db'], results['ber_mean']):
            print(f"  {snr:<12.1f} {ber:<15.6e}")
    
    # Comparison
    print("\n" + "="*70)
    print("OFDM vs SC-FDM Comparison (BER at each SNR)")
    print("="*70)
    print(f"{'SNR (dB)':<12} {'OFDM BER':<15} {'SC-FDM BER':<15} {'Improvement':<15}")
    print("-" * 60)
    
    ofdm_results = results_all['OFDM (QPSK)']
    scfdm_results = results_all['SC-FDM (QPSK)']
    
    for snr, ofdm_ber, scfdm_ber in zip(ofdm_results['snr_db'], 
                                         ofdm_results['ber_mean'],
                                         scfdm_results['ber_mean']):
        # Improvement: how many times better is SC-FDM
        improvement = ofdm_ber / scfdm_ber if scfdm_ber > 0 else float('inf')
        print(f"{snr:<12.1f} {ofdm_ber:<15.6e} {scfdm_ber:<15.6e} {improvement:<15.1f}x")
    
    print("\n" + "="*70)
    print("✓ Example 2 Complete")
    print("="*70 + "\n")


if __name__ == '__main__':
    run_sweep_comparison()
