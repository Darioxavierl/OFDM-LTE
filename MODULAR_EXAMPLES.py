"""
MODULAR ARCHITECTURE - USAGE EXAMPLES

This file contains practical examples showing:
1. Current SISO implementation (working)
2. How to transition to SIMO (Phase 2)
3. How to prepare for MIMO (Phase 3)

Copy and paste these examples to get started!
"""

# ============================================================================
# EXAMPLE 1: Basic SISO with OFDMSimulator
# ============================================================================

from core.ofdm_core import OFDMSimulator
from config import LTEConfig
import numpy as np

# Create LTE config
config = LTEConfig(
    bandwidth=5.0,           # 5 MHz
    modulation='QPSK',       # QPSK (2 bits/symbol)
    cp_type='normal'
)

# Create simulator (SISO, AWGN channel)
sim = OFDMSimulator(
    config=config,
    channel_type='awgn',
    mode='lte',
    enable_sc_fdm=False,
    enable_equalization=True
)

# Generate test bits
num_bits = 10000
bits = np.random.randint(0, 2, num_bits)

# Simulate SISO
result = sim.simulate_siso(bits, snr_db=15.0)

# Access results
print(f"SISO Results:")
print(f"  BER: {result['ber']:.2e}")
print(f"  Bit errors: {result['bit_errors']}")
print(f"  PAPR: {result['papr_db']:.2f} dB")


# ============================================================================
# EXAMPLE 2: SISO with Rayleigh Channel
# ============================================================================

# Create simulator with Rayleigh multipath channel
sim_rayleigh = OFDMSimulator(
    config=config,
    channel_type='rayleigh_mp',
    mode='lte'
)

# Simulate with Rayleigh fading
result_ray = sim_rayleigh.simulate_siso(bits, snr_db=15.0)
print(f"\nSISO Rayleigh Results:")
print(f"  BER: {result_ray['ber']:.2e}")  # Should be worse than AWGN


# ============================================================================
# EXAMPLE 3: BER Sweep (SNR range)
# ============================================================================

# Define SNR range to test
snr_range = np.arange(5, 20, 2)  # 5, 7, 9, 11, 13, 15, 17, 19 dB

# Perform BER sweep
sweep_results = sim.run_ber_sweep(
    num_bits=100000,
    snr_range=snr_range,
    num_trials=1,
    progress_callback=lambda pct, msg: print(f"Progress: {msg}")
)

# Plot BER curve (if matplotlib available)
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(sweep_results['snr_db'], sweep_results['ber_mean'], 'bo-')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('SISO BER vs SNR')
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('ber_curve_siso.png')
    print("BER curve saved to ber_curve_siso.png")
except ImportError:
    print("Matplotlib not available")


# ============================================================================
# EXAMPLE 4: Direct Component Access (Research Mode)
# ============================================================================

# Get individual components for manual control
tx = sim.tx           # OFDMTransmitter
rx = sim.rx           # OFDMReceiver
ch = sim.channels[0]  # First channel (OFDMChannel)

# Manual signal flow
signal_tx, symbols_tx, info = tx.modulate(bits)
print(f"\nDirect Components:")
print(f"  TX signal shape: {signal_tx.shape}")
print(f"  Symbols shape: {symbols_tx.shape}")

# Transmit through channel
ch.set_snr(15.0)
signal_rx = ch.transmit(signal_tx)

# Receive
symbols_rx, bits_rx = rx.demodulate(signal_rx)
ber = rx.calculate_ber(bits, bits_rx)
print(f"  BER: {ber:.2e}")


# ============================================================================
# EXAMPLE 5: SIMO Simulation (Prepared for Phase 2)
# ============================================================================

# SIMO mode: 1 transmitter, 2 receivers, 1 combining
result_simo = sim.simulate_simo(
    bits=bits,
    snr_db=15.0,
    num_rx=2,              # Number of receive antennas
    combining='mrc'        # Maximum Ratio Combining
)

print(f"\nSIMO Results (prepared):")
print(f"  BER: {result_simo['ber']:.2e}")
print(f"  Num RX: {result_simo['num_rx']}")
print(f"  Combining: {result_simo['combining_method']}")

# NOTE: Current implementation is placeholder
# In Phase 2, we will:
# 1. Implement true multi-path fading per antenna
# 2. Add proper MRC with channel estimation
# 3. Show diversity gain (lower BER than SISO)


# ============================================================================
# EXAMPLE 6: MIMO Preparation (Phase 3 - Not Yet Implemented)
# ============================================================================

# This will NOT work yet:
# try:
#     result_mimo = sim.simulate_mimo(
#         bits=bits,
#         snr_db=15.0,
#         num_tx=2,
#         num_rx=2
#     )
# except NotImplementedError as e:
#     print(f"MIMO: {e}")

# In Phase 3, MIMO will support:
# - 2x2 Alamouti space-time coding
# - Spatial multiplexing (V-BLAST)
# - Advanced techniques (SVD precoding)


# ============================================================================
# EXAMPLE 7: Creating Multiple Channels (for SIMO/MIMO)
# ============================================================================

# Create simulator with multiple channels
sim_multi = OFDMSimulator(
    config=config,
    channel_type='rayleigh_mp',
    num_channels=4  # 4 independent channel instances
)

# Access individual channels
for i, ch in enumerate(sim_multi.channels):
    print(f"Channel {i}: {ch}")

# In Phase 2:
# - Each channel will have independent fading
# - SIMO combining will use all 4 paths
# 
# In Phase 3:
# - MIMO will arrange these as N×M matrix


# ============================================================================
# EXAMPLE 8: Backward Compatibility with Original OFDMModule
# ============================================================================

from ofdm_module import OFDMModule

# Old code still works!
module = OFDMModule(config=config, channel_type='awgn')
result = module.transmit(bits, snr_db=15.0)

print(f"\nBackward Compatibility:")
print(f"  BER: {result['ber']:.2e}")

# New: Access components through wrapper
print(f"  TX: {module.tx}")
print(f"  RX: {module.rx}")
print(f"  Channel: {module.channel}")


# ============================================================================
# EXAMPLE 9: Different Modulation Schemes
# ============================================================================

modulations = ['QPSK', '16-QAM', '64-QAM']
ber_results = {}

for mod in modulations:
    config_mod = LTEConfig(
        bandwidth=5.0,
        modulation=mod,
        cp_type='normal'
    )
    
    sim_mod = OFDMSimulator(config=config_mod, channel_type='awgn')
    result = sim_mod.simulate_siso(bits, snr_db=15.0)
    ber_results[mod] = result['ber']
    
    print(f"{mod}: BER = {result['ber']:.2e}")

# Result: Higher modulation = higher BER (less robust)


# ============================================================================
# EXAMPLE 10: Comparing Channels
# ============================================================================

snr = 15.0

# AWGN channel
sim_awgn = OFDMSimulator(config, channel_type='awgn')
result_awgn = sim_awgn.simulate_siso(bits, snr_db=snr)

# Rayleigh channel - different profiles
profiles = ['Pedestrian_A', 'Vehicular_A', 'Bad_Urban']
results_ray = {}

for profile in profiles:
    # Note: Current version doesn't parameterize profile in OFDMSimulator
    # In Phase 2, we'll add this
    sim_ray = OFDMSimulator(config, channel_type='rayleigh_mp')
    result = sim_ray.simulate_siso(bits, snr_db=snr)
    results_ray[profile] = result['ber']

print(f"\nChannel Comparison @ {snr} dB:")
print(f"  AWGN: {result_awgn['ber']:.2e}")
for profile, ber in results_ray.items():
    print(f"  Rayleigh {profile}: {ber:.2e}")


# ============================================================================
# ROADMAP FOR PHASE 2 (SIMO Implementation)
# ============================================================================

"""
What needs to be done to activate SIMO:

1. OFDMChannel.transmit_simo() - Create independent faded signals
   Currently: Returns same signal N times
   Needed: Each signal has independent fading per path
   
2. OFDMReceiver with SIMO support - Handle multiple signals
   Currently: Single input
   Needed: Demodulate combined signal
   
3. _combine_mrc() - Proper Maximum Ratio Combining
   Currently: Returns first signal
   Needed: Weight by channel estimates, sum
   
4. Channel estimation - SIMO paths
   Currently: Placeholder
   Needed: Use pilot subcarriers to estimate H per path
   
5. Testing and validation
   Currently: SISO only
   Needed: Verify diversity gain in SIMO
   
6. Documentation
   Currently: Architecture doc ready
   Needed: SIMO usage examples and theory
"""


# ============================================================================
# ROADMAP FOR PHASE 3 (MIMO Implementation)
# ============================================================================

"""
What needs to be done for MIMO:

1. Extend OFDMTransmitter to OFDMTransmitterArray
   Handle: Multiple TX antennas
   Space-time coding (Alamouti for 2x2)
   
2. Channel matrix modeling
   H matrix: num_rx x num_tx
   Each element: channel from TX_i to RX_j
   
3. MIMO demodulation/detection
   Options:
   - Linear: Zero-forcing, MMSE
   - Non-linear: V-BLAST, ML detection
   
4. Spatial multiplexing
   Transmit different data on each TX antenna
   Receiver separates them
   
5. Advanced techniques
   - SVD-based precoding
   - Water-filling power allocation
   - Interference management
   
6. Benchmarking
   Compare SISO vs SIMO vs MIMO capacity
   Spectral efficiency improvements
"""


# ============================================================================
# KEY INSIGHTS FOR EXTENSION
# ============================================================================

"""
1. Separation of Concerns
   - TX handles modulation only
   - RX handles demodulation only
   - Channel handles fading only
   - Simulator coordinates them
   
   This makes adding SIMO/MIMO straightforward:
   - Just compose multiple channels
   - Add combining/detection logic
   
2. Composition Over Inheritance
   - OFDMSimulator HAS TX, RX, Channels
   - Not: OFDMSimulator INHERITS FROM TX
   
   Benefits:
   - Easy to add components
   - Easy to replace components
   - Flexible composition
   
3. Single Responsibility
   - Each class does ONE thing well
   - Easy to test independently
   - Easy to improve independently
   
4. Backward Compatibility
   - OFDMModule still works
   - Transition gradually
   - No breaking changes
   
5. Preparation for Growth
   - SIMO methods exist (not fully implemented)
   - Multiple channels supported
   - Combining framework ready
"""

