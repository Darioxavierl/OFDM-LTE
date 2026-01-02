# Modular OFDM Architecture - SISO → SIMO → MIMO

## Overview

The OFDM-LTE module has been refactored into a **modular, scalable architecture** that separates concerns and enables future expansion to SIMO and MIMO systems without breaking existing code.

**File**: `core/ofdm_core.py`  
**Main Classes**: 
- `OFDMTransmitter` - Handles transmission (modulation)
- `OFDMReceiver` - Handles reception (demodulation)
- `OFDMChannel` - Wraps channel simulation (AWGN, Rayleigh)
- `OFDMSimulator` - Orchestrates components (SISO/SIMO/MIMO)

---

## Architecture Overview

### Signal Flow by Antenna Configuration

#### SISO (Single-Input Single-Output) - Currently Implemented ✅

```
                        TX                    RX
                        
bits ──→ [Transmitter] ──→ signal_tx ──→ [Channel] ──→ signal_rx ──→ [Receiver] ──→ bits_rx
           (modulate)                      (AWGN/Ray)              (demodulate)
```

**Current State**: Fully operational
- 1 Transmitter, 1 Channel, 1 Receiver
- Uses `OFDMSimulator.simulate_siso(bits, snr_db)`

#### SIMO (Single-Input Multiple-Output) - Prepared ⏳

```
                        TX                      RX
                        
bits ──→ [Transmitter] ──→ signal_tx ──→ ┌─ [CH0] ──→ signal_rx0 ──┐
           (modulate)                    ├─ [CH1] ──→ signal_rx1 ──┤ [Combiner] ──→ [Receiver] ──→ bits_rx
                                        └─ [CH2] ──→ signal_rx2 ──┘   (MRC/EGC)   (demodulate)
```

**Status**: Architecture ready, methods prepared but not fully implemented
- 1 Transmitter, N Channels, 1 Receiver (with combining)
- Method: `OFDMSimulator.simulate_simo(bits, snr_db, num_rx=2, combining='mrc')`
- Current combiner: Placeholder (returns first signal)
- **Next Step**: Implement MRC with proper channel estimation

#### MIMO (Multiple-Input Multiple-Output) - Future 📋

```
                        TX                      RX
                        
bits0 ──→ [TX0]────┐
                   ├──→ [CHANNEL MATRIX H] ──→ [RX0] ──→ [Detector] ──→ bits_rx0
bits1 ──→ [TX1]────┘                           [RX1]                    bits_rx1

(Space-Time Coding, Alamouti, SVD precoding, etc.)
```

**Status**: Placeholder method `OFDMSimulator.simulate_mimo()`
- Not implemented yet
- Roadmap: 2×2 MIMO with Alamouti coding, then advanced techniques

---

## Class Structure

### 1. OFDMTransmitter

**Responsibility**: Modulate bits to OFDM signal

```python
from core.ofdm_core import OFDMTransmitter
from config import LTEConfig

config = LTEConfig(bandwidth=10.0, modulation='64-QAM')
tx = OFDMTransmitter(config, mode='lte', enable_sc_fdm=False)

# Modulate bits
signal_tx, symbols_tx, mapping_infos = tx.modulate(bits)

# Calculate PAPR
papr_info = tx.calculate_papr(signal_tx)
print(f"PAPR: {papr_info['papr_db']:.2f} dB")
```

**Key Methods**:
- `modulate(bits)` → `(signal_tx, symbols_tx, mapping_infos)`
- `calculate_papr(signal)` → PAPR statistics dict
- `get_config()` → Configuration object

**Attributes**:
- `config`: LTEConfig object
- `modulator`: OFDMModulator instance
- `mode`: 'lte' or 'simple'
- `enable_sc_fdm`: SC-FDM enable flag

---

### 2. OFDMReceiver

**Responsibility**: Demodulate OFDM signal to bits

```python
from core.ofdm_core import OFDMReceiver
from config import LTEConfig

config = LTEConfig(bandwidth=10.0, modulation='64-QAM')
rx = OFDMReceiver(config, mode='lte', enable_equalization=True)

# Demodulate received signal
symbols_rx, bits_rx = rx.demodulate(signal_rx)

# Calculate BER
ber = rx.calculate_ber(bits_tx, bits_rx)
print(f"BER: {ber:.2e}")
```

**Key Methods**:
- `demodulate(signal_rx)` → `(symbols_rx, bits_rx)`
- `estimate_channel()` → Channel estimation dict (placeholder for future)
- `calculate_ber(bits_tx, bits_rx)` → BER value
- `get_config()` → Configuration object

**Attributes**:
- `config`: LTEConfig object
- `demodulator`: OFDMDemodulator instance
- `mode`: 'lte' or 'simple'
- `enable_equalization`: Equalization enable flag

---

### 3. OFDMChannel

**Responsibility**: Simulate channel effects (AWGN, Rayleigh multipath)

```python
from core.ofdm_core import OFDMChannel

# SISO Channel
ch = OFDMChannel(channel_type='rayleigh_mp', snr_db=10.0, 
                 itu_profile='Pedestrian_A', frequency_ghz=2.0)

# Transmit SISO
signal_rx = ch.transmit(signal_tx)

# SIMO (prepared)
signals_rx = ch.transmit_simo(signal_tx, num_rx=2)
```

**Key Methods**:
- `set_snr(snr_db)` → Set SNR
- `transmit(signal_tx)` → SISO transmission
- `transmit_simo(signal_tx, num_rx)` → SIMO transmission (prepared)
- `get_config()` → Channel configuration dict

**Attributes**:
- `channel_type`: 'awgn' or 'rayleigh_mp'
- `snr_db`: Signal-to-Noise Ratio
- `profile`: ITU-R M.1225 profile (for Rayleigh)
- `channel`: Underlying ChannelSimulator instance

**ITU Profiles Available**:
- `Pedestrian_A` - Static pedestrian (short delay spread)
- `Pedestrian_B` - Pedestrian with doppler
- `Vehicular_A` - Vehicular (medium delay spread)
- `Vehicular_B` - Vehicular high speed
- `Bad_Urban` - Urban non-LOS (long delay spread)

---

### 4. OFDMSimulator

**Responsibility**: Orchestrate TX, RX, Channels for different antenna configurations

```python
from core.ofdm_core import OFDMSimulator
from config import LTEConfig

config = LTEConfig(bandwidth=10.0, modulation='64-QAM')
sim = OFDMSimulator(config, channel_type='rayleigh_mp')

# ===== SISO =====
result_siso = sim.simulate_siso(bits, snr_db=10.0)
print(f"SISO BER: {result_siso['ber']:.2e}")

# ===== SIMO (prepared) =====
result_simo = sim.simulate_simo(bits, snr_db=10.0, num_rx=2, combining='mrc')
print(f"SIMO BER: {result_simo['ber']:.2e}")
# Note: Currently placeholder, uses first path only

# ===== MIMO (not yet implemented) =====
# result_mimo = sim.simulate_mimo(bits, snr_db=10.0, num_tx=2, num_rx=2)
# raises NotImplementedError
```

**Key Methods**:

#### `simulate_siso(bits, snr_db)`
Transmit through single channel (current working configuration)

**Parameters**:
- `bits`: Input bit array
- `snr_db`: Signal-to-Noise Ratio in dB

**Returns**: Dict with complete results:
```python
{
    'transmitted_bits': int,
    'received_bits': int,
    'bits_received_array': np.ndarray,
    'bit_errors': int,
    'ber': float,
    'snr_db': float,
    'papr_db': float,
    'papr_linear': float,
    'signal_tx': np.ndarray,
    'signal_rx': np.ndarray,
    'symbols_tx': np.ndarray,
    'symbols_rx': np.ndarray
}
```

#### `simulate_simo(bits, snr_db, num_rx, combining)` [PREPARED]
Transmit through multiple receive paths

**Parameters**:
- `bits`: Input bit array
- `snr_db`: Signal-to-Noise Ratio in dB
- `num_rx`: Number of receive antennas (default 2)
- `combining`: Combining method ('mrc', future: 'egc', 'selection')

**Returns**: Same as SISO + additional fields:
```python
{
    'num_rx': int,
    'combining_method': str,
    'diversity_level': int,
    ...
}
```

**Status**: Placeholder implementation
- Currently returns same signal for all RX paths
- Future: Each path will have independent fading

#### `simulate_mimo(bits, snr_db, num_tx, num_rx)` [NOT IMPLEMENTED]
Multiple transmit and receive antennas

**Parameters**:
- `bits`: Input bit array
- `snr_db`: Signal-to-Noise Ratio in dB
- `num_tx`: Number of transmit antennas
- `num_rx`: Number of receive antennas

**Status**: Raises NotImplementedError
- Roadmap: 2×2 with Alamouti, then advanced techniques

#### `_combine_mrc(signals_rx)`
Maximum Ratio Combining (placeholder)

**Current**: Returns first signal  
**Future**: Proper MRC with channel weights

#### `run_ber_sweep(num_bits, snr_range, num_trials, progress_callback)`
Perform BER sweep across SNR values

**Returns**:
```python
{
    'snr_db': np.ndarray,
    'ber_mean': np.ndarray,
    'ber_values': np.ndarray,
    'papr_values': np.ndarray
}
```

---

## Backward Compatibility Layer

The original `OFDMModule` class is now a wrapper around `OFDMSimulator` for backward compatibility.

```python
from ofdm_module import OFDMModule

# Old code still works exactly the same
module = OFDMModule(config)
result = module.transmit(bits, snr_db=10)
print(f"BER: {result['ber']}")

# New: Access individual components
tx = module.tx           # OFDMTransmitter
rx = module.rx           # OFDMReceiver
ch = module.channel      # OFDMChannel (first of potentially many)
```

**Compatibility Properties**:
- `module.channel` → `module.simulator.channels[0]`
- `module.modulator` → `module.simulator.tx.modulator`
- `module.demodulator` → `module.simulator.rx.demodulator`
- `module.tx` → `module.simulator.tx`
- `module.rx` → `module.simulator.rx`

---

## Usage Examples

### Example 1: Basic SISO Simulation

```python
from core.ofdm_core import OFDMSimulator
from config import LTEConfig
import numpy as np

# Create simulator
config = LTEConfig(bandwidth=5.0, modulation='QPSK')
sim = OFDMSimulator(config, channel_type='awgn')

# Generate random bits
bits = np.random.randint(0, 2, 10000)

# Simulate SISO
result = sim.simulate_siso(bits, snr_db=15)
print(f"BER: {result['ber']:.2e}")
```

### Example 2: SIMO with Combining (Prepared)

```python
# Same simulator, SIMO mode
result_simo = sim.simulate_simo(bits, snr_db=15, num_rx=2, combining='mrc')

# Currently: placeholder implementation
# Future: Will show diversity gain
print(f"SIMO BER: {result_simo['ber']:.2e}")  # Expected: lower than SISO
```

### Example 3: Direct Component Usage

```python
from core.ofdm_core import OFDMTransmitter, OFDMReceiver, OFDMChannel
from config import LTEConfig

config = LTEConfig()

# Create independent components
tx = OFDMTransmitter(config)
rx = OFDMReceiver(config)
ch = OFDMChannel(channel_type='rayleigh_mp', snr_db=10.0)

# Manual signal flow
signal_tx, _, _ = tx.modulate(bits)
signal_rx = ch.transmit(signal_tx)
symbols_rx, bits_rx = rx.demodulate(signal_rx)

# Flexible composition for research/experimentation
```

### Example 4: Multiple Channels (for MIMO/SIMO)

```python
# Create simulator with multiple channels (for future SIMO/MIMO)
sim = OFDMSimulator(config, channel_type='rayleigh_mp', num_channels=2)

# Access individual channels
ch0 = sim.channels[0]
ch1 = sim.channels[1]

# Each can be configured independently (future feature)
ch0.set_snr(10.0)
ch1.set_snr(12.0)  # Different SNR per path
```

---

## Extension Roadmap

### Phase 1: SISO (✅ COMPLETE)
- [x] OFDMTransmitter class
- [x] OFDMReceiver class
- [x] OFDMChannel wrapper
- [x] OFDMSimulator orchestrator
- [x] Backward compatibility with OFDMModule
- [x] Test and validate existing functionality

**Current Status**: All tests passing, BER metrics consistent with original

---

### Phase 2: SIMO (⏳ IN PROGRESS)
- [ ] Implement true SIMO channel model (independent fading per path)
- [ ] Implement MRC (Maximum Ratio Combining) with channel estimation
- [ ] Add EGC (Equal Gain Combining) method
- [ ] Add Selection Combining method
- [ ] Test with multiple receive antennas
- [ ] Validate diversity gain metrics
- [ ] Add SIMO examples and documentation

**Estimated**: 2-3 weeks depending on channel estimation complexity

**Key Files to Modify**:
- `core/ofdm_core.py`: OFDMChannel.transmit_simo(), _combine_mrc()
- `core/channel.py`: Add SIMO-specific channel model
- Tests and examples for SIMO

---

### Phase 3: MIMO (📋 FUTURE)
- [ ] Support for multiple transmitters
- [ ] Extend OFDMTransmitter to OFDMTransmitterArray
- [ ] Implement 2×2 Alamouti space-time coding
- [ ] Channel matrix modeling and estimation
- [ ] Implement spatial multiplexing (V-BLAST decoder)
- [ ] Add advanced techniques (SVD precoding, etc.)
- [ ] MIMO simulations and tests

**Estimated**: 4-6 weeks depending on complexity

**Scope**: Start with 2×2, expand to NxM

---

## Design Principles

### 1. Separation of Concerns
Each class has a single responsibility:
- TX: Modulation only
- RX: Demodulation only
- Channel: Channel simulation only
- Simulator: Orchestration only

### 2. Composition Over Inheritance
Uses composition (OFDMSimulator has TX/RX/Channels) rather than deep inheritance hierarchies.

### 3. Flexibility
Components can be used independently or through orchestrator:
```python
# Direct usage for research
tx = OFDMTransmitter(config)
signal = tx.modulate(bits)

# Or through orchestrator for convenience
sim = OFDMSimulator(config)
result = sim.simulate_siso(bits, snr_db=10)
```

### 4. Backward Compatibility
Original API still works without modification, allowing gradual migration.

### 5. Preparation for Growth
- Methods prepared for SIMO/MIMO (not fully implemented)
- Multiple channel instances supported
- Combining methods framework ready
- Architecture extendable without breaking existing code

---

## File Structure

```
core/
├── ofdm_core.py              # New: Modular architecture
│   ├── OFDMTransmitter
│   ├── OFDMReceiver
│   ├── OFDMChannel
│   └── OFDMSimulator
├── ofdm_system.py            # Old: OFDM system (still available)
├── modulator.py              # Used by OFDMTransmitter
├── demodulator.py            # Used by OFDMReceiver
├── channel.py                # Used by OFDMChannel
└── ...

ofdm_module.py                # Compatibility wrapper (new)
```

---

## Testing and Validation

### Current Tests ✅
- `test/final_image_test.py`: Full end-to-end SISO image transmission
  - AWGN channel
  - Rayleigh multipath channel
  - BER calculation
  - Image reconstruction
  - Results saving to `results/` directory

### Validation Results
```
[OK] SISO AWGN @ 20dB:     BER = 1.51e-02, PAPR = 28.68 dB
[OK] SISO Rayleigh @ 20dB: BER = 4.71e-02, PAPR = 28.68 dB
```

Functionality is identical to original implementation.

### Future Tests (for SIMO/MIMO phases)
- SIMO BER sweep with increasing num_rx
- MIMO spatial multiplexing efficiency
- Diversity gain measurements
- Channel estimation accuracy tests

---

## Performance Notes

### SISO Performance
- Consistent with original OFDMModule
- No performance degradation from refactoring
- Same modulation/demodulation algorithms

### Future Optimization Opportunities
1. Vectorize combining operations (SIMO)
2. Parallel channel simulation (MIMO)
3. Precoding optimization
4. Pilot-based channel estimation efficiency

---

## API Reference Quick Start

```python
# === Setup ===
from core.ofdm_core import OFDMSimulator
from config import LTEConfig

config = LTEConfig(bandwidth=10.0, modulation='64-QAM')
sim = OFDMSimulator(config, channel_type='rayleigh_mp')

# === SISO (current) ===
result = sim.simulate_siso(bits, snr_db=10.0)
ber = result['ber']
papr = result['papr_db']

# === SIMO (prepared) ===
result = sim.simulate_simo(bits, snr_db=10.0, num_rx=2)
# Note: Currently placeholder, will be improved in Phase 2

# === BER Sweep ===
sweep = sim.run_ber_sweep(num_bits=100000, 
                          snr_range=np.arange(0, 15, 2))

# === Direct Components ===
tx = sim.tx           # OFDMTransmitter
rx = sim.rx           # OFDMReceiver
ch = sim.channels[0]  # OFDMChannel
```

---

## Glossary

| Term | Meaning |
|------|---------|
| **SISO** | Single-Input Single-Output (1 TX, 1 RX antenna) |
| **SIMO** | Single-Input Multiple-Output (1 TX, N RX antennas) |
| **MIMO** | Multiple-Input Multiple-Output (N TX, M RX antennas) |
| **MRC** | Maximum Ratio Combining (optimal combining for SIMO) |
| **EGC** | Equal Gain Combining (suboptimal but simpler) |
| **SNR** | Signal-to-Noise Ratio (in dB) |
| **PAPR** | Peak-to-Average Power Ratio (power efficiency metric) |
| **BER** | Bit Error Rate (quality metric) |
| **ITU** | International Telecommunication Union |
| **CP** | Cyclic Prefix (guard interval in OFDM) |

---

## Questions & Support

For questions about:
- **SISO implementation**: See `core/ofdm_core.py` OFDMSimulator class
- **SIMO roadmap**: See Phase 2 section above
- **MIMO roadmap**: See Phase 3 section above
- **Original OFDMModule**: Still works, now uses new architecture internally

---

**Last Updated**: 2026-01-01  
**Status**: SISO Phase Complete, Ready for SIMO Phase 2
