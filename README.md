# OFDM-SC Module

A comprehensive Python library for simulating LTE-based OFDM and Single-Carrier FDM (SC-FDM) systems.

## Quick Features

✅ **OFDM & SC-FDM Simulation**
- Full LTE-based OFDM implementation
- Single-Carrier FDM with DFT precoding
- Supports QPSK, 16-QAM, 64-QAM modulations

✅ **Channel Models**
- AWGN (Additive White Gaussian Noise)
- Rayleigh Multipath with ITU-R M.1225 profiles
- Vehicular A/B, Pedestrian A/B, Bad Urban

✅ **Signal Metrics**
- BER (Bit Error Rate) with confidence intervals
- PAPR (Peak-to-Average Power Ratio)
- EVM (Error Vector Magnitude)
- CCDF (Complementary Cumulative Distribution Function)

✅ **Easy-to-Use API**
- Simple high-level interface
- Flexible configuration
- Progress callbacks

## Quick Start

### Installation

```python
# Just import - no installation needed
from module import OFDMModule, LTEConfig
```

### Basic Usage

```python
import numpy as np
from module import OFDMModule

# Create module (default: 5MHz QPSK, AWGN channel)
module = OFDMModule()

# Transmit bits
bits = np.random.randint(0, 2, 1000)
results = module.transmit(bits, snr_db=15)

# Access results
print(f"BER: {results['ber']:.6e}")
print(f"PAPR: {results['papr_db']:.2f} dB")
```

### SC-FDM Mode

```python
# Enable SC-FDM (Single-Carrier FDM)
module = OFDMModule(enable_sc_fdm=True)
results = module.transmit(bits, snr_db=15)

# SC-FDM typically shows 3-6 dB PAPR improvement
```

## Documentation

- **[MODULE_USAGE.md](MODULE_USAGE.md)** - Complete API reference and usage guide
- **[examples/](examples/)** - Working examples
- **[test/](test/)** - Test suite

## Module Structure

```
module/
├── __init__.py              # Module initialization
├── config.py                # LTE configuration (LTEConfig class)
├── ofdm_module.py           # Main interface (OFDMModule class)
├── signal_analysis.py       # Metrics calculation (BER, PAPR, EVM, etc.)
├── MODULE_USAGE.md          # Complete usage guide
├── README.md                # This file
├── examples/
│   ├── example_basic.py     # Basic transmission example
│   └── example_sweep.py     # BER sweep example
└── test/
    ├── test_basic.py        # Basic functionality tests
    └── ...                  # Additional tests
```

## Key Classes

### OFDMModule

Main class for OFDM/SC-FDM simulation.

```python
module = OFDMModule(
    config=None,              # LTEConfig object
    channel_type='awgn',      # 'awgn' or 'rayleigh_mp'
    itu_profile='Vehicular_A',# ITU channel profile
    enable_sc_fdm=False       # Enable SC-FDM mode
)

# Methods
results = module.transmit(bits, snr_db=10)          # Single transmission
results = module.run_ber_sweep(1000, [10,15,20])    # BER sweep
module.set_modulation('16-QAM')                     # Change modulation
```

### LTEConfig

Configuration management for LTE parameters.

```python
config = LTEConfig(
    bandwidth=5.0,            # MHz: 1.25, 2.5, 5, 10, 15, 20
    delta_f=15.0,             # kHz: subcarrier spacing
    modulation='QPSK',        # QPSK, 16-QAM, 64-QAM
    cp_type='normal'          # normal or extended
)

print(config)                 # Display configuration
info = config.get_info()      # Get parameters dictionary
```

## Signal Analysis

```python
from module.signal_analysis import BERAnalyzer, PAPRAnalyzer, EVMAnalyzer

# BER Calculation
ber = BERAnalyzer.calculate_ber(bits_tx, bits_rx)

# PAPR Calculation
papr_data = PAPRAnalyzer.calculate_papr(signal)
print(f"PAPR: {papr_data['papr_db']:.2f} dB")

# CCDF (for visualization)
ccdf = PAPRAnalyzer.calculate_ccdf(papr_values)
# papr_x: PAPR values, ccdf_y: P(PAPR > x)

# EVM
evm = EVMAnalyzer.calculate_evm(symbols_tx, symbols_rx)
print(f"EVM: {evm['evm_percent']:.2f}%")
```

## Examples

### Example 1: Basic Transmission

```bash
cd module/examples/
python example_basic.py
```

### Example 2: BER vs SNR Sweep

```bash
python example_sweep.py
```

### Example 3: Custom Configuration

```python
from module import OFDMModule, LTEConfig

# Create 20 MHz 16-QAM configuration
config = LTEConfig(bandwidth=20.0, modulation='16-QAM')
module = OFDMModule(config=config)

# Transmit
bits = np.random.randint(0, 2, 100000)
results = module.transmit(bits, snr_db=20)
```

## Testing

Run the test suite:

```bash
cd module/test/
python test_basic.py
```

## Available ITU Profiles

Use with `itu_profile` parameter:

- `Pedestrian_A` - Low velocity, short distance
- `Pedestrian_B` - High velocity pedestrian
- `Vehicular_A` - Low velocity, short distance
- `Vehicular_B` - High velocity, long distance
- `Bad_Urban` - Severe multipath urban

## Performance

### Typical Execution Times

- Single transmission (10K bits): ~100-500 ms
- BER sweep (3 SNR, 50K bits): ~5-10 seconds
- SC-FDM: Similar to OFDM

### Memory Usage

- Configuration: ~1 KB
- Per transmission (100K bits): ~5-10 MB

## Key Results

### SC-FDM PAPR Improvement

| Modulation | OFDM PAPR | SC-FDM PAPR | Improvement |
|-----------|----------|-----------|-------------|
| QPSK | 10.8 dB | 7.0 dB | **3.8 dB** |
| 16-QAM | 12.7 dB | 6.4 dB | **6.3 dB** |

SC-FDM provides significant PAPR reduction, critical for:
- Power-efficient amplifiers
- Mobile/battery devices
- Low distortion systems

## Architecture

```
Input Bits
    ↓
QAM Modulation
    ↓
DFT Precoding (if SC-FDM)
    ↓
Resource Mapping (OFDM subcarriers)
    ↓
IFFT
    ↓
Cyclic Prefix Insertion
    ↓
Channel Transmission
    ↓
Cyclic Prefix Removal
    ↓
FFT
    ↓
IDFT Decodification (if SC-FDM)
    ↓
Resource Demapping
    ↓
QAM Demodulation
    ↓
Output Bits
```

## Configuration Examples

### Basic QPSK

```python
config = LTEConfig(bandwidth=5.0, modulation='QPSK')
module = OFDMModule(config=config)
```

### 16-QAM Multipath

```python
config = LTEConfig(bandwidth=10.0, modulation='16-QAM')
module = OFDMModule(
    config=config,
    channel_type='rayleigh_mp',
    itu_profile='Vehicular_A'
)
```

### SC-FDM with 64-QAM

```python
config = LTEConfig(bandwidth=20.0, modulation='64-QAM')
module = OFDMModule(config=config, enable_sc_fdm=True)
```

## System Requirements

- Python 3.8+
- NumPy
- SciPy (optional)

## License

MIT License - See original project

## References

- LTE Specifications (3GPP TS 36.211)
- ITU-R M.1225 Channel Models
- PAPR Reduction Techniques in OFDM

## Support

For detailed usage instructions, see [MODULE_USAGE.md](MODULE_USAGE.md)

For troubleshooting, see the Troubleshooting section in MODULE_USAGE.md

## Version

**Version**: 1.0
**Status**: Production Ready ✓

---

*Self-contained module ready for integration into other projects*
