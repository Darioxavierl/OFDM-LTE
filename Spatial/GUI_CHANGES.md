# Spatial Multiplexing GUI - Changes Summary

## Overview
Adapted the Beamforming GUI (TM6) to support Spatial Multiplexing (TM4) with the following key modifications.

## Changes Made

### 1. UI Labels and Branding
- **Window Title**: Changed from "OFDM-LTE Beamforming (TM6)" → "OFDM-LTE Spatial Multiplexing (TM4)"
- **Info Label**: Updated from "Beamforming TM6 con precoding adaptativo" → "Spatial Multiplexing TM4 con rank adaptativo"
- **Status Messages**: All references updated to "Spatial Multiplexing"

### 2. Antenna Configuration
- **Valid Combinations**: Restricted to TM4-valid configs: 2×2, 4×2, 4×4, 8×4
- **num_rx_combo**: Changed from ['1', '2', '4'] → ['2', '4']
- **Default**: Changed from 2×1 → 2×2
- **Validation**: Added runtime validation in `update_config()` to show error if invalid combination selected

### 3. MIMO Detector Selector
- **New Control**: Added `detector_combo` with options: MMSE, SIC
- **Position**: Row 7 (between antenna config and info label)
- **Default**: MMSE
- **Integration**: Connected to `update_config()` and passed to simulation worker

### 4. SNR Ranges
- **Single Simulation**: Changed default from 15 dB → 25 dB
- **Sweep Start**: Changed from 0 dB → 10 dB
- **Sweep End**: Changed from 20 dB → 35 dB
- **Reason**: Spatial Multiplexing requires higher SNR than Beamforming for good performance

### 5. SimulationWorker: Single Simulation
**Before:**
```python
result = self.ofdm_system.simulate_beamforming(
    bits=bits,
    snr_db=self.params['snr_db'],
    num_tx=self.params['num_tx'],
    num_rx=self.params['num_rx'],
    codebook_type='TM6',
    velocity_kmh=self.params.get('velocity_kmh', 3),
    update_mode='adaptive'
)
```

**After:**
```python
result = self.ofdm_system.simulate_spatial_multiplexing(
    bits=bits,
    snr_db=self.params['snr_db'],
    num_tx=self.params['num_tx'],
    num_rx=self.params['num_rx'],
    rank='adaptive',
    detector_type=self.params.get('detector_type', 'MMSE'),
    velocity_kmh=self.params.get('velocity_kmh', 3)
)
```

### 6. SimulationWorker: Sweep Simulation
**Before (7 configs):**
- 2×1 SFBC
- 2×1 BF, 4×1 BF, 8×1 BF
- 2×2 BF, 4×2 BF, 8×4 BF

**After (8 configs):**
- 2×2 MMSE, 2×2 SIC
- 4×2 MMSE, 4×2 SIC
- 4×4 MMSE, 4×4 SIC
- 8×4 MMSE, 8×4 SIC

All call `simulate_spatial_multiplexing()` with rank='adaptive'

### 7. SimulationWorker: Multiantenna Test
**Before:**
- Single SNR, multiple antenna configs
- 12 configurations: 3 rows (1 RX, 2 RX, 4 RX) × 4 columns
- Generated image mosaic with PSNR

**After:**
- SNR sweep, multiple modulations, multiple antenna configs
- 3 modulations (QPSK, 16-QAM, 64-QAM) × 4 antenna configs × 2 detectors = 24 tests
- Returns BER vs SNR data for plotting (no images)
- Uses SNR range from sweep parameters

### 8. Visualization Updates

#### a) Single Simulation Metrics
- **Removed**: `beamforming_gain_db`
- **Added**: `rank_used`

#### b) Constellation Plot
- Title changed to include detector type and "TM4"
- Legend shows detector type (MMSE/SIC)

#### c) Sweep BER Plot
- Shows 8 lines (4 antenna configs × 2 detectors)
- Color coding by antenna config:
  - 2×2: Blue
  - 4×2: Green
  - 4×4: Orange
  - 8×4: Red
- Line style by detector:
  - MMSE: Solid line (—), circle markers
  - SIC: Dashed line (- -), square markers
- Title: "Spatial Multiplexing Performance (TM4 - 64-QAM, Rank Adaptive)"

#### d) Multiantenna Comparison Plot
- **Layout**: 3 subplots in 1 row (one per modulation)
- **Each subplot**: BER vs SNR with 8 lines (4 antenna × 2 detector)
- **Styling**: Same color/line conventions as sweep plot
- **Title**: "Spatial Multiplexing Performance by Modulation (TM4 - Rank Adaptive)"
- **Legend**: 2 columns, shows all 8 configs

### 9. Configuration Info Panel
**Before:**
- Showed "Beamforming (TM6)"
- Array Gain calculation

**After:**
- Shows "Spatial Multiplexing (TM4)"
- Detector type
- Rank: "Adaptativo (1-4 según canal)"
- Max Multiplexing Gain: min(num_tx, num_rx)

### 10. Channel Configuration
- **Preserved**: All channel controls remain identical
  - Type: Rayleigh Multipath only
  - ITU Profile selector
  - Frequency (GHz)
  - Velocity (km/h)
- **Reason**: Channel model is independent of transmission mode

## File Structure
```
Spatial/
├── gui/
│   ├── main_window.py  (modified for TM4)
│   ├── widgets.py      (unchanged)
│   └── __init__.py     (unchanged)
├── main.py             (entry point)
└── GUI_CHANGES.md      (this file)
```

## Testing Checklist

### Single Simulation
- [ ] Load image
- [ ] Set valid antenna config (2×2, 4×2, 4×4, 8×4)
- [ ] Set detector (MMSE or SIC)
- [ ] Set SNR (25 dB default)
- [ ] Run simulation
- [ ] Check BER, errors, rank_used in metrics panel
- [ ] Verify constellation plot shows detector in title
- [ ] Check received image quality

### SNR Sweep
- [ ] Set SNR range (10-35 dB default)
- [ ] Set iterations (3 default)
- [ ] Run sweep
- [ ] Verify plot shows 8 lines (4 configs × 2 detectors)
- [ ] Check color coding (blue=2×2, green=4×2, orange=4×4, red=8×4)
- [ ] Check line style (solid=MMSE, dashed=SIC)
- [ ] Verify legend and title

### Multiantenna Test
- [ ] Load image
- [ ] Set SNR range (10-35 dB)
- [ ] Set iterations (3 default)
- [ ] Run multiantenna test
- [ ] Verify 3 subplots appear (QPSK, 16-QAM, 64-QAM)
- [ ] Check each subplot has 8 lines
- [ ] Verify performance degrades with higher modulation
- [ ] Check legend shows all configs

### UI Validation
- [ ] Try invalid antenna combo (should show warning)
- [ ] Change detector (MMSE ↔ SIC)
- [ ] Modify channel parameters (ITU profile, velocity)
- [ ] Check info panel shows correct configuration
- [ ] Verify status bar messages

## Known Differences from Beamforming GUI

1. **No SFBC/Diversity Mode**: Spatial Multiplexing only (no fallback to diversity)
2. **Higher SNR Requirements**: Default 25 dB vs 15 dB for Beamforming
3. **No Single-RX Support**: Minimum 2 RX antennas required
4. **Rank Information**: Shows rank used instead of beamforming gain
5. **Detector Selection**: Explicit MMSE/SIC choice (vs implicit in beamforming)
6. **Multiantenna Test**: BER curves vs image mosaics

## Performance Notes

- **Best Detector**: SIC generally performs better than MMSE, especially at high SNR
- **Best Config**: 8×4 provides highest multiplexing gain (up to 4 streams)
- **SNR Threshold**: Need SNR > 20 dB for BER < 0.01 in most cases
- **Rank Adaptation**: System automatically selects rank 1-4 based on channel conditions
