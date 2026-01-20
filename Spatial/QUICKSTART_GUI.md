# Spatial Multiplexing GUI - Quick Start Guide

## Running the GUI

```powershell
cd d:\Proyectos\OFDM-LTE\Spatial
python main.py
```

## Basic Usage

### 1. Single Transmission
1. **Load Image**: Click "Cargar Imagen" and select a test image
2. **Configure System**:
   - Modulation: QPSK, 16-QAM, or 64-QAM
   - Bandwidth: 5, 10, or 20 MHz
   - Antennas: Choose from 2×2, 4×2, 4×4, or 8×4
   - Detector: MMSE (simpler) or SIC (better performance)
   - SNR: 25 dB (default for good quality)
3. **Channel Settings**:
   - ITU Profile: Pedestrian_A (default for low mobility)
   - Velocity: 3 km/h (pedestrian) or higher
4. **Run**: Click "Simulación Única"
5. **View Results**:
   - Metrics panel shows BER, errors, rank used
   - Constellation tab shows I/Q diagram
   - Image comparison shows original vs received

### 2. SNR Sweep (Performance Analysis)
1. **Configure SNR Range**:
   - Start: 10 dB
   - End: 35 dB
   - Step: 2 dB
   - Iterations: 3 (for averaging)
2. **Optional**: Load image (uses image bits) or leave empty (uses random bits)
3. **Run**: Click "Barrido en SNR"
4. **View Plot**: Single graph with 8 curves
   - 4 antenna configs × 2 detectors
   - Solid lines = MMSE, Dashed = SIC
   - Colors: Blue (2×2), Green (4×2), Orange (4×4), Red (8×4)

### 3. Multiantenna Test (Comprehensive Analysis)
1. **Load Image**: Required for this test
2. **Configure SNR Range**: Same as sweep
3. **Run**: Click "Prueba Multiantena"
4. **View Plot**: 3 subplots (QPSK, 16-QAM, 64-QAM)
   - Each shows all 8 configurations
   - Compare performance across modulations
   - Takes longer (24 simulations total)

## Configuration Guide

### Antenna Configurations
| Config | Description | Max Rank | Best Use Case |
|--------|-------------|----------|---------------|
| 2×2 | Entry-level | 2 | Low complexity, moderate SNR |
| 4×2 | Mid-range | 2 | Better diversity, same max streams |
| 4×4 | Advanced | 4 | High capacity, needs high SNR |
| 8×4 | Massive MIMO | 4 | Maximum gain, very high SNR |

### Detector Comparison
| Detector | Complexity | Performance | Best For |
|----------|------------|-------------|----------|
| MMSE | Low | Good | Lower SNR, fast processing |
| SIC | High | Better | High SNR, better BER |

### SNR Guidelines
| Modulation | Min SNR | Target SNR | Expected BER @ Target |
|------------|---------|------------|---------------------|
| QPSK | 10 dB | 15-20 dB | 10^-3 to 10^-4 |
| 16-QAM | 15 dB | 20-25 dB | 10^-3 to 10^-4 |
| 64-QAM | 20 dB | 25-30 dB | 10^-3 to 10^-4 |

### Channel Profiles
| Profile | Type | Delay Spread | Typical Use |
|---------|------|--------------|-------------|
| Pedestrian_A | Low | ~45 ns | Indoor, low mobility |
| Pedestrian_B | Moderate | ~200 ns | Outdoor, low mobility |
| Vehicular_A | Moderate | ~310 ns | Suburban, moderate speed |
| Vehicular_B | High | ~370 ns | Urban, moderate speed |
| Typical_Urban | High | ~710 ns | Dense urban |
| Rural_Area | Very high | ~1280 ns | Open areas, high speed |

### Velocity Settings
| Velocity | Scenario | Doppler @ 2 GHz | Impact |
|----------|----------|-----------------|--------|
| 3 km/h | Pedestrian | ~5.6 Hz | Minimal |
| 30 km/h | City traffic | ~56 Hz | Moderate |
| 120 km/h | Highway | ~222 Hz | Significant |
| 350 km/h | High-speed rail | ~648 Hz | Severe |

## Interpreting Results

### BER (Bit Error Rate)
- **< 10^-4**: Excellent (near perfect)
- **10^-4 to 10^-3**: Good (acceptable for most applications)
- **10^-3 to 10^-2**: Fair (noticeable degradation)
- **> 10^-2**: Poor (significant errors)

### Rank Used
- Shows number of spatial streams actually used
- Rank 1: Poor channel conditions (falls back to SISO-like)
- Rank 2-4: Good spatial multiplexing
- Higher rank ≠ always better (depends on SNR and channel)

### Constellation Diagram
- Tight clusters = low noise/interference
- Spread points = high noise/errors
- Pattern should match modulation (4 points for QPSK, 16 for 16-QAM, 64 for 64-QAM)

## Troubleshooting

### Error: "Configuración Inválida"
**Cause**: Invalid antenna combination
**Solution**: Use only 2×2, 4×2, 4×4, or 8×4

### High BER (> 0.1)
**Possible causes**:
1. SNR too low → Increase SNR
2. Modulation too high → Try QPSK or 16-QAM
3. Velocity too high → Reduce mobility or use lower modulation
4. Wrong detector → Try SIC instead of MMSE

### Simulation Very Slow
**Possible causes**:
1. Too many iterations → Reduce to 1-2 for testing
2. SNR step too small → Use 2-5 dB steps
3. High bandwidth + high modulation → Start with 5 MHz, QPSK

### GUI Not Responding
**Cause**: Long simulation running
**Solution**: Be patient, check console for progress messages

## Tips for Best Results

1. **Start Simple**: Begin with 2×2, QPSK, SNR=25 dB to verify setup
2. **Use Appropriate SNR**: Higher modulation needs higher SNR
3. **Channel Selection**: Match profile to your use case
4. **Detector Choice**: Use MMSE for speed, SIC for performance
5. **Image Selection**: Use moderate-sized images (< 500 KB) for faster tests
6. **Sweep Range**: Use 10-35 dB for comprehensive analysis
7. **Iterations**: 3-5 iterations for reliable averaging

## Common Experiments

### Experiment 1: Detector Comparison
- Config: 4×2
- SNR: 25 dB
- Compare: MMSE vs SIC
- Expected: SIC ~1-2 dB better

### Experiment 2: Antenna Scaling
- Detector: MMSE
- SNR: 25 dB
- Compare: 2×2, 4×2, 4×4, 8×4
- Expected: More antennas = better performance

### Experiment 3: Modulation Analysis
- Config: 4×4
- Detector: SIC
- Run: Multiantenna test
- Expected: Higher modulation needs higher SNR

### Experiment 4: Mobility Impact
- Config: 4×2
- SNR: 25 dB
- Velocities: 3, 30, 120 km/h
- Expected: Higher velocity degrades performance

## Advanced Features

### Custom SNR Range
Modify SNR parameters for specific research needs:
- Fine steps (0.5 dB) for detailed analysis
- Wide range (0-40 dB) for complete characterization

### Multiple Iterations
Increase iterations (5-10) for publication-quality results:
- Reduces variance in BER measurements
- Takes longer but more reliable

### Channel Parameter Tuning
Adjust frequency and velocity for specific scenarios:
- 700 MHz, 3 km/h: Indoor coverage
- 2.0 GHz, 30 km/h: Urban mobile
- 3.5 GHz, 120 km/h: Highway coverage

## Expected Performance

### Typical BER @ SNR=25dB
| Config | QPSK+MMSE | QPSK+SIC | 16-QAM+MMSE | 16-QAM+SIC | 64-QAM+MMSE | 64-QAM+SIC |
|--------|-----------|----------|-------------|------------|-------------|------------|
| 2×2 | 10^-5 | 10^-5 | 10^-3 | 10^-4 | 10^-2 | 10^-3 |
| 4×2 | 10^-5 | 10^-6 | 10^-4 | 10^-4 | 10^-2 | 10^-3 |
| 4×4 | 10^-6 | 10^-6 | 10^-4 | 10^-5 | 10^-3 | 10^-3 |
| 8×4 | 10^-6 | 10^-6 | 10^-5 | 10^-5 | 10^-3 | 10^-4 |

### Typical Rank Used @ SNR=25dB
| Config | QPSK | 16-QAM | 64-QAM |
|--------|------|--------|--------|
| 2×2 | 2 | 2 | 1-2 |
| 4×2 | 2 | 2 | 2 |
| 4×4 | 3-4 | 3-4 | 2-3 |
| 8×4 | 3-4 | 3-4 | 2-3 |

## Contact & Support

For issues or questions:
1. Check GUI_CHANGES.md for implementation details
2. Review console output for debug messages
3. Verify configuration matches valid TM4 requirements
