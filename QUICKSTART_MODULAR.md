# 🚀 QUICK START - Arquitectura Modular OFDM

Aprende las 4 clases en 5 minutos. 💡

---

## Las 4 Clases (Arquitectura Modular)

### 1️⃣ OFDMTransmitter - Modula bits a señal

```python
from core.ofdm_core import OFDMTransmitter
from config import LTEConfig
import numpy as np

config = LTEConfig(bandwidth=5.0, modulation='QPSK')
tx = OFDMTransmitter(config)

# Modular bits
bits = np.random.randint(0, 2, 1000)
signal_tx, symbols_tx, info = tx.modulate(bits)

# Estadísticas de PAPR
papr = tx.calculate_papr(signal_tx)
print(f"PAPR: {papr['papr_db']:.2f} dB")
```

**Métodos clave**:
- `modulate(bits)` → `(signal, symbols, info)`
- `calculate_papr(signal)` → dict con PAPR
- `get_config()` → configuración

---

### 2️⃣ OFDMChannel - Simula el canal (fading)

```python
from core.ofdm_core import OFDMChannel

# SISO: 1 signal in → 1 signal out
ch = OFDMChannel(channel_type='rayleigh_mp', snr_db=10.0)
signal_rx = ch.transmit(signal_tx)

# SIMO (preparado): 1 signal → N signals (para Phase 2)
signals_rx = ch.transmit_simo(signal_tx, num_rx=2)
```

**Tipos de canal**:
- `'awgn'` → Ruido blanco gaussiano
- `'rayleigh_mp'` → Rayleigh multipath

**Métodos clave**:
- `transmit(signal_tx)` → SISO
- `transmit_simo(signal_tx, num_rx)` → SIMO (prepared)
- `set_snr(snr_db)` → cambiar SNR

---

### 3️⃣ OFDMReceiver - Demodula señal a bits

```python
from core.ofdm_core import OFDMReceiver

rx = OFDMReceiver(config)

# Demodula
symbols_rx, bits_rx = rx.demodulate(signal_rx)

# Calcula BER
ber = rx.calculate_ber(bits_tx, bits_rx)
print(f"BER: {ber:.2e}")
```

**Métodos clave**:
- `demodulate(signal_rx)` → `(symbols, bits)`
- `calculate_ber(bits_tx, bits_rx)` → float
- `estimate_channel()` → dict (prepared para Phase 2)
- `get_config()` → configuración

---

### 4️⃣ OFDMSimulator - Orquestador (coordina todo)

```python
from core.ofdm_core import OFDMSimulator

sim = OFDMSimulator(config, channel_type='rayleigh_mp')

# ✅ SISO (funciona ahora)
result = sim.simulate_siso(bits, snr_db=10.0)
print(f"BER: {result['ber']:.2e}")

# ⏳ SIMO (preparado para Phase 2)
result = sim.simulate_simo(bits, snr_db=10.0, num_rx=2, combining='mrc')

# 📋 MIMO (future - no implementado)
# result = sim.simulate_mimo(bits, snr_db=10.0, num_tx=2, num_rx=2)
```

**Métodos clave**:
- `simulate_siso(bits, snr_db)` → ✅ Funciona
- `simulate_simo(bits, snr_db, num_rx, combining)` → ⏳ Prepared
- `simulate_mimo(bits, snr_db, num_tx, num_rx)` → 📋 Future
- `run_ber_sweep(num_bits, snr_range, num_trials)` → BER sweep

**Acceso directo a componentes**:
```python
tx = sim.tx          # OFDMTransmitter
rx = sim.rx          # OFDMReceiver
ch = sim.channels[0] # OFDMChannel (first)
```

---

## 3 Formas de Usar (según necesidad)

### Opción A: Simple (OFDMModule - backward compatible)

```python
# Tu código viejo sigue funcionando exactamente igual
from ofdm_module import OFDMModule

module = OFDMModule(config)
result = module.transmit(bits, snr_db=10)
print(f"BER: {result['ber']:.2e}")
```

**Ventaja**: No cambies nada, todo funciona  
**Ideal para**: Código existente

---

### Opción B: Modular (OFDMSimulator)

```python
# Código nuevo, más claro y escalable
from core.ofdm_core import OFDMSimulator

sim = OFDMSimulator(config, channel_type='rayleigh_mp')
result = sim.simulate_siso(bits, snr_db=10)
print(f"BER: {result['ber']:.2e}")

# En Phase 2, agregar SIMO sin cambiar esto:
# result = sim.simulate_simo(bits, snr_db=10, num_rx=2)
```

**Ventaja**: Código limpio, preparado para SIMO/MIMO  
**Ideal para**: Código nuevo, research

---

### Opción C: Investigación (Componentes independientes)

```python
# Acceso directo a cada componente para experimentos
from core.ofdm_core import OFDMTransmitter, OFDMReceiver, OFDMChannel

tx = OFDMTransmitter(config)
rx = OFDMReceiver(config)
ch = OFDMChannel(channel_type='rayleigh_mp', snr_db=10)

# Control manual del flujo de señal
signal_tx, _, _ = tx.modulate(bits)
signal_corrupted = ch.transmit(signal_tx)
_, bits_rx = rx.demodulate(signal_corrupted)

# Flexible para insertar código custom
```

**Ventaja**: Control total, flexible para research  
**Ideal para**: Experimentación, tesis, papers

---

## Ejemplo Completo (Copy-Paste Ready)

```python
#!/usr/bin/env python3
"""
Simulación OFDM completa con la arquitectura modular
"""

import numpy as np
from core.ofdm_core import OFDMSimulator
from config import LTEConfig

# ============================================================
# 1. Configurar
# ============================================================
config = LTEConfig(
    bandwidth=10.0,      # 10 MHz
    modulation='64-QAM', # 64-QAM (6 bits/symbol)
    cp_type='normal'
)

# ============================================================
# 2. Crear simulador
# ============================================================
sim = OFDMSimulator(
    config=config,
    channel_type='rayleigh_mp',  # Rayleigh multipath
    mode='lte'                   # LTE mode
)

# ============================================================
# 3. Generar bits
# ============================================================
num_bits = 100000
bits = np.random.randint(0, 2, num_bits)

# ============================================================
# 4. Simular para diferentes SNR
# ============================================================
snr_values = [5, 10, 15, 20]
results = {}

for snr in snr_values:
    result = sim.simulate_siso(bits, snr_db=snr)
    results[snr] = result
    
    print(f"SNR = {snr:2d} dB: BER = {result['ber']:.2e}, "
          f"Errors = {result['bit_errors']}, PAPR = {result['papr_db']:.2f} dB")

# ============================================================
# 5. Análisis
# ============================================================
# En Phase 2, comparar SISO vs SIMO:
# result_simo = sim.simulate_simo(bits, snr_db=15, num_rx=2)
# print(f"SIMO diversity gain: {result_siso['ber'] / result_simo['ber']:.2f}x")

# ============================================================
# 6. Graficar (opcional)
# ============================================================
try:
    import matplotlib.pyplot as plt
    
    snrs = list(results.keys())
    bers = [results[s]['ber'] for s in snrs]
    
    plt.semilogy(snrs, bers, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('BER', fontsize=12)
    plt.title('OFDM BER vs SNR (64-QAM, Rayleigh)', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('ofdm_ber_curve.png', dpi=150)
    print("\nGráfica guardada: ofdm_ber_curve.png")
except ImportError:
    print("Matplotlib no disponible")
```

**Salida esperada**:
```
SNR =  5 dB: BER = 2.81e-01, Errors = 28086, PAPR = 28.68 dB
SNR = 10 dB: BER = 1.25e-01, Errors = 12502, PAPR = 28.68 dB
SNR = 15 dB: BER = 2.14e-02, Errors = 2145, PAPR = 28.68 dB
SNR = 20 dB: BER = 1.51e-03, Errors = 151, PAPR = 28.68 dB

Gráfica guardada: ofdm_ber_curve.png
```

---

## Configuraciones Útiles

### Config 1: QPSK (simple, robusta)
```python
config = LTEConfig(
    bandwidth=5.0,
    modulation='QPSK',  # 1 bit/símbolo = más robusto
    cp_type='normal'
)
```

### Config 2: 16-QAM (equilibrio)
```python
config = LTEConfig(
    bandwidth=10.0,
    modulation='16-QAM',  # 4 bits/símbolo = equilibrio
    cp_type='normal'
)
```

### Config 3: 64-QAM (alto SNR)
```python
config = LTEConfig(
    bandwidth=10.0,
    modulation='64-QAM',  # 6 bits/símbolo = más datos
    cp_type='normal'
)
```

### Config 4: SC-FDM (LTE real)
```python
config = LTEConfig(bandwidth=10.0, modulation='16-QAM')
sim = OFDMSimulator(
    config=config,
    enable_sc_fdm=True  # ← SC-FDM (DFT precoding)
)
```

---

## Canales Disponibles

### AWGN (simple, para baseline)
```python
sim = OFDMSimulator(config, channel_type='awgn')
```

### Rayleigh con diferentes perfiles ITU
```python
sim = OFDMSimulator(config, channel_type='rayleigh_mp')

# Los perfiles (multipath profiles):
# - Pedestrian_A: cortos retardos
# - Pedestrian_B: con doppler
# - Vehicular_A: retardos medios
# - Vehicular_B: alta velocidad
# - Bad_Urban: retardos largos
```

---

## Medidas Útiles

### BER (Bit Error Rate)
```python
result = sim.simulate_siso(bits, snr_db=10)
ber = result['ber']
print(f"BER: {ber:.2e}")  # e.g., 1.23e-02 = 1.23%
```

### PAPR (Peak-to-Average Power Ratio)
```python
papr_db = result['papr_db']
print(f"PAPR: {papr_db:.2f} dB")  # Menos es mejor
```

### Número de errores
```python
errors = result['bit_errors']
total = result['transmitted_bits']
print(f"Errores: {errors} de {total}")
```

---

## Próximos Pasos

### Ya hecho ✅
- SISO funciona
- 4 clases modulares
- Backward compatible
- Documentación

### Phase 2 ⏳ (SIMO)
- [ ] Leer `ARCHITECTURE_MODULAR.md`
- [ ] Ejecutar `MODULAR_EXAMPLES.py`
- [ ] Usar `OFDMSimulator` en tus tests
- [ ] Esperar Phase 2 (2-3 semanas)

### Phase 3 📋 (MIMO)
- Roadmap en `IMPLEMENTATION_ROADMAP.py`
- Estimated 4-6 semanas después de Phase 2

---

## Archivos Clave

| Archivo | Qué es |
|---------|--------|
| `core/ofdm_core.py` | ⭐ Las 4 clases (OFDMTransmitter, Receiver, Channel, Simulator) |
| `ofdm_module.py` | Wrapper backward compatible |
| `ARCHITECTURE_MODULAR.md` | 📚 Documentación detallada (léelo!) |
| `MODULAR_EXAMPLES.py` | 💡 10 ejemplos (copy-paste ready) |
| `IMPLEMENTATION_ROADMAP.py` | 🗺️ Cómo implementar SIMO/MIMO |
| `test/final_image_test.py` | ✅ Test SISO funciona |

---

## Diferencias: Antes vs Después

### Antes (Monolítica)
```python
# Todo mezclado
module.modulator
module.demodulator
module.channel
```

### Después (Modular)
```python
# Separado y escalable
sim = OFDMSimulator(config)
sim.tx        # OFDMTransmitter
sim.rx        # OFDMReceiver
sim.channels  # List of OFDMChannel
```

---

## Preguntas Frecuentes

**P: ¿Puedo seguir usando OFDMModule?**  
R: ✅ Sí, funciona igual que antes.

**P: ¿Cuándo hay SIMO?**  
R: Phase 2, en 2-3 semanas (after channel estimation).

**P: ¿Qué cambian los BER?**  
R: ❌ Nada, son idénticos. Test validado.

**P: ¿Puedo experimentar ya?**  
R: ✅ Sí, usa OFDMSimulator o componentes directos.

**P: ¿Dónde aprendo más?**  
R: Léete ARCHITECTURE_MODULAR.md (muy completo).

---

## TL;DR (5 líneas)

```python
from core.ofdm_core import OFDMSimulator
from config import LTEConfig

config = LTEConfig(bandwidth=5.0, modulation='QPSK')
sim = OFDMSimulator(config, channel_type='rayleigh_mp')
result = sim.simulate_siso(bits, snr_db=10)
print(f"BER: {result['ber']:.2e}")  # ← Listo
```

---

**Last Updated**: 1 de Enero de 2026  
**Status**: ✅ SISO Complete, ⏳ SIMO Ready, 📋 MIMO Planned
