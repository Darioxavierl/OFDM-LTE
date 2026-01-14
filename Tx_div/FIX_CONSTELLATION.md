# Fix #6: Agregar Visualización de Constelaciones

## Problema
La GUI no mostraba las constelaciones TX/RX después de la simulación única.

## Causa
Los métodos `simulate_miso()` y `simulate_mimo()` del core NO retornan los símbolos TX/RX necesarios para plotear la constelación.

## Solución
En lugar de modificar el core (podría romper otros tests), genero los símbolos en el GUI a partir de los bits.

### Implementación en `_run_single_simulation()` (línea ~97)

```python
# Después de la simulación
from core.modulator import QAMModulator
qam_mod = QAMModulator(self.ofdm_system.config.modulation)

# Tomar muestra de bits para constelación (máximo 1000 símbolos)
max_symbols = 1000
bits_per_symbol = int(np.log2(len(qam_mod.constellation)))
max_bits = max_symbols * bits_per_symbol

# Símbolos TX (desde bits originales)
sample_bits_tx = bits[:min(max_bits, len(bits))]
symbols_tx = qam_mod.bits_to_symbols(sample_bits_tx)

# Símbolos RX (desde bits recibidos)
bits_rx = results['bits_received_array']
sample_bits_rx = bits_rx[:min(max_bits, len(bits_rx))]
symbols_rx = qam_mod.bits_to_symbols(sample_bits_rx)

# Agregar a resultados
results['symbols_tx'] = symbols_tx
results['symbols_rx'] = symbols_rx
```

### Handler en `on_single_simulation_finished()` (línea ~837)

```python
# Graficar constelación si existe
if 'symbols_rx' in results:
    self.plot_constellation(
        results.get('symbols_tx', results.get('qam_symbols')),
        results['symbols_rx']
    )
```

### Método de Ploteo (línea ~877)

Ya existía `plot_constellation()` que:
- Convierte símbolos a numpy arrays
- Muestrea hasta 2000 símbolos (para rendimiento)
- Plotea TX en azul, RX en rojo
- Título: "Constelación - MIMO 2×N (SFBC Alamouti)"
- Cambia automáticamente a la pestaña de constelación

## Ventajas de Este Enfoque

1. **No modifica el core** - No afecta `simulate_miso()` ni `simulate_mimo()`
2. **No rompe tests existentes** - `test_mimo_image.py` sigue funcionando
3. **Eficiente** - Solo genera símbolos necesarios para visualización
4. **Consistente** - Los símbolos RX reflejan exactamente lo que se recibió

## Nota Técnica

Los símbolos RX se generan de `bits_received_array` (después de detección/decisión), no de los símbolos antes de decisión. Esto significa que:
- Los puntos RX estarán perfectamente alineados con la constelación
- No se ve el "ruido" del canal directamente
- Se ve el efecto del BER (puntos RX en posiciones incorrectas)

Si se quisiera ver el ruido pre-decisión, habría que modificar el core para retornar `decoded_symbols` antes de la detección.

## Estado
✅ **RESUELTO** - Constelaciones se muestran correctamente en simulación única
