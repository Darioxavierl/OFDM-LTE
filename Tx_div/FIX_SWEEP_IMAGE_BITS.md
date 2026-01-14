# Fix: Barrido SNR - Usar Bits de Imagen sin Mostrarla

## Aclaración del Usuario

**Malentendido inicial:** Se removió toda la carga de imagen del barrido.

**Corrección:** El barrido SNR debe:
- ✅ **Usar bits de la imagen** cargada (datos reales para simulación)
- ❌ **No mostrar imagen** reconstruida (solo curvas BER)

**Razón:** El barrido es para **análisis de rendimiento**, no visualización. Pero los bits transmitidos deben ser de imagen real, no aleatorios.

## Solución Final Implementada

### 1. ✅ Cargar Imagen para Obtener Bits (línea ~760)

```python
# Cargar imagen para obtener bits (si existe)
if self.current_image_path:
    try:
        img = ImageProcessor.load_image(self.current_image_path)
        bits, metadata = ImageProcessor.image_to_bits(img)
        params['bits'] = bits
        params['metadata'] = metadata
        print(f"[INFO] Imagen cargada - bits para barrido: {len(bits)} bits ({metadata['width']}×{metadata['height']})")
    except Exception as e:
        print(f"[WARNING] No se pudo cargar imagen, usando bits aleatorios: {e}")
```

**Propósito:** Extraer bits de la imagen para usarlos como datos de transmisión.

### 2. ✅ Usar Bits de Imagen en Barrido (línea ~145)

```python
# Usar bits de imagen si existen, sino aleatorios
if 'bits' in self.params and 'metadata' in self.params:
    bits_to_sweep = self.params['bits']
    print(f"\n[DEBUG TxDiv] Barrido SNR - Usando bits de IMAGEN: {len(bits_to_sweep):,}")
    print(f"  Dimensiones imagen: {self.params['metadata']['width']}×{self.params['metadata']['height']}")
else:
    bits_to_sweep = np.random.randint(0, 2, 50000)
    print(f"\n[DEBUG TxDiv] Barrido SNR - Usando bits ALEATORIOS: {len(bits_to_sweep):,}")
```

**Propósito:** Usar bits de imagen si están disponibles, sino usar bits aleatorios (fallback).

### 3. ❌ No Generar/Mostrar Imagen Reconstruida

**Completamente removido:**
- ❌ Generación de imagen de ejemplo al final (~55 líneas removidas)
- ❌ Visualización en tab "Transmisión/Recepción" (~33 líneas removidas)
- ❌ Conversión de bits recibidos a imagen PIL

**Motivo:** El barrido es para análisis estadístico (curvas BER), no para ver calidad visual.

## Flujo de Datos

### Barrido SNR (Configuración Final)

```
1. Usuario carga imagen en GUI
   ↓
2. run_sweep_simulation() extrae bits de imagen
   ↓
3. _run_sweep_simulation() usa esos bits para TODAS las simulaciones
   ↓
4. Para cada (num_rx, modulación, SNR):
   - Enviar bits de imagen
   - Calcular BER (bits_tx vs bits_rx)
   - Guardar BER en array
   ↓
5. Graficar curvas BER vs SNR (4 subplots)
   ↓
6. NO reconstruir ni mostrar imagen
```

## Comparación: Antes vs Ahora

### ❌ Versión Anterior (Incorrecta)
```python
# Sin imagen - solo bits aleatorios
bits_to_sweep = np.random.randint(0, 2, 50000)
# → Datos sintéticos, no representativos de imagen real
```

### ✅ Versión Actual (Correcta)
```python
# Con imagen - bits reales
if 'bits' in self.params:
    bits_to_sweep = self.params['bits']  # Bits de imagen (ej: 1,920,000 bits)
else:
    bits_to_sweep = np.random.randint(0, 2, 50000)  # Fallback
# → Datos reales, representativos de transmisión de imagen
```

## Ventajas del Enfoque Actual

### Para el Barrido SNR:
1. **Datos realistas** - Bits de imagen real (correlación, patrones)
2. **Sin overhead visual** - No genera/muestra imagen (más rápido)
3. **Análisis puro** - Solo métricas BER vs SNR
4. **Escalable** - Funciona con imágenes grandes sin problemas de visualización

### Para Simulación Única y Comparación MIMO:
- Sí muestran imagen reconstruida (ese es su propósito)
- Permiten ver calidad visual vs métricas

## Casos de Uso

| Modo | Bits Usados | Muestra Imagen | Propósito |
|------|-------------|----------------|-----------|
| **Barrido SNR** | ✅ De imagen | ❌ No | Análisis BER vs SNR |
| **Simulación Única** | ✅ De imagen | ✅ Sí | Ver calidad visual |
| **Comparación MIMO** | ✅ De imagen | ✅ Sí (×4) | Comparar configs |

## Ejemplo de Output Console

### Con imagen cargada:
```
[INFO] Imagen cargada - bits para barrido: 1920000 bits (800×600)

[DEBUG TxDiv] Barrido SNR - Usando bits de IMAGEN: 1,920,000
  Dimensiones imagen: 800×600
  Modulaciones: ['QPSK', '16-QAM', '64-QAM']
  Num TX: 2 (SFBC fijo)
  Num RX: [1, 2, 4, 8]
  Rango SNR: 0.0 a 20.0 dB
```

### Sin imagen cargada:
```
[DEBUG TxDiv] Barrido SNR - Usando bits ALEATORIOS: 50,000
  Modulaciones: ['QPSK', '16-QAM', '64-QAM']
  Num TX: 2 (SFBC fijo)
  Num RX: [1, 2, 4, 8]
  Rango SNR: 0.0 a 20.0 dB
```

## Estado
✅ **RESUELTO** - Barrido SNR usa bits de imagen pero NO muestra imagen reconstruida
