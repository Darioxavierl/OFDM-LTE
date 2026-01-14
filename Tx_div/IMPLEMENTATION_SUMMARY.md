# Tx_div - Implementación Completada

## Resumen de la Implementación

Se ha creado exitosamente la interfaz gráfica para **Transmit Diversity con SFBC Alamouti**, siguiendo la misma estructura que SIMO pero adaptada para MIMO.

## Estructura Creada

```
Tx_div/
├── __init__.py                  ✅ Creado
├── main.py                      ✅ Creado (launcher)
├── README.md                    ✅ Creado (documentación completa)
└── gui/
    ├── __init__.py              ✅ Creado
    ├── main_window.py           ✅ Creado (~1035 líneas)
    └── widgets.py               ✅ Creado (adaptado de SIMO)
```

## Componentes Implementados

### 1. main.py (Launcher)
- Inicializa QApplication
- Crea ventana TxDiversityGUI
- Actualiza configuración inicial
- Ejecuta event loop

### 2. main_window.py (Ventana Principal)

#### SimulationWorker (QThread)
Tres modos de operación:

**a) _run_single_simulation():**
```python
if num_rx == 1:
    results = self.ofdm_system.simulate_miso(bits, snr_db, channel_type, itu_profile)
else:
    results = self.ofdm_system.simulate_mimo(bits, snr_db, num_tx=2, num_rx=num_rx, ...)
```

**b) _run_sweep_simulation():**
```python
num_rx_values = [1, 2, 4, 8]  # 4 subplots
modulations = ['QPSK', '16-QAM', '64-QAM']  # 3 curvas por subplot
# Resultados: all_results[f'2x{num_rx}'][modulation]
```

**c) _run_mimo_comparison():**
```python
configs = [
    ("SISO", 1, 1),    # Referencia
    ("MISO", 2, 1),    # Diversidad TX
    ("MIMO-2x2", 2, 2),
    ("MIMO-2x4", 2, 4)
]
```

#### TxDiversityGUI (QMainWindow)

**Panel de Control:**
- Parámetros LTE: Modulación, BW, CP, Δf
- Parámetros de Simulación: SNR, Iteraciones, **Num RX (1/2/4/8)**
- Label especial: "2 TX fijos con SFBC Alamouti"
- Parámetros de Canal: Tipo, perfil ITU, frecuencia, velocidad
- Carga de imagen

**Botones de Simulación:**
1. **Simulación Única** (verde)
2. **Barrido en SNR** (naranja)
3. **Comparación MIMO** (azul) ← Diferente a SIMO

**Panel de Resultados (5 pestañas):**
1. Transmisión/Recepción: Comparación de imágenes
2. Constelación: Símbolos SFBC decodificados
3. Curvas BER: 4 subplots verticales
4. Comparación MIMO: Grid 2×4 con 8 imágenes
5. Métricas: BER, PAPR, Tiempo

**Métodos de Ploteo:**

```python
def plot_constellation(self, results):
    # Muestra símbolos TX y RX para 2 TX × N RX
    # Subplot por cada antena

def plot_sweep_ber_curves(self, results):
    # 4 subplots apilados verticalmente (12×16 figsize)
    # Subplot 1: MISO (2×1)
    # Subplot 2: MIMO 2×2
    # Subplot 3: MIMO 2×4
    # Subplot 4: MIMO 2×8
    # 3 curvas por subplot (QPSK, 16-QAM, 64-QAM)

def plot_mimo_comparison(self, results):
    # Grid 2 filas × 4 columnas
    # Fila 1: Original (×4)
    # Fila 2: SISO, MISO, 2×2, 2×4 con BER overlay
```

### 3. widgets.py (Widgets Personalizados)

#### PlotWidget
- Contenedor para gráficas matplotlib
- Incluye toolbar de navegación
- Métodos: get_figure(), get_canvas(), clear()

#### MetricsPanel
- Muestra BER, PAPR, Tiempo de procesamiento
- Actualización dinámica con update_metrics()

#### ConfigInfoPanel
- Muestra configuración LTE-MIMO en formato texto
- Estilo terminal (fondo oscuro, fuente monospace)

#### ImageComparisonWidget
- Comparación lado a lado: Original vs Recibida
- Muestra PSNR
- Escala automática manteniendo aspect ratio

## Diferencias Clave vs SIMO

| Característica | SIMO | Tx_div (MIMO) |
|----------------|------|---------------|
| **Título ventana** | "OFDM - SIMO Reception Diversity" | "Transmit Diversity - SFBC Alamouti" |
| **Antenas TX** | 1 (fija) | 2 (SFBC, fija) |
| **Label RX** | "Número de Antenas RX:" | "Número de Antenas RX:<br>(2 TX fijos con SFBC Alamouti)" |
| **Core usado** | simulate_simo() | simulate_miso() / simulate_mimo() |
| **Barrido SNR** | 3 subplots horizontales (por mod) | 4 subplots verticales (por RX) |
| **Botón azul** | "Prueba Multiantena" | "Comparación MIMO" |
| **Comparación** | Compara N RX (1×1 a 1×8) | Compara M×N (1×1 a 2×4) |
| **Orden diversidad** | N | 2×N |
| **ConfigInfoPanel** | "Configuración LTE" | "Configuración LTE-MIMO" |

## Adaptaciones Específicas

### 1. Selección de RX
```python
# SIMO tenía: 1, 2, 4, 8
# Tx_div tiene: 1, 2, 4, 8
# Pero con label "(2 TX fijos con SFBC Alamouti)"
```

### 2. Cálculo de Orden de Diversidad
```python
diversity_order = 2 * self.current_num_rx  # 2 TX × N RX
```

### 3. Configuración en update_config()
```python
config_text = f"""
Modulación: {modulation}
...
Antenas TX: 2 (SFBC Alamouti)
Antenas RX: {num_rx}
Orden de Diversidad: {2 * num_rx}
...
"""
```

### 4. Layout de Barrido
```python
# SIMO: 3 subplots en una fila (uno por modulación)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Tx_div: 4 subplots en una columna (uno por RX)
fig, axes = plt.subplots(4, 1, figsize=(12, 16))
```

## Correcciones Realizadas Durante la Implementación

### Problema 1: Archivo demasiado largo
- **Solución:** Creado en 2 partes y fusionado
- main_window.py (700 líneas) + main_window_part2.txt (500 líneas)

### Problema 2: Sintaxis f-strings
- **Error:** `f'...{results[\"error\"]}'` (escape incorrecto)
- **Solución:** Cambiado a `f'...{results["error"]}'`
- Corregido en 3 lugares (líneas 802, 835, 850)

### Problema 3: Import incorrecto
- **Error:** `from core.ofdm_core import OFDMSystem`
- **Solución:** `from core.ofdm_system import OFDMSystem`

### Problema 4: PyQt6 no instalado
- **Solución:** `pip install PyQt6` en entorno virtual

## Testing

### Verificación de Imports
```bash
✅ python -c "import Tx_div.gui.main_window; print('✓ Imports OK')"
```

### Lanzamiento de GUI
```bash
✅ python -m Tx_div.main
# GUI se abre correctamente
# No errores en consola
```

## Próximos Pasos (Testing de Funcionalidad)

Para validar completamente la implementación:

1. **Test de Simulación Única:**
   - Cargar imagen de prueba (img/entre-ciel-et-terre.jpg)
   - Configurar MISO (2×1) con QPSK, SNR=10dB, AWGN
   - Verificar imagen recibida y BER

2. **Test de Barrido SNR:**
   - Ejecutar barrido completo
   - Verificar 4 subplots verticales
   - Confirmar 3 curvas por subplot
   - Validar tendencia de mejora con más RX

3. **Test de Comparación MIMO:**
   - Lanzar comparación completa
   - Verificar grid 2×4
   - Confirmar mejora: SISO < MISO < 2×2 < 2×4

4. **Test de Canales:**
   - AWGN: Verificar diversidad de fase
   - Rayleigh Pedestrian A: Verificar desvanecimiento
   - ITU perfiles: Vehicular A con alta velocidad

## Documentación

- ✅ README.md completo en Tx_div/
- ✅ Docstrings en todos los métodos
- ✅ Comentarios explicativos en código crítico
- ✅ Este resumen de implementación

## Conclusión

La interfaz gráfica **Tx_div** está **100% implementada** y lista para usar. Replica exactamente la estructura de SIMO pero adaptada para MIMO con SFBC Alamouti:

- 2 TX antenas fijas (SFBC)
- 1-8 RX antenas seleccionables
- 3 modos de simulación completos
- Plotting adaptado para comparación MIMO
- Documentación completa

El sistema está listo para pruebas funcionales y demostraciones.
