# Transmit Diversity - SFBC Alamouti GUI

Interfaz gráfica para simular MIMO con diversidad en transmisión utilizando codificación Space-Frequency Block Code (SFBC) Alamouti.

## Características

### Configuración Fija
- **2 antenas TX** (Alamouti SFBC) - Siempre fijo
- **Variable RX**: 1, 2, 4 u 8 antenas receptoras

### Órdenes de Diversidad
- **MISO (2×1)**: Orden 2
- **MIMO 2×2**: Orden 4
- **MIMO 2×4**: Orden 8
- **MIMO 2×8**: Orden 16

## Uso

### Lanzar la GUI

```bash
python -m Tx_div.main
```

### Modos de Simulación

#### 1. Simulación Única
Transmite una imagen con la configuración actual:
- Selecciona parámetros LTE (modulación, BW, CP, Δf)
- Configura SNR e iteraciones
- Selecciona número de antenas RX (1, 2, 4 u 8)
- Configura canal (AWGN o Rayleigh con perfil ITU)
- Carga imagen (botón "Cargar Imagen")
- Presiona "Simulación Única"

**Resultados:**
- Pestaña "Transmisión/Recepción": Imágenes TX y RX lado a lado
- Pestaña "Constelación": Símbolos SFBC decodificados (2 TX × N RX)
- Pestaña "Métricas": BER, PAPR, tiempo de procesamiento

#### 2. Barrido en SNR
Genera curvas BER vs SNR para todas las configuraciones de RX:
- Configura parámetros LTE y canal
- Presiona "Barrido en SNR"

**Resultados:**
- **4 subplots verticales** (uno por configuración de RX)
  - Subplot 1: MISO (2×1)
  - Subplot 2: MIMO 2×2
  - Subplot 3: MIMO 2×4
  - Subplot 4: MIMO 2×8
- **3 curvas por subplot** (una por modulación):
  - QPSK (azul)
  - 16-QAM (naranja)
  - 64-QAM (verde)

#### 3. Comparación MIMO
Compara rendimiento entre diferentes configuraciones MIMO:
- Carga una imagen
- Presiona "Comparación MIMO"

**Resultados:**
- **Grid 2×4** con 8 imágenes:
  - Fila superior: Imagen original (repetida 4 veces)
  - Fila inferior: Imágenes reconstruidas con:
    - SISO (1×1) - Referencia
    - MISO (2×1) - Diversidad TX
    - MIMO 2×2 - Diversidad espacial
    - MIMO 2×4 - Mayor diversidad
- Cada imagen reconstruida muestra BER en el título

## Arquitectura

```
Tx_div/
├── __init__.py              # Módulo init
├── main.py                  # Launcher de la aplicación
├── README.md                # Este archivo
└── gui/
    ├── __init__.py          # GUI package init
    ├── main_window.py       # Ventana principal (~1000 líneas)
    └── widgets.py           # Widgets personalizados
```

### Core MIMO Utilizado

El GUI usa directamente las funciones del core:
- `OFDMSystem.simulate_miso()`: Para configuración 2×1
- `OFDMSystem.simulate_mimo()`: Para configuraciones 2×N (N>1)

No hay capa de simulación intermedia, a diferencia de otros módulos del proyecto.

## Parámetros LTE

### Modulación
- QPSK (2 bits/símbolo)
- 16-QAM (4 bits/símbolo)
- 64-QAM (6 bits/símbolo)

### Ancho de Banda
- 1.4 MHz (6 PRBs)
- 3 MHz (15 PRBs)
- 5 MHz (25 PRBs)
- 10 MHz (50 PRBs)
- 15 MHz (75 PRBs)
- 20 MHz (100 PRBs)

### Prefijo Cíclico
- Normal: 7 símbolos por slot
- Extendido: 6 símbolos por slot

### Espaciado de Subportadora
- 15 kHz (estándar LTE)
- 30 kHz
- 60 kHz

## Canal

### AWGN
- Canal gaussiano con diversidad de fase (0°, 90°)
- SNR normalizado por número de TX

### Rayleigh Fading
- Perfiles ITU-R M.1225:
  - Pedestrian A (baja dispersión)
  - Vehicular A (media dispersión)
- Configurable: frecuencia portadora (900 MHz - 6 GHz), velocidad (0-500 km/h)

## Diferencias con SIMO

| Aspecto | SIMO | Tx_div (MIMO) |
|---------|------|---------------|
| Antenas TX | 1 (fija) | 2 (SFBC, fija) |
| Antenas RX | 1-8 (variable) | 1-8 (variable) |
| Diversidad | Solo RX | TX + RX |
| Barrido SNR | 3 subplots horizontales (por modulación) | 4 subplots verticales (por RX) |
| Orden diversidad | N (num RX) | 2N (2 TX × N RX) |
| Comparación | Multi-antena (1×N) | MIMO (M×N) |

## Tecnología SFBC Alamouti

### Principio
Space-Frequency Block Coding usa 2 antenas TX y codifica símbolos en bloques:
- TX1 envía: [s₁, -s₂*] en subportadoras consecutivas
- TX2 envía: [s₂, s₁*] en subportadoras consecutivas

### Ventajas
- Ganancia por diversidad de transmisión
- No requiere CSI en transmisor
- Decodificación lineal en receptor
- Compatible con cualquier número de antenas RX

### Rendimiento
En AWGN (según test_mimo_image.py):
- MISO (2×1): +0.83 dB vs SISO
- MIMO 2×2: +3.24 dB vs SISO
- MIMO 2×4: +7.19 dB vs SISO

## Requisitos

```
PyQt6
numpy
matplotlib
Pillow
```

## Notas

- La GUI **NO** crea archivos de simulación intermedios
- Usa directamente `core.ofdm_system.OFDMSystem`
- Todas las simulaciones se ejecutan en threads separados (no bloquea la interfaz)
- Los resultados se almacenan en memoria (no se guardan automáticamente)
