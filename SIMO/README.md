# SIMO - Simulador OFDM-LTE con Múltiples Antenas Receptoras

## Descripción

GUI completa para simulación OFDM-LTE con soporte **SISO** (Single-Input Single-Output) y **SIMO** (Single-Input Multiple-Output). Utiliza el **core local del proyecto** con combinación MRC (Maximum Ratio Combining) para diversidad de antenas.

## Características Principales

### Arquitecturas Soportadas
- **SISO**: 1 antena TX → 1 antena RX (modo clásico)
- **SIMO**: 1 antena TX → Múltiples RX (2, 4, 8 antenas)
- **MRC**: Combinación óptima de señales de múltiples receptores
- **Procesamiento Paralelo**: ThreadPoolExecutor para simulaciones multi-RX

### Interfaz de 3 Paneles
- **Panel Izquierdo**: Controles de configuración
- **Panel Central**: Información de configuración actualizada
- **Panel Derecho**: Resultados en 4 pestañas

### Parámetros LTE Configurables
- **Ancho de banda**: 1.25, 2.5, 5, 10, 15, 20 MHz
- **Separación subportadoras (Δf)**: 15.0, 7.5 kHz
- **Modulación**: QPSK, 16-QAM, 64-QAM
- **Prefijo Cíclico**: normal, extended
- **Número de Receptores**: 1, 2, 4, 8 antenas

El sistema **calcula automáticamente** Nc y N_fft usando LTE_PROFILES.

### Configuración de Canal
- **Canal AWGN**: Ruido gaussiano aditivo
- **Canal Rayleigh Multitrayecto**: Con perfiles ITU-R M.1225
  - Pedestrian A/B (canales peatonales)
  - Vehicular A/B (canales vehiculares)
  - Typical Urban (urbano típico)
  - Rural Area (área rural)
- **Parámetros Doppler**:
  - Frecuencia portadora: 0.5 - 10.0 GHz
  - Velocidad: 0 - 500 km/h
  - Desplazamiento Doppler calculado automáticamente

### Simulaciones Disponibles

#### 1. Simulación Única
- Transmite una imagen con configuración específica
- Soporte SISO (1 RX) o SIMO (2/4/8 RX)
- **Métricas mostradas**: BER, PAPR, Tiempo de ejecución
- **Visualizaciones**:
  - Constelación TX vs RX
  - Comparación de imágenes original/recibida
- Procesamiento paralelo automático para SIMO

#### 2. Barrido en SNR
- Prueba el rango completo de SNR con configuración fija
- **Parámetros**:
  - SNR inicio, fin, paso (configurable)
  - Número de receptores fijo durante el barrido
  - Transmite la imagen completa en cada punto de SNR
- **Salida**: Curva BER vs SNR para analizar rendimiento
- Ideal para caracterizar el sistema en diferentes condiciones de ruido

#### 3. Prueba Multiantena (Nuevo)
- Compara rendimiento de 1, 2, 4 y 8 antenas receptoras simultáneamente
- Misma imagen, mismo SNR, mismo canal para todas
- **Visualización en cuadrícula 2×4**:
  - Fila superior: Imagen original repetida 4 veces
  - Fila inferior: Imágenes recibidas con 1/2/4/8 RX
- Títulos muestran BER y número de receptores
- Procesamiento paralelo habilitado para todas las configuraciones SIMO

### Pestañas de Resultados
1. **Constelación**: Diagramas I/Q de símbolos TX y RX
2. **Imagen**: Comparación visual original vs recibida
3. **BER vs SNR**: Curvas de rendimiento (barrido SNR)
4. **Multiantena**: Comparación lado a lado de diferentes configuraciones RX

## Estructura de Archivos

```
SIMO/
├── __init__.py (8 líneas)
├── main.py (24 líneas) - Punto de entrada
├── README.md - Esta documentación
└── gui/
    ├── __init__.py (3 líneas)
    ├── widgets.py (256 líneas) - Widgets personalizados
    └── main_window.py (975 líneas) - Ventana principal + Workers
```

**Total: ~1,266 líneas de código**

### Archivos Clave

#### `main.py`
- Punto de entrada de la aplicación
- Inicializa QApplication con estilo Fusion
- Maneja excepciones globales

#### `gui/main_window.py`
- **OFDMSimulatorGUI**: Ventana principal con 3 paneles
- **SimulationWorker**: Thread para simulaciones únicas y barridos SNR
- **MultiantennaWorker**: Thread para comparación multi-RX
- **Métodos principales**:
  - `_run_single_simulation()`: Ejecuta SISO o SIMO según configuración
  - `_run_sweep_simulation()`: Barrido de BER vs SNR
  - `_run_multiantenna_test()`: Prueba comparativa 1/2/4/8 RX
  - `plot_constellation()`: Diagrama I/Q
  - `plot_multiantenna_comparison()`: Cuadrícula 2×4 de imágenes

#### `gui/widgets.py`
- **PlotWidget**: Contenedor Matplotlib con toolbar
- **MetricsPanel**: Muestra BER, PAPR, Tiempo
- **ConfigInfoPanel**: Información de configuración formateada
- **ImageComparisonWidget**: Comparación lado a lado de imágenes
- **ConfigPanel**: Panel izquierdo con todos los controles

## Uso

### Instalación Rápida

```powershell
# Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# Instalar dependencias (si es necesario)
pip install -r requirements.txt

# Ejecutar GUI
python SIMO/main.py
```

O directamente:

```powershell
.venv\Scripts\python.exe SIMO\main.py
```

### Flujo de Trabajo Básico

1. **Configurar LTE**
   - Selecciona ancho de banda (1.25 - 20 MHz)
   - Elige modulación (QPSK, 16-QAM, 64-QAM)
   - Configura Δf (15.0 o 7.5 kHz) y tipo de CP
   - **Importante**: Selecciona número de receptores (1, 2, 4, 8)

2. **Configurar Canal**
   - Tipo: AWGN (simple) o Rayleigh Multipath (realista)
   - Si Rayleigh: selecciona perfil ITU (Pedestrian, Vehicular, etc.)
   - Configura frecuencia portadora y velocidad para Doppler
   - Ajusta SNR (dB) para simulación única

3. **Cargar Imagen**
   - Clic en "Cargar Imagen"
   - Selecciona archivo JPG/PNG
   - Verifica que se muestra la ruta en el panel de configuración

4. **Ejecutar Simulaciones**

   **Opción A: Simulación Única**
   - Clic en "Simulación Única"
   - Espera a que termine (progress bar se muestra)
   - Revisa pestañas "Constelación" e "Imagen"
   - Verifica métricas: BER, PAPR, Tiempo

   **Opción B: Barrido en SNR**
   - Configura rango: SNR inicio, fin, paso
   - Clic en "Barrido en SNR"
   - Espera (puede tardar varios minutos)
   - Revisa pestaña "BER vs SNR" para ver curva

   **Opción C: Prueba Multiantena**
   - Configura canal y SNR deseados
   - Clic en "Prueba Multiantena"
   - Compara rendimiento de 1, 2, 4, 8 RX simultáneamente
   - Revisa pestaña "Multiantena" para visualización en cuadrícula

5. **Analizar Resultados**
   - Navegas entre pestañas para ver diferentes visualizaciones
   - Usa toolbar de Matplotlib para zoom, pan, guardar figuras
   - Compara BER entre diferentes configuraciones

### Ejemplos de Uso

#### Ejemplo 1: Evaluar Efecto de Múltiples Antenas
```
1. Config: BW=5MHz, QPSK, 1 RX
2. Canal: Vehicular A, 2 GHz, 60 km/h, SNR=10dB
3. Cargar imagen
4. Simulación Única → Anota BER₁
5. Cambiar a 4 RX
6. Simulación Única → Anota BER₄
7. Observar mejora: BER₄ << BER₁ (típicamente 10-100x menor)
```

#### Ejemplo 2: Caracterización de Canal
```
1. Config: BW=10MHz, 16-QAM, 2 RX
2. Canal: Rayleigh (varía perfil)
3. Barrido: SNR -5 a 25 dB, paso 2.5
4. Repite para cada perfil ITU
5. Compara curvas BER vs SNR
```

#### Ejemplo 3: Comparación Visual Rápida
```
1. Config: Cualquier configuración base
2. Cargar imagen
3. Clic en "Prueba Multiantena"
4. Ver cuadrícula 2×4 con mejora visual conforme aumentan RX
```

## Componentes del Proyecto Local Utilizados

### Config (`config.py`)
- **LTEConfig**: Clase de configuración con todos los parámetros
- **LTE_PROFILES**: Diccionario con perfiles estándar LTE
- Calcula automáticamente Nc y N_fft según ancho de banda

### Core (`core/`)

#### `ofdm_core.py` (Principal)
- **OFDMSimulator**: Clase principal del simulador
- **`simulate_siso(bits, config)`**: Simulación con 1 RX
  - Entrada: bits transmitidos, configuración
  - Salida: bits recibidos, BER, PAPR, símbolos
- **`simulate_simo(bits, config, num_rx, parallel=True)`**: Simulación con múltiples RX
  - Genera canales independientes para cada antena
  - Aplica MRC (Maximum Ratio Combining) a los símbolos recibidos
  - Parámetro `parallel`: ThreadPoolExecutor para procesamiento concurrente
  - Retorna: bits combinados, BER, PAPR promedio, símbolos

#### `channel.py`
- **ChannelSimulator**: Simulador de canal
- Soporte AWGN y Rayleigh multipath
- Carga perfiles ITU-R M.1225 desde JSON

#### `modulator.py` y `demodulator.py`
- **OFDMModulator**: Mapeo de bits → símbolos → OFDM
- **OFDMDemodulator**: Demodulación OFDM → símbolos → bits
- Soporte QPSK, 16-QAM, 64-QAM

#### `itu_r_m1225.py`
- **ITURayChannel**: Canal Rayleigh según ITU-R M.1225
- Perfiles: Pedestrian A/B, Vehicular A/B, Typical Urban, Rural
- Cálculo automático de desplazamiento Doppler

#### `resource_mapper.py` y `dft_precoding.py`
- Mapeo de recursos en frecuencia
- Pre-codificación DFT para SC-FDMA

### Utils (`utils/`)

#### `image_processing.py`
- **ImageProcessor**: Conversión imagen ↔ bits
- **`image_to_bits(image_path)`**: Carga imagen y convierte a bits
  - Usa `np.unpackbits()` para conversión RGB → bits
  - Retorna: array de bits, dimensiones originales
- **`bits_to_image(bits, original_shape)`**: Reconstruye imagen desde bits
  - Usa `np.packbits()` para bits → RGB
  - Maneja padding si el número de bits no coincide exactamente
- **Métricas**: PSNR y SSIM para comparación de calidad (actualmente no usadas en GUI)

## Widgets Personalizados

### PlotWidget
Widget contenedor para gráficos Matplotlib integrados en PyQt6:
- **FigureCanvasQTAgg**: Canvas de Matplotlib embebido
- **NavigationToolbar2QT**: Barra de herramientas con pan, zoom, guardar
- **Métodos públicos**:
  - `get_figure()`: Obtiene figura de Matplotlib
  - `get_canvas()`: Obtiene canvas para drawing
  - `clear()`: Limpia figura y redibuja

### MetricsPanel
Panel de métricas simplificado (3 métricas principales):
- **BER** (Bit Error Rate): Tasa de error de bits
- **PAPR** (Peak-to-Average Power Ratio): Relación pico-promedio en dB
- **Tiempo**: Duración de la simulación en segundos
- **Método**: `update_metrics(metrics_dict)` actualiza labels con formato

### ConfigInfoPanel
Panel informativo de solo lectura (QTextEdit):
- Muestra configuración LTE formateada (BW, modulación, Nc, N_fft, etc.)
- Información del canal (tipo, perfil, Doppler)
- Actualización automática cuando cambian controles
- **Método**: `update_config(config_dict)` actualiza texto formateado

### ImageComparisonWidget
Widget de comparación visual de imágenes:
- Dos subplots lado a lado con Matplotlib
- Subplot izquierdo: Imagen original
- Subplot derecho: Imagen recibida/reconstruida
- Títulos muestran dimensiones y estadísticas
- **Método**: `set_images(original, received)` actualiza ambas imágenes
- **Método**: `clear()` limpia canvas y remueve figuras

### ConfigPanel
Panel izquierdo completo con todos los controles:
- **Grupo LTE**: BW, Δf, Modulación, CP, Num RX
- **Grupo Canal**: Tipo, Perfil ITU, SNR, Freq, Velocidad
- **Callbacks dinámicos**:
  - Cambio de tipo de canal → habilita/deshabilita controles Rayleigh
  - Cambio de frecuencia/velocidad → actualiza label de Doppler
- **Botones de acción**: Cargar Imagen, Simular, Barrido, Prueba Multiantena
- **Señales emitidas**: `config_changed`, `load_image_clicked`, `simulate_clicked`, etc.

## Arquitectura de Workers (QThreads)

### SimulationWorker
Thread para simulaciones únicas y barridos de SNR:
- **Modos de operación**:
  - `'single'`: Simulación única (SISO o SIMO según num_rx)
  - `'sweep'`: Barrido de BER vs SNR
- **Proceso**:
  1. Carga imagen y convierte a bits
  2. Crea/actualiza OFDMSimulator con configuración
  3. Ejecuta simulación(es)
  4. Emite señal de progreso durante ejecución
  5. Emite señal `finished` con resultados al terminar
- **Señales**:
  - `progress(int, str)`: Porcentaje y mensaje de estado
  - `finished(dict)`: Resultados completos (ber, papr, bits, símbolos, imagen)
- **Debug**: Mensajes de consola rastrean bits TX/RX para verificar integridad

### MultiantennaWorker
Thread especializado para prueba comparativa multi-RX:
- **Función**: Ejecuta 4 simulaciones en paralelo (1, 2, 4, 8 RX)
- **Proceso**:
  1. Carga imagen una vez
  2. Crea configuración base
  3. Para cada num_rx en [1, 2, 4, 8]:
     - Actualiza configuración con num_rx actual
     - Ejecuta `simulate_siso()` o `simulate_simo()` según corresponda
     - Habilita `parallel=True` para SIMO
     - Reconstruye imagen desde bits recibidos
     - Almacena BER e imagen
  4. Emite señal `finished` con los 4 resultados
- **Señales**:
  - `progress(int, str)`: Actualiza durante cada simulación
  - `finished(dict)`: Diccionario con resultados de 1/2/4/8 RX
- **Debug**: Rastrea bits esperados vs recibidos, valida longitudes

### Ventajas del Diseño Multi-Thread
- **UI Responsive**: La GUI no se congela durante simulaciones largas
- **Progress Updates**: El usuario ve avance en tiempo real
- **Cancelable**: Posibilidad de abortar simulaciones (no implementado aún)
- **Thread-Safe**: Comunicación mediante señales Qt (signal/slot)
- **Paralelismo**: SIMO usa ThreadPoolExecutor dentro del worker para mayor velocidad

## Notas Técnicas

### Rendimiento
- **SISO (1 RX)**: Velocidad base (~10-15 segundos para imagen 450×600)
- **SIMO (2/4/8 RX)**: Paralelismo con ThreadPoolExecutor reduce overhead
- **GUI vs Script**: La GUI es ~25-45% más lenta que scripts standalone debido a:
  - Overhead de Qt (event loop, signals/slots)
  - Renderizado de gráficos en tiempo real
  - Callbacks de progreso durante ejecución
- **Optimización**: `parallel=True` en `simulate_simo()` usa todos los cores disponibles

### Debugging y Validación
- **Mensajes de Debug**: Formato `[DEBUG NombreWorker] Etapa: detalles`
- **Rastreo de Bits**:
  - Bits TX: longitud esperada = altura × ancho × canales × 8
  - Bits RX: debe coincidir exactamente con bits TX para reconstrucción correcta
  - Status de match: "✓" si coincide, "✗ MISMATCH!" si no
- **Verificación de Paralelismo**: Mensaje indica si está habilitado para SIMO
- **Console Output**: Toda la información de debug se imprime en consola, no en GUI

### Manejo de Errores
- Try-catch exhaustivo en todos los métodos críticos
- Traceback completo en excepciones para debugging
- QMessageBox con errores detallados para el usuario
- Validaciones antes de ejecutar simulaciones:
  - Imagen cargada
  - Configuración válida
  - Parámetros en rangos permitidos

### Formato de Datos
- **Bits**: Arrays NumPy de tipo int (valores 0 o 1)
- **Símbolos**: Arrays NumPy de tipo complex128
- **Imágenes**: Arrays NumPy shape (H, W, 3) dtype uint8
- **BER**: Float entre 0.0 y 1.0
- **PAPR**: Float en dB (típicamente 5-13 dB para OFDM)

### Estilo Visual
- **Tema Qt**: Fusion (moderno, multiplataforma)
- **Colores**: Paleta por defecto de Fusion (gris neutro)
- **Gráficos**: Estilo Matplotlib por defecto con grid
- **Fuentes**: Sistema por defecto (Segoe UI en Windows)
- **Layout**: QSplitter para paneles redimensionables

### Compatibilidad
- **Python**: 3.8+ (recomendado 3.11 o 3.12)
- **PyQt**: PyQt6 (Qt 6.x)
- **Matplotlib**: Backend Qt5Agg (compatible con PyQt6)
- **Sistema Operativo**: Windows, Linux, macOS (testeado en Windows 11)
- **Resolución**: Mínimo 1366×768, recomendado 1920×1080

## Comparación: GUI vs Core Local

| Aspecto | GUI (SIMO/) | Core Local (core/) |
|---------|-------------|-------------------|
| **Propósito** | Interfaz gráfica interactiva | Motor de simulación |
| **Framework** | PyQt6 | NumPy + Python puro |
| **Arquitecturas** | SISO y SIMO (1/2/4/8 RX) | SISO y SIMO configurable |
| **Visualización** | Matplotlib embebido | Sin gráficos (retorna datos) |
| **Paralelismo** | ThreadPoolExecutor en SIMO | Opcional via parámetro `parallel` |
| **Configuración** | Controles GUI interactivos | LTEConfig programático |
| **Entrada** | Carga imágenes (JPG/PNG) | Arrays de bits directos |
| **Salida** | Gráficos + Imágenes + Métricas | Diccionarios con resultados |
| **Uso típico** | Exploración interactiva | Scripts batch, testing |
| **Velocidad** | ~75% del core (overhead Qt) | 100% (sin overhead GUI) |

### Ventajas de la GUI
✅ Interactiva y visual  
✅ No requiere programación para usar  
✅ Comparaciones visuales inmediatas  
✅ Ideal para demos y enseñanza  
✅ Progress feedback en tiempo real

### Ventajas del Core
✅ Máxima velocidad de ejecución  
✅ Scriptable y automatizable  
✅ Flexible para experimentación avanzada  
✅ Fácil integración en pipelines  
✅ Sin dependencias GUI

## Mejoras y Correcciones Implementadas

### Limpieza de Código (Sesión de Debug)
1. ✅ **Eliminación de duplicados**: Removidos 133 líneas duplicadas fuera de clases
2. ✅ **Corrección de corrupciones**: Limpieza de artefactos XML/markdown en archivos
3. ✅ **Validación de sintaxis**: Todos los archivos compilan sin errores

### Correcciones de Bugs
1. ✅ **TypeError en plot_constellation**: Agregado `np.array()` antes de indexación
2. ✅ **AttributeError set_itu_profile**: Removidas llamadas a métodos inexistentes
3. ✅ **Reconstrucción de imagen**: Validación de longitud de bits TX vs RX
4. ✅ **Visualización multiantena**: Corregida de 2×5 (desalineada) a 2×4 (correcta)

### Mejoras de Funcionalidad
1. ✅ **Panel de métricas simplificado**: De 6 métricas a 3 esenciales (BER, PAPR, Tiempo)
2. ✅ **Tab dedicada "Multiantena"**: Comparación visual lado a lado
3. ✅ **Paralelismo explícito**: `parallel=True` en todas las llamadas SIMO
4. ✅ **Debug comprehensivo**: Rastreo completo de bits a través del pipeline

### Mejoras de UX
1. ✅ **Callbacks dinámicos**: Actualización en tiempo real de frecuencia Doppler
2. ✅ **Validación robusta**: Checks antes de ejecutar simulaciones
3. ✅ **Feedback visual mejorado**: Status bar y progress bar informativos
4. ✅ **Mensajes de error detallados**: Traceback completo para debugging

### Optimizaciones de Rendimiento
1. ✅ **ThreadPoolExecutor**: Procesamiento paralelo en SIMO multi-RX
2. ✅ **Paralelismo habilitado por defecto**: Ya no es necesario activarlo manualmente
3. ✅ **Workers optimizados**: SimulationWorker y MultiantennaWorker eficientes

### Correcciones de Visualización
1. ✅ **Layout 2×4**: Coincide exactamente con test_simo_image.py
2. ✅ **Títulos informativos**: Muestran BER y número de receptores
3. ✅ **Fila superior**: Original repetido 4 veces para comparación directa
4. ✅ **Fila inferior**: Reconstrucciones con 1, 2, 4, 8 RX en orden

## Requisitos y Dependencias

### Python y Librerías
- **Python**: 3.8+ (desarrollado y testeado en 3.11 y 3.12)
- **PyQt6**: Framework GUI (Qt 6.x)
- **Matplotlib**: Visualización de gráficos (backend Qt5Agg)
- **NumPy**: Operaciones numéricas y arrays
- **Pillow (PIL)**: Carga y procesamiento de imágenes
- **scikit-image**: Métricas SSIM (opcional, no usado actualmente en GUI)

### Instalación de Dependencias

```powershell
# Método 1: Desde requirements.txt del proyecto
pip install -r requirements.txt

# Método 2: Manual (mínimas para GUI)
pip install PyQt6 matplotlib numpy Pillow
```

### Verificación de Instalación

```python
# Test rápido en Python
python -c "import PyQt6; import matplotlib; import numpy; print('✓ OK')"
```

## Estructura del Proyecto Completo

```
OFDM-LTE/
├── config.py                    # Configuración LTE global
├── requirements.txt             # Dependencias
├── README.md                    # Documentación principal
│
├── core/                        # Motor de simulación
│   ├── __init__.py
│   ├── ofdm_core.py            # Simulador SISO/SIMO principal
│   ├── modulator.py            # Modulación OFDM
│   ├── demodulator.py          # Demodulación OFDM
│   ├── channel.py              # Simulador de canal
│   ├── itu_r_m1225.py          # Canales ITU-R M.1225
│   ├── resource_mapper.py      # Mapeo de recursos
│   └── dft_precoding.py        # Pre-codificación DFT
│
├── utils/                       # Utilidades
│   └── image_processing.py     # Conversión imagen-bits
│
├── test/                        # Scripts de prueba
│   ├── test_basic.py           # Test básico SISO
│   └── test_simo_image.py      # Test SIMO con imagen
│
└── SIMO/                        # 🎯 GUI (este módulo)
    ├── __init__.py
    ├── main.py                 # Punto de entrada
    ├── README.md               # Esta documentación
    └── gui/
        ├── __init__.py
        ├── main_window.py      # Ventana principal + Workers
        └── widgets.py          # Widgets personalizados
```

## Problemas Conocidos y Limitaciones

### Rendimiento
- **Velocidad GUI**: ~25-45% más lenta que scripts por overhead de Qt
- **Barridos largos**: SNR con muchos puntos puede tardar varios minutos
- **Imágenes grandes**: >1000×1000 px pueden causar lentitud significativa

### Funcionalidad
- **Cancelar simulación**: No implementado (el botón no existe)
- **PSNR/SSIM**: Calculados en core pero no mostrados en GUI actualmente
- **Guardar resultados**: No hay export automático de métricas a CSV/Excel
- **Histórico**: No se guarda historial de simulaciones previas

### Visualización
- **Zoom en imágenes**: Toolbar de Matplotlib no es ideal para imágenes grandes
- **Colores**: Paleta fija, no personalizable por el usuario
- **Export**: Guardar figuras requiere usar toolbar de Matplotlib

### Compatibilidad
- **macOS**: No testeado exhaustivamente (debería funcionar)
- **Linux**: No testeado exhaustivamente (debería funcionar)
- **Pantallas pequeñas**: <1366×768 puede requerir scroll

## Trabajo Futuro y Mejoras Potenciales

### Funcionalidad
- [ ] Botón "Cancelar" para abortar simulaciones en progreso
- [ ] Export de resultados a CSV/Excel/JSON
- [ ] Historial de simulaciones con comparación
- [ ] Guardado/carga de configuraciones predefinidas
- [ ] Soporte para múltiples imágenes en batch

### Visualización
- [ ] Mostrar PSNR y SSIM en comparación de imágenes
- [ ] Gráficos de espectro de frecuencia
- [ ] Animaciones de efecto Doppler
- [ ] Visualización 3D de canal multi-trayecto
- [ ] Tema oscuro/claro personalizable

### Rendimiento
- [ ] Caché de simulaciones repetidas
- [ ] Optimización de conversión imagen-bits
- [ ] Procesamiento GPU para modulación/demodulación
- [ ] Progress bar más granular (por subportadora)

### Usabilidad
- [ ] Tooltips explicativos en todos los controles
- [ ] Wizard de configuración para principiantes
- [ ] Presets predefinidos (LTE Cat-1, Cat-3, etc.)
- [ ] Ayuda integrada con ejemplos
- [ ] Logs de debug en ventana separada (no solo consola)

## Preguntas Frecuentes (FAQ)

**P: ¿Por qué la GUI es más lenta que el script test_simo_image.py?**  
R: Es normal. La GUI tiene overhead de Qt (event loop, signals), renderizado en tiempo real, y callbacks de progreso. Típicamente 25-45% más lenta.

**P: ¿Cómo sé si el paralelismo está funcionando?**  
R: Revisa la consola. Deberías ver mensajes "[DEBUG] Paralelismo: ✓ Habilitado" para simulaciones con 2+ RX.

**P: ¿Por qué la imagen reconstruida tiene líneas horizontales?**  
R: Generalmente indica mismatch en longitud de bits. Revisa debug: bits TX vs RX deben coincidir exactamente.

**P: ¿Qué perfil ITU debo usar?**  
R: Depende del escenario:
- **Pedestrian A/B**: Peatones a baja velocidad (~3-5 km/h)
- **Vehicular A/B**: Vehículos urbanos (~30-60 km/h)
- **Typical Urban**: Ciudad genérica
- **Rural Area**: Campo abierto con pocos obstáculos

**P: ¿Cuántas antenas RX debo usar?**  
R: Más antenas = mejor BER, pero más tiempo de cómputo. Prueba con 2 o 4 para balance entre rendimiento y velocidad.

**P: ¿La GUI usa el mismo core que los scripts de test?**  
R: Sí, exactamente el mismo. La GUI es solo una interfaz sobre `core/ofdm_core.py`.

**P: ¿Puedo usar mis propias imágenes?**  
R: Sí, cualquier JPG o PNG. Recomendado: 450×600 px o similar (no muy grande para evitar lentitud).

**P: ¿Qué significa BER = 0.001?**  
R: 0.1% de los bits tienen error. Para imágenes, BER < 0.01 suele dar buena calidad visual.

## Recursos Adicionales

### Documentación Relacionada
- **Proyecto raíz**: `README.md` en directorio principal
- **Core OFDM**: Comentarios en `core/ofdm_core.py`
- **Tests**: Scripts en `test/` con ejemplos de uso

### Referencias Técnicas
- **LTE Standard**: 3GPP TS 36.211 (Physical channels and modulation)
- **ITU-R M.1225**: Guidelines for evaluation of radio transmission technologies
- **OFDM**: "OFDM Baseband Receiver Design for Wireless Communications" (Chiueh & Tsai)

### Contacto y Contribuciones
- **Repositorio**: GitHub Darioxavierl/OFDM-LTE
- **Issues**: Reportar bugs o sugerencias en GitHub Issues
- **Contribuciones**: Pull requests bienvenidos

---

## Autor y Versión

**Proyecto**: OFDM-LTE SIMO Module  
**Versión**: 2.0.0 (Actualizado: Enero 2026)  
**Framework**: PyQt6 + Matplotlib  
**Core**: OFDM-LTE Local Engine  
**Licencia**: (Especificar según proyecto)

---

## Changelog

### v2.0.0 (Enero 2026)
- ✅ Soporte SIMO completo (1/2/4/8 antenas RX)
- ✅ Tab dedicada "Multiantena" con comparación 2×4
- ✅ Paralelismo habilitado por defecto
- ✅ Debug comprehensivo de bits TX/RX
- ✅ Corrección de bugs mayores (TypeError, AttributeError)
- ✅ Simplificación de métricas (6 → 3)
- ✅ Limpieza de código (eliminados 133 líneas duplicadas)
- ✅ Documentación completa actualizada

### v1.0.0 (Versión inicial)
- ✅ GUI básica con 3 paneles
- ✅ Simulación SISO
- ✅ Barrido de SNR
- ✅ Visualización de constelación e imágenes
- ✅ Integración con core local
