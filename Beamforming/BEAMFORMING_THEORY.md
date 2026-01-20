# Closed-Loop Beamforming (TM6) - Implementación LTE

## Introducción

Este documento describe la implementación de **Closed-Loop Beamforming** (Transmission Mode 6, TM6) según el estándar LTE 3GPP. Este modo de transmisión concentra la energía de múltiples antenas en la dirección del receptor, maximizando la potencia recibida y mejorando el SNR efectivo.

A diferencia de **Spatial Multiplexing (TM4)** que transmite múltiples flujos de datos paralelos, TM6 transmite **un solo flujo** pero con mayor robustez gracias a la ganancia de beamforming.

## 1. Conceptos Fundamentales

### 1.1 ¿Qué es Beamforming?

**Beamforming** es una técnica de procesamiento de señales que enfoca la transmisión en una dirección específica ajustando la **fase y amplitud** de las señales en cada antena del array.

**Analogía física:**
Imagina lanzar varias piedras al agua simultáneamente en diferentes puntos. Si controlas el timing exacto de cada lanzamiento, puedes hacer que las ondas lleguen **en fase** a un punto específico, creando una ola grande (interferencia constructiva). En otras direcciones, las ondas se cancelan parcialmente (interferencia destructiva).

**Ventajas sobre transmisión uniforme:**
- ✅ **Mayor SNR**: Concentra energía donde está el receptor
- ✅ **Mayor alcance**: Celdas más grandes o menor potencia TX
- ✅ **Interferencia reducida**: Menos energía hacia otros usuarios
- ✅ **Robustez**: Mejor para un flujo que múltiples flujos débiles

**Desventajas:**
- ❌ **Single stream**: No multiplica throughput como TM4
- ❌ **Requiere CSI**: Necesita conocer el canal (feedback)
- ❌ **Overhead**: PMI feedback consume recursos uplink

### 1.2 Rank-1 en TM6

TM6 es un modo **rank-1** (single-layer):
```
rank = 1 → Una sola capa espacial → Un flujo de datos
```

**Diferencia con TM4:**
- **TM4 adaptativo**: rank ∈ {1, 2, 3, 4} según canal
- **TM6 fijo**: rank = 1 siempre

**¿Cuándo usar TM6?**
- SNR bajo/medio (< 15 dB): Robustez > throughput
- Canales correlacionados (LoS dominante)
- Bordes de celda
- Usuarios de baja movilidad con feedback confiable

**¿Cuándo usar TM4?**
- SNR alto (> 20 dB): Aprovechar multiplexación espacial
- Canales no correlacionados (rich scattering)
- Centro de celda
- Maximizar throughput

### 1.3 CSI Feedback en TM6

El proceso de **Channel State Information (CSI) feedback** es crucial para beamforming:

**Flujo completo:**
```
[1] eNB transmite CRS (Cell-specific Reference Signals)
      ↓
[2] UE estima canal H usando CRS
      ↓
[3] UE busca mejor precoder W del codebook
      ↓
[4] UE calcula PMI/CQI
      ↓
[5] UE envía PMI/CQI al eNB (PUCCH/PUSCH)
      ↓ (delay típico: 4-8 ms)
[6] eNB aplica precoder W[PMI] en transmisión DL
```

**Componentes del feedback:**

1. **PMI (Precoding Matrix Indicator)**: Índice del precoder óptimo
   - Rango: 0 a N-1 (N = tamaño del codebook)
   - TM6 con 2 TX: N = 4
   - TM6 con 4 TX: N = 16
   
2. **CQI (Channel Quality Indicator)**: Calidad del canal post-precoding
   - Rango: 0-15 (16 niveles)
   - Determina el MCS (Modulation and Coding Scheme)
   
3. **RI (Rank Indicator)**: No aplica en TM6 (siempre rank=1)

### 1.4 Perfect CSI vs Realistic CSI

**En nuestra implementación (Perfect CSI):**
```python
# TX conoce H exactamente
W_optimal = f(H_true)
# Sin delay, sin error
```

**En LTE real (Realistic CSI):**
```python
# UE estima H con error
H_estimated = H_true + error
# UE cuantiza usando codebook
PMI = quantize(H_estimated)
# TX recibe PMI con delay (4-8 ms)
W = Codebook[PMI_delayed]
# Canal puede haber cambiado
```

**Impacto del CSI imperfecto:**
- Error de estimación: -1 a -3 dB de pérdida
- Delay de feedback: Pérdida mayor con alta movilidad
- Cuantización del codebook: Pérdida ~0.5-2 dB

## 2. Arquitectura del Transmisor (eNB)

### 2.1 Flujo de Transmisión

```
Bits → QAM Modulation → Beamforming Precoding → Resource Mapping → IFFT → Canal MIMO
      [N_data × 6 bits]  [1 layer] [N_TX ports]  [OFDM grid]       [Time]
```

### 2.2 Modulación QAM

Idéntico a TM4. Los bits se convierten en símbolos QAM:
```
bits_per_OFDM = N_subcarriers_data × bits_per_symbol
```

Para 64-QAM con 249 subportadoras (5 MHz):
```
bits_per_OFDM = 249 × 6 = 1494 bits
```

### 2.3 Selección del Precoder

**Método 1: Basado en feedback PMI (Codebook)**

El transmisor recibe PMI del receptor y selecciona:
```python
W = Codebook[PMI]  # Vector [N_TX × 1]
```

**Método 2: MRT - Maximum Ratio Transmission**

Si el transmisor conoce **H** perfectamente (no realista pero útil para análisis):

```
W_MRT = H* / ||H||
```

donde **H*** es el **conjugado transpuesto** del canal.

**Justificación matemática:**

La señal recibida es:
```
y = H × (W × s) + n
```

Para maximizar SNR, queremos maximizar:
```
SNR = |H × W|² / σ²
```

El máximo se alcanza cuando **W** apunta en la dirección de **H***, es decir:
```
W_optimal ∝ H*
```

**Normalización:**
```
W_MRT = H* / √(H^H × H) = H* / ||H||
```

Esto asegura que la potencia transmitida sea constante: `||W|| = 1`

**Método 3: Eigenbeamforming**

Para canales de múltiples receptores, se usa el **eigenvector dominante**:

```
[1] Calcular matriz Gram: G = H^H × H
[2] Eigenvalue decomposition: G = V × Λ × V^H
[3] Seleccionar eigenvector con mayor eigenvalue:
    W_eigen = v_max / ||v_max||
```

Este método maximiza la **capacidad** del canal.

### 2.4 Aplicación del Precoding

**Operación:**
```
x = W ⊗ s
```

donde:
- **s**: Símbolos QAM de una sola capa [N_symbols]
- **W**: Vector de precoding [N_TX × 1]
- **x**: Señales para antenas TX [N_TX × N_symbols]

**Implementación por elemento:**
```
x₁[k] = W₁ × s[k]
x₂[k] = W₂ × s[k]
x₃[k] = W₃ × s[k]
...
x_N[k] = W_N × s[k]
```

Cada antena transmite el **mismo símbolo** pero con **fase y amplitud diferentes** según W.

**Ejemplo numérico (2 TX):**

```python
s = [s₀, s₁, s₂, ...]  # Símbolos QAM
W = [0.707 + 0j   ]    # Precoder (ej: coherent sum)
    [0.707 + 0j   ]

# Precoding:
x₁ = 0.707 × [s₀, s₁, s₂, ...]
x₂ = 0.707 × [s₀, s₁, s₂, ...]

# Las dos antenas transmiten en fase
# → Interferencia constructiva en el receptor
```

### 2.5 Ganancia de Beamforming

La **ganancia de beamforming** cuantifica la mejora de SNR:

```
Gain_BF = ||H × W||² / (||H||²_F / N_TX)
```

En dB:
```
Gain_BF (dB) = 10 × log₁₀(||H × W||² × N_TX / ||H||²_F)
```

**Interpretación:**
- **Sin beamforming**: Potencia se distribuye uniformemente → `1/N_TX` por antena
- **Con beamforming**: Potencia se concentra hacia el receptor → ganancia > 1

**Valores típicos:**
- **2 TX**: 2-3 dB de ganancia
- **4 TX**: 4-6 dB de ganancia
- **8 TX**: 6-9 dB de ganancia

La ganancia depende de la **correlación del canal**:
- Canal **no correlacionado** (scattering rico): menor ganancia
- Canal **correlacionado** (LoS): mayor ganancia (mejor para beamforming)

### 2.6 Actualización Adaptativa del Precoder

En canales **time-varying** (movilidad), el precoder debe actualizarse periódicamente.

**Coherence Time:**

Es el tiempo durante el cual el canal permanece **aproximadamente constante**:

```
T_c ≈ 1 / f_D
```

donde **f_D** es el **Doppler spread máximo**:

```
f_D = v × f_c / c
```

- **v**: Velocidad del UE (m/s)
- **f_c**: Frecuencia portadora (Hz)
- **c**: Velocidad de la luz (3×10⁸ m/s)

**Ejemplo:**
```
v = 3 km/h = 0.83 m/s
f_c = 2 GHz = 2×10⁹ Hz
c = 3×10⁸ m/s

f_D = (0.83 × 2×10⁹) / (3×10⁸) = 5.5 Hz
T_c ≈ 1 / 5.5 = 182 ms
```

**Regla de actualización:**
```
T_update = 0.1 × T_c  (conservador, 10% del coherence time)
```

Para el ejemplo anterior:
```
T_update = 18.2 ms ≈ 18 slots LTE
```

**En símbolos OFDM:**
```
N_symbols_update = T_update / T_symbol

Para 15 kHz spacing: T_symbol ≈ 67 μs
N_symbols_update = 18.2 ms / 67 μs ≈ 272 símbolos
```

La implementación actualiza W cada **N_symbols_update** símbolos.

## 3. Codebook LTE para TM6

### 3.1 Diseño del Codebook

El codebook es un conjunto **finito de vectores de precoding** predefinidos:

```
Codebook_TM6 = {W₀, W₁, W₂, ..., W_{N-1}}
```

Cada vector es un **beamforming weight** de tamaño [N_TX × 1].

**Propiedades:**
- ✅ **Normalizado**: `||W_i|| = 1` (potencia constante)
- ✅ **Ortogonal/cuasi-ortogonal**: Minimiza interferencia
- ✅ **Cuantización uniforme**: Cubre el espacio de direcciones

### 3.2 Codebook para 2 Antenas TX

**TS 36.211 Table 6.3.4.2.3-2**

```python
W₀ = [1,  1 ]ᵀ / √2    # 0°   - Coherent sum
W₁ = [1, -1 ]ᵀ / √2    # 180° - Coherent difference  
W₂ = [1,  j ]ᵀ / √2    # 90°  - Quadrature lead
W₃ = [1, -j ]ᵀ / √2    # -90° - Quadrature lag
```

**Interpretación geométrica:**

- **W₀**: Ambas antenas en fase → Máxima ganancia en boresight
- **W₁**: Antenas en contrafase → Patrón nulo en boresight, máximo a los lados
- **W₂, W₃**: Rotaciones de fase → Patrones desplazados

**Ejemplo de selección:**

Si el canal es `H = [1, 0.8]` (segunda antena más débil), entonces:
```
|H × W₀|² = |1 + 0.8|² / 2 = 1.62  ← Mejor
|H × W₁|² = |1 - 0.8|² / 2 = 0.02
|H × W₂|² = |1 + 0.8j|² / 2 = 0.82
|H × W₃|² = |1 - 0.8j|² / 2 = 0.82

PMI = 0 (W₀ es el mejor)
```

### 3.3 Codebook para 4 Antenas TX

**TS 36.211 Table 6.3.4.2.3-2**

Se usan **16 vectores** basados en **DFT (Discrete Fourier Transform)**:

```python
W_i[n] = exp(j × 2π × i × n / 16) / 2

para i = 0, 1, ..., 15  (PMI)
    n = 0, 1, 2, 3      (antena)
```

**Ejemplo:**

```python
# PMI = 0: Todas en fase
W₀ = [1, 1, 1, 1]ᵀ / 2

# PMI = 4: Fase lineal (beamsteering)
W₄ = [1, j, -1, -j]ᵀ / 2

# PMI = 8: Alternancia de fase
W₈ = [1, -1, 1, -1]ᵀ / 2
```

**Cobertura angular:**

Los 16 vectores cubren **360°** de direcciones en pasos de **22.5°**:
```
θ_i = 2π × i / 16
```

Esto permite hacer **beamsteering** digital sin mover antenas físicamente.

### 3.4 Selección del Mejor PMI

El receptor (UE) prueba **todos** los precoders del codebook:

```python
best_PMI = argmax_i { Metric(H, W_i) }
```

**Métrica 1: Capacidad (recomendada)**

```
Metric = log₂(1 + SNR × ||H × W_i||²)
```

Simplificado (SNR constante):
```
Metric = ||H × W_i||²
```

**Métrica 2: SINR**

Para múltiples usuarios (no aplicable en TM6 single-user):
```
SINR = |H_desired × W_i|² / Σ|H_interferer × W_i|²
```

**Métrica 3: Frobenius norm**

```
Metric = ||H × W_i||_F
```

**Algoritmo completo:**

```python
def select_best_pmi(H_channel, codebook):
    best_pmi = 0
    best_metric = -inf
    
    for pmi, W in enumerate(codebook):
        # Canal efectivo
        H_eff = H_channel @ W
        
        # Métrica: potencia recibida
        metric = np.sum(np.abs(H_eff)**2)
        
        if metric > best_metric:
            best_metric = metric
            best_pmi = pmi
    
    return best_pmi
```

**Complejidad:** O(N × N_TX × N_RX) donde N = tamaño del codebook.

## 4. Arquitectura del Receptor (UE)

### 4.1 Recepción con Múltiples Antenas RX

**Importante:** TM6 soporta **múltiples antenas RX** aunque transmite un solo flujo.

**Configuraciones comunes:**
- **2×1**: 2 TX, 1 RX → Solo beamforming TX
- **2×2**: 2 TX, 2 RX → Beamforming + diversidad RX
- **4×2**: 4 TX, 2 RX → Mayor ganancia de beamforming + diversidad RX
- **4×4**: 4 TX, 4 RX → Máxima ganancia

### 4.2 Modelo del Canal Efectivo

**Sin beamforming (transmisión uniforme):**
```
y_i = Σ_j H[i,j] × x_j + n_i    para cada RX i
```

**Con beamforming:**
```
y_i = Σ_j H[i,j] × W[j] × s + n_i = (H_eff)_i × s + n_i
```

donde el **canal efectivo** es:
```
H_eff = H × W
```

Dimensiones:
- **H**: [N_RX × N_TX]
- **W**: [N_TX × 1]
- **H_eff**: [N_RX × 1] → un coeficiente complejo por RX

### 4.3 MRC - Maximum Ratio Combining

Para combinar las señales de múltiples antenas RX, se usa **MRC**:

**Receptor óptimo (MMSE con rank-1):**

```
ŝ = (H_eff^H × y) / (H_eff^H × H_eff + σ²)
```

**Simplificación para SNR alto (ZF):**
```
ŝ = (H_eff^H × y) / ||H_eff||²
```

**Implementación MRC por elemento:**

```python
# y: Señal recibida [N_RX]
# H_eff: Canal efectivo [N_RX]

# Suma ponderada por canal conjugado
s_est = 0
for i in range(N_RX):
    s_est += conj(H_eff[i]) × y[i]

# Normalización
power = sum(|H_eff[i]|² for i in range(N_RX))
s_est = s_est / power
```

**SNR post-MRC:**

```
SNR_MRC = (Σ_i |H_eff[i]|²) / σ²
```

**Ganancia de diversidad:**

Con N_RX antenas independientes:
```
SNR_MRC = N_RX × SNR_single  (caso ideal, canales idénticos)
```

En la práctica:
```
Ganancia ≈ 3 dB por cada duplicación de RX (2 RX: +3 dB, 4 RX: +6 dB)
```

### 4.4 Combinación de Beamforming y Diversidad

**TM6 con múltiples RX combina dos beneficios:**

1. **Ganancia de beamforming (TX):**
   ```
   G_BF ≈ 10 log₁₀(N_TX)  dB
   ```

2. **Ganancia de diversidad (RX):**
   ```
   G_diversity ≈ 10 log₁₀(N_RX)  dB
   ```

**Ganancia total aproximada:**
```
G_total ≈ G_BF + G_diversity = 10 log₁₀(N_TX × N_RX)  dB
```

**Ejemplo 4×2:**
```
G_BF ≈ 10 log₁₀(4) = 6 dB
G_diversity ≈ 10 log₁₀(2) = 3 dB
G_total ≈ 9 dB
```

**Comparación con transmisión isótropa:**

Sin beamforming ni diversidad, cada antena TX tiene potencia `P/N_TX`:
```
SNR_isotrópico = (P/N_TX) × |h|² / σ²
```

Con TM6:
```
SNR_TM6 = P × ||H × W||² / σ²
```

Ganancia:
```
Ganancia = SNR_TM6 / SNR_isotrópico ≈ N_TX × ||H × W||² / ||H||²_F
```

### 4.5 Demodulación QAM

Después del MRC, se obtiene un **flujo de símbolos equalizados**:
```
ŝ = [ŝ₀, ŝ₁, ŝ₂, ...]
```

**Hard decision:**
```
Para cada ŝ_i:
    1. Calcular distancia a todos los símbolos de la constelación
    2. Seleccionar el más cercano
    3. Mapear a bits correspondientes
```

Para 64-QAM: 6 bits por símbolo.

## 5. Estimación de Canal y CRS

### 5.1 Cell-Specific Reference Signals (CRS)

Los **CRS** son pilotos conocidos transmitidos periódicamente:

**Ubicación en grid OFDM:**
- Cada 3 subportadoras en frecuencia
- Cada símbolo 0 y 4 de cada slot (símbolos 0, 4, 7, 11 del subframe)

**Secuencia CRS:**
```
r[k] = (1/√2) × (1 - 2×c[2k]) + j×(1/√2) × (1 - 2×c[2k+1])
```

donde `c[k]` es una secuencia pseudo-aleatoria basada en:
- Cell ID
- Slot number
- OFDM symbol number

### 5.2 Estimación del Canal H

**Paso 1: Estimación en pilotos**

```
Ĥ[pilot] = Y[pilot] / R[pilot]
```

donde:
- **Y[pilot]**: Señal recibida en pilotos
- **R[pilot]**: Secuencia CRS conocida

**Paso 2: Interpolación**

Interpolar en frecuencia y tiempo para obtener H en todas las subportadoras:
- **Frecuencia**: Interpolación lineal o cúbica entre pilotos
- **Tiempo**: Filtro de Wiener o interpolación lineal

**Paso 3: Suavizado (opcional)**

Filtrado para reducir ruido de estimación:
```
Ĥ_smooth = filter(Ĥ_raw)
```

### 5.3 Cálculo de PMI/CQI

Una vez estimado **Ĥ**, el UE:

1. **Busca mejor PMI:**
   ```python
   PMI = argmax_i { ||Ĥ × W_i||² }
   ```

2. **Calcula SINR post-precoding:**
   ```python
   W_best = Codebook[PMI]
   H_eff = Ĥ × W_best
   SINR = ||H_eff||² / σ²
   ```

3. **Mapea SINR a CQI** (tabla TS 36.213):
   ```
   CQI ∈ {0, 1, ..., 15}
   ```

4. **Transmite feedback:** PMI + CQI por PUCCH/PUSCH

### 5.4 Errores de Estimación

**Fuentes de error:**

1. **Ruido en pilotos:**
   ```
   Ĥ = H + n/r  donde n ~ CN(0, σ²)
   ```

2. **Interferencia de otras celdas:**
   ```
   Y_pilot = H×r + Σ H_interferer×r_interferer + n
   ```

3. **Interpolación imperfecta:**
   - Entre pilotos espaciados
   - Variación rápida del canal

**Impacto en beamforming:**

Error de estimación → PMI subóptimo → pérdida de ganancia:
```
Loss ≈ σ²_error / ||H||²  (en lineal)
```

Típicamente: **1-3 dB de pérdida** por error de estimación.

## 6. Ecuaciones Clave Implementadas

### 6.1 Transmisión

**Precoding MRT:**
```
W_MRT = H* / ||H||
```

**Precoding codebook:**
```
W = Codebook[PMI]
```

**Señal transmitida:**
```
x = W ⊗ s
x[j] = W[j] × s    para j = 1, ..., N_TX
```

**Canal:**
```
y[i] = Σ_j H[i,j] × x[j] + n[i]
     = Σ_j H[i,j] × W[j] × s + n[i]
     = (H × W)[i] × s + n[i]
```

### 6.2 Recepción

**Canal efectivo:**
```
H_eff = H × W
```
Dimensiones: [N_RX × 1]

**MRC combining:**
```
ŝ = (H_eff^H × y) / (H_eff^H × H_eff + σ²)
```

**Forma simplificada (SNR alto):**
```
ŝ = (Σ_i conj(H_eff[i]) × y[i]) / (Σ_i |H_eff[i]|²)
```

**SNR post-processing:**
```
SNR_out = ||H_eff||² / σ²
        = (Σ_i |H_eff[i]|²) / σ²
```

### 6.3 Métricas

**Ganancia de beamforming:**
```
G_BF (dB) = 10 × log₁₀(||H × W||² / (||H||²_F / N_TX))
```

**Ganancia de diversidad:**
```
G_div (dB) = 10 × log₁₀(N_RX)  (aproximado)
```

**Capacidad del canal:**
```
C = log₂(1 + SNR × ||H × W||²)  bits/s/Hz
```

**CQI calculation:**
```
SINR_dB = 10 × log₁₀(||H × W||² / σ²)
CQI = map_SINR_to_CQI(SINR_dB)
```

### 6.4 Adaptación Temporal

**Doppler shift:**
```
f_D = v × f_c / c
```

**Coherence time:**
```
T_c ≈ 9 / (16π × f_D)  (criterio del 90%)
```

**Período de actualización:**
```
T_update = α × T_c  donde α ∈ [0.05, 0.2]
```

**En símbolos OFDM:**
```
N_update = ⌊T_update / T_symbol⌋
```

## 7. Casos de Uso y Resultados Esperados

### 7.1 Configuración 2×1

**Descripción:** 2 TX, 1 RX

**Ganancia esperada:**
- Beamforming: ~3 dB
- No hay diversidad RX

**Performance (SNR=15 dB, Rayleigh):**
- Sin BF: BER ≈ 5-10%
- Con BF: BER ≈ 1-2%  
- PSNR: ~22-25 dB

**Aplicación:** Dispositivos simples (IoT, wearables)

### 7.2 Configuración 2×2

**Descripción:** 2 TX, 2 RX

**Ganancia esperada:**
- Beamforming: ~3 dB
- Diversidad RX: ~3 dB
- **Total: ~6 dB**

**Performance (SNR=15 dB):**
- BER ≈ 0.1-0.5%
- PSNR: ~28-32 dB

**Aplicación:** Smartphones básicos, configuración estándar

### 7.3 Configuración 4×2

**Descripción:** 4 TX, 2 RX

**Ganancia esperada:**
- Beamforming: ~6 dB
- Diversidad RX: ~3 dB
- **Total: ~9 dB**

**Performance (SNR=15 dB):**
- BER ≈ 0.01-0.1%
- PSNR: ~35-40 dB

**Performance (SNR=10 dB):**
- BER ≈ 1-3%
- PSNR: ~24-28 dB

**Aplicación:** Smartphones premium, tablets

### 7.4 Configuración 4×4

**Descripción:** 4 TX, 4 RX

**Ganancia esperada:**
- Beamforming: ~6 dB
- Diversidad RX: ~6 dB
- **Total: ~12 dB**

**Performance (SNR=10 dB):**
- BER ≈ 0.1-0.5%
- PSNR: ~30-35 dB

**Performance (SNR=15 dB):**
- BER ≈ 0.001-0.01%
- PSNR: ~40-50 dB

**Aplicación:** Devices premium, fixed wireless access

### 7.5 Comparación TM6 vs TM4

**Escenario: 4×2, SNR=15 dB, Rayleigh**

| Métrica | TM6 (rank-1) | TM4 (rank-2) | TM4 (rank-1) |
|---------|--------------|--------------|--------------|
| **Throughput** | 1× | ~1.8× | 1× |
| **BER** | 0.05% | 0.5% | 0.03% |
| **PSNR** | 38 dB | 28 dB | 40 dB |
| **Robustez** | ✅✅ Alta | ⚠️ Media | ✅✅ Alta |
| **Edge performance** | ✅ Excelente | ❌ Pobre | ✅ Excelente |

**Conclusión:**
- **SNR bajo (< 15 dB)**: TM6 es superior (robustez)
- **SNR medio (15-20 dB)**: Empate (depende de prioridad)
- **SNR alto (> 20 dB)**: TM4 es superior (throughput)

## 8. Ventajas y Limitaciones

### 8.1 Ventajas de TM6

✅ **Robustez superior**
- Un solo flujo es más fácil de decodificar
- Menor sensibilidad a error de estimación de canal
- Funciona bien en canales correlacionados

✅ **Mayor cobertura**
- Ganancia de beamforming extiende el alcance
- Permite reducir potencia TX manteniendo QoS

✅ **Menor complejidad del receptor**
- No requiere separación de múltiples flujos
- Detector MRC simple (lineal)

✅ **Compatible con múltiples RX**
- Suma ganancia de beamforming + diversidad
- Escalable (2, 4, 8 RX)

### 8.2 Limitaciones de TM6

❌ **Throughput limitado**
- Rank-1 solo: No multiplica capacidad
- TM4 puede dar 2-4× más throughput en buen canal

❌ **Overhead de feedback**
- PMI/CQI consumen recursos uplink
- Período de feedback afecta latencia

❌ **Sensible a movilidad**
- CSI se vuelve obsoleto rápidamente con alta velocidad
- Beamforming desalineado → pérdida de ganancia

❌ **Requiere codebook**
- Cuantización causa pérdida (~0.5-2 dB)
- Codebook fijo no se adapta perfectamente

### 8.3 Factores que Afectan Performance

**Positivos:**
- ✅ Baja movilidad (< 30 km/h)
- ✅ Canal correlacionado (LoS, poca dispersión)
- ✅ Múltiples antenas RX (2, 4)
- ✅ Feedback frecuente y preciso

**Negativos:**
- ❌ Alta movilidad (> 60 km/h)
- ❌ Canal no correlacionado (rich scattering)
- ❌ Delay de feedback grande (> 10 ms)
- ❌ Error de estimación de canal alto

## 9. Diferencias entre Implementación y LTE Real

### 9.1 Simplificaciones

**En la implementación:**
- ✅ Perfect CSI (H conocido exactamente)
- ✅ Sin delay de feedback
- ✅ Sin error de cuantización PMI
- ✅ Sin codificación de canal (FEC)

**En LTE real:**
- ⚠️ H estimado con error (CRS)
- ⚠️ Delay de feedback: 4-8 ms
- ⚠️ PMI cuantizado (codebook finito)
- ⚠️ Turbo coding (rate 1/3)

**Impacto:**
```
Loss_total = Loss_estimation + Loss_delay + Loss_quantization + Loss_coding
           ≈ 2 dB          + 1-3 dB    + 0.5 dB         + (ganancia de -5 dB)

Net effect: ~1-3 dB peor que implementación ideal
```

### 9.2 Características No Implementadas

❌ **HARQ (Hybrid ARQ)**
- Retransmisiones automáticas
- Soft combining

❌ **Scheduling dinámico**
- Asignación de PRBs según CQI
- Multi-usuario

❌ **Power control**
- Ajuste de potencia según CQI
- Ahorro de batería

❌ **Handover**
- Cambio de celda durante transmisión
- Actualización de PMI durante handover

## 10. Clarificaciones Conceptuales

### 10.1 ¿Beamforming es lo mismo que Precoding?

**Respuesta:** Sí y no.

**Precoding** es el término general para aplicar una matriz W:
```
x = W × s
```

**Beamforming** es un tipo específico de precoding donde:
- W forma un "beam" (haz) directional
- Objetivo: Maximizar SNR en una dirección
- Típicamente rank-1 (single beam)

**Diferencia:**
- TM6: **Beamforming** (rank-1, un beam)
- TM4: **Precoding** general (rank-1/2/3/4, puede no ser beamforming puro)

### 10.2 ¿Por qué usar Codebook en lugar de H* directamente?

**Ventajas del codebook:**

1. **Feedback eficiente:** Solo enviar índice PMI (4-6 bits) en lugar de matriz completa
2. **Cuantización:** Reduce sensibilidad a error de estimación
3. **Estandarización:** Todos los fabricantes usan los mismos precoders

**Desventaja:**
- Pérdida por cuantización (~0.5-2 dB)

**MRT (H*) es óptimo** solo si:
- TX conoce H perfectamente
- Sin restricción de feedback
- No usado en LTE real (solo análisis)

### 10.3 ¿Múltiples RX convierte TM6 en TM4?

**NO.** Son conceptos independientes:

**TM6 con 4 RX:**
- Sigue siendo **rank-1** (un flujo de datos)
- 4 RX se usan para **diversidad** (combinar señal)
- **MRC** en recepción (no detección multi-capa)

**TM4 con 4 RX:**
- **Rank puede ser 1-4** (múltiples flujos)
- 4 RX se usan para **separar capas espaciales**
- **MMSE/SIC** en recepción (detector multi-capa)

**Comparación:**
```
TM6 4×4: 4 TX beamforming → 1 flujo → 4 RX MRC
TM4 4×4: 4 TX spatial mux → 4 flujos → 4 RX MIMO detection
```

### 10.4 ¿Adaptive Beamforming cambia el PMI?

**Sí y no:**

**Adaptive Beamforming:**
- Actualiza **W_MRT** periódicamente según coherence time
- No usa codebook (W calculado directamente de H)
- Es una aproximación de "perfect CSI" adaptativa

**PMI-based Beamforming:**
- Actualiza **PMI** según feedback del UE
- Usa codebook fijo
- Más realista (como LTE real)

**Nuestra implementación:**
- Soporta ambos modos
- `update_mode='adaptive'`: MRT adaptativo
- `update_mode='static'`: PMI fijo (un solo feedback)

## 11. Referencias

### 11.1 Estándares 3GPP

- **TS 36.211 Section 6.3.4.2.3**: Codebook for antenna ports 2/4
- **TS 36.213 Section 7.2**: UE reporting of CSI
- **TS 36.213 Table 7.2.3-1**: CQI indices
- **TS 36.104**: Base station radio transmission and reception

### 11.2 Literatura Académica

1. **MIMO-OFDM Wireless Communications** - Yong Soo Cho et al.
   - Capítulo 8: Beamforming techniques
   
2. **LTE - The UMTS Long Term Evolution** - Stefania Sesia et al.
   - Capítulo 11: Downlink transmission modes
   
3. **Introduction to MIMO Communications** - Jerry Hampton
   - Capítulo 5: Transmit beamforming

### 11.3 Ecuaciones Resumidas

**Beamforming weight (MRT):**
```
W = H* / ||H||
```

**Canal efectivo:**
```
H_eff = H × W  [N_RX × 1]
```

**MRC combining:**
```
ŝ = (H_eff^H × y) / ||H_eff||²
```

**Ganancia de beamforming:**
```
G_BF = 10 log₁₀(N_TX × ||H × W||² / ||H||²_F)  dB
```

**SNR post-MRC:**
```
SNR = ||H_eff||² / σ² = (Σ_i |H_eff[i]|²) / σ²
```

**Capacidad:**
```
C = log₂(1 + SNR × ||H × W||²)  bits/s/Hz
```

**Coherence time:**
```
T_c ≈ 1 / f_D  donde f_D = v×f_c / c
```

---

## Resumen Ejecutivo

Esta implementación de **Closed-Loop Beamforming (TM6)** sigue el estándar LTE 3GPP, incorporando:

✅ **Codebook LTE** (TS 36.211) para 2, 4, 8 antenas TX  
✅ **CSI Feedback** simulado con selección de PMI/CQI  
✅ **MRT** como referencia óptima  
✅ **Beamforming adaptativo** con actualización basada en coherence time  
✅ **MRC** en recepción para combinar múltiples RX  
✅ **Ganancia de beamforming + diversidad** en configuraciones multi-RX  
✅ **Perfect CSI** (límite superior teórico)  

La implementación demuestra los **beneficios clave de beamforming**:
- **Ganancia de SNR**: 3-9 dB según número de TX
- **Mayor cobertura**: Alcance extendido vs transmisión uniforme
- **Robustez**: Superior a TM4 en SNR bajo
- **Diversidad RX**: Beneficio adicional con múltiples RX

**Trade-off principal:** Robustez y ganancia vs throughput (rank-1 solo).

**Aplicabilidad:** Análisis de cobertura, validación de algoritmos de beamforming, educación en MIMO, prototipado de sistemas celulares.
