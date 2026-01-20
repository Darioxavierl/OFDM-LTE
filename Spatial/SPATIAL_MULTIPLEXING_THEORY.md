# Spatial Multiplexing (TM4) - Implementación LTE

## Introducción

Este documento describe la implementación del **Spatial Multiplexing** (Transmission Mode 4, TM4) según el estándar LTE 3GPP. Este modo de transmisión permite enviar múltiples flujos de datos independientes simultáneamente usando múltiples antenas, maximizando el throughput en condiciones de canal favorable.

## 1. Conceptos Fundamentales

### 1.1 ¿Qué es Spatial Multiplexing?

Spatial Multiplexing es una técnica MIMO que **divide los datos en múltiples capas espaciales** que se transmiten simultáneamente por diferentes antenas. A diferencia de técnicas de diversidad (como Alamouti), que envían los mismos datos por múltiples caminos para mejorar confiabilidad, Spatial Multiplexing **multiplica la capacidad** del sistema enviando datos diferentes en paralelo.

**Ventajas:**
- **Mayor throughput**: Transmite hasta `min(N_TX, N_RX)` flujos paralelos
- **Eficiencia espectral**: Más bits/Hz sin aumentar ancho de banda
- **Adaptación dinámica**: Ajusta número de capas según calidad del canal

**Requisitos:**
- Canal con suficiente scattering (no correlacionado)
- SNR adecuado (típicamente ≥ 15-20 dB)
- `N_RX ≥ N_layers` para detección exitosa

### 1.2 El Concepto de Rank (RI - Rank Indicator)

El **rank** representa el **número de capas espaciales** (flujos de datos independientes) que se transmiten simultáneamente. Es el parámetro clave que determina el throughput del sistema.

**Limitaciones:**
```
rank ≤ min(N_TX, N_RX, 4)
```

En LTE, el rank máximo es 4 (aunque el estándar define hasta 8 para configuraciones avanzadas).

**Ejemplos:**
- Sistema **2×2**: rank ∈ {1, 2}
- Sistema **4×2**: rank ∈ {1, 2} (limitado por N_RX)
- Sistema **4×4**: rank ∈ {1, 2, 3, 4}
- Sistema **8×4**: rank ∈ {1, 2, 3, 4} (limitado por estándar)

### 1.3 Channel State Information (CSI)

CSI es el conocimiento que tienen el transmisor (eNB) y receptor (UE) sobre el estado del canal radio. En TM4, el CSI incluye tres componentes:

1. **RI (Rank Indicator)**: Número óptimo de capas
2. **PMI (Precoding Matrix Indicator)**: Índice del precoder óptimo del codebook
3. **CQI (Channel Quality Indicator)**: Calidad del canal (SNR estimado)

**Flujo CSI en LTE:**
```
[UE mide canal H] → [Calcula RI/PMI óptimos] → [Feedback al eNB] → [eNB adapta transmisión]
```

En nuestra implementación, usamos **Perfect CSI**: el transmisor conoce exactamente la matriz de canal **H**, lo cual es ideal para análisis pero no realista (en práctica hay delay y errores de estimación).

## 2. Arquitectura del Transmisor

### 2.1 Flujo de Transmisión

```
Bits → QAM Modulation → Layer Mapping → Precoding → Resource Mapping → IFFT → Canal MIMO
      [N_data × 6 bits]  [Rank layers] [N_TX ports]  [OFDM grid]
```

### 2.2 Modulación QAM

Los bits se convierten en símbolos complejos QAM:
- **QPSK**: 2 bits/símbolo
- **16-QAM**: 4 bits/símbolo  
- **64-QAM**: 6 bits/símbolo

**Capacidad por símbolo OFDM:**
```
bits_per_OFDM = N_subcarriers_data × bits_per_symbol
```

Para 64-QAM con 249 subportadoras de datos (5 MHz):
```
bits_per_OFDM = 249 × 6 = 1494 bits
```

### 2.3 Layer Mapping

Distribuye los símbolos QAM entre las **rank** capas espaciales usando un esquema **round-robin**:

```
symbols = [s₀, s₁, s₂, s₃, s₄, s₅, ...]

Para rank = 2:
  Layer 0: [s₀, s₂, s₄, ...]
  Layer 1: [s₁, s₃, s₅, ...]

Para rank = 3:
  Layer 0: [s₀, s₃, s₆, ...]
  Layer 1: [s₁, s₄, s₇, ...]
  Layer 2: [s₂, s₅, s₈, ...]
```

**Matriz de capas:**
```
x = [x₀(k), x₁(k), ..., x_{rank-1}(k)]ᵀ
```
donde `k` es el índice de subportadora.

### 2.4 Precoding

El precoding aplica una matriz **W** del LTE codebook para:
- Adaptar las capas a las condiciones del canal
- Maximizar SINR en el receptor
- Minimizar interferencia entre capas

**Operación:**
```
x_precoded = W × x
```

donde:
- **W**: Matriz [N_TX × rank] del codebook LTE
- **x**: Símbolos por capa [rank × N_symbols]
- **x_precoded**: Señales para antenas TX [N_TX × N_symbols]

**Selección del precoder:**
El PMI (Precoding Matrix Indicator) selecciona la matriz W del codebook que maximiza la capacidad según el canal medido.

### 2.5 Resource Mapping

Cada antena TX tiene su propia grid OFDM con:
- **Símbolos de datos** en subportadoras específicas
- **Cell-specific Reference Signals (CRS)** para estimación de canal
- **Subportadoras de guarda** en los extremos
- **DC carrier** (subportadora central) nula

### 2.6 Transmisión por Bloques

Las imágenes/datos grandes se dividen en **bloques OFDM**:

```
Total_bits → Divide en bloques de 'bits_per_OFDM'
           → Cada bloque: independiente con su propio canal
           → N_bloques = ⌈Total_bits / bits_per_OFDM⌉
```

**Ventajas:**
- Simula canal time-varying realista
- Cada bloque experimenta un canal diferente (fading)
- Permite procesamiento eficiente de imágenes grandes

## 3. Canal MIMO

### 3.1 Modelo del Canal

La señal recibida en cada antena RX es la combinación de las señales de todas las antenas TX:

```
y = H × x + n
```

donde:
- **y**: Señal recibida [N_RX × 1]
- **H**: Matriz de canal [N_RX × N_TX]
- **x**: Señal transmitida [N_TX × 1]
- **n**: Ruido AWGN complejo [N_RX × 1]

**Elemento H(i,j):**
Representa el coeficiente complejo del canal desde la antena TX_j a la antena RX_i:
```
H(i,j) = α(i,j) × exp(jφ(i,j))
```
- **α**: Atenuación (Rayleigh fading)
- **φ**: Rotación de fase

### 3.2 Rayleigh Multipath Channel

Implementamos el modelo **ITU-R M.1225** con múltiples trayectorias:

**Pedestrian A** (baja velocidad, entorno urbano):
```
Delays: [0.0, 0.11, 0.19, 0.41] μs
Powers: [0.0, -9.7, -19.2, -22.8] dB
```

Cada trayectoria tiene:
- Delay específico
- Potencia relativa
- Fading Rayleigh independiente
- Doppler shift (si hay movilidad)

### 3.3 Perfect CSI

En nuestra implementación, el receptor conoce **exactamente** la matriz H:
```
H_perfect = H_actual (sin error de estimación)
```

Esto representa el **límite superior de performance** y simplifica el análisis. En sistemas reales, H se estima usando los CRS (Cell-specific Reference Signals).

## 4. Rank Adaptation

### 4.1 ¿Por qué adaptar el rank?

El rank óptimo depende de las condiciones del canal:
- **Canal bueno (SNR alto, baja correlación)**: Usar rank alto → más throughput
- **Canal malo (SNR bajo, alta correlación)**: Usar rank bajo → más robustez

Un rank demasiado alto en mal canal causa **interferencia entre capas** excesiva.

### 4.2 Método Eigenvalue

Calculamos el **Singular Value Decomposition (SVD)** de H:

```
H = U × Σ × Vᴴ
```

donde Σ contiene los **eigenvalues** (valores singulares) del canal:
```
σ₁² ≥ σ₂² ≥ σ₃² ≥ ... ≥ σₙ²
```

**Criterio de selección:**
```
rank = count(σᵢ² > threshold × σ₁²)
```

Los eigenvalues representan la **potencia** de cada modo espacial del canal. Solo contamos modos "fuertes".

**Threshold típico:** 0.15 (15% del eigenvalue máximo)

### 4.3 Consideración del SNR

El rank también se ajusta según SNR:

```python
if SNR < 10 dB:
    rank = 1  # Forzar rank bajo, modo conservador
elif SNR < 15 dB:
    rank = min(rank_eigenvalue, 2)  # Limitar a 2 capas
else:
    rank = rank_eigenvalue  # Usar cálculo completo
```

### 4.4 PMI Selection

Para el rank seleccionado, probamos todos los precoders del codebook LTE:

```
PMI = argmax_i { capacity(H, W_i) }
```

donde la capacidad de Shannon para MIMO es:

```
C = log₂ det(I + (SNR/rank) × H × W × Wᴴ × Hᴴ)
```

El precoder que maximiza C es el óptimo.

## 5. Arquitectura del Receptor

### 5.1 Demodulación OFDM

Cada antena RX:
1. Remove Cyclic Prefix
2. FFT → dominio frecuencia
3. Extrae símbolos de subportadoras de datos

Resultado: **y[N_RX, N_subcarriers]** en frecuencia

### 5.2 Canal Efectivo

Con precoding, el canal efectivo es:

```
H_eff = H × W
```

Dimensiones: [N_RX × rank]

Esto transforma el problema de separar N_TX señales en separar rank capas.

### 5.3 MIMO Detection

El objetivo es estimar las capas originales **x** dado:
- Señal recibida **y**
- Canal **H_eff**
- Varianza de ruido **σ²**

**Problema:**
```
y = H_eff × x + n
Estimar: x̂ dado (y, H_eff, σ²)
```

## 6. Detectores Implementados

### 6.1 MMSE (IRC) - Interference Rejection Combining

**Minimum Mean Square Error** minimiza el error cuadrático medio:

```
MSE = E[||x - x̂||²]
```

**Filtro MMSE:**
```
W_MMSE = (H_effᴴ × H_eff + σ² × I)⁻¹ × H_effᴴ
```

**Detección:**
```
x̂ = W_MMSE × y
```

**Características:**
- **Linear detector**: Complejidad O(rank³) por inversión matricial
- **Balance bias-variance**: Regularización σ²I evita noise amplification
- **Performance**: Mejor que ZF en SNR bajo/medio
- **Robustez**: Funciona bien con diferentes ranks

**También llamado IRC (Interference Rejection Combining)** en contextos LTE porque suprime interferencia inter-capa mientras considera el ruido.

### 6.2 SIC - Successive Interference Cancellation

**Non-linear detector** que detecta capas **secuencialmente**:

**Algoritmo:**
```
Para k = 1 hasta rank:
    1. Proyectar y en subespacio ortogonal a capas ya detectadas
    2. Detectar capa k (símbolo más cercano en constelación)
    3. Cancelar contribución de capa k: y ← y - H_k × x̂_k
```

**Ordenamiento óptimo:**
Detectar primero las capas con **mayor SNR post-detección** (mejor condición):

```
SNR_k = |h_k|² / σ²
```

donde h_k es la k-ésima columna de H_eff.

**Ecuación por capa:**

Para la capa k:
```
ỹ_k = y - Σ(j<k) H_j × x̂_j   (cancelar capas previas)

W_k = H_kᴴ / (H_kᴴ × H_k + σ²)   (proyector MMSE)

z_k = W_k × ỹ_k   (soft symbol)

x̂_k = quantize(z_k)   (hard decision)
```

**Características:**
- **Complejidad**: O(rank⁴) - más complejo que MMSE
- **Performance**: Mejor que MMSE/ZF en SNR medio/alto
- **Error propagation**: Si una capa se detecta mal, afecta las siguientes
- **Mejor en**: Configuraciones con rank alto (3-4 capas)

### 6.3 Comparación MMSE vs SIC

| Aspecto | MMSE (IRC) | SIC |
|---------|-----------|-----|
| **Tipo** | Linear | Non-linear |
| **Complejidad** | O(rank³) | O(rank⁴) |
| **SNR bajo** | ✅ Mejor | ⚠️ Propagación de errores |
| **SNR alto** | ✅ Bueno | ✅✅ Mejor |
| **Rank bajo (1-2)** | ✅ Similar | ✅ Similar |
| **Rank alto (3-4)** | ✅ Bueno | ✅✅ Mejor |
| **Robustez** | ✅✅ Muy robusto | ⚠️ Sensible a errores |

**En la práctica:**
- SNR < 20 dB: **MMSE** es más seguro
- SNR ≥ 25 dB: **SIC** aprovecha mejor el canal
- 4×4 rank=4: **SIC** puede dar hasta 1-2 dB de ganancia

## 7. Reconstrucción y Métricas

### 7.1 Layer Demapping

Proceso inverso del Layer Mapping:
```
[x̂₀, x̂₁, ..., x̂_{rank-1}] → [ŝ₀, ŝ₁, ŝ₂, ŝ₃, ...]
```

Reordena los símbolos detectados de todas las capas al orden original.

### 7.2 Demodulación QAM

Símbolos complejos → bits:
```
ŝ → [b̂₀, b̂₁, ..., b̂₅]   (para 64-QAM)
```

Usando **hard decision**: selecciona el símbolo más cercano de la constelación.

### 7.3 Bit Error Rate (BER)

```
BER = (Número de bits erróneos) / (Total de bits transmitidos)
```

**Rangos típicos según SNR (64-QAM, rank=2):**
- SNR = 15 dB: BER ≈ 1-5% (imagen reconocible con ruido)
- SNR = 20 dB: BER ≈ 0.1-1% (buena calidad)
- SNR = 25 dB: BER ≈ 0.01-0.1% (excelente calidad)
- SNR = 30 dB: BER ≈ 0.001% (casi perfecta)

### 7.4 Peak Signal-to-Noise Ratio (PSNR)

Para evaluación de calidad de imagen:

```
MSE = (1/N) × Σ(pixel_original - pixel_reconstruido)²

PSNR = 10 × log₁₀(MAX²/MSE)
```

donde MAX = 255 para imágenes 8-bit.

**Interpretación:**
- PSNR < 20 dB: Mala calidad
- PSNR = 20-30 dB: Calidad aceptable  
- PSNR = 30-40 dB: Buena calidad
- PSNR > 40 dB: Excelente calidad

## 8. Casos de Uso y Resultados Esperados

### 8.1 Configuración 2×2

**Rank típico:** 1-2

**Performance (SNR=20 dB):**
- Rank=1: BER ≈ 0.5%, PSNR ≈ 28 dB
- Rank=2: BER ≈ 1-2%, PSNR ≈ 25 dB

**Características:**
- Sistema balanceado, simple
- Buena robustez
- Throughput: 2× vs SISO

### 8.2 Configuración 4×2

**Rank típico:** 1-2 (limitado por N_RX)

**Performance (SNR=20 dB):**
- Rank=1: BER ≈ 0.3%, PSNR ≈ 30 dB (mejor que 2×2 por diversidad TX)

**Características:**
- **4 TX, solo 2 RX**: sistema subdeterminado
- Extra TXs ayudan con precoding pero no aumentan rank
- Requiere más SNR que 4×4 para mismo rank

### 8.3 Configuración 4×4

**Rank típico:** 2-4

**Performance (SNR=20 dB):**
- Rank=2: BER ≈ 0.005%, PSNR ≈ 45 dB ✅ **Excelente**
- Rank=4: BER ≈ 5-10% (necesita SNR más alto)

**Características:**
- Sistema **sobredeterminado** cuando rank < 4
- Mejor performance de todas las configs a SNR medio
- Throughput: hasta 4× vs SISO

### 8.4 Configuración 8×4

**Rank típico:** 2-3

**Performance (SNR=20 dB):**
- Rank=3: BER ≈ 5-8% ⚠️ (SNR insuficiente)

**Performance (SNR=30 dB):**
- Rank=4: BER ≈ 0.1%, PSNR ≈ 35 dB ✅

**Características:**
- Configuración más compleja
- **Necesita SNR ≥ 25 dB** para trabajar bien
- Mayor throughput potencial pero más demandante

## 9. Limitaciones y Consideraciones

### 9.1 Suposiciones de la Implementación

1. **Perfect CSI**: Conocimiento exacto de H (no realista)
2. **Sincronización perfecta**: No hay CFO (Carrier Frequency Offset)
3. **Canal flat-fading por subportadora**: Válido con OFDM y CP adecuado
4. **No hay codificación de canal**: No FEC (Forward Error Correction)

### 9.2 Diferencias con LTE Real

- **Real**: CSI feedback tiene delay (4-8 ms típico)
- **Real**: Canal se estima con error usando CRS
- **Real**: Incluye Turbo coding (rate 1/3)
- **Real**: HARQ (retransmisiones automáticas)
- **Real**: Scheduling dinámico de recursos

### 9.3 Factores que Afectan Performance

**Positivos:**
- ✅ Alto SNR
- ✅ Canal no correlacionado (rich scattering)
- ✅ N_RX ≥ N_TX
- ✅ Baja velocidad (Doppler bajo)

**Negativos:**
- ❌ Bajo SNR
- ❌ Canal correlacionado (LoS dominante)
- ❌ N_TX > N_RX
- ❌ Alta velocidad (Doppler shift)

## 10. Referencias

### 10.1 Estándares 3GPP

- **TS 36.211**: Physical channels and modulation
  - Section 6.3: Modulation and layer mapping
  - Section 6.10: Reference signals
  
- **TS 36.213**: Physical layer procedures
  - Section 7.2: CSI reporting
  - Section 7.2.2: Rank indicator

- **TS 36.101**: UE radio transmission and reception
  - Section 8.2: MIMO performance requirements

### 10.2 Referencias Académicas

1. **MIMO Wireless Communications** - Ezio Biglieri et al.
   - Capítulo 4: Spatial Multiplexing
   
2. **Fundamentals of Wireless Communication** - David Tse, Pramod Viswanath
   - Capítulo 7: MIMO capacity
   
3. **LTE - The UMTS Long Term Evolution** - Stefania Sesia et al.
   - Capítulo 9: MIMO techniques

### 10.3 Ecuaciones Clave Resumidas

**Capacidad MIMO:**
```
C = log₂ det(I_NR + (SNR/N_TX) × H × Hᴴ) bits/s/Hz
```

**MMSE Filter:**
```
W_MMSE = (Hᴴ H + σ²I)⁻¹ Hᴴ
```

**Rank Selection:**
```
rank = argmax_r { C_r } sujeto a r ≤ min(N_TX, N_RX)
```

**BER Aproximado (64-QAM, AWGN):**
```
BER ≈ (4/6) × Q(√(6×SNR/42))
```

---

## Resumen Ejecutivo

Esta implementación de **Spatial Multiplexing (TM4)** sigue fielmente el estándar LTE 3GPP, incorporando:

✅ **Rank Adaptation** basado en eigenvalues del canal  
✅ **Precoding** con codebook LTE  
✅ **Layer Mapping** round-robin estándar  
✅ **Dos detectores**: MMSE (robusto) y SIC (alta performance)  
✅ **Canal realista**: Rayleigh multipath (ITU-R M.1225)  
✅ **Perfect CSI**: Límite superior teórico  

La implementación demuestra el **trade-off fundamental** de MIMO:
- **Más capas** = más throughput pero requiere **mejor canal**
- **Configuraciones balanceadas** (N_TX ≈ N_RX) dan **mejor performance**
- **Detectores avanzados** (SIC) aprovechan mejor el **SNR alto**

**Aplicabilidad:** Análisis de performance, validación de algoritmos, educación en MIMO, prototipado de sistemas LTE.
