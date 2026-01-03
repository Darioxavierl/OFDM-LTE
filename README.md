# Modulo OFDM-SC para Simulacion de Sistemas LTE

Biblioteca Python completa para simular sistemas OFDM y SC-FDM basados en LTE, con soporte para transmision SISO Downlink, Uplink con SC-FDM e SIMO con combinacion de maxima razon (MRC).

## Tabla de Contenidos

1. [Descripcion General](#descripcion-general)
2. [Características Principales](#características-principales)
3. [Arquitectura del Modulo](#arquitectura-del-modulo)
4. [Estructura de Archivos](#estructura-de-archivos)
5. [Instalacion y Requisitos](#instalacion-y-requisitos)
6. [Guia de Uso Rapido](#guia-de-uso-rapido)
7. [SISO Downlink (OFDM Estandar)](#siso-downlink-ofdm-estandar)
8. [SISO Uplink (SC-FDM)](#siso-uplink-sc-fdm)
9. [SIMO Downlink (Multiples Antenas)](#simo-downlink-multiples-antenas)
10. [Referencia de API](#referencia-de-api)
11. [Modelos de Canal](#modelos-de-canal)
12. [Esquemas de Modulacion](#esquemas-de-modulacion)
13. [Scripts de Prueba](#scripts-de-prueba)
14. [Preguntas Frecuentes](#preguntas-frecuentes)

---

## Descripcion General

Este modulo implementa una simulacion completa de sistemas LTE (Long Term Evolution) con capacidad para:

- **Transmision SISO Downlink**: Sistema de entrada simple y salida simple usando OFDM estandar
- **Transmision SISO Uplink**: Sistema con precodificacion DFT (SC-FDM) para reducir PAPR
- **Transmision SIMO**: Sistema con multiples antenas receptoras y combinacion MRC para mejorar robustez

El modulo esta diseñado para ser flexible, modular y facil de usar. Soporta diferentes esquemas de modulacion, modelos de canal y configuraciones de ancho de banda.

---

## Características Principales

### Modos de Operacion

| Modo | Caracteristicas | Uso |
|------|-----------------|-----|
| SISO OFDM | Una antena TX, una antena RX, OFDM | Downlink LTE |
| SISO SC-FDM | Una antena TX, una antena RX, SC-FDM con DFT | Uplink LTE |
| SIMO | Una antena TX, N antenas RX, combinacion MRC | Recepcion mejorada |

### Canales Soportados

| Canal | Descripcion | Parametros |
|-------|------------|-----------|
| AWGN | Ruido Gaussiano Blanco Aditivo | SNR en dB |
| Rayleigh Multipath | Desvanecimiento Rayleigh con perfiles ITU | Perfil ITU, velocidad, frecuencia |

### Esquemas de Modulacion

| Esquema | Bits/Simbolo | Eficiencia | Robustez |
|---------|--------------|-----------|----------|
| QPSK | 2 | 2 bps/Hz | Muy alta |
| 16-QAM | 4 | 4 bps/Hz | Alta |
| 64-QAM | 6 | 6 bps/Hz | Normal |

### Configuraciones de Ancho de Banda

| BW (MHz) | Subportadoras | FFT Size |
|----------|---------------|----------|
| 1.25 | 76 | 128 |
| 2.5 | 150 | 256 |
| 5.0 | 300 | 512 |
| 10.0 | 600 | 1024 |
| 15.0 | 900 | 2048 |
| 20.0 | 1200 | 2048 |

### Metricas Soportadas

- BER (Bit Error Rate): Tasa de error de bit
- PAPR (Peak-to-Average Power Ratio): Relacion de potencia maxima a promedio en dB
- EVM (Error Vector Magnitude): Magnitud del vector de error
- CCDF (Complementary Cumulative Distribution Function): Distribucion acumulada complementaria

---

## Arquitectura del Modulo

### Componentes Principales

```
OFDMModule (Capa de Compatibilidad)
    └─ OFDMSimulator (Orquestador)
            │
    ├─ OFDMTransmitter (Modulacion)
    ├─ OFDMReceiver (Demodulacion)
    └─ OFDMChannel (Simulacion de canal)
            │
    ├─ OFDMModulator (mapeo IFFT)
    ├─ OFDMDemodulator (demapeo FFT)
    ├─ ResourceMapper (mapeo LTE)
    ├─ SC_FDMPrecodifier (precodificacion DFT)
    └─ ChannelSimulator (AWGN, Rayleigh)
            │
    └─ LTEConfig (parametros LTE)
```

### Flujo de Procesamiento SISO

```
Entrada de bits
    ↓ [Modulador QAM]
Simbolos complejos QAM
    ↓ [Mapeo de Recursos LTE]
    ├─ Subportadoras de datos
    ├─ Pilotos CRS
    ├─ Subportadora DC nula
    └─ Subportadoras de guarda nulas
    ↓ [IFFT de 1024 puntos]
Señal temporal
    ↓ [Agregar Prefijo Ciclico]
Señal OFDM transmitida
    ↓ [CANAL: AWGN o Rayleigh]
    ├─ Desvanecimiento
    └─ Ruido AWGN
    ↓ [Remover Prefijo Ciclico]
Señal temporal recibida
    ↓ [FFT de 1024 puntos]
    ↓ [Estimacion de Canal (CRS)]
    ↓ [Demapeo de Recursos]
Simbolos recibidos
    ↓ [Demodulador QAM]
Bits detectados
    ↓ [Calculo de BER]
```

### Flujo de Procesamiento SISO SC-FDM (Uplink)

El flujo es identico al SISO OFDM, excepto que despues del mapeo QAM se aplica precodificacion DFT:

```
Simbolos QAM
    ↓ [Precodificacion DFT]  <- Diferencia con OFDM
Simbolos precodificados
    ↓ [Resto del flujo igual...]
```

### Flujo de Procesamiento SIMO

```
Entrada de bits
    ↓ [Modulacion TX SISO]
    ↓
Señal TX
    ↓ [CANAL MULTIPLE]
    ├─ Antenna 0 (Thread 0)
    ├─ Antenna 1 (Thread 1)
    ├─ Antenna 2 (Thread 2)
    └─ Antenna 3 (Thread 3)
    ↓ [Sincronizacion]
    ↓ [Combinacion MRC en Frecuencia]
    ├─ Para cada subportadora k:
    │  ├─ w_i[k] = conj(H_i[k]) / |H_i[k]|²
    │  └─ Y_combined[k] = sum_i(w_i[k] * Y_i[k]) / num_rx
    ↓ [Demodulacion Final]
Bits detectados
```

---

## Estructura de Archivos

```
OFDM-LTE/
├── README.md                      Esta documentacion
├── config.py                      Configuracion LTE (LTEConfig)
├── ofdm_module.py                 Interfaz OFDMModule
├── signal_analysis.py             Analisis de señal (BER, PAPR, EVM)
├── validate.py                    Validacion de sistema
│
├── core/
│   ├── ofdm_core.py               Componentes principales (TX, RX, Channel, Simulator)
│   ├── modulator.py               Modulador OFDM y SC-FDM
│   ├── demodulator.py             Demodulador OFDM
│   ├── channel.py                 Simulador de canal
│   ├── resource_mapper.py         Mapeo de recursos LTE
│   ├── dft_precoding.py           Precodificacion/decodificacion DFT
│   ├── lte_receiver.py            Receptor LTE especializado
│   ├── rayleighchannel.py         Canal Rayleigh
│   ├── itu_r_m1225.py             Perfiles ITU-R M.1225
│   └── itu_r_m1225_channels.json  Datos de perfiles ITU
│
├── test/
│   ├── test_basic.py              Pruebas basicas
│   ├── test_simo_image.py         Pruebas SIMO con imagen
│   ├── final_image_test.py        Prueba final con imagen
│   └── test_modular_image.py      Prueba modular con imagen
│
├── examples/
│   ├── example_basic.py           Ejemplo basico
│   └── example_sweep.py           Ejemplo con barrido SNR
│
└── results/                       Resultados de simulaciones
```

---

## Instalacion y Requisitos

### Requisitos

- Python 3.8 o superior
- NumPy 1.21+
- Matplotlib (opcional, para visualizacion)
- SciPy (opcional, para analisis avanzado)

### Instalacion

El modulo esta completamente contenido en la carpeta del proyecto. No requiere instalacion:

```bash
git clone <repository>
cd OFDM-LTE
```

### Verificar Instalacion

```bash
python -c "from config import LTEConfig; from ofdm_module import OFDMModule; print('OK')"
```

---

## Guia de Uso Rapido

### Transmision SISO Basica

```python
import numpy as np
from ofdm_module import OFDMModule

# Crear modulo con configuracion por defecto (5 MHz, QPSK, AWGN)
module = OFDMModule()

# Generar bits aleatorios
bits = np.random.randint(0, 2, 10000)

# Transmitir
results = module.transmit(bits, snr_db=15)

# Ver resultados
print(f"BER: {results['ber']:.2e}")
print(f"Errores: {results['errors']}")
print(f"PAPR: {results['papr_db']:.2f} dB")
```

### Transmision SC-FDM

```python
from config import LTEConfig
from ofdm_module import OFDMModule

config = LTEConfig(bandwidth=5.0, modulation='QPSK')
module = OFDMModule(config=config, enable_sc_fdm=True)

bits = np.random.randint(0, 2, 10000)
results = module.transmit(bits, snr_db=15)

# SC-FDM reduce PAPR de ~10 dB a ~7 dB
print(f"PAPR (SC-FDM): {results['papr_db']:.2f} dB")
```

### Transmision SIMO

```python
from core.ofdm_core import OFDMSimulator
from config import LTEConfig

config = LTEConfig(bandwidth=5.0, modulation='QPSK')
simulator = OFDMSimulator(config=config, channel_type='awgn')

bits = np.random.randint(0, 2, 50000)

# SIMO con 4 antenas receptoras
results = simulator.simulate_simo(bits, snr_db=15, num_rx=4)

print(f"SIMO BER: {results['ber']:.2e}")
```

---

## SISO Downlink (OFDM Estandar)

El modo SISO Downlink usa OFDM estandar. Es la configuracion mas simple y se usa en el downlink LTE.

### Uso Basico

```python
import numpy as np
from ofdm_module import OFDMModule, LTEConfig

# Opcion 1: Usar configuracion por defecto
module = OFDMModule()

# Opcion 2: Especificar configuracion
config = LTEConfig(bandwidth=10.0, modulation='16-QAM')
module = OFDMModule(config=config, channel_type='awgn')

# Generar bits
bits = np.random.randint(0, 2, 10000)

# Transmitir
results = module.transmit(bits, snr_db=15)

print(f"BER: {results['ber']:.2e}")
print(f"PAPR: {results['papr_db']:.2f} dB")
```

### Usar Canal Rayleigh

```python
config = LTEConfig(bandwidth=5.0, modulation='QPSK')
module = OFDMModule(
    config=config,
    channel_type='rayleigh_mp'  # Activa canal Rayleigh
)

bits = np.random.randint(0, 2, 10000)
results = module.transmit(bits, snr_db=20)

print(f"BER (Rayleigh): {results['ber']:.2e}")
```

### Barrido de SNR

```python
import numpy as np
import matplotlib.pyplot as plt
from ofdm_module import OFDMModule

module = OFDMModule(channel_type='awgn')
bits = np.random.randint(0, 2, 10000)

snr_range = np.arange(5, 25, 2)
ber_values = []

for snr in snr_range:
    results = module.transmit(bits.copy(), snr_db=snr)
    ber_values.append(results['ber'])

plt.semilogy(snr_range, ber_values, 'o-')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.grid(True)
plt.show()
```

### Diferentes Modulaciones

```python
from ofdm_module import OFDMModule, LTEConfig

modulaciones = ['QPSK', '16-QAM', '64-QAM']
snr_db = 20

for mod in modulaciones:
    config = LTEConfig(bandwidth=5.0, modulation=mod)
    module = OFDMModule(config=config, channel_type='awgn')
    
    bits = np.random.randint(0, 2, 10000)
    results = module.transmit(bits, snr_db=snr_db)
    
    print(f"{mod}: BER = {results['ber']:.2e}")
```

---

## SISO Uplink (SC-FDM)

El modo Uplink usa SC-FDM (Single-Carrier FDM) que es OFDM con precodificacion DFT. Reduce significativamente el PAPR.

### Configuracion Basica

```python
import numpy as np
from ofdm_module import OFDMModule, LTEConfig

config = LTEConfig(bandwidth=5.0, modulation='QPSK')
module = OFDMModule(
    config=config,
    channel_type='awgn',
    enable_sc_fdm=True  # Activar SC-FDM
)

bits = np.random.randint(0, 2, 10000)
results = module.transmit(bits, snr_db=15)

print(f"PAPR (SC-FDM): {results['papr_db']:.2f} dB")
```

### Comparacion OFDM vs SC-FDM

```python
import numpy as np
from ofdm_module import OFDMModule, LTEConfig

config = LTEConfig(bandwidth=10.0, modulation='16-QAM')
bits = np.random.randint(0, 2, 50000)

# OFDM
module_ofdm = OFDMModule(config=config, enable_sc_fdm=False)
results_ofdm = module_ofdm.transmit(bits.copy(), snr_db=20)

# SC-FDM
module_scfdm = OFDMModule(config=config, enable_sc_fdm=True)
results_scfdm = module_scfdm.transmit(bits.copy(), snr_db=20)

print(f"OFDM PAPR: {results_ofdm['papr_db']:.2f} dB")
print(f"SC-FDM PAPR: {results_scfdm['papr_db']:.2f} dB")
print(f"Reduccion: {results_ofdm['papr_db'] - results_scfdm['papr_db']:.2f} dB")
```

SC-FDM tipicamente reduce el PAPR de 7-10 dB a 3-4 dB.

### SC-FDM en Canal Rayleigh

```python
from ofdm_module import OFDMModule, LTEConfig

config = LTEConfig(bandwidth=5.0, modulation='QPSK')
module = OFDMModule(
    config=config,
    channel_type='rayleigh_mp',
    enable_sc_fdm=True
)

bits = np.random.randint(0, 2, 20000)
results = module.transmit(bits, snr_db=18)

print(f"BER (Rayleigh, SC-FDM): {results['ber']:.2e}")
print(f"PAPR (SC-FDM): {results['papr_db']:.2f} dB")
```

---

## SIMO Downlink (Multiples Antenas)

SIMO (Single-Input Multiple-Output) usa una antena transmisora y multiples antenas receptoras para mejorar robustez mediante combinacion de maxima razon (MRC).

### Uso Basico de SIMO

```python
import numpy as np
from core.ofdm_core import OFDMSimulator
from config import LTEConfig

config = LTEConfig(bandwidth=5.0, modulation='QPSK')
simulator = OFDMSimulator(
    config=config,
    channel_type='awgn',
    num_channels=1
)

bits = np.random.randint(0, 2, 10000)

# SIMO con 4 antenas receptoras
results_simo = simulator.simulate_simo(
    bits=bits,
    snr_db=15,
    num_rx=4,           # 4 antenas receptoras
    combining='mrc'     # Maximum Ratio Combining
)

print(f"SIMO (4 RX) BER: {results_simo['ber']:.2e}")
```

### Comparacion SISO vs SIMO

```python
import numpy as np
from core.ofdm_core import OFDMSimulator
from config import LTEConfig

config = LTEConfig(bandwidth=5.0, modulation='QPSK')
bits = np.random.randint(0, 2, 50000)

simulator = OFDMSimulator(config=config, channel_type='awgn')

# SISO (1 RX)
results_siso = simulator.simulate_siso(bits.copy(), snr_db=10)
ber_siso = results_siso['ber']

# SIMO (2 RX)
results_simo_2 = simulator.simulate_simo(bits.copy(), snr_db=10, num_rx=2)
ber_simo_2 = results_simo_2['ber']

# SIMO (4 RX)
results_simo_4 = simulator.simulate_simo(bits.copy(), snr_db=10, num_rx=4)
ber_simo_4 = results_simo_4['ber']

print(f"SISO BER:        {ber_siso:.2e}")
print(f"SIMO (2 RX) BER: {ber_simo_2:.2e} ({ber_siso/ber_simo_2:.1f}x mejor)")
print(f"SIMO (4 RX) BER: {ber_simo_4:.2e} ({ber_siso/ber_simo_4:.1f}x mejor)")
```

### SIMO en Canal Rayleigh

```python
import numpy as np
from core.ofdm_core import OFDMSimulator
from config import LTEConfig

config = LTEConfig(bandwidth=5.0, modulation='QPSK')
bits = np.random.randint(0, 2, 100000)

simulator = OFDMSimulator(config=config, channel_type='rayleigh_mp')

# SISO
results_siso = simulator.simulate_siso(bits.copy(), snr_db=20)

# SIMO
results_simo = simulator.simulate_simo(bits.copy(), snr_db=20, num_rx=4)

print(f"SISO (Rayleigh) BER: {results_siso['ber']:.2e}")
print(f"SIMO (Rayleigh, 4 RX) BER: {results_simo['ber']:.2e}")
print(f"Mejora: {results_siso['ber'] / results_simo['ber']:.1f}x")
```

Nota: La mejora SIMO en Rayleigh es menor que en AWGN (~20-30% vs ~99%) porque a veces todos los receptores desvanecen simultaneamente. Esto es comportamiento normal.

### Variacion de Numero de Antenas

```python
import numpy as np
from core.ofdm_core import OFDMSimulator
from config import LTEConfig

config = LTEConfig(bandwidth=5.0, modulation='QPSK')
bits = np.random.randint(0, 2, 50000)

simulator = OFDMSimulator(config=config, channel_type='awgn')
snr_db = 15

for num_rx in [1, 2, 4, 8]:
    if num_rx == 1:
        results = simulator.simulate_siso(bits.copy(), snr_db=snr_db)
    else:
        results = simulator.simulate_simo(bits.copy(), snr_db=snr_db, num_rx=num_rx)
    
    print(f"{num_rx} RX: BER = {results['ber']:.2e}")
```

---

## Referencia de API

### Clase LTEConfig

Define todos los parametros de configuracion LTE.

#### Constructor

```python
config = LTEConfig(
    bandwidth=5.0,           # MHz: 1.25, 2.5, 5, 10, 15, 20
    delta_f=15.0,            # kHz: 15.0 o 7.5
    modulation='QPSK',       # 'QPSK', '16-QAM', '64-QAM'
    cp_type='normal'         # 'normal' o 'extended'
)
```

#### Parametros Principales

| Parametro | Tipo | Descripcion |
|-----------|------|-----------|
| bandwidth | float | Ancho de banda en MHz |
| delta_f | float | Espaciado de subportadora en kHz |
| modulation | str | Esquema de modulacion |
| cp_type | str | Tipo de prefijo ciclico |
| Nc | int | Numero de subportadoras de datos |
| N | int | Tamaño FFT |
| fs | float | Frecuencia de muestreo en Hz |
| cp_length | int | Longitud de prefijo ciclico en muestras |
| bits_per_symbol | int | Bits por simbolo QAM |

#### Metodos

```python
# Obtener informacion de configuracion
info = config.get_info()

# Obtener representacion string
print(config)
```

#### Ejemplo

```python
from config import LTEConfig

config = LTEConfig(bandwidth=10.0, modulation='16-QAM')
print(config)
print(f"Subportadoras: {config.Nc}")
print(f"FFT Size: {config.N}")
print(f"Bits/simbolo: {config.bits_per_symbol}")
```

### Clase OFDMModule

Interfaz de alto nivel para transmision SISO. Mantiene compatibilidad hacia atras.

#### Constructor

```python
module = OFDMModule(
    config=None,                    # LTEConfig, default 5MHz QPSK
    channel_type='awgn',            # 'awgn' o 'rayleigh_mp'
    mode='lte',                     # 'lte' o 'simple'
    enable_sc_fdm=False,            # Habilitar SC-FDM
    enable_equalization=True        # Habilitar igualacion
)
```

#### Metodo: transmit()

```python
results = module.transmit(
    bits=np.random.randint(0, 2, 10000),
    snr_db=15.0
)

# Resultados disponibles:
results['ber']              # Tasa de error de bit (float)
results['errors']           # Numero de errores (int)
results['papr_db']          # PAPR en dB (float)
results['papr_linear']      # PAPR lineal (float)
results['signal_tx']        # Señal transmitida (np.ndarray)
results['signal_rx']        # Señal recibida (np.ndarray)
results['symbols_tx']       # Simbolos transmitidos (list)
results['symbols_rx']       # Simbolos recibidos (list)
results['transmitted_bits'] # Bits transmitidos (int)
results['received_bits']    # Bits recibidos (int)
```

### Clase OFDMSimulator

Orquestrador de simulaciones, soporta SISO y SIMO.

#### Constructor

```python
simulator = OFDMSimulator(
    config=None,                # LTEConfig, default 5MHz QPSK
    channel_type='awgn',        # 'awgn' o 'rayleigh_mp'
    mode='lte',                 # 'lte' o 'simple'
    enable_sc_fdm=False,        # SC-FDM enable
    enable_equalization=True,   # Equalization enable
    num_channels=1              # Numero de canales (SIMO)
)
```

#### Metodo: simulate_siso()

```python
results = simulator.simulate_siso(bits, snr_db=10.0)

# Estructura de resultados (dict)
{
    'transmitted_bits': int,
    'received_bits': int,
    'bit_errors': int,
    'ber': float,
    'snr_db': float,
    'papr_db': float,
    'papr_linear': float,
    'signal_tx': np.ndarray,
    'signal_rx': np.ndarray,
    'symbols_tx': list,
    'symbols_rx': list
}
```

#### Metodo: simulate_simo()

```python
results = simulator.simulate_simo(
    bits=bits,
    snr_db=15.0,
    num_rx=4,               # 4 antenas receptoras
    combining='mrc'         # Combinacion MRC
)

# Resultados identicos a simulate_siso()
```

---

## Modelos de Canal

### Canal AWGN

Ruido Gaussiano Blanco Aditivo. No hay desvanecimiento, solo ruido.

```python
module = OFDMModule(channel_type='awgn')
results = module.transmit(bits, snr_db=20)
```

Caracteristicas:
- SNR uniforme en todas las subportadoras
- BER predecible y bajo con buena SNR
- Referencia teorica

### Canal Rayleigh Multipath

Desvanecimiento Rayleigh con multiples caminos. Simula propagacion realista.

```python
module = OFDMModule(channel_type='rayleigh_mp')
results = module.transmit(bits, snr_db=20)
```

### Perfiles ITU-R M.1225

El modulo incluye 5 perfiles de canal estandar ITU:

#### Pedestrian_A

Peaton baja velocidad, distancia corta.

```python
from core.ofdm_core import OFDMChannel

channel = OFDMChannel(
    channel_type='rayleigh_mp',
    itu_profile='Pedestrian_A',
    velocity_kmh=3
)
```

Caracteristicas:
- 4 caminos
- Deltas de tiempo: 0, 0.11, 0.19, 0.41 microsegundos
- Potencia: 0, -9.7, -19.2, -22.8 dB

#### Pedestrian_B

Peaton alta velocidad. Mas severo que Pedestrian_A.

#### Vehicular_A

Vehiculo baja velocidad, distancia corta.

#### Vehicular_B

Vehiculo alta velocidad, distancia larga. El mas severo.

Caracteristicas:
- 8 caminos (maximo)
- Propagacion severa

#### Bad_Urban

Urbano severo con multitrayecto extremo.

---

## Esquemas de Modulacion

### QPSK

Modulacion mas robusta. 2 bits por simbolo.

```python
config = LTEConfig(modulation='QPSK')
module = OFDMModule(config=config)
```

Ventajas: Maxima robustez, menor BER
Desventajas: Menor eficiencia espectral (2 bps/Hz)

### 16-QAM

Modulacion intermedia. 4 bits por simbolo.

```python
config = LTEConfig(modulation='16-QAM')
module = OFDMModule(config=config)
```

Ventajas: Balance entre robustez y eficiencia
Desventajas: Sensible a ruido

### 64-QAM

Modulacion de alta eficiencia. 6 bits por simbolo.

```python
config = LTEConfig(modulation='64-QAM')
module = OFDMModule(config=config)
```

Ventajas: Alta eficiencia espectral (6 bps/Hz)
Desventajas: Requiere SNR muy buena

---

## Scripts de Prueba

El directorio test/ contiene varios scripts de prueba automatizadas.

### test_basic.py

Pruebas basicas del modulo.

```bash
python test/test_basic.py
```

Pruebas incluidas:
1. Instanciacion del modulo
2. Transmision basica
3. Barrido de BER
4. Calculo de metricas

### test_simo_image.py

Prueba SIMO transmitiendo una imagen.

```bash
python test/test_simo_image.py
```

Pruebas incluidas:
1. Codificacion de imagen a bits
2. Transmision SISO vs SIMO
3. Comparacion de resultados
4. Decodificacion de imagen recibida

### final_image_test.py

Test final completo con imagen.

```bash
python test/final_image_test.py
```

### test_modular_image.py

Test con arquitectura modular.

```bash
python test/test_modular_image.py
```

---

## Preguntas Frecuentes

### P1: Cual es la diferencia entre OFDM y SC-FDM?

OFDM: Multiples portadoras. Cada bit se transmite en una portadora independiente.

SC-FDM: Una portadora equivalente. Los datos se precodifican con DFT antes de mapearlos, reduciendo el PAPR.

Ventajas SC-FDM:
- PAPR mas bajo (3-4 dB vs 7-10 dB)
- Mejor eficiencia de potencia en Uplink

Desventajas SC-FDM:
- Equipo transmisor mas complejo
- Menos flexible para asignacion de recursos

### P2: Por que SIMO mejora mas en AWGN que en Rayleigh?

En AWGN: Las 4 antenas ven la misma señal con ruido independiente. Se suman linealmente.

En Rayleigh: A veces todos los receptores desvanecen simultaneamente (evento raro pero probable).
En esos momentos, la diversidad no ayuda.

Esto es comportamiento normal. La mejora SIMO en Rayleigh (~20-30%) es realista.

### P3: Que es la interpolacion temporal de canal?

Estimacion Periodica (actual):
- Se estima el canal una vez por simbolo OFDM
- Los 1024 simbolos en la subportadora usan la MISMA estimacion

Interpolacion Temporal (mejora opcional):
- Se interpola la estimacion entre simbolos
- El canal varia suavemente dentro del simbolo
- Ganancia: +2-4 dB en fading rapido

Actualmente se usa estimacion periodica. Interpolacion es mejora futura.

### P4: Como selecciono el perfil de canal ITU correcto?

Selecciona segun el escenario:

- Pedestrian_A: Pruebas de laboratorio, referencia
- Pedestrian_B: Peatones en entorno urbano
- Vehicular_A: Vehiculos lenta, autopista
- Vehicular_B: Alta velocidad, escenario mas severo
- Bad_Urban: Centro urbano con muchos obstaculos

Para la mayoria de casos: Pedestrian_A es buena referencia.

### P5: Como mejoro el BER?

Opciones (en orden de efectividad):

1. Aumentar SNR: Mas potencia de transmision
2. Usar modulacion mas robusta: QPSK en lugar de 64-QAM
3. Agregar antenas receptoras: SIMO con 4-8 RX
4. Implementar codificacion: Codigos correctores (FEC)
5. Interpolacion temporal: +2-4 dB en fading rapido
6. Igualacion avanzada: ZF, MMSE

### P6: Como optimizo para diferentes anchos de banda?

Los parametros se actualizan automaticamente:

```python
# 5 MHz: 300 subportadoras
config_5mhz = LTEConfig(bandwidth=5.0)

# 20 MHz: 1200 subportadoras
config_20mhz = LTEConfig(bandwidth=20.0)
```

Mas ancho de banda = mas datos pero mas sensible al ruido.

### P7: Cual es el maximo numero de antenas SIMO?

Teoricamente no hay limite. Practicamente:

- 2-4 RX: Ganancia buena, complejidad manejable
- 4-8 RX: Ganancia excelente, complejidad moderada
- 8+ RX: Muy pocas mejoras, complejidad alta

Para la mayoria de casos: 4 antenas es optimo.

### P8: Que significa "PAPR"?

PAPR (Peak-to-Average Power Ratio): Relacion entre la potencia maxima instantanea y la potencia promedio.

Valores tipicos:
- OFDM: 7-10 dB
- SC-FDM: 3-4 dB

PAPR bajo es mejor para amplificadores (menos distorsion).

### P9: Como validar mis resultados?

Comparar con referencias teoricas:

- QPSK en AWGN deberia dar ~1% BER @ SNR 6dB
- 16-QAM deberia dar ~1% BER @ SNR 10dB
- 64-QAM deberia dar ~1% BER @ SNR 16dB

Si tus resultados son similares: implementacion correcta.

### P10: Como extender el modulo?

El modulo esta diseñado para extension:

1. Nuevo canal: Hereda ChannelSimulator
2. Nuevo esquema de modulacion: Modifica QAMModulator
3. Nuevo receptor: Hereda OFDMDemodulator
4. MIMO: Extiende OFDMSimulator con simulate_mimo()

---

VERSION: 1.0
ULTIMA ACTUALIZACION: Enero 2026
ESTADO: Produccion
