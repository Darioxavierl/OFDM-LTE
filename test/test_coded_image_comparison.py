"""
Test Visual de Channel Coding con Imágenes
===========================================

Transmite una imagen a través del sistema OFDM con y sin channel coding,
probando diferentes SNRs. Genera una imagen comparativa:
- Fila superior: Sin codificación (uncoded)
- Fila inferior: Con codificación (coded - CRC + Turbo)
- Columnas: Diferentes SNRs

Autor: Sistema OFDM-LTE
Fecha: 2026-01-22
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

# Agregar path del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LTEConfig
from core.ofdm_core import OFDMSimulator
from utils.image_processing import ImageProcessor

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

# Imagen a transmitir
IMAGE_PATH = "img/entre-ciel-et-terre.jpg"  # Cambiar según imagen disponible

# Configuración LTE
BANDWIDTH = 20.0  # MHz: 1.25, 2.5, 5, 10, 15, 20
SUBCARRIER_SPACING = 15.0  # kHz: 15.0 o 7.5
CYCLIC_PREFIX_TYPE = 'normal'  # 'normal' o 'extended'
MODULATION = 'QPSK'  # Fijo en QPSK para este test

# Configuración del canal
CHANNEL_TYPE = 'rayleigh_mp'  # 'awgn' o 'rayleigh_mp'
ITU_PROFILE = 'Pedestrian_A'  # 'Pedestrian_A', 'Pedestrian_B', 'Vehicular_A', etc.
FREQUENCY_GHZ = 2.0  # Frecuencia de portadora en GHz
VELOCITY_KMH = 3.0  # Velocidad del móvil en km/h

# SNRs a probar (en dB)
SNR_VALUES = [9, 12, 15]

# Configuración de visualización
OUTPUT_PATH = "results/coded/coded_vs_uncoded_comparison.png"
FIGURE_SIZE = (20, 8)  # Ancho x Alto en pulgadas
DPI = 150


def create_comparison_figure(results, original_img, snr_values):
    """
    Crea figura comparativa con imágenes reconstruidas
    
    Args:
        results: Dict con keys 'uncoded' y 'coded', cada uno con lista de imágenes
        original_img: Imagen original
        snr_values: Lista de SNRs probados
    """
    num_snrs = len(snr_values)
    
    # Crear figura con GridSpec para mejor control
    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = GridSpec(3, num_snrs + 1, figure=fig, 
                  height_ratios=[1, 1, 1],
                  width_ratios=[1] * (num_snrs + 1),
                  hspace=0.3, wspace=0.15)
    
    # Fila 0: Imagen original (centrada)
    ax_orig = fig.add_subplot(gs[0, num_snrs // 2])
    # Convertir PIL Image a array si es necesario
    orig_array = np.array(original_img) if isinstance(original_img, Image.Image) else original_img
    ax_orig.imshow(orig_array, cmap='gray', vmin=0, vmax=255)
    ax_orig.set_title('Original', fontsize=14, fontweight='bold')
    ax_orig.axis('off')
    
    # Fila 1: Sin codificación (uncoded)
    for col, (snr, img_uncoded) in enumerate(zip(snr_values, results['uncoded'])):
        ax = fig.add_subplot(gs[1, col])
        # Convertir PIL Image a array si es necesario
        img_array = np.array(img_uncoded['image']) if isinstance(img_uncoded['image'], Image.Image) else img_uncoded['image']
        ax.imshow(img_array, cmap='gray', vmin=0, vmax=255)
        
        psnr = img_uncoded['psnr']
        ber = img_uncoded['ber']
        
        title = f"SNR={snr}dB\nPSNR={psnr:.1f}dB\nBER={ber:.2%}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        
        if col == 0:
            ax.text(-0.15, 0.5, 'Uncoded', rotation=90, 
                   va='center', ha='center', fontsize=14, fontweight='bold',
                   transform=ax.transAxes)
    
    # Fila 2: Con codificación (coded)
    for col, (snr, img_coded) in enumerate(zip(snr_values, results['coded'])):
        ax = fig.add_subplot(gs[2, col])
        # Convertir PIL Image a array si es necesario
        img_array = np.array(img_coded['image']) if isinstance(img_coded['image'], Image.Image) else img_coded['image']
        ax.imshow(img_array, cmap='gray', vmin=0, vmax=255)
        
        psnr = img_coded['psnr']
        ber = img_coded['ber']
        crc = '✓' if img_coded['crc_pass'] else '✗'
        
        title = f"SNR={snr}dB\nPSNR={psnr:.1f}dB\nBER={ber:.2%}\nCRC={crc}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        
        if col == 0:
            ax.text(-0.15, 0.5, 'Coded\n(CRC+Turbo)', rotation=90, 
                   va='center', ha='center', fontsize=14, fontweight='bold',
                   transform=ax.transAxes)
    
    # Título general
    channel_name = f"{CHANNEL_TYPE.upper()} - {ITU_PROFILE}" if CHANNEL_TYPE == 'rayleigh_mp' else 'AWGN'
    fig.suptitle(f'Comparación Channel Coding - {MODULATION} - {channel_name}\n'
                 f'BW={BANDWIDTH}MHz, Δf={SUBCARRIER_SPACING}kHz, CP={CYCLIC_PREFIX_TYPE}',
                 fontsize=16, fontweight='bold', y=0.98)
    
    return fig


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal del test"""
    
    print("="*80)
    print("TEST VISUAL: Channel Coding con Imágenes (QPSK)")
    print("="*80)
    print()
    
    # Configuración
    print("Configuración:")
    print(f"  Imagen: {IMAGE_PATH}")
    print(f"  Bandwidth: {BANDWIDTH} MHz")
    print(f"  Subcarrier Spacing: {SUBCARRIER_SPACING} kHz")
    print(f"  Cyclic Prefix: {CYCLIC_PREFIX_TYPE}")
    print(f"  Modulation: {MODULATION}")
    print(f"  Channel: {CHANNEL_TYPE}")
    if CHANNEL_TYPE == 'rayleigh_mp':
        print(f"  ITU Profile: {ITU_PROFILE}")
        print(f"  Frequency: {FREQUENCY_GHZ} GHz")
        print(f"  Velocity: {VELOCITY_KMH} km/h")
    print(f"  SNR values: {SNR_VALUES}")
    print()
    
    # Cargar imagen usando ImageProcessor
    print("Cargando imagen...")
    try:
        # Cargar y redimensionar a 64x64
        original_img = Image.open(IMAGE_PATH)
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        original_size = original_img.size
        original_img = original_img.resize((64, 64), Image.Resampling.LANCZOS)
        print(f"  Dimensiones originales: {original_size[0]}x{original_size[1]}")
        print(f"  Redimensionada a: 64x64")
        
        # Guardar temporalmente y reconvertir a bits
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            original_img.save(tmp_path)
        
        bits_tx, metadata = ImageProcessor.image_to_bits(tmp_path)
        os.remove(tmp_path)
        
        img_height = metadata['height']
        img_width = metadata['width']
        channels = metadata['channels']
        num_pixels = img_height * img_width
        num_bits = len(bits_tx)
        print(f"  Canales: {channels}")
        print(f"  Pixels: {num_pixels:,}")
        print(f"  Bits: {num_bits:,}")
        
        bits = bits_tx  # Renombrar para compatibilidad con código existente
        print(f"  ✓ Imagen cargada y convertida a {len(bits)} bits")
    except FileNotFoundError:
        print(f"  Imagen no encontrada: {IMAGE_PATH}")
        print("   Usa una imagen del folder img/ o actualiza IMAGE_PATH")
        return
    print()
    
    # Crear configuración LTE
    print("Creando configuración LTE...")
    config = LTEConfig(
        bandwidth=BANDWIDTH,
        delta_f=SUBCARRIER_SPACING,
        modulation=MODULATION,
        cp_type=CYCLIC_PREFIX_TYPE
    )
    print(f"  ✓ FFT: {config.N}, Subportadoras: {config.Nc}")
    print(f"  ✓ Bits por símbolo: {config.bits_per_symbol}")
    print()
    
    # Crear simulador
    print(f"Creando simulador ({CHANNEL_TYPE})...")
    sim = OFDMSimulator(
        config=config,
        channel_type=CHANNEL_TYPE,
        itu_profile=ITU_PROFILE if CHANNEL_TYPE == 'rayleigh_mp' else None,
        frequency_ghz=FREQUENCY_GHZ,
        velocity_kmh=VELOCITY_KMH
    )
    print(f"  ✓ Simulador creado")
    print()
    
    # Almacenar resultados
    results = {
        'uncoded': [],
        'coded': []
    }
    
    # Probar cada SNR
    print("="*80)
    print("TRANSMISIONES")
    print("="*80)
    
    for snr in SNR_VALUES:
        print(f"\n{'─'*80}")
        print(f"SNR = {snr} dB")
        print(f"{'─'*80}")
        
        # Transmisión sin codificación
        print("  [Uncoded] Transmitiendo...")
        try:
            result_uncoded = sim.simulate_siso(bits, snr_db=snr)
            bits_rx_uncoded = result_uncoded['bits_received_array']
            ber_uncoded = result_uncoded['ber']
            
            # Reconstruir imagen usando ImageProcessor
            img_rx_uncoded = ImageProcessor.bits_to_image(bits_rx_uncoded, metadata)
            psnr_uncoded = ImageProcessor.calculate_psnr(original_img, img_rx_uncoded)
            
            results['uncoded'].append({
                'image': img_rx_uncoded,
                'ber': ber_uncoded,
                'psnr': psnr_uncoded,
                'snr': snr
            })
            
            print(f"    BER: {ber_uncoded:.4f} ({int(ber_uncoded*num_bits)}/{num_bits} bits)")
            print(f"    PSNR: {psnr_uncoded:.2f} dB")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            # Imagen en negro si falla
            results['uncoded'].append({
                'image': Image.new('RGB', (img_width, img_height), color='black'),
                'ber': 1.0,
                'psnr': 0.0,
                'snr': snr
            })
        
        # Transmisión con codificación
        print("  [Coded] Transmitiendo...")
        try:
            result_coded = sim.simulate_siso_coded(bits, snr_db=snr)
            bits_rx_coded = result_coded['bits_received_array']
            ber_coded = result_coded['ber']
            crc_pass = result_coded['crc_pass']
            
            # Reconstruir imagen usando ImageProcessor
            img_rx_coded = ImageProcessor.bits_to_image(bits_rx_coded, metadata)
            psnr_coded = ImageProcessor.calculate_psnr(original_img, img_rx_coded)
            
            results['coded'].append({
                'image': img_rx_coded,
                'ber': ber_coded,
                'psnr': psnr_coded,
                'crc_pass': crc_pass,
                'snr': snr
            })
            
            crc_symbol = '✓' if crc_pass else '✗'
            print(f"    BER: {ber_coded:.4f} ({int(ber_coded*num_bits)}/{num_bits} bits)")
            print(f"    PSNR: {psnr_coded:.2f} dB")
            print(f"    CRC: {crc_symbol}")
            
            # Comparación
            if ber_uncoded > 0:
                improvement = ((ber_uncoded - ber_coded) / ber_uncoded) * 100
                if improvement > 5:
                    print(f"    → Mejora: {improvement:.1f}% ✓")
                elif improvement < -5:
                    print(f"    → Empeora: {improvement:.1f}% ✗")
                else:
                    print(f"    → Similar: {improvement:.1f}%")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            # Imagen en negro si falla
            results['coded'].append({
                'image': Image.new('RGB', (img_width, img_height), color='black'),
                'ber': 1.0,
                'psnr': 0.0,
                'crc_pass': False,
                'snr': snr
            })
    
    # Crear figura comparativa
    print()
    print("="*80)
    print("GENERANDO FIGURA COMPARATIVA")
    print("="*80)
    
    fig = create_comparison_figure(results, original_img, SNR_VALUES)
    
    # Guardar figura
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight')
    print(f"✓ Figura guardada: {OUTPUT_PATH}")
    
    # Mostrar figura
    plt.show()
    
    print()
    print("="*80)
    print("TEST COMPLETADO")
    print("="*80)


if __name__ == "__main__":
    main()
