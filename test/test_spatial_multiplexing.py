"""
Test de Spatial Multiplexing (TM4) con transmisión de imagen

Prueba el sistema completo de multiplexación espacial con:
- Múltiples configuraciones de antenas (2×2, 4×2, 4×4, 8×4)
- Rank adaptativo
- Detectores MMSE y SIC
- Transmisión y reconstrucción de imagen
- Análisis de BER vs configuración

Usa utils.image_processing.ImageProcessor para:
- Carga de imagen (image_to_bits)
- Reconstrucción de imagen (bits_to_image)
- Cálculo de métricas (calculate_psnr_bits)

Salida: results/mimo/
"""

import numpy as np
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Backend sin display
import matplotlib.pyplot as plt
from PIL import Image

# Agregar path del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.ofdm_core import simulate_spatial_multiplexing
from config import LTEConfig
from utils.image_processing import ImageProcessor


# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================
SNR_DB = 20  # SNR en dB - Modifica este valor para probar diferentes BER
             # Valores sugeridos: 5-10 (BER alto), 15-20 (medio), 25-30 (bajo)
# ============================================================================


def test_spatial_multiplexing_single(
    bits,
    metadata,
    num_tx,
    num_rx,
    rank,
    detector_type,
    snr_db=None,  # Si es None, usa SNR_DB global
    modulation='64-QAM',
    output_dir='results/mimo'
):
    """
    Test de spatial multiplexing con una configuración específica.
    Soporta transmisión de imágenes grandes mediante múltiples símbolos OFDM.
    
    Args:
        bits: Array de bits de la imagen (de ImageProcessor.image_to_bits)
        metadata: Diccionario con información de la imagen
        num_tx, num_rx: Configuración de antenas
        rank: Rank a usar o 'adaptive'
        detector_type: 'MMSE', 'SIC', etc.
        snr_db: SNR en dB (usa SNR_DB global si es None)
        modulation: Modulación a usar
        output_dir: Directorio de salida
    """
    # Usar SNR global si no se especifica
    if snr_db is None:
        snr_db = SNR_DB
    
    total_bits = len(bits)
    print(f"      Transmitiendo {total_bits:,} bits ({metadata['width']}x{metadata['height']}px) para {num_tx}x{num_rx}")
    
    # Calcular bits por bloque basado en el número REAL de subportadoras de datos
    # El sistema LTE tiene 249 subportadoras de datos (de 300 totales - CRS, DC, guardas)
    # Con 64-QAM: 6 bits por símbolo
    # IMPORTANTE: En spatial multiplexing, el número de bits por OFDM es CONSTANTE
    # El rank solo divide los símbolos en capas paralelas, NO aumenta la capacidad por OFDM symbol
    data_subcarriers_per_symbol = 249  # Número real de subportadoras de datos en LTE
    bits_per_ofdm_symbol = data_subcarriers_per_symbol * 6  # 6 bits por símbolo 64-QAM
    
    # Calcular número de símbolos OFDM necesarios
    num_ofdm_symbols = int(np.ceil(total_bits / bits_per_ofdm_symbol))
    print(f"      Necesita ~{num_ofdm_symbols} símbolos OFDM (capacidad: {bits_per_ofdm_symbol} bits/símbolo con {data_subcarriers_per_symbol} subportadoras)")

    
    # Transmitir en bloques
    all_bits_rx = []
    total_errors = 0
    
    for block_idx in range(num_ofdm_symbols):
        start_bit = block_idx * bits_per_ofdm_symbol
        end_bit = min(start_bit + bits_per_ofdm_symbol, total_bits)
        bits_block = bits[start_bit:end_bit]
        
        # Pad si es necesario para completar el bloque
        if len(bits_block) < bits_per_ofdm_symbol:
            bits_block = np.concatenate([bits_block, np.zeros(bits_per_ofdm_symbol - len(bits_block), dtype=int)])
        
        try:
            # Simular transmisión de este bloque
            results = simulate_spatial_multiplexing(
                bits=bits_block,
                num_tx=num_tx,
                num_rx=num_rx,
                rank=rank,
                detector_type=detector_type,
                modulation=modulation,
                snr_db=snr_db,
                channel_type='rayleigh_mp',
                itu_profile='Pedestrian_A',
                velocity_kmh=3,
                frequency_ghz=2.0,
                enable_csi_feedback=True,
                enable_parallel=True
            )
            
            # Acumular bits recibidos
            bits_rx_block = results['bits_received_array']
            all_bits_rx.append(bits_rx_block)
            total_errors += results['bit_errors']
            
        except Exception as e:
            print(f"        [ERROR] Bloque {block_idx+1}/{num_ofdm_symbols}: {e}")
            # Usar bits con errores como fallback
            all_bits_rx.append(np.random.randint(0, 2, len(bits_block)))
            total_errors += len(bits_block) // 2
    
    # Concatenar todos los bits recibidos
    bits_rx = np.concatenate(all_bits_rx)[:total_bits]
    
    # Calcular BER total
    ber = total_errors / total_bits if total_bits > 0 else 0.0
    
    # Reconstruir imagen usando ImageProcessor
    img_rx = ImageProcessor.bits_to_image(bits_rx, metadata)
    
    # Calcular PSNR usando ImageProcessor
    psnr = ImageProcessor.calculate_psnr_bits(bits, bits_rx)
    
    # Obtener rank usado del último bloque
    rank_used = results['rank'] if 'results' in locals() else 1
    
    return {
        'img_rx': img_rx,
        'config': f"{num_tx}x{num_rx}",
        'num_tx': num_tx,
        'num_rx': num_rx,
        'rank': rank_used,
        'detector': detector_type,
        'snr_db': snr_db,
        'ber': ber,
        'psnr': psnr,
        'bits_transmitted': total_bits,
        'bit_errors': total_errors
    }


def test_all_configurations(image_path='img/entre-ciel-et-terre.jpg', snr_db=None):
    """
    Prueba todas las configuraciones y genera UNA imagen con todas las comparaciones.
    
    Formato de salida:
    - Primera columna: Imagen TX original
    - Fila 1 (MMSE): 2x2, 4x2, 4x4, 8x4
    - Fila 2 (SIC):  2x2, 4x2, 4x4, 8x4
    
    Args:
        image_path: Ruta de la imagen a transmitir
        snr_db: SNR en dB. Si es None, usa SNR_DB global
               (recomendado: 25-30 para 64-QAM, 15-20 para 16-QAM, 5-10 para BER alto)
    """
    # Usar SNR global si no se especifica
    if snr_db is None:
        snr_db = SNR_DB
    
    print(f"\n{'#'*80}")
    print(f"  TEST COMPLETO: SPATIAL MULTIPLEXING (TM4)")
    print(f"  Una sola imagen de salida con todas las configuraciones")
    print(f"  SNR: {snr_db} dB (variable global SNR_DB = {SNR_DB} dB)")
    print(f"{'#'*80}\n")
    
    output_dir = 'results/mimo'
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar imagen original UNA VEZ usando ImageProcessor
    print("[1/3] Cargando imagen original...")
    bits, metadata = ImageProcessor.image_to_bits(image_path)
    img_tx_original = ImageProcessor.load_image_pil(image_path)
    print(f"      Imagen: {metadata['width']}x{metadata['height']} pixels, {len(bits):,} bits")
    
    # Configuraciones: una lista por detector
    antenna_configs = [
        (2, 2),
        (4, 2),
        (4, 4),
        (8, 4)
    ]
    
    detectors = ['MMSE', 'SIC']
    
    # Almacenar resultados
    results_grid = {
        'MMSE': [],
        'SIC': []
    }
    
    # Ejecutar todas las configuraciones
    print(f"\n[2/3] Ejecutando simulaciones...")
    total = len(antenna_configs) * len(detectors)
    idx = 0
    
    for detector in detectors:
        for num_tx, num_rx in antenna_configs:
            idx += 1
            print(f"\n  [{idx}/{total}] {num_tx}x{num_rx} MIMO - {detector}...")
            
            try:
                result = test_spatial_multiplexing_single(
                    bits=bits,
                    metadata=metadata,
                    num_tx=num_tx,
                    num_rx=num_rx,
                    rank='adaptive',
                    detector_type=detector,
                    snr_db=snr_db,
                    modulation='64-QAM',
                    output_dir=output_dir
                )
                results_grid[detector].append(result)
                print(f"      BER={result['ber']:.4e}, Rank={result['rank']}, PSNR={result['psnr']:.2f}dB")
                
            except Exception as e:
                print(f"      [ERROR] {e}")
                import traceback
                traceback.print_exc()
                # Agregar resultado vacío
                results_grid[detector].append(None)
    
    # Crear imagen única con todas las comparaciones
    print(f"\n[3/3] Generando imagen unificada...")
    
    # Layout: 2 filas (MMSE, SIC) × 5 columnas (TX original + 4 configuraciones)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for row_idx, detector in enumerate(detectors):
        # Primera columna: TX original
        axes[row_idx, 0].imshow(img_tx_original, cmap='gray')
        axes[row_idx, 0].set_title(f'TX Original\n{metadata["width"]}x{metadata["height"]}px', 
                                     fontsize=10, fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        # Columnas 1-4: Resultados RX para cada configuración
        for col_idx, result in enumerate(results_grid[detector]):
            ax = axes[row_idx, col_idx + 1]
            
            if result is not None:
                ax.imshow(result['img_rx'], cmap='gray')
                title = (f"{result['config']} | Rank {result['rank']}\n"
                        f"BER={result['ber']:.2e}\n"
                        f"PSNR={result['psnr']:.1f}dB")
                ax.set_title(title, fontsize=9, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'ERROR', ha='center', va='center', fontsize=12)
                ax.set_title('Failed', fontsize=9)
            
            ax.axis('off')
        
        # Etiqueta de fila
        axes[row_idx, 0].text(-0.1, 0.5, f'{detector}', 
                               transform=axes[row_idx, 0].transAxes,
                               fontsize=14, fontweight='bold', rotation=90,
                               va='center', ha='right')
    
    plt.suptitle(f'Spatial Multiplexing TM4 - Todas las Configuraciones (SNR={snr_db}dB)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_file = os.path.join(output_dir, 'all_configurations_comparison.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n[DONE] Imagen unificada guardada: {output_file}")
    
    # Crear también el resumen de métricas
    all_results = results_grid['MMSE'] + results_grid['SIC']
    all_results = [r for r in all_results if r is not None]
    
    print(f"\n{'#'*80}")
    print(f"  RESUMEN DE RESULTADOS")
    print(f"{'#'*80}\n")
    print(f"{'Config':<15} {'Rank':<6} {'Detector':<8} {'BER':<12} {'PSNR (dB)'}")
    print(f"{'-'*80}")
    for res in all_results:
        print(f"{res['config']:<15} {res['rank']:<6} {res['detector']:<8} "
              f"{res['ber']:<12.2e} {res['psnr']:<10.2f}")
    
    print(f"\n{'#'*80}\n")
    print(f"[SUCCESS] Test completado. Total configuraciones: {len(all_results)}")
    print(f"          Resultados en: {output_dir}/")
    
    return all_results


if __name__ == '__main__':
    # Verificar imagen de entrada
    image_path = 'img/entre-ciel-et-terre.jpg'
    
    if not os.path.exists(image_path):
        print(f"[ERROR] Imagen no encontrada: {image_path}")
        print(f"        Usando imagen de prueba sintética...")
        # Crear imagen de prueba
        os.makedirs('img', exist_ok=True)
        test_img = Image.new('L', (128, 128), color=128)
        # Agregar patrón
        for i in range(0, 128, 16):
            for j in range(0, 128, 16):
                test_img.putpixel((i, j), 255 if (i+j)//16 % 2 == 0 else 0)
        test_img.save(image_path)
        print(f"        Imagen de prueba creada: {image_path}")
    
    # Ejecutar test completo
    # Para modificar el SNR, edita la variable SNR_DB al inicio del archivo
    # Valores sugeridos:
    #   - SNR_DB = 5-10:  BER alto (~10-30%), para pruebas de robustez
    #   - SNR_DB = 15-20: BER medio (~1-5%), imagen reconocible con ruido
    #   - SNR_DB = 25-30: BER bajo (~0.001-0.1%), imagen clara
    print(f"\n[INFO] Usando SNR_DB = {SNR_DB} dB (configurado al inicio del archivo)")
    print(f"       Para cambiar el SNR, modifica la variable global SNR_DB\n")
    
    results = test_all_configurations(image_path=image_path)
    
    print("\n[SUCCESS] Test de Spatial Multiplexing completado [OK]")
    print(f"          Resultados en: results/mimo/")
    print(f"          SNR usado: {SNR_DB} dB\n")
