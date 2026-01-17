"""
Test Beamforming con Imagen - Comparación Múltiple
===================================================

Compara transmisión de imagen con diferentes configuraciones de antenas:
1. 2×1 TX Diversity (SFBC Alamouti) - baseline
2. 2×1 Beamforming (TM6)
3. 4×1 Beamforming (TM6)
4. 8×1 Beamforming (TM6)

Métricas:
- BER (Bit Error Rate)
- PSNR (Peak Signal-to-Noise Ratio)
- Ganancia de beamforming en dB
- Comparación visual en una sola imagen

Resultados guardados en: results/beamforming/
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LTEConfig
from core.ofdm_core import OFDMSimulator
from utils.image_processing import ImageProcessor


def test_beamforming_image():
    """
    Test principal: comparar beamforming vs sin beamforming.
    """
    print("\n" + "="*80)
    print("TEST BEAMFORMING - Transmisión de Imagen")
    print("="*80)
    
    # Crear directorio de resultados
    results_dir = Path("results/beamforming")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuración LTE
    config = LTEConfig(
        bandwidth=10,        # MHz
        modulation='64-QAM', # 64-QAM
        delta_f=15,          # kHz
        cp_type='normal'
    )
    
    print(f"\nConfiguración LTE:")
    print(f"  Bandwidth: {config.bandwidth} MHz")
    print(f"  Modulation: {config.modulation}-QAM")
    print(f"  Subcarrier spacing: {config.delta_f} kHz")
    print(f"  FFT size: {config.N}")
    
    # Cargar imagen
    image_path = "img/entre-ciel-et-terre.jpg"
    if not os.path.exists(image_path):
        print(f"\n[ERROR] Imagen no encontrada: {image_path}")
        print("Por favor, asegúrese de tener una imagen en img/")
        return
    
    print(f"\nCargando imagen: {image_path}")
    bits, metadata = ImageProcessor.image_to_bits(image_path)
    
    print(f"  Dimensiones: {metadata['width']}×{metadata['height']}")
    print(f"  Bits totales: {len(bits):,}")
    print(f"  Tamaño: {len(bits) / 8 / 1024:.2f} KB")
    
    # Parámetros de simulación
    snr_db = 15  # SNR medio para ver diferencias claras
    velocity_kmh = 3  # Pedestrian
    
    print(f"\nParámetros de simulación:")
    print(f"  SNR: {snr_db} dB")
    print(f"  Velocidad: {velocity_kmh} km/h")
    
    # Configuraciones a probar: múltiples RX (1, 2, 4)
    test_configs = [
        # Fila 1: 1 RX
        {'name': '2×1 TX Diversity (SFBC)', 'num_tx': 2, 'num_rx': 1, 'mode': 'diversity'},
        {'name': '2×1 Beamforming', 'num_tx': 2, 'num_rx': 1, 'mode': 'beamforming'},
        {'name': '4×1 Beamforming', 'num_tx': 4, 'num_rx': 1, 'mode': 'beamforming'},
        {'name': '8×1 Beamforming', 'num_tx': 8, 'num_rx': 1, 'mode': 'beamforming'},
        
        # Fila 2: 2 RX
        {'name': '2×2 TX Diversity (SFBC)', 'num_tx': 2, 'num_rx': 2, 'mode': 'diversity'},
        {'name': '2×2 Beamforming', 'num_tx': 2, 'num_rx': 2, 'mode': 'beamforming'},
        {'name': '4×2 Beamforming', 'num_tx': 4, 'num_rx': 2, 'mode': 'beamforming'},
        {'name': '8×2 Beamforming', 'num_tx': 8, 'num_rx': 2, 'mode': 'beamforming'},
        
        # Fila 3: 4 RX
        {'name': '2×4 TX Diversity (SFBC)', 'num_tx': 2, 'num_rx': 4, 'mode': 'diversity'},
        {'name': '2×4 Beamforming', 'num_tx': 2, 'num_rx': 4, 'mode': 'beamforming'},
        {'name': '4×4 Beamforming', 'num_tx': 4, 'num_rx': 4, 'mode': 'beamforming'},
        {'name': '8×4 Beamforming', 'num_tx': 8, 'num_rx': 4, 'mode': 'beamforming'},
    ]
    
    results = []
    images_received = []
    
    # ========================================================================
    # EJECUTAR SIMULACIONES
    # ========================================================================
    total_configs = len(test_configs)
    for idx, test_config in enumerate(test_configs, 1):
        print(f"\n" + "="*80)
        print(f"SIMULACIÓN {idx}/{total_configs}: {test_config['name']}")
        print("="*80)
        
        simulator = OFDMSimulator(
            config=config,
            mode='lte',
            enable_equalization=True,
            num_channels=1
        )
        
        num_tx = test_config['num_tx']
        num_rx = test_config['num_rx']
        mode = test_config['mode']
        
        # Ejecutar simulación según el modo
        if mode == 'diversity':
            # TX Diversity (SFBC Alamouti) - baseline comparable
            print(f"  Modo: SFBC Alamouti (TX Diversity)")
            sim_results = simulator.simulate_miso(
                bits=bits,
                snr_db=snr_db
            )
        else:
            # Beamforming
            print(f"  Modo: Beamforming TM6")
            sim_results = simulator.simulate_beamforming(
                bits=bits,
                snr_db=snr_db,
                num_tx=num_tx,
                num_rx=num_rx,
                codebook_type='TM6',
                velocity_kmh=velocity_kmh,
                update_mode='adaptive'
            )
        
        # Reconstruir imagen
        img_received = ImageProcessor.bits_to_image(
            sim_results['bits_received_array'], 
            metadata
        )
        
        # Guardar resultados
        results.append({
            'name': test_config['name'],
            'num_tx': num_tx,
            'num_rx': num_rx,
            'mode': mode,
            'ber': sim_results['ber'],
            'bit_errors': sim_results['bit_errors'],
            'psnr': ImageProcessor.calculate_psnr_bits(bits, sim_results['bits_received_array']),
            'gain_db': sim_results.get('beamforming_gain_db', 0.0)
        })
        images_received.append(img_received)
        
        print(f"\n  BER: {sim_results['ber']:.4e}")
        print(f"  Errores: {sim_results['bit_errors']:,} / {len(bits):,}")
        if 'beamforming_gain_db' in sim_results:
            print(f"  Ganancia BF: {sim_results['beamforming_gain_db']:.2f} dB")
    
    # ========================================================================
    # CREAR IMAGEN COMPARATIVA
    # ========================================================================
    print(f"\n" + "="*80)
    print("CREANDO IMAGEN COMPARATIVA")
    print("="*80)
    
    from PIL import Image, ImageDraw, ImageFont
    
    # Cargar imagen original
    img_original = ImageProcessor.load_image_pil(image_path)
    
    # Dimensiones
    img_width = img_original.width
    img_height = img_original.height
    
    # Organizar en 3 filas: 1 RX, 2 RX, 4 RX (cada fila tiene original + 4 configs)
    images_per_row = 5  # Original + 4 configs
    num_rows = 3
    
    # Crear canvas para todas las imágenes
    margin = 20
    text_height = 100
    canvas_width = images_per_row * (img_width + margin) + margin
    canvas_height = num_rows * (img_height + text_height + margin) + margin
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Intentar cargar fuente, si falla usar default
    try:
        font_title = ImageFont.truetype("arial.ttf", 16)
        font_metrics = ImageFont.truetype("arial.ttf", 11)
        font_row_title = ImageFont.truetype("arialbd.ttf", 18)
    except:
        font_title = ImageFont.load_default()
        font_metrics = ImageFont.load_default()
        font_row_title = ImageFont.load_default()
    
    # Procesar por filas (cada fila = una configuración de RX)
    row_configs = [
        {'rx': 1, 'title': '1 RX ANTENNA', 'indices': [0, 1, 2, 3]},
        {'rx': 2, 'title': '2 RX ANTENNAS', 'indices': [4, 5, 6, 7]},
        {'rx': 4, 'title': '4 RX ANTENNAS', 'indices': [8, 9, 10, 11]},
    ]
    
    for row_idx, row_config in enumerate(row_configs):
        y_pos = margin + row_idx * (img_height + text_height + margin)
        
        # Título de la fila (a la izquierda)
        row_title_y = y_pos + img_height // 2
        
        # Colocar imagen original al inicio de cada fila
        x_pos = margin
        canvas.paste(img_original, (x_pos, y_pos))
        
        # Texto para original
        text_y = y_pos + img_height + 10
        draw.text((x_pos + img_width//2, text_y), "ORIGINAL", 
                  fill='black', font=font_title, anchor='mt')
        
        # Colocar imágenes de esta fila
        for col_idx, result_idx in enumerate(row_config['indices']):
            img_rx = images_received[result_idx]
            result = results[result_idx]
            
            x_pos = margin + (col_idx + 1) * (img_width + margin)
            canvas.paste(img_rx, (x_pos, y_pos))
            
            # Texto con nombre y métricas
            text_y = y_pos + img_height + 10
            name_short = result['name'].split(' ')[0]  # "2×1", "4×1", etc.
            mode_short = "SFBC" if result['mode'] == 'diversity' else "BF"
            display_name = f"{name_short} {mode_short}"
            
            draw.text((x_pos + img_width//2, text_y), display_name,
                      fill='black', font=font_title, anchor='mt')
            
            # Métricas (sin anchor para multilinea)
            metrics_y = text_y + 25
            draw.text((x_pos + 5, metrics_y), 
                      f"BER: {result['ber']:.2e}",
                      fill='blue', font=font_metrics)
            draw.text((x_pos + 5, metrics_y + 15), 
                      f"PSNR: {result['psnr']:.1f} dB",
                      fill='blue', font=font_metrics)
            if result['gain_db'] > 0:
                draw.text((x_pos + 5, metrics_y + 30), 
                          f"Gain: {result['gain_db']:.1f} dB",
                          fill='green', font=font_metrics)
        
        # Añadir título de fila al lado izquierdo
        draw.text((5, row_title_y), row_config['title'],
                  fill='red', font=font_row_title, anchor='lm')
    
    # Guardar imagen comparativa
    comparison_path = results_dir / f"comparacion_completa_snr{snr_db}dB.png"
    canvas.save(str(comparison_path))
    print(f"  ✓ Imagen comparativa guardada: {comparison_path}")
    
    # ========================================================================
    # MOSTRAR TABLA DE RESULTADOS
    # ========================================================================
    print(f"\n" + "="*80)
    print("TABLA DE RESULTADOS")
    print("="*80)
    
    # Organizar por número de RX
    for rx_count in [1, 2, 4]:
        print(f"\n{'='*65}")
        print(f"  {rx_count} RX ANTENNA(S)")
        print(f"{'='*65}")
        print(f"{'Configuración':<25} {'BER':<12} {'PSNR (dB)':<12} {'Gain (dB)':<12}")
        print("-" * 65)
        
        # Filtrar resultados para este RX count
        rx_results = [r for r in results if r['num_rx'] == rx_count]
        
        # Baseline para esta fila (primera config = TX Diversity)
        baseline_ber = rx_results[0]['ber']
        baseline_psnr = rx_results[0]['psnr']
        
        for result in rx_results:
            print(f"{result['name']:<25} {result['ber']:<12.2e} {result['psnr']:<12.1f} ", end='')
            
            if result['mode'] == 'diversity':
                print(f"{'(baseline)':<12}")
            else:
                print(f"{result['gain_db']:<12.1f}")
    
    # Mejoras respecto a baselines
    print(f"\n" + "="*80)
    print("MEJORAS RESPECTO A TX DIVERSITY (baseline de cada fila)")
    print("="*80)
    
    for rx_count in [1, 2, 4]:
        rx_results = [r for r in results if r['num_rx'] == rx_count]
        baseline_ber = rx_results[0]['ber']
        baseline_psnr = rx_results[0]['psnr']
        
        print(f"\n{'='*65}")
        print(f"  {rx_count} RX ANTENNA(S) - Mejoras respecto a {rx_results[0]['name']}")
        print(f"{'='*65}")
        
        for result in rx_results[1:]:  # Skip baseline itself
            ber_improvement = baseline_ber / result['ber'] if result['ber'] > 0 else np.inf
            psnr_improvement = result['psnr'] - baseline_psnr
            
            print(f"\n{result['name']}:")
            print(f"  BER mejora: {ber_improvement:.2f}× ({10*np.log10(ber_improvement):.2f} dB)")
            print(f"  PSNR mejora: {psnr_improvement:+.2f} dB")
            print(f"  Array gain: {result['gain_db']:.2f} dB")
    
    # ========================================================================
    # GUARDAR RESULTADOS EN ARCHIVO
    # ========================================================================
    results_file = results_dir / "resultados_comparacion.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("RESULTADOS TEST BEAMFORMING - COMPARACIÓN MÚLTIPLE\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURACIÓN\n")
        f.write(f"  Imagen: {image_path}\n")
        f.write(f"  Dimensiones: {metadata['width']}×{metadata['height']}\n")
        f.write(f"  Bits: {len(bits):,}\n")
        f.write(f"  Modulación: {config.modulation}\n")
        f.write(f"  Bandwidth: {config.bandwidth} MHz\n")
        f.write(f"  SNR: {snr_db} dB\n")
        f.write(f"  Velocidad: {velocity_kmh} km/h\n\n")
        
        f.write("CONFIGURACIONES PROBADAS\n")
        f.write("-" * 80 + "\n")
        
        # Organizar por RX
        for rx_count in [1, 2, 4]:
            f.write(f"\n{'='*80}\n")
            f.write(f"{rx_count} RX ANTENNA(S)\n")
            f.write(f"{'='*80}\n")
            
            rx_results = [r for r in results if r['num_rx'] == rx_count]
            
            for result in rx_results:
                f.write(f"\n{result['name']} ({result['num_tx']}×{result['num_rx']}):\n")
                f.write(f"  BER: {result['ber']:.4e}\n")
                f.write(f"  Errores: {result['bit_errors']:,} / {len(bits):,}\n")
                f.write(f"  PSNR: {result['psnr']:.2f} dB\n")
                if result['gain_db'] > 0:
                    f.write(f"  Array Gain: {result['gain_db']:.2f} dB\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("MEJORAS RESPECTO A TX DIVERSITY (baseline de cada fila)\n")
        f.write("="*80 + "\n")
        
        for rx_count in [1, 2, 4]:
            rx_results = [r for r in results if r['num_rx'] == rx_count]
            baseline_ber = rx_results[0]['ber']
            baseline_psnr = rx_results[0]['psnr']
            
            f.write(f"\n{rx_count} RX - Baseline: {rx_results[0]['name']}\n")
            f.write("-" * 80 + "\n")
            
            for result in rx_results[1:]:
                ber_improvement = baseline_ber / result['ber'] if result['ber'] > 0 else np.inf
                psnr_improvement = result['psnr'] - baseline_psnr
                
                f.write(f"\n{result['name']}:\n")
                f.write(f"  BER mejora: {ber_improvement:.2f}× ({10*np.log10(ber_improvement):.2f} dB)\n")
                f.write(f"  PSNR mejora: {psnr_improvement:+.2f} dB\n")
                f.write(f"  Array gain: {result['gain_db']:.2f} dB\n")
    
    print(f"\n[✓] Resultados guardados en: {results_file}")
    
    print(f"\n" + "="*80)
    print("TEST COMPLETADO")
    print("="*80)
    print(f"\nArchivo principal generado:")
    print(f"  - {comparison_path.name}")
    print(f"\nArchivo de texto:")
    print(f"  - {results_file.name}")
    print()


if __name__ == "__main__":
    test_beamforming_image()
